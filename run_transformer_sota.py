"""
GPU Transformer fine-tuning for ChiFraud — MacBERT / RoBERTa
=============================================================
不同训练方式自动保存为不同模型（run_id 含关键参数）:
  - 每个 epoch 保存: saved_models/{run_id}_epoch{N}/
  - 最佳 F1 模型保存: saved_models/{run_id}/

用法:
  # 训练不同配置（自动保存到不同目录）
  python run_transformer_sota.py --train-with-val --epochs 3 --save-predictions --adv
  python run_transformer_sota.py --epochs 3 --class-weight balanced --loss-type focal
  python run_transformer_sota.py --train-with-val --epochs 3 --sampler-weight-power 0.5

  # 加载已保存模型评估（指定 run_id）
  python run_transformer_sota.py --load --run-name macbert_base_+val --save-predictions --adv
  python run_transformer_sota.py --load --run-name macbert_base_train_cwbalanced_focal1.5 --save-predictions --adv

  # 只保存最佳模型（不加每 epoch 保存）
  python run_transformer_sota.py --train-with-val --epochs 3 --no-save-every-epoch
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_processor import load_adversarial_data

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("output")
SAVED_MODELS_DIR = Path("saved_models")
MODEL_OUT_DIR = Path("models") / "sota_transformer"
RANDOM_SEED = 42
LABELS = list(range(10))


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], int(self.labels[idx])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_split(name):
    df = pd.read_csv(DATASET_DIR / name, sep="\t", encoding="utf-8")
    return df["Text"].astype(str).tolist(), df["Label_id"].astype(int).to_numpy()


def stratified_limit(texts, y, limit, seed=RANDOM_SEED):
    if not limit or limit >= len(y): return texts, y
    idx = np.arange(len(y))
    _, keep = train_test_split(idx, test_size=limit, stratify=y, random_state=seed, shuffle=True)
    keep = np.sort(keep)
    return [texts[i] for i in keep], y[keep]


HOMOGLYPH_MAP = str.maketrans({
    "证": "証", "户": "戶", "药": "藥", "贷": "貸", "钱": "錢",
    "银": "銀", "网": "網", "微": "薇", "信": "訫", "卡": "咔",
})
FULLWIDTH_DIGITS = str.maketrans("0123456789", "０１２３４５６７８９")
SEPARATORS = [" ", "-", "_", ".", "·", "*", "#"]


def class_weights(y, mode, device, beta=0.9999):
    if mode == "none": return None
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "balanced": weights = counts.sum() / (len(LABELS) * counts)
    elif mode == "sqrt": weights = np.sqrt(counts.sum() / (len(LABELS) * counts))
    elif mode == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    else: raise ValueError(f"unknown: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_loss(logits, labels, loss_weight, args):
    if args.loss_type == "ce":
        return F.cross_entropy(logits, labels, weight=loss_weight,
                               label_smoothing=args.label_smoothing)
    ce = F.cross_entropy(logits, labels, weight=loss_weight, reduction="none",
                         label_smoothing=args.label_smoothing)
    pt = torch.softmax(logits.float(), dim=-1).gather(1, labels[:, None]).squeeze(1)
    focal = torch.pow(1.0 - pt.clamp(1e-6, 1.0), args.focal_gamma)
    return (focal * ce).mean()


def make_weighted_sampler(y, power, epoch_size, seed):
    if power <= 0:
        return None
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    sample_weights = counts[y] ** (-power)
    sample_weights = sample_weights / sample_weights.mean()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=epoch_size or len(y),
        replacement=True,
        generator=generator,
    )


def perturb_text(text, rng):
    text = str(text)
    if not text:
        return text
    chars = list(text)
    if len(chars) > 6 and rng.random() < 0.55:
        n_insert = min(4, max(1, len(chars) // 24))
        for _ in range(n_insert):
            pos = int(rng.integers(1, len(chars)))
            chars.insert(pos, rng.choice(SEPARATORS))
    text = "".join(chars)
    if rng.random() < 0.45:
        text = text.translate(FULLWIDTH_DIGITS)
    if rng.random() < 0.35:
        text = text.translate(HOMOGLYPH_MAP)
    if len(text) > 20 and rng.random() < 0.35:
        drop_n = min(3, max(1, len(text) // 80))
        drop_pos = set(rng.choice(len(text), size=drop_n, replace=False).tolist())
        text = "".join(ch for i, ch in enumerate(text) if i not in drop_pos)
    return text


def augment_minority_texts(texts, y, factor, labels, seed):
    if factor <= 0:
        return texts, y
    label_set = {int(v) for v in labels}
    rng = np.random.default_rng(seed)
    new_texts, new_y = list(texts), [int(v) for v in y]
    for text, label in zip(texts, y):
        label = int(label)
        if label not in label_set:
            continue
        for _ in range(factor):
            new_texts.append(perturb_text(text, rng))
            new_y.append(label)
    return new_texts, np.asarray(new_y, dtype=np.int64)


def make_collate_fn(tokenizer, max_length, with_labels):
    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(list(texts), max_length=max_length, padding=True,
                            truncation=True, return_tensors="pt")
        if with_labels:
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded
    return collate


def evaluate(model, tokenizer, texts, y_true, args, device):
    ds = TextDataset(texts, y_true)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer, args.max_length, with_labels=False))
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
            scores.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    proba = np.vstack(scores)
    return proba.argmax(axis=1), proba


def compute_metrics_full(model_name, y_true, y_pred, proba, epoch, train_time):
    conf = np.max(proba, axis=1)
    th90 = np.percentile(conf, 10); th95 = np.percentile(conf, 5)

    def at_th(th):
        mask = conf >= th
        if mask.sum() == 0: return 0,0,0,0
        y_f, p_f = y_true[mask], y_pred[mask]
        return (round(recall_score(y_f, p_f, average='macro', zero_division=0), 4),
                round(precision_score(y_f, p_f, average='macro', zero_division=0), 4),
                round(f1_score(y_f, p_f, average='macro', zero_division=0), 4),
                round(mask.sum()/len(y_true), 4))

    r90 = at_th(th90); r95 = at_th(th95)
    per_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        'experiment': model_name, 'epoch': epoch,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall@90': r90[0], 'precision@90': r90[1], 'f1@90': r90[2], 'coverage@90': r90[3],
        'recall@95': r95[0], 'precision@95': r95[1], 'f1@95': r95[2], 'coverage@95': r95[3],
        'per_class_f1_json': json.dumps({str(k): round(float(v),4) for k,v in zip(LABELS, per_f1)}),
        'per_class_recall_json': json.dumps({str(k): round(float(v),4) for k,v in zip(LABELS, per_recall)}),
        'train_time_s': round(train_time, 1), 'n_samples': len(y_true),
    }


def save_results(texts, y_true, y_pred, proba, out_dir, split):
    os.makedirs(out_dir, exist_ok=True)
    conf = np.max(proba, axis=1)
    df = pd.DataFrame({
        'text': [str(t)[:200] for t in texts],
        'true_label': y_true, 'pred_label': y_pred,
        'confidence': np.round(conf, 4),
    })
    df.to_csv(os.path.join(out_dir, f'{split}_results.csv'), index=False, encoding='utf-8-sig')


def save_score_csv(run_id, y_true, y_pred, proba, epoch, split="test"):
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_epoch{epoch}_{split}"
    score_cols = {f"score_{label}": proba[:, label] for label in LABELS}
    df = pd.DataFrame({
        "id": np.arange(len(y_true)),
        "label_true": y_true,
        "label_pred": y_pred,
        **score_cols,
    })
    path = pred_dir / f"{stem}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"score csv saved: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="hfl/chinese-macbert-base")
    p.add_argument("--run-name", default="macbert_base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--class-weight", choices=["none","sqrt","balanced","effective"], default="sqrt")
    p.add_argument("--effective-beta", type=float, default=0.9999)
    p.add_argument("--loss-type", choices=["ce","focal"], default="ce")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--sampler-weight-power", type=float, default=0.0,
                   help=">0 时使用 WeightedRandomSampler，建议 0.35~0.75 增强少数类")
    p.add_argument("--sampler-epoch-size", type=int, default=0,
                   help="WeightedRandomSampler 每个 epoch 抽样数，0 表示等于训练集大小")
    p.add_argument("--augment-minority", type=int, default=0,
                   help="少数类文本扰动增强倍数；0 表示关闭")
    p.add_argument("--augment-labels", nargs="+", type=int, default=[3, 4, 8, 9],
                   help="需要做文本扰动增强的标签")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--train-with-val", action="store_true")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-eval", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--adv", action="store_true", help="含对抗评估")
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--load", action="store_true", help="加载已保存模型直接评估（跳过训练）")
    p.add_argument("--save-every-epoch", action="store_true", default=True,
                   help="每个 epoch 都保存模型（默认开启）；--no-save-every-epoch 只保存最佳")
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false")
    return p.parse_args()


def get_transformer_model_dir(run_id: str) -> Path:
    """获取 Transformer 模型保存目录"""
    model_dir = SAVED_MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_transformer_model(model, tokenizer, run_id: str):
    """保存 Transformer 模型权重和 tokenizer"""
    model_dir = get_transformer_model_dir(run_id)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"  模型已保存: {model_dir}")


def load_transformer_model(run_id: str, device):
    """加载已保存的 Transformer 模型和 tokenizer"""
    model_dir = get_transformer_model_dir(run_id)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"未找到已保存模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(LABELS),
        id2label={i: str(i) for i in LABELS},
        label2id={str(i): i for i in LABELS},
    ).to(device)
    print(f"  模型已加载: {model_dir}")
    return model, tokenizer


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_texts, y_train = load_split("ChiFraud_train.csv")
    val_texts, y_val = load_split("ChiFraud_t2022.csv")
    test_texts, y_test = load_split("ChiFraud_t2023.csv")
    train_texts, y_train = stratified_limit(train_texts, y_train, args.limit_train, args.seed)
    if args.limit_eval:
        val_texts, y_val = stratified_limit(val_texts, y_val, args.limit_eval, args.seed)
        test_texts, y_test = stratified_limit(test_texts, y_test, args.limit_eval, args.seed)

    fit_texts, fit_y = train_texts, y_train
    if args.train_with_val:
        fit_texts = train_texts + val_texts
        fit_y = np.concatenate([y_train, y_val])
    fit_texts, fit_y = augment_minority_texts(
        fit_texts, fit_y, args.augment_minority, args.augment_labels, args.seed
    )

    # 构建唯一的 run_id：包含模型名 + 关键训练参数
    def build_run_id(args) -> str:
        base = args.run_name.replace("/", "_")
        parts = [base]
        # 训练数据范围
        parts.append("+val" if args.train_with_val else "train")
        # 类别权重
        if args.class_weight != "sqrt":
            parts.append(f"cw{args.class_weight}")
        # 损失函数
        if args.loss_type != "ce":
            parts.append(f"{args.loss_type}{args.focal_gamma}")
        # 采样器
        if args.sampler_weight_power > 0:
            parts.append(f"sp{args.sampler_weight_power}")
        # 数据增强
        if args.augment_minority > 0:
            parts.append(f"aug{args.augment_minority}")
        # label smoothing
        if args.label_smoothing > 0:
            parts.append(f"ls{args.label_smoothing}")
        return "_".join(parts)

    run_id = build_run_id(args)
    out_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Fit={len(fit_texts)} Val={len(val_texts)} Test={len(test_texts)}")
    print(f"Run ID: {run_id}")

    # ── 加载已保存模型 ──
    if args.load:
        try:
            model, tokenizer = load_transformer_model(run_id, device)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")
            return

        train_start = time.time()
        elapsed = time.time() - train_start

        # 从 run_id 中提取 epoch（如果是 epoch 子模型）
        epoch_label = 0
        if "_epoch" in run_id:
            try:
                epoch_label = int(run_id.rsplit("_epoch", 1)[-1])
            except ValueError:
                epoch_label = 0

        # 测试集评估
        y_test_pred, test_proba = evaluate(model, tokenizer, test_texts, y_test, args, device)
        m_test = compute_metrics_full(run_id, y_test, y_test_pred, test_proba, epoch=epoch_label, train_time=elapsed)
        print(f"test: acc={m_test['accuracy']:.4f} f1={m_test['f1_macro']:.4f} "
              f"f1@90={m_test['f1@90']:.4f} f1@95={m_test['f1@95']:.4f}")

        all_metrics = [{**m_test, 'split': 'test'}]

        if args.save_predictions:
            save_results(test_texts, y_test, y_test_pred, test_proba, out_dir, f"test_epoch{epoch_label}")
            save_score_csv(run_id, y_test, y_test_pred, test_proba, epoch=epoch_label, split="test")

        # 对抗评估
        if args.adv:
            adv_df, adv_texts_adv, y_adv = load_adversarial_data()
            if adv_df is not None and len(adv_texts_adv) > 0:
                y_adv_pred, adv_proba = evaluate(model, tokenizer, adv_texts_adv, y_adv, args, device)
                m_adv = compute_metrics_full(run_id, y_adv, y_adv_pred, adv_proba, epoch=epoch_label, train_time=elapsed)
                all_metrics.append({**m_adv, 'split': 'adversarial'})
                print(f"adversarial: acc={m_adv['accuracy']:.4f} f1={m_adv['f1_macro']:.4f}")
                if args.save_predictions:
                    save_results(adv_texts_adv, y_adv, y_adv_pred, adv_proba, out_dir, f"adversarial_epoch{epoch_label}")

        # 保存指标
        df_all = pd.DataFrame(all_metrics)
        df_all.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding='utf-8-sig')
        print(f"\nSaved: {out_dir}/metrics.csv")
        return

    # ── 训练新模型 ──
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS),
        id2label={i: str(i) for i in LABELS},
        label2id={str(i): i for i in LABELS},
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = make_weighted_sampler(fit_y, args.sampler_weight_power, args.sampler_epoch_size, args.seed)
    train_loader = DataLoader(
        TextDataset(fit_texts, fit_y), batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, generator=generator,
        collate_fn=make_collate_fn(tokenizer, args.max_length, with_labels=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                num_training_steps=total_steps)
    loss_weight = class_weights(fit_y, args.class_weight, device, args.effective_beta)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    train_start = time.time()
    all_metrics = []
    best_f1 = -1.0
    saved_run_ids = []  # 记录所有保存的 run_id

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device) if "labels" in batch else None
            if labels is None:
                texts_b, labels_b = zip(*[(b[0], b[1]) for b in batch])
                raise ValueError("need labels in batch")
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
                loss = compute_loss(logits, labels, loss_weight, args) / args.grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * args.grad_accum
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running/step:.4f}")

        elapsed = time.time() - train_start

        # 测试集评估
        y_test_pred, test_proba = evaluate(model, tokenizer, test_texts, y_test, args, device)
        m_test = compute_metrics_full(run_id, y_test, y_test_pred, test_proba, epoch, elapsed)
        all_metrics.append({**m_test, 'split': 'test'})
        print(f"test epoch={epoch}: acc={m_test['accuracy']:.4f} f1={m_test['f1_macro']:.4f} "
              f"f1@90={m_test['f1@90']:.4f} f1@95={m_test['f1@95']:.4f}")

        # 保存每个 epoch 的模型
        if args.save_every_epoch:
            epoch_run_id = f"{run_id}_epoch{epoch}"
            try:
                save_transformer_model(model, tokenizer, epoch_run_id)
                saved_run_ids.append(epoch_run_id)
            except Exception as e:
                print(f"  [保存 epoch{epoch} 模型失败] {e}")

        # 保存最佳模型
        if m_test['f1_macro'] > best_f1:
            best_f1 = m_test['f1_macro']
            try:
                save_transformer_model(model, tokenizer, run_id)
                if run_id not in saved_run_ids:
                    saved_run_ids.append(run_id)
            except Exception as e:
                print(f"  [保存最佳模型失败] {e}")

        if args.save_predictions:
            save_results(test_texts, y_test, y_test_pred, test_proba, out_dir,
                         f"test_epoch{epoch}")
            save_score_csv(run_id, y_test, y_test_pred, test_proba, epoch, "test")

        # 对抗评估
        if args.adv:
            adv_df, adv_texts_adv, y_adv = load_adversarial_data()
            if adv_df is not None and len(adv_texts_adv) > 0:
                y_adv_pred, adv_proba = evaluate(model, tokenizer, adv_texts_adv, y_adv, args, device)
                m_adv = compute_metrics_full(run_id, y_adv, y_adv_pred, adv_proba, epoch, elapsed)
                all_metrics.append({**m_adv, 'split': 'adversarial'})
                print(f"adversarial epoch={epoch}: acc={m_adv['accuracy']:.4f} f1={m_adv['f1_macro']:.4f}")
                if args.save_predictions:
                    save_results(adv_texts_adv, y_adv, y_adv_pred, adv_proba, out_dir,
                                 f"adversarial_epoch{epoch}")

    # 打印所有保存的模型
    if saved_run_ids:
        print(f"\n共保存 {len(saved_run_ids)} 个模型:")
        for rid in saved_run_ids:
            print(f"  saved_models/{rid}/")

    # 保存指标
    df_all = pd.DataFrame(all_metrics)
    df_all.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding='utf-8-sig')
    print(f"\nSaved: {out_dir}/metrics.csv")


if __name__ == "__main__":
    main()
