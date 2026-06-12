"""
GPU Transformer fine-tuning for ChiFraud — MacBERT / RoBERTa
=============================================================
用法:
  python run_transformer_sota.py --train-with-val --epochs 2 --save-predictions --adv
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data_processor import load_adversarial_data

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("output")
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


def stratified_limit(texts, y, limit):
    if not limit or limit >= len(y): return texts, y
    idx = np.arange(len(y))
    _, keep = train_test_split(idx, test_size=limit, stratify=y, random_state=RANDOM_SEED, shuffle=True)
    keep = np.sort(keep)
    return [texts[i] for i in keep], y[keep]


def class_weights(y, mode, device):
    if mode == "none": return None
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "balanced": weights = counts.sum() / (len(LABELS) * counts)
    elif mode == "sqrt": weights = np.sqrt(counts.sum() / (len(LABELS) * counts))
    else: raise ValueError(f"unknown: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, tokenizer, texts, y_true, args, device):
    ds = TextDataset(texts, y_true)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=lambda batch: tokenizer([b[0] for b in batch],
                                                           max_length=args.max_length, padding=True,
                                                           truncation=True, return_tensors="pt"))
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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
        'recall@90': r90[1], 'precision@90': r90[0], 'f1@90': r90[2], 'coverage@90': r90[3],
        'recall@95': r95[1], 'precision@95': r95[0], 'f1@95': r95[2], 'coverage@95': r95[3],
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
    p.add_argument("--class-weight", choices=["none","sqrt","balanced"], default="sqrt")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--train-with-val", action="store_true")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-eval", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--adv", action="store_true", help="含对抗评估")
    p.add_argument("--eval-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_texts, y_train = load_split("ChiFraud_train.csv")
    val_texts, y_val = load_split("ChiFraud_t2022.csv")
    test_texts, y_test = load_split("ChiFraud_t2023.csv")
    train_texts, y_train = stratified_limit(train_texts, y_train, args.limit_train)
    if args.limit_eval:
        val_texts, y_val = stratified_limit(val_texts, y_val, args.limit_eval)
        test_texts, y_test = stratified_limit(test_texts, y_test, args.limit_eval)

    fit_texts, fit_y = train_texts, y_train
    if args.train_with_val:
        fit_texts = train_texts + val_texts
        fit_y = np.concatenate([y_train, y_val])

    print(f"Fit={len(fit_texts)} Val={len(val_texts)} Test={len(test_texts)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS),
        id2label={i: str(i) for i in LABELS},
        label2id={str(i): i for i in LABELS},
    ).to(device)

    train_loader = DataLoader(
        TextDataset(fit_texts, fit_y), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tokenizer([b[0] for b in batch],
                                           max_length=args.max_length, padding=True,
                                           truncation=True, return_tensors="pt"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                num_training_steps=total_steps)
    loss_weight = class_weights(fit_y, args.class_weight, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    run_id = args.run_name.replace("/", "_")
    out_dir = os.path.join(OUTPUT_DIR, run_id)
    train_start = time.time()
    all_metrics = []

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
                loss = loss_fn(logits, labels) / args.grad_accum
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

        if args.save_predictions:
            save_results(test_texts, y_test, y_test_pred, test_proba, out_dir,
                         f"test_epoch{epoch}")

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

    # 保存指标
    df_all = pd.DataFrame(all_metrics)
    df_all.to_csv(os.path.join(out_dir, "metrics.csv"), index=False, encoding='utf-8-sig')
    print(f"\nSaved: {out_dir}/metrics.csv")


if __name__ == "__main__":
    main()
