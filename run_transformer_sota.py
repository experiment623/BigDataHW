"""
GPU Transformer fine-tuning for ChiFraud.

Run this with elevated execution when CUDA is sandbox-blocked:

  .venv/bin/python -u run_transformer_sota.py --limit-train 2000 --limit-eval 1000 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from run_sota import (
    DATASET_DIR,
    LABELS,
    OUTPUT_DIR,
    PRED_DIR,
    apply_exact_fallback,
    build_exact_label_map,
    normalize_text,
)


MODEL_OUT_DIR = Path("models") / "sota_transformer"
RANDOM_SEED = 42


@dataclass
class TransformerMetrics:
    experiment: str
    split: str
    epoch: int
    protocol: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    exact_matches: int
    n_samples: int
    train_time_s: float
    predict_time_s: float
    config_json: str
    per_class_f1_json: str
    per_class_recall_json: str


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(DATASET_DIR / name, sep="\t", encoding="utf-8")
    return [normalize_text(t) for t in df["Text"].astype(str).tolist()], df["Label_id"].astype(int).to_numpy()


def stratified_limit(texts: list[str], y: np.ndarray, limit: int) -> tuple[list[str], np.ndarray]:
    if not limit or limit >= len(y):
        return texts, y
    idx = np.arange(len(y))
    _, keep = train_test_split(
        idx,
        test_size=limit,
        stratify=y,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    keep = np.sort(keep)
    return [texts[i] for i in keep], y[keep]


def collate_batch(tokenizer, max_length: int):
    def _collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return _collate


def class_weights(y: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "balanced":
        weights = counts.sum() / (len(LABELS) * counts)
    elif mode == "sqrt":
        weights = np.sqrt(counts.sum() / (len(LABELS) * counts))
    else:
        raise ValueError(f"unknown class weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(
    experiment: str,
    split: str,
    epoch: int,
    protocol: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exact_matches: int,
    train_time_s: float,
    predict_time_s: float,
    config: dict,
) -> TransformerMetrics:
    per_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return TransformerMetrics(
        experiment=experiment,
        split=split,
        epoch=epoch,
        protocol=protocol,
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        exact_matches=exact_matches,
        n_samples=len(y_true),
        train_time_s=train_time_s,
        predict_time_s=predict_time_s,
        config_json=json.dumps(config, ensure_ascii=False, sort_keys=True),
        per_class_f1_json=json.dumps({str(k): float(v) for k, v in zip(LABELS, per_class_f1)}, sort_keys=True),
        per_class_recall_json=json.dumps({str(k): float(v) for k, v in zip(LABELS, per_class_recall)}, sort_keys=True),
    )


def evaluate(
    model,
    tokenizer,
    texts: list[str],
    y_true: np.ndarray,
    args,
    device: torch.device,
    exact_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    ds = TextDataset(texts, y_true)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch(tokenizer, args.max_length),
    )
    preds = []
    scores = []
    start = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
            prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            scores.append(prob)
            preds.append(prob.argmax(axis=1))
    y_pred = np.concatenate(preds)
    score = np.vstack(scores)
    if not args.no_exact_fallback:
        y_pred, score, exact_matches = apply_exact_fallback(texts, y_pred, score, exact_map)
    else:
        exact_matches = 0
    return y_pred, score, exact_matches, time.time() - start


def save_metrics(rows: list[TransformerMetrics]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "transformer_results.csv"
    df_new = pd.DataFrame([asdict(r) for r in rows])
    if out_path.exists():
        df_old = pd.read_csv(out_path, encoding="utf-8-sig")
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_predictions(experiment: str, split: str, y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": np.arange(len(y_true)), "label_true": y_true, "label_pred": y_pred})
    for label in LABELS:
        df[f"score_{label}"] = score[:, label]
    df.to_csv(PRED_DIR / f"{experiment}_{split}.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Chinese Transformer on ChiFraud.")
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--run-name", default="macbert_base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--no-exact-fallback", action="store_true")
    parser.add_argument("--train-with-val", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate t2023 after each epoch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda={torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        torch.set_float32_matmul_precision("high")

    train_texts, y_train = load_split("ChiFraud_train.csv")
    val_texts, y_val = load_split("ChiFraud_t2022.csv")
    test_texts, y_test = load_split("ChiFraud_t2023.csv")
    train_texts, y_train = stratified_limit(train_texts, y_train, args.limit_train)
    if args.limit_eval:
        val_texts, y_val = stratified_limit(val_texts, y_val, args.limit_eval)
        test_texts, y_test = stratified_limit(test_texts, y_test, args.limit_eval)

    fit_texts, fit_y = train_texts, y_train
    eval_splits = [("val", val_texts, y_val)]
    protocol = "train->val"
    if args.eval_test:
        eval_splits.append(("test", test_texts, y_test))
        protocol = "train->val/test"
    if args.train_with_val:
        fit_texts = train_texts + val_texts
        fit_y = np.concatenate([y_train, y_val])
        eval_splits = [("test", test_texts, y_test)]
        protocol = "train+val->test"

    print(f"Fit={len(fit_texts)} Val={len(val_texts)} Test={len(test_texts)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={i: str(i) for i in LABELS},
        label2id={str(i): i for i in LABELS},
    ).to(device)

    train_loader = DataLoader(
        TextDataset(fit_texts, fit_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch(tokenizer, args.max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_weight = class_weights(fit_y, args.class_weight, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    exact_map = build_exact_label_map(fit_texts, fit_y)

    config = vars(args) | {"device": str(device), "total_steps": total_steps}
    all_metrics: list[TransformerMetrics] = []
    best_val_f1 = -1.0
    run_id = args.run_name.replace("/", "_")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * args.grad_accum
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                if device.type == "cuda":
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running/step:.4f} max_mem={mem:.2f}GB", flush=True)
                else:
                    print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running/step:.4f}", flush=True)

        elapsed = time.time() - train_start
        for split, texts, y_true in eval_splits:
            y_pred, score, exact_matches, predict_time = evaluate(model, tokenizer, texts, y_true, args, device, exact_map)
            metric = compute_metrics(
                experiment=run_id,
                split=split,
                epoch=epoch,
                protocol=protocol + " exact_fallback=" + str(not args.no_exact_fallback),
                y_true=y_true,
                y_pred=y_pred,
                exact_matches=exact_matches,
                train_time_s=elapsed,
                predict_time_s=predict_time,
                config=config,
            )
            all_metrics.append(metric)
            print(
                f"{split} epoch={epoch}: acc={metric.accuracy:.6f} p={metric.precision_macro:.6f} "
                f"r={metric.recall_macro:.6f} f1={metric.f1_macro:.6f} wf1={metric.f1_weighted:.6f} "
                f"exact={exact_matches}",
                flush=True,
            )
            worst = sorted(json.loads(metric.per_class_f1_json).items(), key=lambda item: item[1])[:3]
            print("worst_f1:", ", ".join(f"{k}={v:.4f}" for k, v in worst), flush=True)
            if args.save_predictions:
                save_predictions(f"{run_id}_epoch{epoch}", split, y_true, y_pred, score)
            if split == "val" and metric.f1_macro > best_val_f1:
                best_val_f1 = metric.f1_macro
                save_dir = MODEL_OUT_DIR / f"{run_id}_best"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"saved best val model to {save_dir}", flush=True)

    save_metrics(all_metrics)
    print(f"saved metrics: {OUTPUT_DIR / 'transformer_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
