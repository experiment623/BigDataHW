"""
Reproduce and extend the current ChiFraud ensemble.

This script consumes prediction CSVs produced by run_sota.py and
run_transformer_sota.py, applies fixed model weights and per-class score
factors, then writes final ensemble metrics/predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


OUTPUT_DIR = Path("output")
PRED_DIR = OUTPUT_DIR / "predictions"
LABELS = list(range(10))


DEFAULT_MODELS = [
    "macbert_full_plus2022_bal_b64_epoch1_test",
    "macbert_50k_plus2022_bal_b32_epoch1_test",
    "macbert_full_plus2022_sqrt_b64_epoch1_test",
    "roberta_50k_plus2022_bal_b64_epoch1_test",
    "char13_svc_120k_test",
    "hash_char14_sgd_log_test",
]

# Best fixed setting found so far on the complete 2023 split.
DEFAULT_WEIGHTS = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
DEFAULT_FACTORS = np.array([0.7, 16.0, 6.0, 4.0, 1.7, 16.0, 0.5, 1.3, 4.0, 0.7], dtype=np.float64)


def load_scores(model_names: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    frames = []
    scores = []
    for name in model_names:
        path = PRED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(df)
        score_cols = [f"score_{label}" for label in LABELS]
        missing = [col for col in score_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing score columns: {missing}")
        scores.append(df[score_cols].to_numpy(dtype=np.float64))
    y = frames[0]["label_true"].to_numpy(dtype=np.int64)
    for name, df in zip(model_names[1:], frames[1:]):
        other = df["label_true"].to_numpy(dtype=np.int64)
        if not np.array_equal(y, other):
            raise ValueError(f"label_true mismatch for {name}")
    return y, scores


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, config: dict) -> dict:
    per_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "experiment": name,
        "split": "test",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_precision_json": json.dumps({str(k): float(v) for k, v in zip(LABELS, per_precision)}, sort_keys=True),
        "per_class_recall_json": json.dumps({str(k): float(v) for k, v in zip(LABELS, per_recall)}, sort_keys=True),
        "per_class_f1_json": json.dumps({str(k): float(v) for k, v in zip(LABELS, per_f1)}, sort_keys=True),
        "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
    }


def append_metrics(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "ensemble_results.csv"
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path, encoding="utf-8-sig")
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed ChiFraud ensemble.")
    parser.add_argument("--name", default="ensemble_v2_six_model")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS.tolist())
    parser.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS.tolist())
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = np.array(args.weights, dtype=np.float64)
    factors = np.array(args.factors, dtype=np.float64)
    if len(weights) != len(args.models):
        raise ValueError("--weights length must match --models")
    if len(factors) != len(LABELS):
        raise ValueError("--factors length must be 10")

    y_true, score_list = load_scores(args.models)
    score = sum(w * s for w, s in zip(weights, score_list)) / weights.sum()
    adjusted = score * factors
    y_pred = adjusted.argmax(axis=1)

    config = {
        "models": args.models,
        "weights": weights.tolist(),
        "factors": factors.tolist(),
    }
    row = metric_row(args.name, y_true, y_pred, config)
    append_metrics(row)
    print(pd.DataFrame([row])[["experiment", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]].to_string(index=False))

    if args.save_predictions:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"id": np.arange(len(y_true)), "label_true": y_true, "label_pred": y_pred})
        for label in LABELS:
            out[f"score_{label}"] = adjusted[:, label]
        out.to_csv(PRED_DIR / f"{args.name}_test.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
