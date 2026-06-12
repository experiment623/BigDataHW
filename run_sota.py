"""
Strong ChiFraud experiments.

This runner is intentionally separate from the original homework baselines.  It
focuses on fast, reproducible SOTA-seeking experiments:

  - strict split protocol: train on ChiFraud_train, select on t2022, report t2023
  - character n-gram TF-IDF models that are strong for Chinese short text
  - exact train-text fallback, reported separately, for duplicated train/test rows
  - unified CSV outputs for metrics and predictions
"""
from __future__ import annotations

import argparse
import json
import os
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
BASELINE_DIR = ROOT / "baseline"
OUTPUT_DIR = ROOT / "output"
PRED_DIR = OUTPUT_DIR / "predictions"
MODEL_DIR = ROOT / "models" / "sota"
RANDOM_SEED = 42
LABELS = list(range(10))


@dataclass
class Metrics:
    experiment: str
    split: str
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


def normalize_text(text: object) -> str:
    """Keep signal, but normalize width/case/whitespace for duplicate matching."""
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.lower().split())


def load_split(name: str) -> tuple[list[str], np.ndarray, list[str]]:
    path = DATASET_DIR / name
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    labels = df["Label_id"].astype(int).to_numpy()
    texts_raw = df["Text"].astype(str).tolist()
    texts_norm = [normalize_text(t) for t in texts_raw]
    invalid = sorted(set(labels.tolist()) - set(LABELS))
    if invalid:
        raise ValueError(f"{name} has labels outside 0-9: {invalid}")
    return texts_norm, labels, texts_raw


def build_exact_label_map(texts: Iterable[str], y: np.ndarray) -> dict[str, int]:
    """Map exact normalized training text to its majority label."""
    counts: dict[str, dict[int, int]] = {}
    for text, label in zip(texts, y):
        bucket = counts.setdefault(text, {})
        bucket[int(label)] = bucket.get(int(label), 0) + 1
    return {
        text: max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        for text, label_counts in counts.items()
    }


def apply_exact_fallback(
    texts: list[str],
    pred: np.ndarray,
    score: np.ndarray | None,
    exact_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray | None, int]:
    pred = pred.copy()
    score = None if score is None else score.copy()
    matches = 0
    for i, text in enumerate(texts):
        label = exact_map.get(text)
        if label is None:
            continue
        matches += 1
        pred[i] = label
        if score is not None:
            score[i, :] = 0.0
            score[i, label] = 1.0
    return pred, score, matches


def make_pipeline(name: str) -> tuple[Pipeline, dict]:
    configs = {
        "char13_svc_120k": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.98,
                max_features=120_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=5000,
                dual="auto",
            ),
        },
        "char14_svc_160k": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 4),
                min_df=3,
                max_df=0.98,
                max_features=160_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=6000,
                dual="auto",
            ),
        },
        "char15_svc_c1": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 5),
                min_df=2,
                max_df=0.98,
                max_features=300_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=6000,
                dual="auto",
            ),
        },
        "char25_svc_c1": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_df=0.98,
                max_features=300_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=6000,
                dual="auto",
            ),
        },
        "char15_svc_c2": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 5),
                min_df=2,
                max_df=0.98,
                max_features=400_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": LinearSVC(
                C=2.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=8000,
                dual="auto",
            ),
        },
        "char15_sgd_log": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 5),
                min_df=2,
                max_df=0.98,
                max_features=300_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": SGDClassifier(
                loss="log_loss",
                alpha=3e-6,
                penalty="l2",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=35,
                tol=1e-4,
                n_jobs=-1,
            ),
        },
        "hash_char14_sgd_log": {
            "vectorizer": HashingVectorizer(
                analyzer="char",
                ngram_range=(1, 4),
                n_features=2**19,
                alternate_sign=False,
                norm=None,
                dtype=np.float32,
            ),
            "transformer": TfidfTransformer(sublinear_tf=True),
            "classifier": SGDClassifier(
                loss="log_loss",
                alpha=1e-6,
                penalty="l2",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=45,
                tol=1e-4,
                n_jobs=-1,
            ),
        },
        "char15_lr_saga": {
            "vectorizer": TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 5),
                min_df=2,
                max_df=0.98,
                max_features=250_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "classifier": LogisticRegression(
                C=4.0,
                solver="saga",
                penalty="l2",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=700,
                n_jobs=-1,
                verbose=0,
            ),
        },
    }
    if name not in configs:
        raise KeyError(f"Unknown experiment {name!r}; choose from {sorted(configs)}")
    cfg = configs[name]
    steps = [("tfidf", cfg["vectorizer"])]
    if "transformer" in cfg:
        steps.append(("tfidf_norm", cfg["transformer"]))
    steps.append(("clf", cfg["classifier"]))
    pipe = Pipeline(steps)
    serializable = {
        "vectorizer": {
            key: value
            for key, value in cfg["vectorizer"].get_params().items()
            if key in {
                "analyzer",
                "ngram_range",
                "min_df",
                "max_df",
                "max_features",
                "sublinear_tf",
                "n_features",
                "alternate_sign",
            }
        },
        "classifier": {
            key: value
            for key, value in cfg["classifier"].get_params().items()
            if key in {"C", "alpha", "loss", "class_weight", "max_iter", "penalty", "solver"}
        },
    }
    return pipe, serializable


def model_scores(model: Pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    pred = model.predict(texts)
    clf = model.named_steps["clf"]
    if hasattr(model, "predict_proba") and hasattr(clf, "predict_proba"):
        return pred, model.predict_proba(texts)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(texts)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        return pred, softmax(decision, axis=1)
    return pred, None


def compute_metrics(
    experiment: str,
    split: str,
    protocol: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exact_matches: int,
    train_time_s: float,
    predict_time_s: float,
    config: dict,
) -> Metrics:
    per_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return Metrics(
        experiment=experiment,
        split=split,
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


def save_predictions(
    experiment: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None,
) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "id": np.arange(len(y_true)),
        "label_true": y_true,
        "label_pred": y_pred,
    })
    if score is not None and score.shape[1] == len(LABELS):
        for label in LABELS:
            df[f"score_{label}"] = score[:, label]
    df.to_csv(PRED_DIR / f"{experiment}_{split}.csv", index=False, encoding="utf-8-sig")


def baseline_thresholds() -> dict[str, float]:
    if not BASELINE_DIR.exists():
        return {}
    rows = []
    for path in BASELINE_DIR.glob("*.csv"):
        rows.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    return {metric: float(df[metric].max()) for metric in metrics if metric in df}


def append_results(results: list[Metrics]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sota_results.csv"
    df_new = pd.DataFrame([asdict(r) for r in results])
    if out_path.exists():
        df_old = pd.read_csv(out_path, encoding="utf-8-sig")
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def print_result_table(results: list[Metrics]) -> None:
    cols = [
        "experiment",
        "split",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "exact_matches",
        "train_time_s",
    ]
    df = pd.DataFrame([asdict(r) for r in results])
    print(df[cols].sort_values(["split", "f1_macro"], ascending=[True, False]).to_string(index=False))
    thresholds = baseline_thresholds()
    if thresholds:
        best_test = df[df["split"] == "test"].sort_values("f1_macro", ascending=False).head(1)
        if not best_test.empty:
            row = best_test.iloc[0]
            print("\nBaseline maxima:")
            for metric, value in thresholds.items():
                current = float(row[metric])
                flag = "PASS" if current > value else "MISS"
                print(f"  {metric:<16s} current={current:.6f} baseline={value:.6f} {flag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong ChiFraud SOTA experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["char15_svc_c1"],
        help="Experiment names. Use 'all' for the full CPU set.",
    )
    parser.add_argument("--no-exact-fallback", action="store_true", help="Disable exact train-text fallback.")
    parser.add_argument("--save-predictions", action="store_true", help="Write per-sample prediction CSV files.")
    parser.add_argument("--append-results", action="store_true", help="Append metrics to output/sota_results.csv.")
    parser.add_argument("--train-with-val", action="store_true", help="Train final model on train + t2022; report t2023 only.")
    parser.add_argument("--limit-train", type=int, default=0, help="Debug: use first N train rows only.")
    parser.add_argument("--limit-eval", type=int, default=0, help="Debug: use first N val/test rows only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiments == ["all"]:
        experiments = [
            "char13_svc_120k",
            "char14_svc_160k",
            "char15_svc_c1",
            "char25_svc_c1",
            "char15_svc_c2",
            "hash_char14_sgd_log",
            "char15_sgd_log",
            "char15_lr_saga",
        ]
    else:
        experiments = args.experiments

    train_texts, y_train, _ = load_split("ChiFraud_train.csv")
    val_texts, y_val, _ = load_split("ChiFraud_t2022.csv")
    test_texts, y_test, _ = load_split("ChiFraud_t2023.csv")

    if args.limit_train:
        train_texts = train_texts[: args.limit_train]
        y_train = y_train[: args.limit_train]
    if args.limit_eval:
        val_texts = val_texts[: args.limit_eval]
        y_val = y_val[: args.limit_eval]
        test_texts = test_texts[: args.limit_eval]
        y_test = y_test[: args.limit_eval]

    fit_texts = train_texts
    fit_y = y_train
    eval_splits = [("val", val_texts, y_val), ("test", test_texts, y_test)]
    protocol = "train->val/test"
    if args.train_with_val:
        fit_texts = train_texts + val_texts
        fit_y = np.concatenate([y_train, y_val])
        eval_splits = [("test", test_texts, y_test)]
        protocol = "train+val->test"

    print(
        f"Train={len(train_texts)} Val={len(val_texts)} Test={len(test_texts)} Fit={len(fit_texts)}",
        flush=True,
    )
    exact_map = build_exact_label_map(fit_texts, fit_y)
    print(f"Exact text map entries={len(exact_map)} fallback={'off' if args.no_exact_fallback else 'on'}", flush=True)

    all_results: list[Metrics] = []
    for experiment in experiments:
        print(f"\n=== {experiment} ===", flush=True)
        model, config = make_pipeline(experiment)
        start = time.time()
        model.fit(fit_texts, fit_y)
        train_time = time.time() - start
        vectorizer = model.named_steps["tfidf"]
        if hasattr(vectorizer, "vocabulary_"):
            vocab_size = len(vectorizer.vocabulary_)
        else:
            vocab_size = int(getattr(vectorizer, "n_features", 0))
        print(f"trained in {train_time:.1f}s, vocab_or_features={vocab_size}", flush=True)

        for split, texts, y_true in eval_splits:
            pred_start = time.time()
            y_pred, score = model_scores(model, texts)
            exact_matches = 0
            if not args.no_exact_fallback:
                y_pred, score, exact_matches = apply_exact_fallback(texts, y_pred, score, exact_map)
            predict_time = time.time() - pred_start
            metric = compute_metrics(
                experiment=experiment,
                split=split,
                protocol=protocol + " exact_fallback=" + str(not args.no_exact_fallback),
                y_true=y_true,
                y_pred=y_pred,
                exact_matches=exact_matches,
                train_time_s=train_time,
                predict_time_s=predict_time,
                config=config | {"vocab_size": vocab_size},
            )
            all_results.append(metric)
            print(
                f"{split}: acc={metric.accuracy:.6f} p={metric.precision_macro:.6f} "
                f"r={metric.recall_macro:.6f} f1={metric.f1_macro:.6f} "
                f"wf1={metric.f1_weighted:.6f} exact={exact_matches}",
                flush=True,
            )
            f1_by_class = json.loads(metric.per_class_f1_json)
            worst = sorted(f1_by_class.items(), key=lambda item: item[1])[:3]
            print("worst_f1:", ", ".join(f"{k}={v:.4f}" for k, v in worst), flush=True)
            if args.save_predictions:
                save_predictions(experiment, split, y_true, y_pred, score)

    print("\n=== Summary ===", flush=True)
    print_result_table(all_results)
    if args.append_results:
        append_results(all_results)
        print(f"\nSaved metrics: {OUTPUT_DIR / 'sota_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
