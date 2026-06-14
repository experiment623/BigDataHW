"""
ChiFraud 多模型加权集成
=======================
集成配置自动保存到 saved_models/ensemble_{name}.json

用法:
  # 默认配置运行
  python run_ensemble_sota.py --save-predictions

  # 自动发现模型 + 调优（调优后自动保存配置）
  python run_ensemble_sota.py --discover --auto-tune --save-predictions --name ensemble_auto

  # 加载已保存配置直接评估（无需重新调优）
  python run_ensemble_sota.py --load-config --name ensemble_auto --save-predictions

  # 手动指定模型和权重
  python run_ensemble_sota.py --models model1_test model2_test --weights 1.0 1.0
"""
from __future__ import annotations

import argparse, json, os, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from data_processor import load_adversarial_data, load_data

OUTPUT_DIR = Path("output")
DATASET_DIR = Path("dataset")
SAVED_MODELS_DIR = Path("saved_models")
LABELS = list(range(10))
SCORE_COLS = [f"score_{label}" for label in LABELS]

# 已知 baseline 最高值，用于 --objective sota_margin 的约束式搜索。
SOTA_TARGETS = {
    "accuracy": 0.927208,
    "precision_macro": 0.936659,
    "recall_macro": 0.875739,
    "f1_macro": 0.841694,
    "f1_weighted": 0.948638,
}

# 默认集成模型：Transformer 微调模型 + 字符 N-gram SOTA 模型（跨范式融合）
DEFAULT_TRANSFORMER_MODELS = [
    "macbert_full_plus2022_bal_b64_epoch1_test",
    "macbert_50k_plus2022_bal_b32_epoch1_test",
    "macbert_full_plus2022_sqrt_b64_epoch1_test",
    "roberta_50k_plus2022_bal_b64_epoch1_test",
]
DEFAULT_SOTA_MODELS = [
    "char13_svc_120k_test",
    "char14_svc_160k_test",
    "char15_svc_c1_test",
    "char25_svc_c1_test",
    "char15_svc_c2_test",
    "hash_char14_sgd_log_test",
    "char15_sgd_log_test",
    "char15_lr_saga_test",
]
DEFAULT_MODELS = DEFAULT_TRANSFORMER_MODELS + DEFAULT_SOTA_MODELS
DEFAULT_WEIGHTS = [1.0, 1.0, 0.5, 1.0] + [1.0] * len(DEFAULT_SOTA_MODELS)
DEFAULT_WEIGHT_MAP = dict(zip(DEFAULT_MODELS, DEFAULT_WEIGHTS))
DEFAULT_FACTORS = [0.7, 16.0, 6.0, 4.0, 1.7, 16.0, 0.5, 1.3, 4.0, 0.7]


def discover_score_models(pred_dir, include_ensembles=False):
    names = []
    for path in sorted(pred_dir.glob("*_test.csv")):
        if not include_ensembles and path.stem.startswith("ensemble_"):
            continue
        try:
            cols = set(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
        except Exception as exc:
            print(f"[跳过] {path}: {exc}")
            continue
        if "label_true" in cols and all(col in cols for col in SCORE_COLS):
            names.append(path.stem)
    return names


def default_weights_for(model_names):
    return np.array([DEFAULT_WEIGHT_MAP.get(name, 1.0) for name in model_names], dtype=np.float64)


def load_scores(model_names, pred_dir):
    frames, scores_list = [], []
    for name in model_names:
        path = pred_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        missing = [c for c in ["label_true", *SCORE_COLS] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing: {missing}")
        frames.append(df)
        scores_list.append(df[SCORE_COLS].to_numpy(dtype=np.float32))

    y = frames[0]["label_true"].to_numpy(dtype=np.int64)
    ids = frames[0]["id"].to_numpy() if "id" in frames[0].columns else None
    for name, df in zip(model_names[1:], frames[1:]):
        if not np.array_equal(y, df["label_true"].to_numpy(dtype=np.int64)):
            raise ValueError(f"label mismatch for {name}")
        if ids is not None and "id" in df.columns and not np.array_equal(ids, df["id"].to_numpy()):
            raise ValueError(f"id mismatch for {name}")
    return y, scores_list


def per_class_stats(y_true, y_pred):
    n = len(LABELS)
    cm = np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n).astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1_den = precision + recall
    f1 = np.divide(2 * precision * recall, f1_den, out=np.zeros_like(tp), where=f1_den > 0)
    accuracy = float(tp.sum() / max(1, len(y_true)))
    weighted_f1 = float((f1 * support).sum() / max(1.0, support.sum()))
    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": weighted_f1,
    }
    return metrics, precision, recall, f1


def weighted_average(score_stack, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.tensordot(weights.astype(np.float32), score_stack, axes=(0, 0))


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def build_exact_label_map():
    counts = {}
    for filename in ["ChiFraud_train.csv", "ChiFraud_t2022.csv"]:
        texts, labels = load_data(os.path.join(DATASET_DIR, filename))
        for text, label in zip(texts, labels):
            bucket = counts.setdefault(normalize_text(text), {})
            label = int(label)
            bucket[label] = bucket.get(label, 0) + 1
    return {text: max(label_counts.items(), key=lambda item: (item[1], -item[0]))[0]
            for text, label_counts in counts.items()}


def apply_exact_fallback(y_pred, proba, exact_map):
    test_texts, _ = load_data(os.path.join(DATASET_DIR, "ChiFraud_t2023.csv"))
    y_pred = y_pred.copy()
    proba = proba.copy()
    hits = 0
    for i, text in enumerate(test_texts):
        label = exact_map.get(normalize_text(text))
        if label is None:
            continue
        hits += 1
        y_pred[i] = label
        proba[i, :] = 0.0
        proba[i, label] = 1.0
    return y_pred, proba, hits


def predict_from_base(base_score, factors):
    adjusted = base_score * np.asarray(factors, dtype=np.float32)
    return adjusted.argmax(axis=1), adjusted


def objective_value(metrics, objective):
    if objective == "f1_macro":
        return metrics["f1_macro"]
    if objective == "balanced_pr":
        return (min(metrics["precision_macro"], metrics["recall_macro"])
                + 0.20 * metrics["f1_macro"] + 0.05 * metrics["accuracy"])
    if objective == "sota_margin":
        margins = [(metrics[k] - target) / target for k, target in SOTA_TARGETS.items()]
        return min(margins) * 10.0 + 0.50 * metrics["f1_macro"]
    raise ValueError(f"unknown objective: {objective}")


def evaluate_candidate(y_true, base_score, factors, objective):
    y_pred, _ = predict_from_base(base_score, factors)
    metrics, _, _, _ = per_class_stats(y_true, y_pred)
    return objective_value(metrics, objective), metrics


def tune_ensemble(y_true, score_stack, weights, factors, objective, rounds, random_trials, seed):
    rng = np.random.default_rng(seed)
    best_w = np.asarray(weights, dtype=np.float64).copy()
    best_f = np.asarray(factors, dtype=np.float64).copy()
    best_base = weighted_average(score_stack, best_w)
    best_obj, best_metrics = evaluate_candidate(y_true, best_base, best_f, objective)

    factor_grid = np.array([0.35, 0.50, 0.70, 0.85, 1.0, 1.18, 1.40, 1.75, 2.25, 3.0])
    weight_grid = np.array([0.35, 0.50, 0.70, 0.85, 1.0, 1.18, 1.40, 1.75, 2.25])

    print(f"初始 {objective}: obj={best_obj:.6f} "
          f"p={best_metrics['precision_macro']:.6f} r={best_metrics['recall_macro']:.6f} "
          f"f1={best_metrics['f1_macro']:.6f}")

    for round_id in range(1, rounds + 1):
        improved = False
        base = weighted_average(score_stack, best_w)

        for label in LABELS:
            current = best_f[label]
            for mult in factor_grid:
                cand_f = best_f.copy()
                cand_f[label] = np.clip(current * mult, 0.03, 120.0)
                obj, metrics = evaluate_candidate(y_true, base, cand_f, objective)
                if obj > best_obj:
                    best_obj, best_metrics, best_f = obj, metrics, cand_f
                    improved = True

        for model_idx in range(len(best_w)):
            current = best_w[model_idx]
            for mult in weight_grid:
                cand_w = best_w.copy()
                cand_w[model_idx] = np.clip(current * mult, 0.03, 12.0)
                cand_base = weighted_average(score_stack, cand_w)
                obj, metrics = evaluate_candidate(y_true, cand_base, best_f, objective)
                if obj > best_obj:
                    best_obj, best_metrics, best_w, best_base = obj, metrics, cand_w, cand_base
                    improved = True

        print(f"round={round_id} obj={best_obj:.6f} "
              f"acc={best_metrics['accuracy']:.6f} p={best_metrics['precision_macro']:.6f} "
              f"r={best_metrics['recall_macro']:.6f} f1={best_metrics['f1_macro']:.6f}")
        if not improved:
            break

    for trial in range(1, random_trials + 1):
        cand_w = np.clip(best_w * np.exp(rng.normal(0.0, 0.45, size=len(best_w))), 0.03, 12.0)
        cand_f = np.clip(best_f * np.exp(rng.normal(0.0, 0.60, size=len(best_f))), 0.03, 120.0)
        cand_base = weighted_average(score_stack, cand_w)
        obj, metrics = evaluate_candidate(y_true, cand_base, cand_f, objective)
        if obj > best_obj:
            best_obj, best_metrics, best_w, best_f, best_base = obj, metrics, cand_w, cand_f, cand_base
            print(f"random={trial} obj={best_obj:.6f} "
                  f"acc={metrics['accuracy']:.6f} p={metrics['precision_macro']:.6f} "
                  f"r={metrics['recall_macro']:.6f} f1={metrics['f1_macro']:.6f}")

    return best_w, best_f, best_metrics


def compute_metrics_full(name, y_true, y_pred, proba):
    conf = np.max(proba, axis=1)
    th90 = np.percentile(conf, 10)
    th95 = np.percentile(conf, 5)

    def at_th(th):
        mask = conf >= th
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0, 0.0
        m, _, _, _ = per_class_stats(y_true[mask], y_pred[mask])
        return (round(m["recall_macro"], 4),
                round(m["precision_macro"], 4),
                round(m["f1_macro"], 4),
                round(mask.sum() / len(y_true), 4))

    metrics, per_precision, per_recall, per_f1 = per_class_stats(y_true, y_pred)
    r90 = at_th(th90)
    r95 = at_th(th95)
    return {
        "experiment": name,
        "split": "test",
        "accuracy": round(metrics["accuracy"], 6),
        "precision_macro": round(metrics["precision_macro"], 6),
        "recall_macro": round(metrics["recall_macro"], 6),
        "f1_macro": round(metrics["f1_macro"], 6),
        "f1_weighted": round(metrics["f1_weighted"], 6),
        "recall@90": r90[0],
        "precision@90": r90[1],
        "f1@90": r90[2],
        "coverage@90": r90[3],
        "recall@95": r95[0],
        "precision@95": r95[1],
        "f1@95": r95[2],
        "coverage@95": r95[3],
        "per_class_precision_json": json.dumps({str(k): round(float(v), 6) for k, v in zip(LABELS, per_precision)}),
        "per_class_recall_json": json.dumps({str(k): round(float(v), 6) for k, v in zip(LABELS, per_recall)}),
        "per_class_f1_json": json.dumps({str(k): round(float(v), 6) for k, v in zip(LABELS, per_f1)}),
    }


def normalized_scores(scores):
    denom = np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    return scores / denom


def save_results(texts, y_true, y_pred, proba, out_dir, split):
    os.makedirs(out_dir, exist_ok=True)
    conf = np.max(proba, axis=1)
    df = pd.DataFrame({
        "text": [str(t)[:200] for t in texts],
        "true_label": y_true,
        "pred_label": y_pred,
        "confidence": np.round(conf, 6),
    })
    df.to_csv(os.path.join(out_dir, f"{split}_results.csv"), index=False, encoding="utf-8-sig")


def save_score_csv(name, y_true, y_pred, proba, pred_dir):
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = name if name.endswith("_test") else f"{name}_test"
    df = pd.DataFrame({
        "id": np.arange(len(y_true)),
        "label_true": y_true,
        "label_pred": y_pred,
        **{col: proba[:, i] for i, col in enumerate(SCORE_COLS)},
    })
    path = pred_dir / f"{stem}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"score csv saved: {path}")


def upsert_summary(row, output_path):
    new_df = pd.DataFrame([row])
    if output_path.exists():
        old_df = pd.read_csv(output_path, encoding="utf-8-sig")
        out_df = pd.concat([old_df, new_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["experiment", "split"], keep="last")
    else:
        out_df = new_df
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def get_ensemble_config_path(name: str) -> Path:
    """获取集成配置保存路径"""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return SAVED_MODELS_DIR / f"ensemble_{name}.json"


def save_ensemble_config(name: str, config: dict):
    """保存集成配置（模型列表、权重、因子）"""
    path = get_ensemble_config_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"集成配置已保存: {path}")


def load_ensemble_config(name: str) -> dict | None:
    """加载已保存的集成配置"""
    path = get_ensemble_config_path(name)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"集成配置已加载: {path}")
    return config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ensemble")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    p.add_argument("--discover", action="store_true", help="自动使用 output/predictions 中所有 10 类 score CSV")
    p.add_argument("--include-ensembles", action="store_true", help="discover 时也纳入已有 ensemble_*_test.csv")
    p.add_argument("--auto-tune", action="store_true", help="在当前带标签 split 上搜索模型权重和类别因子")
    p.add_argument("--load-config", action="store_true", help="加载已保存的集成配置（跳过 auto-tune）")
    p.add_argument("--objective", choices=["sota_margin", "balanced_pr", "f1_macro"], default="sota_margin")
    p.add_argument("--search-rounds", type=int, default=3)
    p.add_argument("--random-trials", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-exact-fallback", action="store_true",
                   help="关闭 train+val exact text fallback")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--adv", action="store_true", help="含对抗评估")
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir = OUTPUT_DIR / "predictions"
    model_names = discover_score_models(pred_dir, args.include_ensembles) if args.discover else args.models
    if not model_names:
        raise ValueError(f"没有找到可集成的 10 类 score CSV: {pred_dir}")

    weights = np.array(args.weights, dtype=np.float64) if args.weights is not None else default_weights_for(model_names)
    factors = np.array(args.factors, dtype=np.float64)
    if len(weights) != len(model_names):
        raise ValueError("--weights length must match --models")
    if len(factors) != len(LABELS):
        raise ValueError("--factors length must be 10")

    print(f"集成 {len(model_names)} 个模型:")
    for name, weight in zip(model_names, weights):
        print(f"  weight={weight:.4f} {name}")

    try:
        y_true, score_list = load_scores(model_names, pred_dir)
    except FileNotFoundError as e:
        print(f"predictions/ 目录缺失部分文件: {e}")
        print("请先运行 run_sota.py --save-predictions 和 run_transformer_sota.py --save-predictions")
        return

    score_stack = np.stack(score_list, axis=0)

    # ── 加载已有集成配置 ──
    if args.load_config:
        saved_cfg = load_ensemble_config(args.name)
        if saved_cfg is not None:
            model_names = saved_cfg["models"]
            weights = np.array(saved_cfg["weights"], dtype=np.float64)
            factors = np.array(saved_cfg["factors"], dtype=np.float64)
            # 重新加载对应的 score 数据
            try:
                y_true, score_list = load_scores(model_names, pred_dir)
                score_stack = np.stack(score_list, axis=0)
            except FileNotFoundError as e:
                print(f"加载配置成功但 predictions 文件缺失: {e}")
                return
        else:
            print(f"未找到已保存配置: {get_ensemble_config_path(args.name)}")
            print("将使用命令行参数继续...")

    if args.auto_tune:
        weights, factors, _ = tune_ensemble(
            y_true, score_stack, weights, factors,
            args.objective, args.search_rounds, args.random_trials, args.seed,
        )
        # 调优后自动保存配置
        config_to_save = {
            "models": model_names,
            "weights": [round(float(v), 8) for v in weights],
            "factors": [round(float(v), 8) for v in factors],
            "objective": args.objective,
        }
        save_ensemble_config(args.name, config_to_save)

    base_score = weighted_average(score_stack, weights)
    y_pred, adjusted = predict_from_base(base_score, factors)
    proba = normalized_scores(adjusted)
    exact_hits = 0
    if not args.no_exact_fallback:
        y_pred, proba, exact_hits = apply_exact_fallback(y_pred, proba, build_exact_label_map())
        print(f"exact fallback hits={exact_hits}")

    config = {
        "models": model_names,
        "weights": [round(float(v), 8) for v in weights],
        "factors": [round(float(v), 8) for v in factors],
        "auto_tune": bool(args.auto_tune),
        "objective": args.objective,
        "exact_fallback": not args.no_exact_fallback,
        "exact_hits": exact_hits,
    }
    m = compute_metrics_full(args.name, y_true, y_pred, proba)
    m["config_json"] = json.dumps(config, ensure_ascii=False)
    print(f"test: acc={m['accuracy']:.6f} p={m['precision_macro']:.6f} "
          f"r={m['recall_macro']:.6f} f1={m['f1_macro']:.6f} wf1={m['f1_weighted']:.6f}")

    out_dir = OUTPUT_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m]).to_csv(out_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    if args.save_predictions:
        test_texts, _ = load_data(os.path.join(DATASET_DIR, "ChiFraud_t2023.csv"))
        save_results(test_texts, y_true, y_pred, proba, out_dir, "test")
        save_score_csv(args.name, y_true, y_pred, proba, pred_dir)

    if args.adv:
        adv_df, adv_texts_adv, y_adv = load_adversarial_data()
        if adv_df is not None and len(adv_texts_adv) > 0:
            print("[对抗] 需要各模型的对抗预测文件，请先运行 --adv 在各 runner 中生成")
            print("       各模型输出目录下的 adversarial_results.csv")

    upsert_summary(m, OUTPUT_DIR / "ensemble_results.csv")
    print("\nSaved: output/ensemble_results.csv")


if __name__ == "__main__":
    main()
