"""
ChiFraud 多模型加权集成
=======================
用法:
  python run_ensemble_sota.py --save-predictions --adv
  python run_ensemble_sota.py --models model1_test model2_test --weights 1.0 1.0
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_processor import load_adversarial_data, load_data

OUTPUT_DIR = Path("output")
DATASET_DIR = Path("dataset")
LABELS = list(range(10))

DEFAULT_MODELS = [
    "macbert_full_plus2022_bal_b64_epoch1_test",
    "macbert_50k_plus2022_bal_b32_epoch1_test",
    "macbert_full_plus2022_sqrt_b64_epoch1_test",
    "roberta_50k_plus2022_bal_b64_epoch1_test",
    "char13_svc_120k_test",
    "hash_char14_sgd_log_test",
]
DEFAULT_WEIGHTS = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
DEFAULT_FACTORS = [0.7, 16.0, 6.0, 4.0, 1.7, 16.0, 0.5, 1.3, 4.0, 0.7]


def load_scores(model_names, pred_dir):
    frames, scores_list = [], []
    for name in model_names:
        path = pred_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(df)
        score_cols = [f"score_{l}" for l in LABELS]
        missing = [c for c in score_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing: {missing}")
        scores_list.append(df[score_cols].to_numpy(dtype=np.float64))
    y = frames[0]["label_true"].to_numpy(dtype=np.int64)
    for name, df in zip(model_names[1:], frames[1:]):
        if not np.array_equal(y, df["label_true"].to_numpy(dtype=np.int64)):
            raise ValueError(f"label mismatch for {name}")
    return y, scores_list


def compute_metrics_full(name, y_true, y_pred, proba):
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
        'experiment': name, 'split': 'test',
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall@90': r90[1], 'precision@90': r90[0], 'f1@90': r90[2], 'coverage@90': r90[3],
        'recall@95': r95[1], 'precision@95': r95[0], 'f1@95': r95[2], 'coverage@95': r95[3],
        'per_class_f1_json': json.dumps({str(k): round(float(v),4) for k,v in zip(LABELS, per_f1)}),
        'per_class_recall_json': json.dumps({str(k): round(float(v),4) for k,v in zip(LABELS, per_recall)}),
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
    p.add_argument("--name", default="ensemble")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    p.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--adv", action="store_true", help="含对抗评估")
    return p.parse_args()


def main():
    args = parse_args()
    weights = np.array(args.weights, dtype=np.float64)
    factors = np.array(args.factors, dtype=np.float64)
    if len(weights) != len(args.models):
        raise ValueError("--weights length must match --models")
    if len(factors) != len(LABELS):
        raise ValueError("--factors length must be 10")

    pred_dir = OUTPUT_DIR / "predictions"

    # ── 测试集集成 ──
    print(f"集成 {len(args.models)} 个模型...")
    try:
        y_true, score_list = load_scores(args.models, pred_dir)
    except FileNotFoundError as e:
        # 尝试从 run_sota 和 run_transformer 的输出目录读取
        print(f"predictions/ 目录缺失部分文件: {e}")
        print("请先运行 run_sota.py --save-predictions 和 run_transformer_sota.py --save-predictions")
        return

    score = sum(w * s for w, s in zip(weights, score_list)) / weights.sum()
    adjusted = score * factors
    y_pred = adjusted.argmax(axis=1)

    config = {"models": args.models, "weights": weights.tolist(), "factors": factors.tolist()}
    m = compute_metrics_full(args.name, y_true, y_pred, adjusted)
    print(f"test: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
          f"f1@90={m['f1@90']:.4f} f1@95={m['f1@95']:.4f}")

    out_dir = os.path.join(OUTPUT_DIR, args.name)
    pd.DataFrame([m]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False, encoding='utf-8-sig')

    if args.save_predictions:
        test_texts, _ = load_data(os.path.join(DATASET_DIR, "ChiFraud_t2023.csv"))
        save_results(test_texts, y_true, y_pred, adjusted, out_dir, "test")

    # ── 对抗集集成 ──
    if args.adv:
        adv_df, adv_texts_adv, y_adv = load_adversarial_data()
        if adv_df is not None and len(adv_texts_adv) > 0:
            # 对抗集无法从已有的 prediction CSV 读（不同 texts），需要用各个模型的原始分数
            # 简化：仅记录占位
            print("[对抗] 需要各模型的对抗预测文件，请先运行 --adv 在各 runner 中生成")
            print("       各模型输出目录下的 adversarial_results.csv")

    # 保存汇总
    pd.DataFrame([m]).to_csv(os.path.join(OUTPUT_DIR, "ensemble_results.csv"),
                             index=False, encoding='utf-8-sig')
    print(f"\nSaved: output/ensemble_results.csv")


if __name__ == "__main__":
    main()
