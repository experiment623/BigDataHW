"""
Strong ChiFraud SOTA experiments — 字符级 N-gram + 强分类器
============================================================
用法:
  python run_sota.py --experiments char15_svc_c1         # 跑单个
  python run_sota.py --experiments all                   # 全跑
  python run_sota.py --experiments all --save-predictions --adv
  python run_sota.py --experiments all --load            # 加载已保存模型评估
"""
from __future__ import annotations

import argparse, json, os, time, unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_processor import load_adversarial_data

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"
SAVED_MODELS_DIR = ROOT / "saved_models"
RANDOM_SEED = 42
LABELS = list(range(10))


@dataclass
class Metrics:
    experiment: str; split: str; protocol: str
    accuracy: float; precision_macro: float; recall_macro: float
    f1_macro: float; f1_weighted: float
    recall_90: float; precision_90: float; f1_90: float; coverage_90: float
    recall_95: float; precision_95: float; f1_95: float; coverage_95: float
    exact_matches: int; n_samples: int
    train_time_s: float; predict_time_s: float
    config_json: str; per_class_f1_json: str; per_class_recall_json: str


def normalize_text(text: object) -> str:
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
    return texts_norm, labels, texts_raw


def build_exact_label_map(texts: Iterable[str], y: np.ndarray) -> dict[str, int]:
    counts: dict[str, dict[int, int]] = {}
    for text, label in zip(texts, y):
        bucket = counts.setdefault(text, {})
        bucket[int(label)] = bucket.get(int(label), 0) + 1
    return {text: max(lc.items(), key=lambda item: (item[1], -item[0]))[0]
            for text, lc in counts.items()}


def apply_exact_fallback(texts, pred, score, exact_map):
    pred, score = pred.copy(), None if score is None else score.copy()
    matches = 0
    for i, text in enumerate(texts):
        label = exact_map.get(text)
        if label is None: continue
        matches += 1
        pred[i] = label
        if score is not None:
            score[i, :] = 0.0; score[i, label] = 1.0
    return pred, score, matches


def make_pipeline(name: str) -> tuple[Pipeline, dict]:
    configs = {
        "char13_svc_120k": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                                          max_df=0.98, max_features=120_000, sublinear_tf=True, dtype=np.float32),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000, dual="auto"),
        },
        "char14_svc_160k": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3,
                                          max_df=0.98, max_features=160_000, sublinear_tf=True, dtype=np.float32),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=6000, dual="auto"),
        },
        "char15_svc_c1": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2,
                                          max_df=0.98, max_features=300_000, sublinear_tf=True, dtype=np.float32),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=6000, dual="auto"),
        },
        "char25_svc_c1": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                          max_df=0.98, max_features=300_000, sublinear_tf=True, dtype=np.float32),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=6000, dual="auto"),
        },
        "char15_svc_c2": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2,
                                          max_df=0.98, max_features=400_000, sublinear_tf=True, dtype=np.float32),
            "classifier": LinearSVC(C=2.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=8000, dual="auto"),
        },
        "char15_sgd_log": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2,
                                          max_df=0.98, max_features=300_000, sublinear_tf=True, dtype=np.float32),
            "classifier": SGDClassifier(loss="log_loss", alpha=3e-6, penalty="l2",
                                        class_weight="balanced", random_state=RANDOM_SEED, max_iter=35, tol=1e-4, n_jobs=-1),
        },
        "hash_char14_sgd_log": {
            "vectorizer": HashingVectorizer(analyzer="char", ngram_range=(1, 4),
                                            n_features=2**19, alternate_sign=False, norm=None, dtype=np.float32),
            "transformer": TfidfTransformer(sublinear_tf=True),
            "classifier": SGDClassifier(loss="log_loss", alpha=1e-6, penalty="l2",
                                        class_weight="balanced", random_state=RANDOM_SEED, max_iter=45, tol=1e-4, n_jobs=-1),
        },
        "char15_lr_saga": {
            "vectorizer": TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2,
                                          max_df=0.98, max_features=250_000, sublinear_tf=True, dtype=np.float32),
            "classifier": LogisticRegression(C=4.0, solver="saga", penalty="l2",
                                             class_weight="balanced", random_state=RANDOM_SEED, max_iter=700, n_jobs=-1),
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
        "vectorizer": {k: v for k, v in cfg["vectorizer"].get_params().items()
                       if k in {"analyzer","ngram_range","min_df","max_df","max_features","sublinear_tf","n_features","alternate_sign"}},
        "classifier": {k: v for k, v in cfg["classifier"].get_params().items()
                       if k in {"C","alpha","loss","class_weight","max_iter","penalty","solver"}},
    }
    return pipe, serializable


def get_sota_model_path(experiment_name: str) -> Path:
    """获取 SOTA 模型保存路径"""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return SAVED_MODELS_DIR / f"{experiment_name}.pkl"


def save_sota_model(model: Pipeline, experiment_name: str):
    """保存 sklearn Pipeline 模型"""
    import pickle
    path = get_sota_model_path(experiment_name)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  模型已保存: {path}")


def load_sota_model(experiment_name: str) -> Pipeline:
    """加载 sklearn Pipeline 模型"""
    import pickle
    path = get_sota_model_path(experiment_name)
    if not path.exists():
        raise FileNotFoundError(f"未找到已保存模型: {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"  模型已加载: {path}")
    return model


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


def compute_threshold_metrics(y_true, y_pred, proba):
    conf = np.max(proba, axis=1)
    th90 = np.percentile(conf, 10)
    th95 = np.percentile(conf, 5)

    def at_th(th):
        mask = conf >= th
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0, 0.0
        y_f, p_f = y_true[mask], y_pred[mask]
        return (round(recall_score(y_f, p_f, average='macro', zero_division=0), 6),
                round(precision_score(y_f, p_f, average='macro', zero_division=0), 6),
                round(f1_score(y_f, p_f, average='macro', zero_division=0), 6),
                round(mask.sum() / len(y_true), 6))

    r90_rec, r90_prec, r90_f1, r90_cov = at_th(th90)
    r95_rec, r95_prec, r95_f1, r95_cov = at_th(th95)
    return r90_rec, r90_prec, r90_f1, r90_cov, r95_rec, r95_prec, r95_f1, r95_cov


def compute_metrics(experiment, split, protocol, y_true, y_pred, proba,
                    exact_matches, train_time_s, predict_time_s, config):
    per_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    r90_rec, r90_prec, r90_f1, r90_cov, r95_rec, r95_prec, r95_f1, r95_cov = \
        compute_threshold_metrics(y_true, y_pred, proba) if proba is not None else (0,)*8
    return Metrics(
        experiment=experiment, split=split, protocol=protocol,
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_90=r90_rec, precision_90=r90_prec, f1_90=r90_f1, coverage_90=r90_cov,
        recall_95=r95_rec, precision_95=r95_prec, f1_95=r95_f1, coverage_95=r95_cov,
        exact_matches=exact_matches, n_samples=len(y_true),
        train_time_s=train_time_s, predict_time_s=predict_time_s,
        config_json=json.dumps(config, ensure_ascii=False, sort_keys=True),
        per_class_f1_json=json.dumps({str(k): float(v) for k, v in zip(LABELS, per_class_f1)}, sort_keys=True),
        per_class_recall_json=json.dumps({str(k): float(v) for k, v in zip(LABELS, per_class_recall)}, sort_keys=True),
    )


def save_result_csv(experiment, split, texts_raw, y_true, y_pred, proba, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    conf = np.max(proba, axis=1) if proba is not None else np.ones(len(y_true))
    df = pd.DataFrame({
        'text': [str(t)[:200] for t in texts_raw],
        'true_label': y_true, 'pred_label': y_pred,
        'confidence': np.round(conf, 4),
    })
    df.to_csv(os.path.join(out_dir, f'{split}_results.csv'), index=False, encoding='utf-8-sig')


def save_score_csv_for_ensemble(experiment, y_true, y_pred, proba):
    """保存 10 类概率 CSV 到 output/predictions/，供 ensemble 集成使用"""
    pred_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    stem = f"{experiment}_test"
    if proba is None:
        proba = np.eye(10)[y_pred]
    df = pd.DataFrame({
        "id": np.arange(len(y_true)),
        "label_true": y_true,
        "label_pred": y_pred,
        **{f"score_{i}": proba[:, i] for i in range(10)},
    })
    df.to_csv(os.path.join(pred_dir, f"{stem}.csv"), index=False, encoding="utf-8-sig")
    print(f"  score csv saved: output/predictions/{stem}.csv")


def save_metrics_csv(metrics_list: list[Metrics], out_path):
    df = pd.DataFrame([asdict(m) for m in metrics_list])
    df.to_csv(out_path, index=False, encoding='utf-8-sig')


def parse_args():
    parser = argparse.ArgumentParser(description="Run strong ChiFraud SOTA experiments.")
    parser.add_argument("--experiments", nargs="+", default=["char15_svc_c1"],
                        help="Experiment names. Use 'all' for the full CPU set.")
    parser.add_argument("--no-exact-fallback", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--train-with-val", action="store_true")
    parser.add_argument("--adv", action="store_true", help="含对抗评估")
    parser.add_argument("--load", action="store_true", help="加载已保存模型直接评估（跳过训练）")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.experiments == ["all"]:
        experiments = ["char13_svc_120k","char14_svc_160k","char15_svc_c1","char25_svc_c1",
                       "char15_svc_c2","hash_char14_sgd_log","char15_sgd_log","char15_lr_saga"]
    else:
        experiments = args.experiments

    train_texts, y_train, train_raw = load_split("ChiFraud_train.csv")
    val_texts, y_val, val_raw = load_split("ChiFraud_t2022.csv")
    test_texts, y_test, test_raw = load_split("ChiFraud_t2023.csv")

    if args.limit_train:
        train_texts, y_train = train_texts[:args.limit_train], y_train[:args.limit_train]
    if args.limit_eval:
        val_texts, y_val = val_texts[:args.limit_eval], y_val[:args.limit_eval]
        test_texts, y_test = test_texts[:args.limit_eval], y_test[:args.limit_eval]

    fit_texts, fit_y = train_texts, y_train
    eval_splits = [("test", test_texts, y_test, test_raw)]
    protocol = "train->test"
    if args.train_with_val:
        fit_texts = train_texts + val_texts
        fit_y = np.concatenate([y_train, y_val])
        protocol = "train+val->test"

    # 对抗数据
    adv_df, adv_texts_adv, y_adv = None, [], np.array([])
    if args.adv:
        adv_df, adv_texts_adv, y_adv = load_adversarial_data()

    print(f"Train={len(train_texts)} Val={len(val_texts)} Test={len(test_texts)} Fit={len(fit_texts)}")
    exact_map = build_exact_label_map(fit_texts, fit_y)

    all_results = []
    for experiment in experiments:
        print(f"\n=== {experiment} ===", flush=True)
        model, config = make_pipeline(experiment)

        if args.load:
            # 加载已保存模型
            try:
                model = load_sota_model(experiment)
            except FileNotFoundError as e:
                print(f"  [跳过] {e}")
                continue
            train_time = 0.0
        else:
            # 训练新模型
            start = time.time()
            model.fit(fit_texts, fit_y)
            train_time = time.time() - start
            # 保存模型
            try:
                save_sota_model(model, experiment)
            except Exception as e:
                print(f"  [保存模型失败] {e}")

        vec = model.named_steps["tfidf"]
        vsize = len(vec.vocabulary_) if hasattr(vec, "vocabulary_") else int(getattr(vec, "n_features", 0))
        if not args.load:
            print(f"trained in {train_time:.1f}s, vocab={vsize}")

        for split, texts, y_true, raw_texts in eval_splits:
            y_pred, score = model_scores(model, texts)
            exact_m = 0
            if not args.no_exact_fallback:
                y_pred, score, exact_m = apply_exact_fallback(texts, y_pred, score, exact_map)

            metric = compute_metrics(experiment, split, protocol, y_true, y_pred, score,
                                     exact_m, train_time, 0, config | {"vocab_size": vsize})
            all_results.append(metric)
            print(f"{split}: acc={metric.accuracy:.6f} f1={metric.f1_macro:.6f} "
                  f"f1@90={metric.f1_90:.6f} f1@95={metric.f1_95:.6f}")

            if args.save_predictions:
                out_dir = os.path.join(OUTPUT_DIR, experiment)
                save_result_csv(experiment, split, raw_texts, y_true, y_pred, score, out_dir)
                save_metrics_csv([metric], os.path.join(out_dir, f'{split}_metrics.csv'))
                if split == "test" and score is not None:
                    save_score_csv_for_ensemble(experiment, y_true, y_pred, score)

        # 对抗评估
        if args.adv and adv_df is not None and len(adv_texts_adv) > 0:
            adv_norm = [normalize_text(t) for t in adv_texts_adv]
            y_adv_pred, adv_score = model_scores(model, adv_norm)
            adv_metric = compute_metrics(experiment, "adversarial", protocol, y_adv, y_adv_pred, adv_score,
                                         0, train_time, 0, config | {"vocab_size": vsize})
            all_results.append(adv_metric)
            print(f"adversarial: acc={adv_metric.accuracy:.6f} f1={adv_metric.f1_macro:.6f}")
            if args.save_predictions:
                out_dir = os.path.join(OUTPUT_DIR, experiment)
                save_result_csv(experiment, "adversarial", adv_texts_adv, y_adv, y_adv_pred, adv_score, out_dir)
                save_metrics_csv([adv_metric], os.path.join(out_dir, 'adversarial_metrics.csv'))

    print("\n=== Summary ===")
    cols = ["experiment","split","accuracy","f1_macro","f1_90","coverage_90","f1_95","coverage_95","train_time_s"]
    df = pd.DataFrame([asdict(r) for r in all_results])
    print(df[[c for c in cols if c in df.columns]].sort_values(["split","f1_macro"], ascending=[True,False]).to_string(index=False))
    save_metrics_csv(all_results, os.path.join(OUTPUT_DIR, "sota_results.csv"))
    print(f"\nSaved: output/sota_results.csv")


if __name__ == "__main__":
    main()
