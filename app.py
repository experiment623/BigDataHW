"""
ChiFraud 垃圾文本多模型对比系统
================================
Flask + Jinja2 构建，支持 8 个模型并行对比预测。
明星模型: ensemble_cross (22模型加权融合)
"""

from __future__ import annotations

import json
import time
import threading
import traceback
import unicodedata
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

# ── sklearn 跨版本 pickle 兼容 ──
import sys as _sys
import warnings as _warnings
_warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1. SGDClassifier loss='log_loss' 旧版引用 Log 类
import sklearn.linear_model._sgd_fast as _sgd_fast
if not hasattr(_sgd_fast, "Log"):
    class _DummyLog:
        pass
    _sgd_fast.Log = _DummyLog

# 2. HistGradientBoostingClassifier 旧版 pickle 引用顶层 _loss 模块
try:
    import sklearn._loss._loss as _loss_ext
    _sys.modules["_loss"] = _loss_ext
except ImportError:
    pass

# 项目根
ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = ROOT / "saved_models"
OUTPUT_DIR = ROOT / "output"
DATASET_DIR = ROOT / "dataset"

# 类别映射
LABEL_MAP = {
    0: "正常", 1: "赌博博彩", 2: "招嫖色情", 3: "办假证", 4: "虚假办卡",
    5: "违禁药品交易", 6: "违规提现", 7: "虚假证明", 8: "虚假手机卡", 9: "地下黑贷",
}
NUM_CLASSES = 10
LABELS = list(range(NUM_CLASSES))

# 文本长度限制
MAX_TEXT_LEN_DISPLAY = 120

# ===================== 文本规范化 =====================

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===================== 模型加载缓存 =====================

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()
_loading_status: dict[str, str] = {}  # key -> "unloaded" | "loading" | "loaded" | "error"
_texts_by_label: dict[int, list] | None = None  # 随机文本按标签分组缓存

# ===================== 辅助函数 (从 run_ensemble_sota.py 移植) =====================

def weighted_average(score_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """多模型概率加权平均 (score_stack shape: [N_models, N_samples, N_classes])"""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.tensordot(weights.astype(np.float32), score_stack, axes=(0, 0))


def predict_from_base(base_score: np.ndarray, factors: np.ndarray):
    """乘以 per-class 因子后 argmax"""
    adjusted = base_score * np.asarray(factors, dtype=np.float32)
    return adjusted.argmax(axis=1), adjusted


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    """行归一化为概率分布"""
    denom = np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    return scores / denom


# ===================== 基线模型加载器 =====================

def _load_d2v_gbdt():
    from models.baselines import Doc2VecCharGBDT
    model = Doc2VecCharGBDT()
    model.load(str(SAVED_MODELS_DIR / "d2v_gbdt.pkl"))
    return model


def _load_w2v_w():
    from models.baselines import Word2VecWordLR
    model = Word2VecWordLR()
    model.load(str(SAVED_MODELS_DIR / "w2v_w.pkl"))
    return model


def _load_w2v_c():
    from models.baselines import Word2VecCharLR
    model = Word2VecCharLR()
    model.load(str(SAVED_MODELS_DIR / "w2v_c.pkl"))
    return model


def _load_w2v_gbdt():
    from models.baselines import Word2VecCharGBDT
    model = Word2VecCharGBDT()
    model.load(str(SAVED_MODELS_DIR / "w2v_gbdt.pkl"))
    return model


def _load_gas():
    from models.baselines import GAS
    import pickle
    model = GAS()
    model.load(str(SAVED_MODELS_DIR / "gas.pth"))
    # 使用预训练的 TF-IDF vectorizer，避免每次 predict_proba 新建导致维度不匹配
    model._tfidf_vec = pickle.load(open(str(OUTPUT_DIR / "gas" / "tfidf_vec.pkl"), "rb"))

    # 保存原始方法并替换为使用预训练 vectorizer 的版本
    _orig_predict_proba = model.predict_proba
    _orig_predict = model.predict
    _tfidf = model._tfidf_vec

    def _patched_predict_proba(X):
        """使用预训练 vectorizer 转换文本后预测概率"""
        if isinstance(X[0], str):
            X_sp = _tfidf.transform(X)
            X_dense = X_sp.toarray() if hasattr(X_sp, "toarray") else X_sp
        elif hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X
        return _orig_predict_proba(X_dense)

    def _patched_predict(X):
        """使用预训练 vectorizer 转换文本后预测标签"""
        if isinstance(X[0], str):
            X_sp = _tfidf.transform(X)
            X_dense = X_sp.toarray() if hasattr(X_sp, "toarray") else X_sp
        elif hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X
        return _orig_predict(X_dense)

    model.predict_proba = _patched_predict_proba
    model.predict = _patched_predict
    return model


def _load_char15_sgd_log():
    """加载 sklearn Pipeline（兼容不同 sklearn 版本）"""
    from run_sota import load_sota_model
    return load_sota_model("char15_sgd_log")


def _load_macbert_cwb():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = str(SAVED_MODELS_DIR / "macbert_base_+val_cwbalanced")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=NUM_CLASSES,
        id2label={i: str(i) for i in LABELS},
        label2id={str(i): i for i in LABELS},
    ).to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


def _load_ensemble_cross():
    """加载 ensemble_cross 所需全部 22 个子模型 + 集成配置"""
    # 1. 加载配置
    with open(str(SAVED_MODELS_DIR / "ensemble_ensemble_cross.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. 解析子模型
    sub_models = {}  # name -> loaded model
    for csv_name in config["models"]:
        # csv_name 形如 "char13_svc_120k_test" 或 "macbert_base_+val_aug2_epoch1_test"
        sub_models[csv_name] = _load_ensemble_submodel(csv_name)

    return {"config": config, "sub_models": sub_models}


def _load_ensemble_submodel(csv_name: str):
    """根据预测 CSV 文件名加载对应模型"""
    # 去掉 _test 后缀得到模型 key
    model_key = csv_name
    if model_key.endswith("_test"):
        model_key = model_key[:-5]

    # 字符 N-gram SOTA 模型
    if model_key in {"char13_svc_120k", "char14_svc_160k", "char15_lr_saga",
                      "char15_sgd_log", "char15_svc_c1", "char15_svc_c2",
                      "char25_svc_c1", "hash_char14_sgd_log"}:
        from run_sota import load_sota_model
        return {"type": "sota", "model": load_sota_model(model_key)}

    # Transformer 模型
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = SAVED_MODELS_DIR / model_key
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=NUM_CLASSES,
        id2label={i: str(i) for i in LABELS},
        label2id={str(i): i for i in LABELS},
    ).to(device)
    model.eval()
    return {"type": "transformer", "model": model, "tokenizer": tokenizer, "device": device}


# ===================== 模型注册表 =====================

MODEL_REGISTRY = {
    "d2v_gbdt": {
        "display": "Doc2Vec-c+GBDT",
        "loader": _load_d2v_gbdt,
        "type": "baseline",
        "star": False,
    },
    "w2v_w": {
        "display": "Word2Vec-w+LR",
        "loader": _load_w2v_w,
        "type": "baseline",
        "star": False,
    },
    "w2v_c": {
        "display": "Word2Vec-c+LR",
        "loader": _load_w2v_c,
        "type": "baseline",
        "star": False,
    },
    "gas": {
        "display": "GAS (GCN)",
        "loader": _load_gas,
        "type": "baseline",
        "star": False,
    },
    "w2v_gbdt": {
        "display": "Word2Vec-c+GBDT",
        "loader": _load_w2v_gbdt,
        "type": "baseline",
        "star": False,
    },
    "char15_sgd_log": {
        "display": "char15_sgd_log",
        "loader": _load_char15_sgd_log,
        "type": "sota",
        "star": False,
    },
    "macbert_cwb": {
        "display": "macbert_cwb",
        "loader": _load_macbert_cwb,
        "type": "transformer",
        "star": False,
    },
    "ensemble_cross": {
        "display": "ensemble_cross",
        "loader": _load_ensemble_cross,
        "type": "ensemble_cross",
        "star": True,
    },
}


def ensure_loaded(model_key: str):
    """线程安全的懒加载"""
    with _cache_lock:
        if model_key in _model_cache:
            return
        if _loading_status.get(model_key) == "loading":
            return  # 其他线程正在加载
        _loading_status[model_key] = "loading"

    try:
        loader = MODEL_REGISTRY[model_key]["loader"]
        model = loader()
        with _cache_lock:
            _model_cache[model_key] = model
            _loading_status[model_key] = "loaded"
    except Exception as e:
        with _cache_lock:
            _loading_status[model_key] = f"error: {e}"
        raise


def get_loading_status() -> dict:
    with _cache_lock:
        return dict(_loading_status)


# ===================== 预测函数 =====================

def _predict_baseline(model, text: str) -> np.ndarray:
    """基线模型预测，返回概率向量 (10,)"""
    proba = model.predict_proba([clean_text(text)])
    return proba[0].astype(np.float64)


def _predict_sota(model, text: str) -> np.ndarray:
    """SOTA sklearn Pipeline 预测"""
    from scipy.special import softmax
    text_norm = normalize_text(text)
    try:
        proba = model.predict_proba([text_norm])
        return proba[0].astype(np.float64)
    except (AttributeError, Exception):
        decision = model.decision_function([text_norm])
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        return softmax(decision, axis=1)[0].astype(np.float64)


def _predict_transformer(model_info: dict, text: str) -> np.ndarray:
    """Transformer 模型预测"""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    device = model_info["device"]

    encoded = tokenizer(
        clean_text(text), max_length=160, padding=True,
        truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
        proba = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return proba[0].astype(np.float64)


def _predict_ensemble_cross(model_info: dict, text: str) -> np.ndarray:
    """ensemble_cross: 22 模型加权融合"""
    config = model_info["config"]
    sub_models = model_info["sub_models"]

    # 逐个模型预测，收集概率向量
    score_list = []
    for csv_name in config["models"]:
        sub = sub_models[csv_name]
        if sub["type"] == "sota":
            proba = _predict_sota(sub["model"], text)
        else:  # transformer
            proba = _predict_transformer(sub, text)
        score_list.append(proba)

    # 加权融合
    score_stack = np.stack(score_list, axis=0)  # [22, 10]
    weights = np.array(config["weights"], dtype=np.float64)
    factors = np.array(config["factors"], dtype=np.float64)

    base_score = weighted_average(score_stack[:, np.newaxis, :], weights)[0]  # [10]
    _, adjusted = predict_from_base(base_score[np.newaxis, :], factors)
    proba = normalized_scores(adjusted)[0]
    return proba.astype(np.float64)


def predict_single(model_key: str, text: str) -> dict:
    """对单条文本执行预测，返回统一格式的结果字典"""
    ensure_loaded(model_key)
    model = _model_cache[model_key]
    model_type = MODEL_REGISTRY[model_key]["type"]

    t0 = time.time()
    if model_type == "baseline":
        proba = _predict_baseline(model, text)
    elif model_type == "sota":
        proba = _predict_sota(model, text)
    elif model_type == "transformer":
        proba = _predict_transformer(model, text)
    elif model_type == "ensemble_cross":
        proba = _predict_ensemble_cross(model, text)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    elapsed_ms = round((time.time() - t0) * 1000, 1)
    pred_label = int(np.argmax(proba))
    confidence = round(float(np.max(proba)), 4)

    # Top-3 概率
    top_indices = np.argsort(proba)[::-1][:3]
    proba_list = [
        {"label": int(i), "label_name": LABEL_MAP[int(i)], "prob": round(float(proba[i]), 4)}
        for i in top_indices
    ]

    return {
        "model_key": model_key,
        "display_name": MODEL_REGISTRY[model_key]["display"],
        "star": MODEL_REGISTRY[model_key]["star"],
        "pred_label": pred_label,
        "label_name": LABEL_MAP[pred_label],
        "confidence": confidence,
        "proba_list": proba_list,
        "elapsed_ms": elapsed_ms,
    }


def predict_batch(model_keys: list[str], texts: list[str]) -> list[dict]:
    """批量预测: 每条文本 × 每个模型"""
    # 先确保所有模型已加载（并行首条文本触发加载）
    for key in model_keys:
        ensure_loaded(key)

    results = []
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        model_results = []
        for key in model_keys:
            try:
                r = predict_single(key, text)
                model_results.append(r)
            except Exception as e:
                model_results.append({
                    "model_key": key,
                    "display_name": MODEL_REGISTRY[key]["display"],
                    "star": MODEL_REGISTRY[key]["star"],
                    "error": str(e),
                })
        results.append({
            "index": i,
            "text": text[:MAX_TEXT_LEN_DISPLAY] + ("..." if len(text) > MAX_TEXT_LEN_DISPLAY else ""),
            "full_text": text,
            "model_results": model_results,
        })
    return results


# ===================== Flask 应用 =====================

app = Flask(__name__)


@app.route("/")
def index():
    """主页：文本输入 + 模型选择（左侧），结果展示（右侧）"""
    models_meta = [
        {"key": k, "display": v["display"], "star": v["star"], "type": v["type"]}
        for k, v in MODEL_REGISTRY.items()
    ]
    return render_template("index.html", models=models_meta)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API：执行预测并返回结果"""
    texts_raw = request.form.get("texts", "")
    selected_models = request.form.getlist("models")

    if not texts_raw.strip():
        return jsonify({"error": "请输入要分类的文本"}), 400
    if not selected_models:
        return jsonify({"error": "请至少选择一个模型"}), 400

    valid_models = [m for m in selected_models if m in MODEL_REGISTRY]
    if not valid_models:
        return jsonify({"error": "未选择有效模型"}), 400

    text = texts_raw.strip()
    if not text:
        return jsonify({"error": "请输入要分类的文本"}), 400
    texts = [text]

    try:
        results = predict_batch(valid_models, texts)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify({"results": results})


@app.route("/api/random-text")
def api_random_text():
    """返回数据集中随机一条文本（十个类别等概率均匀抽样）"""
    import random as _random
    global _texts_by_label
    if _texts_by_label is None:
        from data_processor import load_data
        from config import TEST_PATH
        texts, labels = load_data(TEST_PATH)
        _texts_by_label = {l: [] for l in range(NUM_CLASSES)}
        for t, l in zip(texts, labels):
            _texts_by_label[int(l)].append(t)
    label = _random.randint(0, NUM_CLASSES - 1)
    pool = _texts_by_label[label]
    text = pool[_random.randint(0, len(pool) - 1)]
    return jsonify({
        "text": text,
        "label": label,
        "label_name": LABEL_MAP[label],
    })


@app.route("/api/status")
def api_status():
    """返回模型加载状态"""
    return jsonify(get_loading_status())


if __name__ == "__main__":
    print("=" * 60)
    print("ChiFraud 多模型对比系统")
    print("=" * 60)
    print(f"注册模型: {len(MODEL_REGISTRY)} 个")
    for k, v in MODEL_REGISTRY.items():
        star = " ★" if v["star"] else ""
        print(f"  [{v['type']}] {v['display']}{star}")
    print(f"\n启动后首次预测将触发模型加载（约需 60-90 秒）")
    print(f"后续预测会在已缓存模型上快速执行")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
