"""
评估工具模块 — 含置信度阈值指标 + 统一输出格式
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from config import LABEL_MAP, N_FOLDS, RANDOM_SEED, NUM_CLASSES


def evaluate_model(model, X, y, label_names=None):
    """基础评估：打印指标并返回 dict"""
    pred = model.predict(X)
    y = np.array(y)
    pred = np.array(pred)

    unique_labels = sorted(set(list(y) + list(pred)))
    target_names = [LABEL_MAP.get(l, str(l)) for l in unique_labels]

    acc = accuracy_score(y, pred)
    prec_macro = precision_score(y, pred, average='macro', zero_division=0)
    rec_macro = recall_score(y, pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y, pred, average='weighted', zero_division=0)

    print(f'  Accuracy:          {acc:.4f}')
    print(f'  Precision(macro):  {prec_macro:.4f}')
    print(f'  Recall(macro):     {rec_macro:.4f}')
    print(f'  F1(macro):         {f1_macro:.4f}')
    print(f'  F1(weighted):      {f1_weighted:.4f}')

    return {
        'model': model.name,
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def evaluate_with_confidence_threshold(model, X, y, threshold=0.90):
    """在特定置信度阈值下评估"""
    y = np.array(y)
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    mask = confidence >= threshold
    coverage = np.mean(mask)

    if np.sum(mask) == 0:
        return {
            f'recall@{int(threshold*100)}%': 0.0,
            f'precision@{int(threshold*100)}%': 0.0,
            f'f1@{int(threshold*100)}%': 0.0,
            'coverage': 0.0
        }

    y_filtered = y[mask]
    pred_filtered = pred[mask]

    return {
        f'recall@{int(threshold*100)}%': round(recall_score(y_filtered, pred_filtered, average='macro', zero_division=0), 4),
        f'precision@{int(threshold*100)}%': round(precision_score(y_filtered, pred_filtered, average='macro', zero_division=0), 4),
        f'f1@{int(threshold*100)}%': round(f1_score(y_filtered, pred_filtered, average='macro', zero_division=0), 4),
        'coverage': round(coverage, 4)
    }


def compute_metrics_full(model_name, y_true, y_pred, proba, train_time_s=0, predict_time_s=0):
    """完整指标计算，包含置信度阈值指标"""
    per_f1 = f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)

    confidence = np.max(proba, axis=1)
    # 按置信度排序，取百分位阈值
    sorted_conf = np.sort(confidence)[::-1]
    th90 = sorted_conf[min(int(len(sorted_conf) * 0.10), len(sorted_conf) - 1)]
    th95 = sorted_conf[min(int(len(sorted_conf) * 0.05), len(sorted_conf) - 1)]

    def metrics_at_threshold(th):
        mask = confidence >= th
        if mask.sum() == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'coverage': 0.0}
        y_f, p_f = y_true[mask], y_pred[mask]
        return {
            'precision': round(precision_score(y_f, p_f, average='macro', zero_division=0), 4),
            'recall': round(recall_score(y_f, p_f, average='macro', zero_division=0), 4),
            'f1': round(f1_score(y_f, p_f, average='macro', zero_division=0), 4),
            'coverage': round(mask.sum() / len(y_true), 4)
        }

    r90 = metrics_at_threshold(th90)
    r95 = metrics_at_threshold(th95)

    return {
        'model': model_name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'recall@90': r90['recall'], 'precision@90': r90['precision'], 'f1@90': r90['f1'], 'coverage@90': r90['coverage'],
        'recall@95': r95['recall'], 'precision@95': r95['precision'], 'f1@95': r95['f1'], 'coverage@95': r95['coverage'],
        'per_class_f1_json': json.dumps({str(k): round(float(v), 4) for k, v in zip(range(NUM_CLASSES), per_f1)}, ensure_ascii=False),
        'per_class_recall_json': json.dumps({str(k): round(float(v), 4) for k, v in zip(range(NUM_CLASSES), per_recall)}, ensure_ascii=False),
        'per_class_precision_json': json.dumps({str(k): round(float(v), 4) for k, v in zip(range(NUM_CLASSES), per_precision)}, ensure_ascii=False),
        'train_time_s': round(train_time_s, 1),
        'predict_time_s': round(predict_time_s, 1),
        'n_samples': len(y_true),
    }


def save_predictions_csv(texts, y_true, y_pred, confidences, output_path):
    """保存预测结果 CSV：text, true_label, pred_label, confidence"""
    df = pd.DataFrame({
        'text': [str(t)[:200] for t in texts],
        'true_label': y_true,
        'pred_label': y_pred,
        'confidence': np.round(confidences, 4),
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  预测结果已保存: {output_path}')


def save_metrics_csv(metrics_dict, output_path):
    """保存指标 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([metrics_dict])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  指标已保存: {output_path}')


def compare_models(results: list, dataset_name: str = ''):
    """模型对比汇总"""
    df = pd.DataFrame(results)
    print(f'\n{"="*70}')
    print(f'  模型对比汇总 ({dataset_name})')
    print(f'{"="*70}')
    print(df.to_string(index=False))
    return df


def cross_validate(model_factory, X, y, n_folds=N_FOLDS, verbose=True):
    """K-Fold 分层交叉验证"""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    y = np.array(y)
    metrics_list = []
    model_name = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        if hasattr(X, 'shape') and hasattr(X, 'todense'):
            X_train, X_val = X[train_idx], X[val_idx]
        elif isinstance(X, np.ndarray):
            X_train, X_val = X[train_idx], X[val_idx]
        else:
            X_train, X_val = [X[i] for i in train_idx], [X[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        if model_name is None:
            model_name = model.name
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        metrics = {
            'fold': fold + 1,
            'accuracy': accuracy_score(y_val, pred),
            'f1_macro': f1_score(y_val, pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_val, pred, average='macro', zero_division=0),
            'precision_macro': precision_score(y_val, pred, average='macro', zero_division=0),
        }
        metrics_list.append(metrics)
        if verbose:
            print(f'  Fold {fold+1}/{n_folds}: Acc={metrics["accuracy"]:.4f}, F1={metrics["f1_macro"]:.4f}')

    df_folds = pd.DataFrame(metrics_list)
    summary = {
        'model': model_name,
        'folds': metrics_list,
        'mean_accuracy': round(df_folds['accuracy'].mean(), 4),
        'std_accuracy': round(df_folds['accuracy'].std(), 4),
        'mean_f1_macro': round(df_folds['f1_macro'].mean(), 4),
        'std_f1_macro': round(df_folds['f1_macro'].std(), 4),
    }
    if verbose:
        print(f'\n  ★ {model_name} {n_folds}-Fold CV: F1={summary["mean_f1_macro"]}±{summary["std_f1_macro"]}')
    return summary
