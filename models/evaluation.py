"""
评估工具模块
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from config import LABEL_MAP


def evaluate_model(model, X, y, label_names: list = None):
    """评估模型，打印详细指标"""
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

    print(f'\n{"="*60}')
    print(f'  模型: {model.name}')
    print(f'{"="*60}')
    print(f'  Accuracy:          {acc:.4f}')
    print(f'  Precision(macro):  {prec_macro:.4f}')
    print(f'  Recall(macro):     {rec_macro:.4f}')
    print(f'  F1(macro):         {f1_macro:.4f}')
    print(f'  F1(weighted):      {f1_weighted:.4f}')
    print(f'\n  分类报告:')
    print(classification_report(y, pred, labels=unique_labels,
          target_names=target_names, zero_division=0))

    return {
        'model': model.name,
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def evaluate_with_confidence_threshold(model, X, y, threshold=0.90):
    """
    在特定置信度阈值下评估模型
    只考虑模型置信度 >= threshold 的样本
    """
    y = np.array(y)
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    mask = confidence >= threshold
    coverage = np.mean(mask)

    if np.sum(mask) == 0:
        print(f'  [警告] 阈值 {threshold:.0%} 下无样本满足条件')
        return {
            f'precision@{int(threshold*100)}%': 0.0,
            f'recall@{int(threshold*100)}%': 0.0,
            f'f1@{int(threshold*100)}%': 0.0,
            'coverage': 0.0
        }

    y_filtered = y[mask]
    pred_filtered = pred[mask]

    return {
        f'precision@{int(threshold*100)}%': round(precision_score(y_filtered, pred_filtered, average='macro', zero_division=0), 4),
        f'recall@{int(threshold*100)}%': round(recall_score(y_filtered, pred_filtered, average='macro', zero_division=0), 4),
        f'f1@{int(threshold*100)}%': round(f1_score(y_filtered, pred_filtered, average='macro', zero_division=0), 4),
        'coverage': round(coverage, 4)
    }


def compare_models(results: list, dataset_name: str = ''):
    """模型对比汇总"""
    df = pd.DataFrame(results)
    print(f'\n{"="*70}')
    print(f'  模型对比汇总 ({dataset_name})')
    print(f'{"="*70}')
    print(df.to_string(index=False))
    return df
