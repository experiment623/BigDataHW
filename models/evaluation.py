"""
评估工具模块
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from config import LABEL_MAP, N_FOLDS, RANDOM_SEED


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


def cross_validate(model_factory, X, y, n_folds=N_FOLDS, verbose=True):
    """
    K-Fold 分层交叉验证

    参数:
        model_factory: 可调用对象，每次调用返回一个新的模型实例 (如 lambda: TfidfLogisticRegression())
        X: 特征矩阵 / 文本列表
        y: 标签数组
        n_folds: 折数
        verbose: 是否打印每折结果

    返回:
        dict: {
            'model': 模型名称,
            'folds': 每折指标列表,
            'mean_*': 平均值,
            'std_*': 标准差,
        }
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    y = np.array(y)

    metrics_list = []
    model_name = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train = X[train_idx] if isinstance(X, np.ndarray) else [X[i] for i in train_idx]
        X_val = X[val_idx] if isinstance(X, np.ndarray) else [X[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 处理稀疏矩阵
        if hasattr(X, 'shape') and hasattr(X, 'todense'):
            X_train = X[train_idx]
            X_val = X[val_idx]

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

    # 汇总
    df_folds = pd.DataFrame(metrics_list)
    summary = {
        'model': model_name,
        'folds': metrics_list,
        'mean_accuracy': round(df_folds['accuracy'].mean(), 4),
        'std_accuracy': round(df_folds['accuracy'].std(), 4),
        'mean_f1_macro': round(df_folds['f1_macro'].mean(), 4),
        'std_f1_macro': round(df_folds['f1_macro'].std(), 4),
        'mean_recall_macro': round(df_folds['recall_macro'].mean(), 4),
        'std_recall_macro': round(df_folds['recall_macro'].std(), 4),
    }

    if verbose:
        print(f'\n  ★ {model_name} {n_folds}-Fold CV 结果:')
        print(f'    Accuracy: {summary["mean_accuracy"]} ± {summary["std_accuracy"]}')
        print(f'    F1(macro): {summary["mean_f1_macro"]} ± {summary["std_f1_macro"]}')
        print(f'    Recall(macro): {summary["mean_recall_macro"]} ± {summary["std_recall_macro"]}')

    return summary
