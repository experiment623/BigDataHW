"""
后处理脚本：十分类 → 二分类转换
================================
读取所有模型的 test_results.csv，将 label>0 映射为 spam，
计算二分类指标并输出汇总表。

用法:
  python postprocess_binary.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from config import OUTPUT_DIR

# 尝试导入 proba 相关（对抗数据不适用）
try:
    from run_sota import load_split
except ImportError:
    load_split = None


def convert_to_binary(y_true, y_pred, confidence=None, proba_10=None):
    """十分类 → 二分类：label 0=正常, label>0=垃圾"""
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    # 二分类置信度：垃圾类概率之和
    if proba_10 is not None:
        spam_conf = proba_10[:, 1:].sum(axis=1)
    elif confidence is not None:
        # 保守估计：如果预测正确，用原置信度；否则用 1-confidence
        spam_conf = np.where(y_pred_bin == 1, confidence, 1 - confidence)
    else:
        spam_conf = np.ones(len(y_true_bin))

    return y_true_bin, y_pred_bin, spam_conf


def compute_binary_metrics(y_true_bin, y_pred_bin, spam_conf):
    """计算二分类指标"""
    return {
        'binary_accuracy': round(accuracy_score(y_true_bin, y_pred_bin), 4),
        'binary_precision': round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        'binary_recall': round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        'binary_f1': round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        'binary_auc': round(roc_auc_score(y_true_bin, spam_conf) if len(set(y_true_bin)) > 1 else 0.5, 4),
    }


def main():
    print('=' * 60)
    print('  十分类 → 二分类 后处理')
    print('=' * 60)

    all_results = []

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == 'test_results.csv':
                path = os.path.join(root, f)
                model_name = os.path.basename(root)

                df = pd.read_csv(path, encoding='utf-8-sig')
                if 'true_label' not in df.columns or 'pred_label' not in df.columns:
                    print(f'  [跳过] {path}: 缺少必要列')
                    continue

                y_true = df['true_label'].values
                y_pred = df['pred_label'].values
                confidence = df['confidence'].values if 'confidence' in df.columns else None

                y_true_bin, y_pred_bin, spam_conf = convert_to_binary(y_true, y_pred, confidence)

                metrics = compute_binary_metrics(y_true_bin, y_pred_bin, spam_conf)
                metrics['model'] = model_name

                # 类别分布
                n_normal = (y_true_bin == 0).sum()
                n_spam = (y_true_bin == 1).sum()
                metrics['n_samples'] = len(y_true_bin)
                metrics['n_normal'] = n_normal
                metrics['n_spam'] = n_spam

                all_results.append(metrics)
                print(f'  {model_name}: Acc={metrics["binary_accuracy"]:.4f} '
                      f'F1={metrics["binary_f1"]:.4f} AUC={metrics["binary_auc"]:.4f}')

    if not all_results:
        print('[未找到任何 test_results.csv 文件]')
        return

    # 汇总表
    df_summary = pd.DataFrame(all_results)
    cols_order = ['model', 'binary_accuracy', 'binary_precision', 'binary_recall',
                  'binary_f1', 'binary_auc', 'n_samples', 'n_normal', 'n_spam']
    df_summary = df_summary[[c for c in cols_order if c in df_summary.columns]]
    df_summary = df_summary.sort_values('binary_f1', ascending=False)

    print('\n' + '=' * 60)
    print('  二分类模型对比汇总')
    print('=' * 60)
    print(df_summary.to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, 'binary_classification_summary.csv')
    df_summary.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\n汇总已保存: {out_path}')


if __name__ == '__main__':
    main()
