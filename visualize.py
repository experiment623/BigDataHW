"""
可视化脚本 — 混淆矩阵、模型对比、对抗鲁棒性、学习曲线
===================================================
用法:
  python visualize.py                    # 生成全部图表
  python visualize.py --type confusion   # 仅混淆矩阵
  python visualize.py --type compare     # 仅模型对比
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

from config import OUTPUT_DIR, LABEL_MAP, NUM_CLASSES

sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 中文字体
for fname in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
    try:
        fm.findfont(fname, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [fname, 'DejaVu Sans']
        break
    except Exception:
        continue

FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = [LABEL_MAP.get(i, str(i)) for i in range(NUM_CLASSES)]


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_name='confusion_matrix.png'):
    """混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title(f'{title} (Count)', fontsize=14)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmax=1.0)
    axes[1].set_title(f'{title} (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  混淆矩阵: {path}')


def plot_per_class_f1(metrics_row, save_name='per_class_f1.png'):
    """各类别 F1 柱状图"""
    if isinstance(metrics_row, str):
        per_f1 = json.loads(metrics_row)
    elif isinstance(metrics_row, dict):
        per_f1 = metrics_row
    else:
        return

    labels = [CLASS_NAMES[int(k)] for k in per_f1.keys()]
    values = list(per_f1.values())

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#2ecc71' if v > 0.8 else '#f39c12' if v > 0.5 else '#e74c3c' for v in values]
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class F1 Score', fontsize=14)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  各类别F1: {path}')


def plot_model_comparison():
    """模型横向对比柱状图 — 读取所有 metrics.csv"""
    all_metrics = []

    # 收集所有 test_metrics.csv
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f in ('test_metrics.csv', 'metrics.csv'):
                path = os.path.join(root, f)
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                    if len(df) > 0:
                        row = df.iloc[0].to_dict()
                        if 'experiment' not in row:
                            row['experiment'] = os.path.basename(root)
                        all_metrics.append(row)
                except Exception:
                    pass

    # 也读取 sota_results.csv
    sota_path = os.path.join(OUTPUT_DIR, 'sota_results.csv')
    if os.path.exists(sota_path):
        df_sota = pd.read_csv(sota_path, encoding='utf-8-sig')
        if 'experiment' in df_sota.columns and 'split' in df_sota.columns:
            df_test = df_sota[df_sota['split'] == 'test']
            for _, row in df_test.iterrows():
                all_metrics.append(row.to_dict())

    if not all_metrics:
        print('[可视化] 未找到任何指标文件，请先运行模型')
        return

    df_all = pd.DataFrame(all_metrics)
    df_all = df_all.drop_duplicates(subset=['experiment'], keep='last')

    metrics_to_plot = ['accuracy', 'f1_macro', 'f1@90', 'f1@95']
    available = [m for m in metrics_to_plot if m in df_all.columns]
    if not available:
        print('[可视化] 无可用指标列')
        return

    n_models = len(df_all)
    n_metrics = len(available)
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.2), 6))

    for i, metric in enumerate(available):
        values = [df_all.iloc[j].get(metric, 0) for j in range(n_models)]
        bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
                      color=colors[i % len(colors)], edgecolor='white')

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(df_all['experiment'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Model Comparison', fontsize=14)
    ax.legend(loc='lower right')
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  模型对比: {path}')


def plot_confidence_distribution():
    """置信度分布直方图 — 读取所有 test_results.csv"""
    all_conf = {}
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == 'test_results.csv':
                path = os.path.join(root, f)
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                    if 'confidence' in df.columns:
                        name = os.path.basename(root) or os.path.basename(os.path.dirname(root))
                        all_conf[name] = df['confidence'].values
                except Exception:
                    pass

    if not all_conf:
        print('[可视化] 未找到 test_results.csv')
        return

    n = len(all_conf)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows * cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (name, confs) in enumerate(all_conf.items()):
        r, c = idx // cols, idx % cols
        ax = axes[r, c] if r < axes.shape[0] and c < axes.shape[1] else None
        if ax is None:
            continue
        ax.hist(confs, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(confs), color='red', linestyle='--', label=f'mean={np.mean(confs):.3f}')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Confidence'); ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    for idx in range(len(all_conf), rows * cols):
        r, c = idx // cols, idx % cols
        if r < axes.shape[0] and c < axes.shape[1]:
            axes[r, c].set_visible(False)

    plt.suptitle('Prediction Confidence Distribution', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'confidence_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  置信度分布: {path}')


def plot_learning_curve():
    """学习曲线 — 读取 transformer metrics"""
    tf_path = os.path.join(OUTPUT_DIR, 'transformer_results.csv')
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == 'metrics.csv' and 'macbert' in root.lower():
                tf_path = os.path.join(root, f)
                break

    if not os.path.exists(tf_path):
        print(f'[可视化] 未找到 Transformer 指标文件')
        return

    df = pd.read_csv(tf_path, encoding='utf-8-sig')
    if 'epoch' not in df.columns or 'f1_macro' not in df.columns:
        return

    df_test = df[df.get('split', 'test') == 'test'].sort_values('epoch')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_test['epoch'], df_test['f1_macro'], 'o-', color='#2ecc71', linewidth=2, markersize=8, label='F1 Macro')
    if 'f1@90' in df_test.columns:
        ax.plot(df_test['epoch'], df_test['f1@90'], 's--', color='#3498db', linewidth=1.5, label='F1@90')
    if 'f1@95' in df_test.columns:
        ax.plot(df_test['epoch'], df_test['f1@95'], '^--', color='#e74c3c', linewidth=1.5, label='F1@95')

    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Transformer Fine-tuning Learning Curve', fontsize=14)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'learning_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  学习曲线: {path}')


def plot_adversarial_robustness():
    """对抗鲁棒性对比 — 读取各模型的 adversarial_metrics.csv"""
    all_adv = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f in ('adversarial_metrics.csv',):
                path = os.path.join(root, f)
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                    if len(df) > 0:
                        row = df.iloc[0].to_dict()
                        row['model'] = os.path.basename(root)
                        all_adv.append(row)
                except Exception:
                    pass

    if not all_adv:
        print('[可视化] 未找到 adversarial_metrics.csv，请用 --adv 运行模型')
        return

    df_adv = pd.DataFrame(all_adv)
    df_adv = df_adv.drop_duplicates(subset=['model'], keep='last')

    if 'f1_macro' not in df_adv.columns:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df_adv)*1.0), 5))
    x = np.arange(len(df_adv))
    width = 0.35

    bars1 = ax.bar(x - width/2, [df_adv.iloc[i].get('f1_macro', 0) for i in range(len(df_adv))],
                   width, label='F1 (Adversarial)', color='#e74c3c', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(df_adv['model'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Adversarial Robustness Comparison', fontsize=14)
    ax.legend()

    for bar, v in zip(bars1, [df_adv.iloc[i].get('f1_macro', 0) for i in range(len(df_adv))]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'adversarial_robustness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  对抗鲁棒性: {path}')


def plot_all():
    print('\n' + '=' * 50)
    print('  生成可视化图表')
    print('=' * 50)
    plot_model_comparison()
    plot_confidence_distribution()
    plot_learning_curve()
    plot_adversarial_robustness()

    # 如果有 sota_results.csv，生成各类别 F1 图
    sota_path = os.path.join(OUTPUT_DIR, 'sota_results.csv')
    if os.path.exists(sota_path):
        df = pd.read_csv(sota_path, encoding='utf-8-sig')
        for _, row in df.iterrows():
            if 'per_class_f1_json' in row and pd.notna(row['per_class_f1_json']):
                name = row.get('experiment', 'model')
                plot_per_class_f1(row['per_class_f1_json'], f'per_class_f1_{name}.png')
                break

    # 如果某个模型目录下有 test_results.csv，生成混淆矩阵
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == 'test_results.csv':
                path = os.path.join(root, f)
                df = pd.read_csv(path, encoding='utf-8-sig')
                if 'true_label' in df.columns and 'pred_label' in df.columns:
                    name = os.path.basename(root)
                    plot_confusion_matrix(df['true_label'].values, df['pred_label'].values,
                                          title=name, save_name=f'confusion_{name}.png')
                break

    print(f'\n全部图表已保存至: {FIG_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='all',
                        choices=['all', 'confusion', 'compare', 'confidence', 'learning', 'adversarial', 'f1'])
    args = parser.parse_args()

    if args.type == 'all':
        plot_all()
    elif args.type == 'compare':
        plot_model_comparison()
    elif args.type == 'confidence':
        plot_confidence_distribution()
    elif args.type == 'learning':
        plot_learning_curve()
    elif args.type == 'adversarial':
        plot_adversarial_robustness()
    elif args.type == 'f1':
        sota_path = os.path.join(OUTPUT_DIR, 'sota_results.csv')
        if os.path.exists(sota_path):
            df = pd.read_csv(sota_path, encoding='utf-8-sig')
            for _, row in df.iterrows():
                if 'per_class_f1_json' in row and pd.notna(row['per_class_f1_json']):
                    plot_per_class_f1(row['per_class_f1_json'])
                    break
    else:
        # 对单个 test_results.csv 画混淆矩阵
        print('用法: python visualize.py --type all')
