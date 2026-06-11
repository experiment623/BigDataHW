"""
结果可视化模块
提供混淆矩阵、模型对比、对抗鲁棒性等图表
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import OUTPUT_DIR, LABEL_MAP


def _find_chinese_font():
    """自动搜索系统中的中文字体"""
    # 1. 先找系统已安装的字体
    all_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
                 'Droid Sans Fallback', 'AR PL UMing CN', 'Source Han Sans CN']:
        if name in all_fonts:
            return all_fonts[name]

    # 2. 搜常见路径
    paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for p in paths:
        if os.path.exists(p):
            return p

    return None


_font_path = _find_chinese_font()
if _font_path:
    fm.fontManager.addfont(_font_path)
    _font_name = fm.FontProperties(fname=_font_path).get_name()
    plt.rcParams['font.sans-serif'] = [_font_name, 'DejaVu Sans']
    print(f'[Visualize] 中文字体: {_font_name} ({_font_path})')
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print('[Visualize] ⚠ 未找到中文字体! 图表中文会显示为方块')
    print('  安装: apt-get install fonts-wqy-microhei (Linux)')
    print('  或手动下载字体放到 /usr/share/fonts/ 下')

plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """
    绘制单个模型的混淆矩阵热力图
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        save_path: 图片保存路径（None 则自动生成）
    """
    labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 归一化（按行）
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    label_names = [LABEL_MAP.get(l, str(l)) for l in labels]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[0], cbar_kws={'label': '样本数'})
    axes[0].set_title(f'{model_name} - 混淆矩阵(计数)', fontsize=14)
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    
    # 归一化
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[1], cbar_kws={'label': '召回率'})
    axes[1].set_title(f'{model_name} - 混淆矩阵(归一化)', fontsize=14)
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f'cm_{model_name.replace(" ", "_")}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  混淆矩阵已保存: {save_path}')
    return save_path


def plot_model_comparison(results_df: pd.DataFrame, metric: str = 'f1_macro',
                          save_path: str = None):
    """
    多模型性能对比柱状图
    
    参数:
        results_df: 包含 'model' 列和各指标列的 DataFrame
        metric: 对比的主指标 (f1_macro / accuracy 等)
        save_path: 保存路径
    """
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
    available = [m for m in metrics_to_plot if m in results_df.columns]
    
    if not available:
        print('[警告] 无可绘制指标')
        return
    
    df_sorted = results_df.sort_values(metric, ascending=True)
    
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette('viridis', len(df_sorted))
    
    for ax, met in zip(axes, available):
        bars = ax.barh(df_sorted['model'], df_sorted[met], color=colors)
        for bar, val in zip(bars, df_sorted[met]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_title(met.replace('_', ' ').title(), fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Score')
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  模型对比图已保存: {save_path}')
    return save_path


def plot_adversarial_robustness(adv_results: dict, save_path: str = None):
    """
    对抗鲁棒性对比图
    
    参数:
        adv_results: {model_name: {'accuracy': float, 'detail': DataFrame}}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：各模型对抗总体准确率
    model_names = list(adv_results.keys())
    accuracies = [adv_results[n]['accuracy'] for n in model_names]
    
    colors = ['#2ecc71' if a >= 0.7 else '#f39c12' if a >= 0.5 else '#e74c3c'
              for a in accuracies]
    bars = axes[0].barh(model_names, accuracies, color=colors)
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                      f'{val:.3f}', va='center', fontsize=10)
    axes[0].set_title('各模型对抗样本总体准确率', fontsize=14)
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 右图：首选第一个有效模型的攻击类型分解
    ax2 = axes[1]
    for name in model_names:
        detail = adv_results[name]['detail']
        if 'attack_method' in detail.columns and 'accuracy' in detail.columns:
            detail_sorted = detail.sort_values('accuracy')
            bars = ax2.barh(detail_sorted['attack_method'], detail_sorted['accuracy'],
                           alpha=0.7, label=name)
            for bar, val in zip(bars, detail_sorted['accuracy']):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', va='center', fontsize=8)
            break  # 只画第一个模型的细节
    
    ax2.set_title('攻击类型鲁棒性分解', fontsize=14)
    ax2.set_xlim(0, 1.0)
    ax2.legend(loc='lower right')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'adversarial_robustness.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  对抗鲁棒性图已保存: {save_path}')
    return save_path


def plot_label_distribution(train_labels, val_labels, test_labels,
                            save_path: str = None):
    """
    绘制标签分布对比图
    
    参数:
        train_labels, val_labels, test_labels: 各数据集的标签列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        ('训练集', train_labels),
        ('验证集(2022)', val_labels),
        ('测试集(2023)', test_labels)
    ]
    
    for ax, (name, labels) in zip(axes, datasets):
        label_names = [LABEL_MAP.get(l, str(l)) for l in sorted(set(labels))]
        counts = pd.Series(labels).value_counts().sort_index()
        
        colors = sns.color_palette('Set3', len(counts))
        bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='white')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{name}\n(总样本: {len(labels)})', fontsize=12)
        ax.set_ylabel('样本数')
        
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha='center', fontsize=8)
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'label_distribution.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  标签分布图已保存: {save_path}')
    return save_path


def generate_all_visualizations(raw_data, proc_data, ml_models, ml_results,
                                 mlp_val, mlp_test, adv_results,
                                 bert_results=None):
    """
    一键生成所有可视化图表
    
    参数:
        raw_data: (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
        proc_data: 预处理后的数据字典
        ml_models: 训练好的模型字典
        ml_results: 传统 ML 结果
        mlp_val, mlp_test: MLP 的验证/测试结果
        adv_results: 对抗评估结果
        bert_results: (可选) BERT 的 (bert_val, bert_test, bert_data)
    """
    print('\n' + '='*60)
    print('  [可视化] 生成结果图表')
    print('='*60)
    
    (_, train_labels), (_, val_labels), (_, test_labels) = raw_data
    
    # 1. 标签分布
    plot_label_distribution(train_labels, val_labels, test_labels)
    
    # 2. 模型对比
    all_results = ml_results['test'].copy()
    all_results.append(mlp_test)
    if bert_results is not None:
        bert_val, bert_test, _ = bert_results
        all_results.append(bert_test)
    results_df = pd.DataFrame(all_results)
    plot_model_comparison(results_df)
    
    # 3. 混淆矩阵（对每个模型在测试集上生成）
    X_test = proc_data['X_test']
    y_test = proc_data['y_test']
    test_texts = proc_data.get('test_texts', [])
    for name, model in ml_models.items():
        if model.input_type == 'text':
            # 文本输入模型
            _, _, bert_data = bert_results if bert_results else (None, None, None)
            if bert_data and 'BERT' in name:
                _, _, sub_test_texts, sub_test_labels = bert_data
                y_pred = model.predict(sub_test_texts)
                plot_confusion_matrix(sub_test_labels, y_pred, name)
            else:
                # Word2Vec/Doc2Vec 等文本模型用全量测试文本
                y_pred = model.predict(test_texts)
                plot_confusion_matrix(y_test, y_pred, name)
        else:
            y_pred = model.predict(X_test)
            plot_confusion_matrix(y_test, y_pred, name)
    
    # 4. 对抗鲁棒性
    if adv_results:
        plot_adversarial_robustness(adv_results)
    
    print('  所有图表已保存至:', OUTPUT_DIR)
