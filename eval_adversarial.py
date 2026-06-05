"""
对抗数据集评估脚本
===================
加载 make_adversarial_dataset.py 生成的对抗测试集，
评估所有已训练模型并输出对比报告。

用法:
  python eval_adversarial.py                          # 评估全部模型
  python eval_adversarial.py --model LR --model BERT  # 只评估指定模型
  python eval_adversarial.py --export                 # 导出详细结果CSV
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR, MODEL_DIR, OUTPUT_DIR, LABEL_MAP, NUM_CLASSES
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from scipy.sparse import load_npz

# ==================== 模型加载 ====================

def load_sklearn_model(path):
    """加载 sklearn pickle 模型"""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_mlp_model(path):
    """加载 PyTorch MLP 模型"""
    from models.neural_net import SimpleMLP
    # 先创建实例再加载
    import torch
    ckpt = torch.load(path, map_location='cpu')
    mlp = SimpleMLP(
        input_dim=ckpt['input_dim'],
        num_classes=ckpt['num_classes']
    )
    mlp.load(path)
    return mlp


def load_bert_model(path):
    """加载 BERT 模型"""
    from models.bert_model import BERTClassifier
    bert = BERTClassifier(num_classes=NUM_CLASSES)
    bert.load(path)
    return bert


def load_gca_model(path):
    """加载 GCA-Net 模型"""
    from models.gca_net import GCANet
    gca = GCANet(name='GCA-Net')
    gca.load(path)
    return gca


# 模型注册表
MODEL_REGISTRY = [
    {
        'key': 'TF-IDF + LogisticRegression',
        'file': 'TF-IDF_+_LogisticRegression.pkl',
        'loader': load_sklearn_model,
        'use_text': False,
    },
    {
        'key': 'TF-IDF + LinearSVC',
        'file': 'TF-IDF_+_LinearSVC.pkl',
        'loader': load_sklearn_model,
        'use_text': False,
    },
    {
        'key': 'TF-IDF + RandomForest',
        'file': 'TF-IDF_+_RandomForest.pkl',
        'loader': load_sklearn_model,
        'use_text': False,
    },
    {
        'key': 'TF-IDF + MultinomialNB',
        'file': 'TF-IDF_+_MultinomialNB.pkl',
        'loader': load_sklearn_model,
        'use_text': False,
    },
    {
        'key': 'TF-IDF + MLP',
        'file': 'TF-IDF_+_MLP.pkl',
        'loader': load_mlp_model,
        'use_text': False,
    },
    {
        'key': 'BERT(bert-base-chinese)',
        'file': 'bert_model',
        'loader': load_bert_model,
        'use_text': True,
    },
    {
        'key': 'GCA-Net (Glyph+TF-IDF)',
        'file': 'gca_net_model.pt',
        'loader': load_gca_model,
        'use_text': False,
    },
]


# ==================== 评估逻辑 ====================

def load_adversarial_dataset():
    """加载对抗数据集"""
    csv_path = os.path.join(DATASET_DIR, 'adversarial_test.csv')
    vec_path = os.path.join(DATASET_DIR, 'adversarial_test_vectorized.npz')

    if not os.path.exists(csv_path):
        print(f'[错误] 对抗数据集不存在: {csv_path}')
        print(f'       请先运行: python make_adversarial_dataset.py')
        return None, None, None

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    y_true = df['label'].values

    # 加载向量化版本
    if os.path.exists(vec_path):
        X_vec = load_npz(vec_path)
    else:
        print(f'[警告] 向量化版本不存在，将跳过需要 TF-IDF 的模型')
        X_vec = None

    return df, y_true, X_vec


def evaluate_model_on_adversarial(model, df, y_true, X_vec, use_text=False):
    """
    评估单个模型在对抗数据集上的表现
    返回按攻击类型分组的指标
    """
    # 预测
    if use_text:
        texts = df['adv_text'].tolist()
        y_pred = model.predict(texts)
    else:
        if X_vec is None:
            return None
        y_pred = model.predict(X_vec)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 总体指标
    overall = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }

    # 按攻击类型分组
    attack_results = []
    for atk_idx in sorted(df['attack_idx'].unique()):
        mask = df['attack_idx'] == atk_idx
        if mask.sum() == 0:
            continue
        atk_name = df.loc[df['attack_idx'] == atk_idx, 'attack_method'].iloc[0]
        atk_desc = df.loc[df['attack_idx'] == atk_idx, 'attack_desc'].iloc[0]
        y_t = y_true[mask]
        y_p = y_pred[mask]

        attack_results.append({
            'attack_idx': atk_idx,
            'attack_method': atk_name,
            'attack_desc': atk_desc,
            'samples': len(y_t),
            'accuracy': round(accuracy_score(y_t, y_p), 4),
            'precision_macro': round(precision_score(y_t, y_p, average='macro', zero_division=0), 4),
            'recall_macro': round(recall_score(y_t, y_p, average='macro', zero_division=0), 4),
            'f1_macro': round(f1_score(y_t, y_p, average='macro', zero_division=0), 4),
        })

    return {
        'overall': overall,
        'by_attack': pd.DataFrame(attack_results),
    }


def evaluate_confidence_threshold(model, df, y_true, X_vec, use_text=False, 
                                   thresholds=[0.90, 0.95]):
    """
    置信度阈值评估（对抗数据集上）
    """
    import torch
    
    # 获取概率
    if use_text:
        texts = df['adv_text'].tolist()
        proba = model.predict_proba(texts)
    else:
        if X_vec is None:
            return {}
        proba = model.predict_proba(X_vec)
    
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    
    results = {}
    for th in thresholds:
        mask = confidence >= th
        coverage = np.mean(mask)
        
        if np.sum(mask) == 0:
            results[f'recall@{int(th*100)}%'] = 0.0
            results[f'precision@{int(th*100)}%'] = 0.0
            results[f'f1@{int(th*100)}%'] = 0.0
            results[f'coverage@{int(th*100)}%'] = 0.0
        else:
            y_f = y_true[mask]
            p_f = pred[mask]
            results[f'recall@{int(th*100)}%'] = round(recall_score(y_f, p_f, average='macro', zero_division=0), 4)
            results[f'precision@{int(th*100)}%'] = round(precision_score(y_f, p_f, average='macro', zero_division=0), 4)
            results[f'f1@{int(th*100)}%'] = round(f1_score(y_f, p_f, average='macro', zero_division=0), 4)
            results[f'coverage@{int(th*100)}%'] = round(coverage, 4)
    
    return results


def print_adversarial_report(all_results):
    """打印对抗评估报告"""
    print(f'\n{"="*80}')
    print(f'  ★ 对抗数据集评估报告')
    print(f'{"="*80}')
    
    # ---- 表1: 总体对比 ----
    print(f'\n{"="*80}')
    print(f'  表1: 对抗样本总体指标')
    print(f'{"="*80}')
    rows = []
    for name, res in all_results.items():
        if res is None:
            continue
        rows.append({'Model': name, **res['overall']})
    df_overall = pd.DataFrame(rows)
    df_overall = df_overall.sort_values('f1_macro', ascending=False)
    print(df_overall.to_string(index=False))
    
    # ---- 表2: 按攻击类型分解（显示所有模型的） ----
    print(f'\n{"="*80}')
    print(f'  表2: 各攻击类型 F1 对比')
    print(f'{"="*80}')
    
    # 收集所有攻击类型
    all_attacks = set()
    for name, res in all_results.items():
        if res is None:
            continue
        for _, row in res['by_attack'].iterrows():
            all_attacks.add(row['attack_method'])
    
    # 构建以攻击类型为索引的对比表
    pivot_data = {}
    for name, res in all_results.items():
        if res is None:
            continue
        pivot_data[name] = {}
        for _, row in res['by_attack'].iterrows():
            pivot_data[name][row['attack_method']] = row['f1_macro']
    
    attack_df = pd.DataFrame(pivot_data).T
    attack_df.index.name = 'Model'
    print(attack_df.round(4).to_string())
    
    # ---- 表3: 置信度阈值（对抗数据集） ----
    print(f'\n{"="*80}')
    print(f'  表3: 对抗数据集 置信度阈值指标')
    print(f'{"="*80}')
    th_rows = []
    for name, res in all_results.items():
        if res is None or 'confidence' not in res:
            continue
        th_rows.append({'Model': name, **res['confidence']})
    if th_rows:
        th_df = pd.DataFrame(th_rows)
        print(th_df.to_string(index=False))
    
    return df_overall


def main():
    parser = argparse.ArgumentParser(description='对抗数据集评估脚本')
    parser.add_argument('--model', '-m', action='append', default=None,
                        help='指定评估的模型（可多次使用）')
    parser.add_argument('--export', '-e', action='store_true',
                        help='导出详细报告 CSV')
    args = parser.parse_args()
    
    # ---- 加载对抗数据集 ----
    print('='*60)
    print('  对抗数据集评估器')
    print('='*60)
    
    df, y_true, X_vec = load_adversarial_dataset()
    if df is None:
        return
    
    print(f'  数据集: {len(df)} 条对抗样本')
    print(f'  攻击类型数: {df["attack_idx"].nunique()}')
    print(f'  类别数: {df["label"].nunique()}')
    
    # ---- 加载模型 ----
    print(f'\n[模型加载]')
    all_results = {}
    
    for entry in MODEL_REGISTRY:
        model_path = os.path.join(MODEL_DIR, entry['file'])
        
        # 如果用户指定了模型，跳过不匹配的
        if args.model and entry['key'] not in args.model:
            continue
        
        if not os.path.exists(model_path):
            print(f'  [跳过] {entry["key"]}: 模型文件不存在 ({entry["file"]})')
            continue
        
        try:
            print(f'  加载: {entry["key"]} ...')
            model = entry['loader'](model_path)
            
            print(f'  评估: {entry["key"]} ...')
            res = evaluate_model_on_adversarial(
                model, df, y_true, X_vec,
                use_text=entry['use_text']
            )
            
            # 置信度阈值评估
            conf_res = evaluate_confidence_threshold(
                model, df, y_true, X_vec,
                use_text=entry['use_text'],
                thresholds=[0.90, 0.95]
            )
            
            all_results[entry['key']] = {
                'overall': res['overall'] if res else {},
                'by_attack': res['by_attack'] if res else pd.DataFrame(),
                'confidence': conf_res,
            }
            
            print(f'    总体 F1: {res["overall"]["f1_macro"] if res else "N/A"}')
            
        except Exception as e:
            print(f'  [错误] {entry["key"]}: {e}')
            import traceback
            traceback.print_exc()
    
    # ---- 打印报告 ----
    if not all_results:
        print(f'\n[错误] 没有成功加载任何模型！')
        print(f'       请先运行 main.py 训练模型，再运行本脚本评估。')
        return
    
    df_report = print_adversarial_report(all_results)
    
    # ---- 导出 ----
    if args.export and df_report is not None:
        export_path = os.path.join(OUTPUT_DIR, 'adversarial_eval_report.csv')
        df_report.to_csv(export_path, index=False, encoding='utf-8-sig')
        print(f'\n  报告已导出: {export_path}')


if __name__ == '__main__':
    main()
