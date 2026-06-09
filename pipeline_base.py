"""
Pipeline 共享基础模块
=====================
提供所有独立运行脚本共用的数据加载、预处理、评估工具。
每个模型脚本只需 import 此模块，就能获得准备好的数据。
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_DIR, MODEL_DIR, DATASET_DIR,
    RANDOM_SEED, LABEL_MAP, NUM_CLASSES, BERT_MODEL_NAME,
    SUBSET_TRAIN_SIZE, SUBSET_BERT_TRAIN, SUBSET_BERT_VAL, SUBSET_BERT_TEST,
    GLYPH_WEIGHT, CONTRAST_WEIGHT, GCA_EPOCHS, N_FOLDS,
)
from data_processor import (
    load_data, preprocess_texts, build_tfidf_vectorizer,
    stratified_sample_indices,
)
from models.evaluation import evaluate_model, evaluate_with_confidence_threshold

np.random.seed(RANDOM_SEED)

# 全局缓存：避免重复加载数据
_data_cache = {}


def load_all_data():
    """加载全部原始数据 (带缓存)"""
    if 'raw' in _data_cache:
        return _data_cache['raw']

    print('\n[Pipeline] 加载数据集...')
    train_texts, train_labels = load_data(TRAIN_PATH)
    val_texts, val_labels = load_data(VAL_PATH)
    test_texts, test_labels = load_data(TEST_PATH)

    print(f'  训练集: {len(train_texts)} 条')
    print(f'  验证集: {len(val_texts)} 条')
    print(f'  测试集: {len(test_texts)} 条')

    _data_cache['raw'] = (
        (train_texts, train_labels),
        (val_texts, val_labels),
        (test_texts, test_labels),
    )
    return _data_cache['raw']


def build_tfidf_data(train_texts, val_texts, test_texts):
    """构建 TF-IDF 特征矩阵 (带缓存)"""
    if 'tfidf' in _data_cache:
        return _data_cache['tfidf']

    print('\n[Pipeline] 构建 TF-IDF 特征...')
    train_tokens = preprocess_texts(train_texts)
    val_tokens = preprocess_texts(val_texts)
    test_tokens = preprocess_texts(test_texts)

    vectorizer = build_tfidf_vectorizer(
        train_tokens,
        save_path=os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    )
    print(f'  词汇表: {len(vectorizer.vocabulary_)}')

    X_train = vectorizer.transform(train_tokens)
    X_val = vectorizer.transform(val_tokens)
    X_test = vectorizer.transform(test_tokens)

    result = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'vectorizer': vectorizer,
        'train_texts': train_texts, 'val_texts': val_texts, 'test_texts': test_texts,
    }
    _data_cache['tfidf'] = result
    return result


def get_prepared_data():
    """
    一键获取全部准备好的数据

    返回:
        proc: {
            'X_train', 'X_val', 'X_test',
            'y_train', 'y_val', 'y_test',
            'vectorizer',
            'train_texts', 'val_texts', 'test_texts',
        }
    """
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_all_data()
    proc = build_tfidf_data(train_texts, val_texts, test_texts)
    proc['y_train'] = np.array(train_labels)
    proc['y_val'] = np.array(val_labels)
    proc['y_test'] = np.array(test_labels)
    return proc


def subset_data(texts, labels, X_tfidf, n_samples):
    """分层采样子集"""
    if n_samples is None or n_samples >= len(texts):
        return texts, labels, X_tfidf
    indices = stratified_sample_indices(labels, n_samples, RANDOM_SEED)
    sub_texts = [texts[i] for i in indices]
    sub_labels = labels[indices]
    sub_X = X_tfidf[indices]
    return sub_texts, sub_labels, sub_X


def load_adversarial_testset():
    """加载预生成对抗测试集"""
    from scipy.sparse import load_npz
    
    adv_csv = os.path.join(DATASET_DIR, 'adversarial_test.csv')
    adv_vec = os.path.join(DATASET_DIR, 'adversarial_test_vectorized.npz')

    if not os.path.exists(adv_csv):
        print('[跳过] 对抗测试集不存在, 请先运行: python make_adversarial_dataset.py')
        return None, None, []

    adv_df = pd.read_csv(adv_csv, encoding='utf-8-sig')
    y_adv = np.array(adv_df['label'].values)
    X_adv = load_npz(adv_vec) if os.path.exists(adv_vec) else None

    print(f'[Pipeline] 加载对抗测试集: {len(adv_df)} 条')
    return adv_df, y_adv, X_adv


def evaluate_on_adversarial_testset(model, adv_df, y_adv, X_adv,
                                     adv_texts_list=None):
    """在预生成对抗测试集上评估单个模型"""
    from sklearn.metrics import accuracy_score, f1_score

    if model.input_type == 'text' and adv_texts_list:
        y_pred = np.array(model.predict(adv_texts_list))
    elif X_adv is not None:
        y_pred = np.array(model.predict(X_adv))
    else:
        return None

    # 总体指标
    overall = {
        'accuracy': round(accuracy_score(y_adv, y_pred), 4),
        'f1_macro': round(f1_score(y_adv, y_pred, average='macro', zero_division=0), 4),
    }

    # 按攻击类型分解
    by_attack = {}
    for atk_idx in sorted(adv_df['attack_idx'].unique()):
        mask = adv_df['attack_idx'] == atk_idx
        if mask.sum() == 0: continue
        atk_name = adv_df.loc[mask, 'attack_method'].iloc[0]
        y_t, y_p = y_adv[mask], y_pred[mask]
        by_attack[atk_name] = round(
            f1_score(y_t, y_p, average='macro', zero_division=0), 4
        )

    return {'overall': overall, 'by_attack': by_attack}


def evaluate_thresholds(model, X, y, texts=None):
    """评估置信度阈值指标 @90% @95%"""
    if model.input_type == 'text' and texts is not None:
        r90 = evaluate_with_confidence_threshold(model, texts, y, 0.90)
        r95 = evaluate_with_confidence_threshold(model, texts, y, 0.95)
    else:
        r90 = evaluate_with_confidence_threshold(model, X, y, 0.90)
        r95 = evaluate_with_confidence_threshold(model, X, y, 0.95)
    return {'r90': r90, 'r95': r95}


def print_single_model_report(model, test_result, adv_result=None, thr_result=None):
    """打印单个模型的完整报告"""
    print(f'\n{"="*60}')
    print(f'  ★ {model.name} 评估报告')
    print(f'{"="*60}')

    print('\n  清洁测试集:')
    for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']:
        if k in test_result:
            print(f'    {k:<20s}: {test_result[k]:.4f}')

    if adv_result:
        print('\n  对抗测试集:')
        print(f'    总体准确率: {adv_result["overall"]["accuracy"]:.4f}')
        print(f'    总体F1:     {adv_result["overall"]["f1_macro"]:.4f}')

    if thr_result:
        print(f'\n  置信度阈值:')
        r90 = thr_result.get('r90', {})
        r95 = thr_result.get('r95', {})
        print(f'    recall@90%: {r90.get("recall@90%", 0):.4f}  coverage: {r90.get("coverage", 0):.4f}')
        print(f'    recall@95%: {r95.get("recall@95%", 0):.4f}  coverage: {r95.get("coverage", 0):.4f}')
