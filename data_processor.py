"""
数据处理模块：数据加载、清洗、预处理
"""
import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, CLASS_TXT, LABEL_MAP,
    MAX_VOCAB_SIZE, NGRAM_RANGE, MAX_TEXT_LEN, OUTPUT_DIR
)

# 中文标点/特殊字符正则
CHINESE_PUNCT = re.compile(r'[，。！？；：""''【】《》（）…—\s]+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
NUM_PATTERN = re.compile(r'\d+')


def clean_text(text: str) -> str:
    """清洗中文文本"""
    if not isinstance(text, str):
        return ''
    text = URL_PATTERN.sub(' ', text)          # 去URL
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()   # 合并空格
    return text


def tokenize(text: str) -> str:
    """jieba 分词，空格分隔"""
    return ' '.join(jieba.cut(text))


def load_data(filepath: str, text_col: str = 'Text', label_col: str = 'Label_id') -> tuple:
    """
    加载 tab 分隔的 CSV 数据
    返回: (texts, labels) 原始文本和标签
    """
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    if label_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"列名不匹配，实际列: {list(df.columns)}")
    
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


def preprocess_texts(texts: list) -> list:
    """对文本列表做清洗+分词"""
    cleaned = [clean_text(t) for t in texts]
    tokenized = [tokenize(t) for t in cleaned]
    return tokenized


def build_tfidf_vectorizer(train_texts: list, save_path: str = None) -> TfidfVectorizer:
    """
    构建 TF-IDF 向量化器
    """
    vectorizer = TfidfVectorizer(
        max_features=MAX_VOCAB_SIZE,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        max_df=0.9,
        min_df=3
    )
    vectorizer.fit(train_texts)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
    return vectorizer


def load_vectorizer(path: str) -> TfidfVectorizer:
    with open(path, 'rb') as f:
        return pickle.load(f)


def stratified_sample(texts, labels, n_samples, random_state=42):
    """
    分层采样：保持类别分布的前提下，等比例抽取 n_samples 条样本

    参数:
        texts: 文本列表
        labels: 标签列表
        n_samples: 目标采样总数
        random_state: 随机种子

    返回:
        (sampled_texts, sampled_labels)
    """
    from sklearn.model_selection import train_test_split
    n_total = len(texts)
    if n_samples >= n_total:
        return texts, labels

    # 按比例分层分割（train_test_split 的 stratify 保持分布）
    _, sampled_texts, _, sampled_labels = train_test_split(
        texts, labels,
        train_size=(n_total - n_samples) / n_total,
        stratify=labels,
        random_state=random_state,
        shuffle=True
    )
    # 取未被选入 train 的部分作为样本
    return sampled_texts, sampled_labels


def stratified_sample_indices(labels, n_samples, random_state=42):
    """
    分层采样返回索引（用于 numpy 数组/稀疏矩阵的下标采样）

    返回:
        indices: 采样索引数组
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    n_total = len(labels)
    if n_samples >= n_total:
        return np.arange(n_total)
    
    indices = np.arange(n_total)
    _, sampled_idx = train_test_split(
        indices, train_size=(n_total - n_samples) / n_total,
        stratify=labels, random_state=random_state, shuffle=True
    )
    return sampled_idx


if __name__ == '__main__':
    # 快速测试
    train_texts, train_labels = load_data(TRAIN_PATH)
    print(f'Train: {len(train_texts)} samples')
    print(f'Label distribution: {pd.Series(train_labels).value_counts().sort_index().to_dict()}')

    val_texts, val_labels = load_data(VAL_PATH)
    print(f'Val: {len(val_texts)} samples')

    test_texts, test_labels = load_data(TEST_PATH)
    print(f'Test: {len(test_texts)} samples')
    print(f'Test labels unique: {sorted(set(test_labels))}')

    # 分词测试
    sample = train_texts[0]
    print(f'\n原始: {sample[:100]}')
    print(f'清洗: {clean_text(sample)[:100]}')
    print(f'分词: {tokenize(clean_text(sample))[:100]}')
