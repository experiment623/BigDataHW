"""
数据处理模块：数据加载、清洗、预处理
"""
import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, CLASS_TXT, LABEL_MAP,
    MAX_VOCAB_SIZE, NGRAM_RANGE, MAX_TEXT_LEN, OUTPUT_DIR, DATASET_DIR
)

CHINESE_PUNCT = re.compile(r'[，。！？；：""''【】《》（）…—\s]+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = URL_PATTERN.sub(' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> str:
    return ' '.join(jieba.cut(text))


def load_data(filepath: str, text_col: str = 'Text', label_col: str = 'Label_id') -> tuple:
    """加载 tab 分隔的 CSV 数据，返回 (texts_list, labels_array)"""
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    if label_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"列名不匹配，实际列: {list(df.columns)}")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).to_numpy()
    return texts, labels


def preprocess_texts(texts: list) -> list:
    cleaned = [clean_text(t) for t in texts]
    return [tokenize(t) for t in cleaned]


def build_tfidf_vectorizer(train_texts: list, save_path: str = None) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=MAX_VOCAB_SIZE, ngram_range=NGRAM_RANGE,
        sublinear_tf=True, max_df=0.9, min_df=3
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
    from sklearn.model_selection import train_test_split
    n_total = len(texts)
    if n_samples >= n_total:
        return texts, labels
    _, sampled_texts, _, sampled_labels = train_test_split(
        texts, labels, train_size=(n_total - n_samples) / n_total,
        stratify=labels, random_state=random_state, shuffle=True
    )
    return sampled_texts, sampled_labels


def stratified_sample_indices(labels, n_samples, random_state=42):
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


def load_adversarial_data():
    """加载对抗测试集，返回 (df, adv_texts_list, y_labels_array)"""
    adv_path = os.path.join(DATASET_DIR, 'adversarial_test.csv')
    if not os.path.exists(adv_path):
        print(f'[跳过] 对抗测试集不存在: {adv_path}')
        return None, [], np.array([])
    df = pd.read_csv(adv_path, encoding='utf-8-sig')
    adv_texts = df['adv_text'].astype(str).tolist()
    y_adv = df['label'].astype(int).to_numpy()
    print(f'[对抗数据] 加载 {len(adv_texts)} 条对抗样本')
    return df, adv_texts, y_adv


if __name__ == '__main__':
    train_texts, train_labels = load_data(TRAIN_PATH)
    print(f'Train: {len(train_texts)} samples')
    val_texts, val_labels = load_data(VAL_PATH)
    print(f'Val: {len(val_texts)} samples')
    test_texts, test_labels = load_data(TEST_PATH)
    print(f'Test: {len(test_texts)} samples')

    sample = train_texts[0]
    print(f'\n原始: {sample[:100]}')
    print(f'清洗: {clean_text(sample)[:100]}')
    print(f'分词: {tokenize(clean_text(sample))[:100]}')
