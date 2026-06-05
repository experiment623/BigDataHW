"""快速测试脚本"""
import sys, traceback
sys.path.insert(0, r'c:\Users\86132\Desktop\hw\大数据原理与技术\Final_hw')

def test_load():
    from data_processor import load_data
    from config import TRAIN_PATH, VAL_PATH, TEST_PATH
    
    for name, path in [('Train', TRAIN_PATH), ('Val', VAL_PATH), ('Test', TEST_PATH)]:
        texts, labels = load_data(path)
        print(f'{name}: {len(texts)} samples, labels={sorted(set(labels))}')

def test_preprocess():
    from data_processor import clean_text, tokenize, preprocess_texts
    from config import TRAIN_PATH
    from data_processor import load_data
    
    texts, _ = load_data(TRAIN_PATH)
    sample = texts[0]
    print(f'Raw: {sample[:100]}')
    cleaned = clean_text(sample)
    print(f'Cleaned: {cleaned[:100]}')
    tokens = tokenize(cleaned)
    print(f'Tokenized: {tokens[:100]}')

def test_vectorizer():
    from data_processor import load_data, preprocess_texts, build_tfidf_vectorizer
    from config import TRAIN_PATH
    import numpy as np
    from scipy.sparse import issparse
    
    texts, labels = load_data(TRAIN_PATH)
    tokens = preprocess_texts(texts[:5000])  # 子集测试
    vec = build_tfidf_vectorizer(tokens)
    print(f'Vocab size: {len(vec.vocabulary_)}')
    X = vec.transform(tokens)
    print(f'TF-IDF shape: {X.shape}, is sparse: {issparse(X)}')

def test_adversarial():
    from adversarial import (
        char_similar_substitution, char_deletion, number_obfuscation,
        add_filler_text, generate_adversarial_sample
    )
    
    fraud_text = '专业办证，信用卡提现，联系微信12345快速办理'
    normal_text = '今天天气真好，适合出去散步锻炼身体'
    
    print('=== 欺诈文本对抗 ===')
    variants = generate_adversarial_sample(fraud_text, label=1)
    for v_text, v_label, v_method in variants:
        print(f'  [{v_method}] {v_text[:80]}')
    
    print('\n=== 正常文本对抗 ===')
    variants2 = generate_adversarial_sample(normal_text, label=0)
    for v_text, v_label, v_method in variants2:
        print(f'  [{v_method}] {v_text[:80]}')

if __name__ == '__main__':
    try:
        print('=== 测试1: 数据加载 ===')
        test_load()
        
        print('\n=== 测试2: 预处理 ===')
        test_preprocess()
        
        print('\n=== 测试3: TF-IDF ===')
        test_vectorizer()
        
        print('\n=== 测试4: 对抗样本 ===')
        test_adversarial()
        
        print('\n✅ 所有测试通过!')
    except Exception as e:
        print(f'❌ 错误: {e}')
        traceback.print_exc()
