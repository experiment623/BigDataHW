"""
对抗数据集生成脚本
===================
从原始测试集(2023)生成独立的对抗测试集，持久化为 CSV 文件。

用法:
  python make_adversarial_dataset.py                      # 默认参数
  python make_adversarial_dataset.py --samples 300        # 每类300条
  python make_adversarial_dataset.py --all                # 使用全部测试集

输出文件:
  dataset/adversarial_test.csv           对抗样本（原始文本）
  dataset/adversarial_test_vectorized.npz 向量化版本（预计算 TF-IDF）

数据列:
  adv_text       对抗扰动后的文本
  label          原始标签 (0-10)
  original_text  扰动前原文
  attack_method  攻击策略名称
  attack_idx     攻击策略序号 (1-7)
  1=char_sub, 2=char_del, 3=char_ins, 4=text_mask,
  5=add_filler, 6=num_obf, 7=url_obf
"""
import os
import sys
import random
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 确保工作目录正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TEST_PATH, DATASET_DIR, TRAIN_PATH,
    RANDOM_SEED, LABEL_MAP, MAX_VOCAB_SIZE, NGRAM_RANGE
)
from data_processor import load_data, preprocess_texts, build_tfidf_vectorizer

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== 对抗攻击策略（从 adversarial.py 精简内嵌） ====================

HOMOGRAPH_MAP = {
    '零': '〇', '一': '壹', '二': '贰', '三': '叁', '四': '肆', '五': '伍',
    '六': '陆', '七': '柒', '八': '捌', '九': '玖', '十': '拾',
    '万': '萬', '千': '仟', '百': '佰',
    '元': '圆', '块': '元', '钱': '銭',
    '卡': '咔', '证': '証', '贷': '貸', '款': '欵',
    '药': '薬', '博': '愽', '彩': '採', '赌': '賭',
    '码': '碼', '微': '薇', '信': '伩', '加': '伽',
    '电': '電', '话': '話', '手': '扌',
}

HOMOPHONE_MAP = {
    '博': '搏', '彩': '采', '赌': '堵', '码': '马', '号': '好',
    '加': '家', '微': '危', '信': '心', '贷': '代', '款': '宽',
    '证': '正', '卡': '咖', '药': '要', '钱': '前', '元': '原',
    '下': '夏', '中': '忠', '上': '尚', '大': '达', '小': '晓',
    '新': '辛', '快': '块', '来': '莱', '手': '首', '网': '往',
    '送': '宋', '出': '初', '入': '如', '开': '凯', '关': '冠',
}

CONFUSABLE_CHARS = {
    '0': 'O', '1': 'l', '2': 'Z', '5': 'S', '8': 'B',
    'O': '0', 'l': '1', 'Z': '2', 'S': '5', 'B': '8',
    '。': '.', '，': ',', '！': '!', '？': '?', '；': ';', '：': ':',
}

NUM_OBF_MAP = {
    '0': '０', '1': '１', '2': '２', '3': '３', '4': '４',
    '5': '５', '6': '６', '7': '７', '8': '８', '9': '９',
}


def char_similar_substitution(text: str, ratio: float = 0.3) -> str:
    """策略1：形近/音近字替换"""
    all_map = {**HOMOGRAPH_MAP, **HOMOPHONE_MAP}
    chars = list(text)
    n = len(chars)
    if n == 0:
        return text
    indices = random.sample(range(n), max(1, int(n * ratio)))
    for idx in indices:
        c = chars[idx]
        if c in all_map:
            chars[idx] = all_map[c]
    return ''.join(chars)


def char_deletion(text: str, ratio: float = 0.1) -> str:
    """策略2：随机删除字符"""
    chars = list(text)
    n = len(chars)
    if n <= 2:
        return text
    indices = random.sample(range(n), max(1, int(n * ratio)))
    return ''.join([c for i, c in enumerate(chars) if i not in indices])


def char_insertion(text: str, ratio: float = 0.05) -> str:
    """策略3：随机插入无关字符"""
    chars = list(text)
    n = len(chars)
    insert_count = max(1, int(n * ratio))
    for _ in range(insert_count):
        pos = random.randint(0, len(chars))
        insert_char = random.choice([' ', '的', '了', '是', '一', '。', '！'])
        chars.insert(pos, insert_char)
    return ''.join(chars)


def text_masking(text: str, ratio: float = 0.15) -> str:
    """策略4：关键词掩码"""
    chars = list(text)
    n = len(chars)
    mask_count = max(1, int(n * ratio))
    indices = random.sample(range(n), mask_count)
    for idx in sorted(indices):
        chars[idx] = random.choice(['*', '#', 'X'])
    return ''.join(chars)


def add_filler_text(text: str, filler: str = '祝您生活愉快，如有疑问请联系客服。') -> str:
    """策略5：添加无关正常文本伪装"""
    pos = random.choice(['start', 'end', 'middle'])
    if pos == 'start':
        return filler + text
    elif pos == 'end':
        return text + filler
    else:
        mid = len(text) // 2
        return text[:mid] + filler + text[mid:]


def number_obfuscation(text: str) -> str:
    """策略6：数字全角混淆"""
    chars = list(text)
    for i, c in enumerate(chars):
        if c.isdigit():
            chars[i] = NUM_OBF_MAP.get(c, c)
    return ''.join(chars)


def url_obfuscation(text: str) -> str:
    """策略7：添加伪装 URL"""
    prefixes = ['详情点击', '更多请访问', '查看', '链接', '网址', '点我看']
    suffix = random.choice(['.com', '.cn', '.net', '.top', '.vip', '.xyz'])
    fake_url = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6)) + suffix
    append = f' {random.choice(prefixes)} {fake_url}'
    return text + append


# 攻击策略注册表
ATTACK_STRATEGIES = [
    (1, 'char_sub',   '形近字替换',   lambda t: char_similar_substitution(t)),
    (2, 'char_del',   '随机删字',     lambda t: char_deletion(t)),
    (3, 'char_ins',   '插入无关字符', lambda t: char_insertion(t)),
    (4, 'text_mask',  '关键词掩码',   lambda t: text_masking(t)),
    (5, 'add_filler', '正常文本伪装', lambda t: add_filler_text(t)),
    (6, 'num_obf',    '数字全角混淆', lambda t: number_obfuscation(t)),
    (7, 'url_obf',    '伪装URL添加',  lambda t: url_obfuscation(t)),
]

ATTACK_TARGET_MAP = {
    # label=0 正常文本：只做轻微扰动（插入/掩码），不生成欺诈类攻击
    0: [3, 4],  # char_ins, text_mask
    # label!=0 欺诈文本：使用全部 7 种攻击
    1: [1, 2, 3, 4, 5, 6, 7],
}


# ==================== 主逻辑 ====================

def load_label_info():
    """加载类别信息"""
    class_file = os.path.join(DATASET_DIR, 'class.txt')
    id_to_name = {}
    if os.path.exists(class_file):
        with open(class_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        id_to_name[int(parts[0])] = parts[1]
    return id_to_name


def make_adversarial_dataset(samples_per_class: int = 200, use_all: bool = False):
    """
    构建持久化的对抗测试集

    Args:
        samples_per_class: 每类从测试集中采样的数量
        use_all: 是否使用全部测试集（会生成大量样本）

    Returns:
        adv_df: 对抗样本 DataFrame
    """
    print('=' * 60)
    print('  对抗数据集生成器')
    print('=' * 60)

    # ---- 1. 加载原始测试集 ----
    print('\n[1/5] 加载原始测试集...')
    test_texts, test_labels = load_data(TEST_PATH)
    print(f'  测试集大小: {len(test_texts)} 条')
    print(f'  标签分布: {dict(pd.Series(test_labels).value_counts().sort_index())}')

    # ---- 2. 按标签分组采样 ----
    print(f'\n[2/5] 分组采样（每类{samples_per_class}条）...')
    label_to_texts = {}
    for t, l in zip(test_texts, test_labels):
        label_to_texts.setdefault(l, []).append(t)

    sampled = {}
    for label in sorted(label_to_texts.keys()):
        texts = label_to_texts[label]
        if use_all:
            sampled[label] = texts
        else:
            n = min(samples_per_class, len(texts))
            sampled[label] = random.sample(texts, n)
        label_name = LABEL_MAP.get(label, f'类别{label}')
        print(f'  类别 {label}({label_name}): {len(sampled[label])} 条原文')

    total_original = sum(len(v) for v in sampled.values())
    print(f'  采样总计: {total_original} 条原文')

    # ---- 3. 生成对抗样本 ----
    print(f'\n[3/5] 对每条原文应用攻击策略...')
    rows = []
    for label, texts in sorted(sampled.items()):
        attack_ids = ATTACK_TARGET_MAP.get(label, [1, 2, 3, 4, 5, 6, 7])
        label_name = LABEL_MAP.get(label, f'类别{label}')

        for original_text in texts:
            for atk_id in attack_ids:
                _, atk_code, atk_desc, atk_func = ATTACK_STRATEGIES[atk_id - 1]
                adv_text = atk_func(original_text)
                rows.append({
                    'adv_text': adv_text,
                    'label': label,
                    'label_name': label_name,
                    'original_text': original_text,
                    'attack_method': atk_code,
                    'attack_desc': atk_desc,
                    'attack_idx': atk_id,
                })

    adv_df = pd.DataFrame(rows)
    print(f'  生成对抗样本: {len(adv_df)} 条')
    print(f'  攻击类型分布:')
    for atk_id, atk_code, atk_desc, _ in ATTACK_STRATEGIES:
        count = (adv_df['attack_idx'] == atk_id).sum()
        print(f'    {atk_id}. {atk_code:<12s} ({atk_desc:<10s}): {count} 条')

    # ---- 4. 保存原始文本版 ----
    print(f'\n[4/5] 保存对抗数据集 (文本)...')
    adv_text_path = os.path.join(DATASET_DIR, 'adversarial_test.csv')
    adv_df.to_csv(adv_text_path, index=False, encoding='utf-8-sig')
    print(f'  已保存: {adv_text_path} ({adv_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)')

    # ---- 5. 生成并保存向量化版本 ----
    print(f'\n[5/5] 预计算 TF-IDF 向量化...')
    # 用训练集构建 vectorizer（保证一致性）
    train_texts, _ = load_data(TRAIN_PATH)
    train_tokens = preprocess_texts(train_texts)
    vectorizer = build_tfidf_vectorizer(
        train_tokens,
        save_path=os.path.join(DATASET_DIR, 'tfidf_vectorizer.pkl')
    )
    print(f'  词汇表大小: {len(vectorizer.vocabulary_)}')

    # 向量化对抗文本
    adv_tokens = preprocess_texts(adv_df['adv_text'].tolist())
    X_adv = vectorizer.transform(adv_tokens)
    print(f'  向量化维度: {X_adv.shape}')

    # 保存稀疏矩阵
    from scipy.sparse import save_npz
    vec_path = os.path.join(DATASET_DIR, 'adversarial_test_vectorized.npz')
    save_npz(vec_path, X_adv)
    print(f'  已保存: {vec_path} ({os.path.getsize(vec_path)/1024**2:.1f} MB)')

    # ---- 6. 打印摘要 ----
    print(f'\n{"="*60}')
    print(f'  对抗数据集生成完成！')
    print(f'{"="*60}')
    print(f'')
    print(f'  输出文件:')
    print(f'    {adv_text_path}')
    print(f'    {vec_path}')
    print(f'    {os.path.join(DATASET_DIR, "tfidf_vectorizer.pkl")}')
    print(f'')
    print(f'  数据集统计:')
    print(f'    总样本数:   {len(adv_df)}')
    print(f'    原始文本数: {total_original}')
    print(f'    攻击策略数: {len(ATTACK_STRATEGIES)}')
    print(f'    列: {list(adv_df.columns)}')
    print(f'')
    print(f'  标签分布:')
    for label in sorted(adv_df['label'].unique()):
        cnt = (adv_df['label'] == label).sum()
        name = LABEL_MAP.get(label, f'类别{label}')
        print(f'    类别 {label}({name}): {cnt} 条')

    return adv_df, X_adv, vectorizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='对抗数据集生成器')
    parser.add_argument('--samples', '-n', type=int, default=200,
                        help='每类采样数量 (默认: 200)')
    parser.add_argument('--all', action='store_true',
                        help='使用全部测试集（会生成大量样本）')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    make_adversarial_dataset(
        samples_per_class=args.samples,
        use_all=args.all,
    )
