"""
标签分布查看脚本
================
用法: python show_label_dist.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import TRAIN_PATH, VAL_PATH, TEST_PATH, DATASET_DIR, LABEL_MAP

print('=' * 70)
print('  数据集标签分布')
print('=' * 70)

for name, path in [
    ('训练集 (ChiFraud_train)', TRAIN_PATH),
    ('验证集 (ChiFraud_t2022)', VAL_PATH),
    ('测试集 (ChiFraud_t2023)', TEST_PATH),
]:
    if not os.path.exists(path):
        print(f'\n  [跳过] {name}: 文件不存在')
        continue

    df = pd.read_csv(path, sep='\t', encoding='utf-8')
    label_col = 'Label_id' if 'Label_id' in df.columns else 'label'
    counts = df[label_col].value_counts().sort_index()
    total = len(df)

    print(f'\n  {name}  ({total} 条)')
    print(f'  {"Label":<6s} {"类别":<12s} {"数量":>8s}  {"占比":>8s}')
    print(f'  {"-"*40}')

    for lbl in range(counts.index.min(), counts.index.max() + 1):
        cnt = counts.get(lbl, 0)
        name_str = LABEL_MAP.get(lbl, '?')
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        print(f'  {lbl:<6d} {name_str:<12s} {cnt:>8d}  {pct:>6.1f}% {bar}')

# ── 对抗数据集 ──
adv_csv = os.path.join(DATASET_DIR, 'adversarial_test.csv')
if os.path.exists(adv_csv):
    adv_df = pd.read_csv(adv_csv, encoding='utf-8-sig')
    print(f'\n  对抗测试集 ({len(adv_df)} 条)')
    if 'label' in adv_df.columns:
        adv_counts = adv_df['label'].value_counts().sort_index()
        for lbl in sorted(adv_counts.index):
            cnt = adv_counts[lbl]
            print(f'    label={lbl:<4d} {LABEL_MAP.get(lbl, "?"):<12s} {cnt:>6d}')

    if 'attack_idx' in adv_df.columns:
        print(f'\n  攻击策略分布:')
        atk_counts = adv_df['attack_idx'].value_counts().sort_index()
        atk_names = {
            1: '形近字替换', 2: '随机删字', 3: '插入无关字符',
            4: '关键词掩码', 5: '正常文本伪装', 6: '数字全角混淆',
            7: '伪装URL添加', 8: '拼音字母简写',
        }
        for idx in sorted(atk_counts.index):
            print(f'    #{idx:<3d} {atk_names.get(idx, "?"):<12s} {atk_counts[idx]:>6d}')

print()
