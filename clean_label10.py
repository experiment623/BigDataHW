"""
标签10清理脚本
==============
从所有数据集中删除 label=10 的行, 并同步更新配置文件和对抗数据集。

用法:
  python clean_label10.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (
    DATASET_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, CLASS_TXT,
    LABEL_MAP, NUM_CLASSES,
)

LABEL_TO_DROP = 10

print('=' * 60)
print(f'  清理 label={LABEL_TO_DROP} (新类型)')
print('=' * 60)

# ===================================================================
# 1. 清理数据集 CSV
# ===================================================================
stats = {}

for name, path in [
    ('训练集', TRAIN_PATH),
    ('验证集', VAL_PATH),
    ('测试集', TEST_PATH),
]:
    if not os.path.exists(path):
        print(f'  [跳过] {name}: 文件不存在')
        continue

    df = pd.read_csv(path, sep='\t', encoding='utf-8')
    before = len(df)

    # 查找 label 列名 (可能是 Label_id)
    label_col = 'Label_id' if 'Label_id' in df.columns else 'label'
    if label_col not in df.columns:
        print(f'  [跳过] {name}: 未找到标签列, 实际列: {list(df.columns)}')
        continue

    count_10 = (df[label_col] == LABEL_TO_DROP).sum()
    df = df[df[label_col] != LABEL_TO_DROP]
    after = len(df)

    # 保存
    df.to_csv(path, sep='\t', index=False, encoding='utf-8')
    stats[name] = (before, after, count_10)
    print(f'  {name}: {before} → {after} (删除 {count_10} 条 label={LABEL_TO_DROP})')

# ===================================================================
# 2. 更新 class.txt
# ===================================================================
if os.path.exists(CLASS_TXT):
    with open(CLASS_TXT, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = [l for l in lines if not l.strip().startswith(f'{LABEL_TO_DROP}\t')]
    with open(CLASS_TXT, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f'  class.txt: 已移除 label={LABEL_TO_DROP} 行')

# ===================================================================
# 3. 更新 config.py
# ===================================================================
config_path = os.path.join(os.path.dirname(__file__), 'config.py')
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 3a. 移除 LABEL_MAP 中的 10 行
    old_map_line = f"    {LABEL_TO_DROP}: '新类型'"
    new_content = content.replace(old_map_line + '\n', '')
    if old_map_line not in new_content:
        new_content = new_content.replace(old_map_line, '')

    # 3b. 修改 NUM_CLASSES: 11→10
    new_content = new_content.replace(
        'NUM_CLASSES = 11',
        'NUM_CLASSES = 10'
    )
    new_content = new_content.replace(
        '# 0-10 共11个类别',
        '# 0-9 共10个类别'
    )

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'  config.py: NUM_CLASSES=10, 已移除LABEL_MAP[10]')

# 同时更新运行时缓存
LABEL_MAP.pop(10, None)

# ===================================================================
# 4. 删除对抗数据集 (含 label=10 的旧版本)
# ===================================================================
for fname in ['adversarial_test.csv', 'adversarial_test_vectorized.npz']:
    path = os.path.join(DATASET_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        print(f'  已删除: dataset/{fname} (需重新生成)')

# ===================================================================
# 5. 汇总
# ===================================================================
print(f'\n{"=" * 60}')
print(f'  清理完成!')
print(f'{"=" * 60}')
for name, (before, after, count) in stats.items():
    print(f'  {name}: {before} → {after} 条 (删除{count}条 label=10)')
print(f'  NUM_CLASSES: 11 → 10')
print(f'  LABEL_MAP: 已移除 10:新类型')
print(f'\n  下一步: python make_adversarial_dataset.py (重新生成对抗数据)')
print(f'         python run_single.py --model lr  (验证清理成功)')
