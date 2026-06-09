"""
对抗样本生成模块
通过多种策略生成对抗数据集，测试模型鲁棒性
"""
import random
import numpy as np
import pandas as pd
from config import ADVERSARIAL_RATIO, ADVERSARIAL_NUM, RANDOM_SEED, LABEL_MAP

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----- 形近字字典（基于 Unicode 异体/形近字符） -----
HOMOGRAPH_MAP = {
    # 数字大写
    '零': '〇', '一': '壹', '二': '贰', '三': '叁', '四': '肆', '五': '伍',
    '六': '陆', '七': '柒', '八': '捌', '九': '玖', '十': '拾',
    '万': '萬', '千': '仟', '百': '佰',
    # 金融/欺诈常见字
    '元': '圆', '块': '元', '钱': '銭', '付': '付', '费': '費',
    '卡': '咔', '证': '証', '贷': '貸', '款': '欵', '欠': '欠',
    # 药品/博彩/色情
    '药': '薬', '博': '愽', '彩': '採', '赌': '賭', '色': '脃',
    '码': '碼', '微': '薇', '信': '伩', '加': '伽', '群': '羣',
    '电': '電', '话': '話', '手': '扌', '网': '網', '视': '視',
    # 扩展形近字（Unicode CJK Compatibility）
    '国': '國', '对': '対', '写': '冩', '买': '買', '卖': '賣',
    '号': '號', '学': '學', '宝': '寶', '实': '實', '专': '專',
    '业': '業', '发': '發', '报': '報', '车': '車', '转': '轉',
    '账': '賬', '账': '帳', '银': '銀', '行': '行', '提': '提',
    '现': '現', '点': '點', '击': '擊', '链': '鏈', '接': '接',
    '红': '紅', '包': '飽', '赠': '贈', '送': '餸', '免': '俛',
    '费': '費', '优': '優', '惠': '憄', '秒': '皊', '杀': '殺',
}

# ----- 中文同音近音字（扩展版） -----
HOMOPHONE_MAP = {
    '博': '搏', '彩': '采', '赌': '堵', '码': '马', '号': '好',
    '加': '家', '微': '危', '信': '心', '贷': '代', '款': '宽',
    '证': '正', '卡': '咖', '药': '要', '钱': '前', '元': '原',
    '下': '夏', '中': '忠', '上': '尚', '大': '达', '小': '晓',
    '新': '辛', '快': '块', '来': '莱', '手': '首', '网': '往',
    '送': '宋', '出': '初', '入': '如', '开': '凯', '关': '冠',
    # 扩展同音字
    '提': '题', '现': '线', '点': '典', '秒': '妙', '包': '保',
    '红': '宏', '免': '棉', '费': '飞', '优': '幽', '惠': '慧',
    '专': '砖', '业': '页', '银': '吟', '转': '赚', '接': '街',
    '赠': '曾', '群': '裙', '发': '罚', '国': '果', '买': '埋',
}

# ----- 混淆字符（看起来像但不同的字符） -----
CONFUSABLE_CHARS = {
    '0': 'O', '1': 'l', '2': 'Z', '5': 'S', '8': 'B',
    'O': '0', 'l': '1', 'Z': '2', 'S': '5', 'B': '8',
    '。': '.', '，': ',', '！': '!', '？': '?', '；': ';', '：': ':',
}


def char_similar_substitution(text: str, ratio: float = ADVERSARIAL_RATIO) -> str:
    """
    策略1：形近/音近字替换
    """
    all_map = {**HOMOGRAPH_MAP, **HOMOPHONE_MAP}
    chars = list(text)
    n = len(chars)
    indices = random.sample(range(n), max(1, int(n * ratio)))
    for idx in indices:
        c = chars[idx]
        if c in all_map:
            chars[idx] = all_map[c]
    return ''.join(chars)


def char_deletion(text: str, ratio: float = 0.1) -> str:
    """
    策略2：随机删除字符
    """
    chars = list(text)
    n = len(chars)
    if n <= 2:
        return text
    indices = random.sample(range(n), max(1, int(n * ratio)))
    return ''.join([c for i, c in enumerate(chars) if i not in indices])


def char_insertion(text: str, ratio: float = 0.05) -> str:
    """
    策略3：随机插入无关字符/空格
    """
    chars = list(text)
    n = len(chars)
    insert_count = max(1, int(n * ratio))
    for _ in range(insert_count):
        pos = random.randint(0, len(chars))
        # 插入空格或常用字
        insert_char = random.choice([' ', '的', '了', '是', '一', '。', '！'])
        chars.insert(pos, insert_char)
    return ''.join(chars)


def text_masking(text: str, ratio: float = 0.15) -> str:
    """
    策略4：关键词部分掩码（用 [MASK] 或 *** 替代）
    """
    chars = list(text)
    n = len(chars)
    mask_count = max(1, int(n * ratio))
    indices = random.sample(range(n), mask_count)
    for i, idx in enumerate(sorted(indices)):
        # 保持位置，用掩码符号
        chars[idx] = random.choice(['*', '#', 'X'])
    return ''.join(chars)


def add_filler_text(text: str, filler: str = '祝您生活愉快，如有疑问请联系客服。') -> str:
    """
    策略5：添加无关正常文本，混淆检测（模拟欺诈文本伪装）
    """
    pos = random.choice(['start', 'end', 'middle'])
    if pos == 'start':
        return filler + text
    elif pos == 'end':
        return text + filler
    else:
        mid = len(text) // 2
        return text[:mid] + filler + text[mid:]


def number_obfuscation(text: str) -> str:
    """
    策略6：数字混淆（用全角数字或中文数字替换，规避规则检测）
    """
    num_map = {
        '0': '０', '1': '１', '2': '２', '3': '３', '4': '４',
        '5': '５', '6': '６', '7': '７', '8': '８', '9': '９',
    }
    chars = list(text)
    for i, c in enumerate(chars):
        if c.isdigit():
            chars[i] = num_map.get(c, c)
    return ''.join(chars)


def url_obfuscation(text: str) -> str:
    """
    策略7：URL 混淆（空格分隔、短链接样式）
    """
    # 给文本添加伪装的 URL
    prefixes = ['详情点击', '更多请访问', '查看', '链接', '网址', '点我看']
    suffix = random.choice(['.com', '.cn', '.net', '.top', '.vip', '.xyz'])
    fake_url = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6)) + suffix
    append = f' {random.choice(prefixes)} {fake_url}'
    return text + append


# ----- 拼音/字母简写映射 (策略8) -----
PINYIN_ABBREV_MAP = {
    '微': 'v',     '信': 'x',      '微信': 'vx',
    '支': 'z',     '付': 'f',      '宝': 'b',
    '扣': 'k',     '扣扣': 'kk',   '扣扣': 'qq',
    '加': '+',     '钱': 'q',      '元': 'y',
    '万': 'w',     '新': 'x',      '人': 'r',
    '手': 's',     '机': 'j',
}


def pinyin_abbreviation(text: str, ratio: float = 0.15) -> str:
    """
    策略8：拼音首字母/简写替换
    将部分关键词替换为拼音首字母或符号简写
    例: '微信'→'v信'  '支付宝'→'z付b'  '加微信'→'+vx'
    """
    result = list(text)
    n = len(result)
    # 对较长词先做整体替换
    for word, abbrev in [('微信', 'vx'), ('扣扣', 'qq')]:
        if word in text and random.random() < 0.5:
            result = ''.join(result).replace(word, abbrev)
            result = list(result)
    
    # 单个字符替换
    indices = random.sample(range(n), max(1, int(n * ratio)))
    for idx in indices:
        ch = result[idx]
        if ch in PINYIN_ABBREV_MAP:
            result[idx] = PINYIN_ABBREV_MAP[ch]
    return ''.join(result)


def generate_adversarial_sample(text: str, label: int) -> list:
    """
    对一个文本生成多种对抗变体
    返回: [(adversarial_text, original_label, attack_method), ...]
    """
    variants = []
    # 只对欺诈文本（label != 0）生成对抗样本
    # 正常文本也做轻微扰动，观察模型是否误判
    if label != 0:
        variants.append((char_similar_substitution(text), label, 'char_sub'))
        variants.append((char_deletion(text), label, 'char_del'))
        variants.append((number_obfuscation(text), label, 'num_obf'))
        variants.append((add_filler_text(text), label, 'add_filler'))
        variants.append((url_obfuscation(text), label, 'url_obf'))
        variants.append((pinyin_abbreviation(text), label, 'pinyin_abv'))  # 新增
    else:
        # 正常文本加轻微扰动
        variants.append((char_insertion(text), label, 'char_ins'))
        variants.append((text_masking(text), label, 'text_mask'))

    return variants


def build_adversarial_dataset(texts: list, labels: list, samples_per_class: int = ADVERSARIAL_NUM) -> tuple:
    """
    构建完整对抗数据集
    返回: (adv_texts, adv_labels, adv_attack_types)
    """
    adv_texts = []
    adv_labels = []
    adv_attack_types = []

    # 按标签分组
    label_to_texts = {}
    for t, l in zip(texts, labels):
        label_to_texts.setdefault(l, []).append(t)

    for label, t_list in label_to_texts.items():
        n_sample = min(samples_per_class, len(t_list))
        sampled = random.sample(t_list, n_sample)
        for text in sampled:
            variants = generate_adversarial_sample(text, label)
            for v_text, v_label, v_method in variants:
                adv_texts.append(v_text)
                adv_labels.append(v_label)
                adv_attack_types.append(v_method)

    print(f'\n[对抗样本生成] 总计 {len(adv_texts)} 条对抗样本')
    if adv_attack_types:
        from collections import Counter
        print(f'  攻击类型分布: {dict(Counter(adv_attack_types))}')

    return adv_texts, adv_labels, adv_attack_types


def evaluate_on_adversarial(model, adv_texts, adv_labels, adv_attack_types):
    """
    评估模型在对抗数据集上的表现，按攻击类型分组
    """
    from models.evaluation import evaluate_model
    import pandas as pd

    all_pred = model.predict(adv_texts)
    all_pred = np.array(all_pred)
    adv_labels = np.array(adv_labels)

    # 按攻击类型统计
    attack_results = []
    unique_attacks = sorted(set(adv_attack_types))
    for atk in unique_attacks:
        idx = [i for i, a in enumerate(adv_attack_types) if a == atk]
        pred_atk = all_pred[idx]
        true_atk = adv_labels[idx]
        acc = np.mean(pred_atk == true_atk)
        attack_results.append({
            'attack_method': atk,
            'samples': len(idx),
            'accuracy': round(acc, 4)
        })

    df = pd.DataFrame(attack_results)
    print(f'\n{"="*60}')
    print(f'  对抗鲁棒性评估 - {model.name}')
    print(f'{"="*60}')
    print(df.to_string(index=False))

    # 总体
    overall_acc = np.mean(all_pred == adv_labels)
    print(f'\n  对抗样本总体准确率: {overall_acc:.4f}')
    return df, overall_acc


if __name__ == '__main__':
    # 对抗样本生成测试
    test_text = "专业办证，信用卡提现，联系微信12345"
    print(f'原始: {test_text}')
    for method, func in [
        ('char_sub', char_similar_substitution),
        ('char_del', char_deletion),
        ('num_obf', number_obfuscation),
        ('add_filler', add_filler_text),
    ]:
        print(f'  {method}: {func(test_text)}')
