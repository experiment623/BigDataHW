"""
全局配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 数据集路径
TRAIN_PATH = os.path.join(DATASET_DIR, 'ChiFraud_train.csv')
VAL_PATH = os.path.join(DATASET_DIR, 'ChiFraud_t2022.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'ChiFraud_t2023.csv')
CLASS_TXT = os.path.join(DATASET_DIR, 'class.txt')

# 类别映射（从 class.txt）
LABEL_MAP = {
    0: '正常',
    1: '赌博博彩',
    2: '招嫖色情',
    3: '办假证',
    4: '虚假办卡',
    5: '违禁药品交易',
    6: '违规提现',
    7: '虚假证明',
    8: '虚假手机卡',
    9: '地下黑贷',
    10: '新类型'
}

NUM_CLASSES = 11  # 0-10 共11个类别

# 数据预处理参数
MAX_TEXT_LEN = 256          # 文本最大长度（字符数）
MAX_VOCAB_SIZE = 30000      # TF-IDF 词表大小
NGRAM_RANGE = (1, 2)        # n-gram 范围

# 训练参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# BERT 模型（中文预训练模型）
BERT_MODEL_NAME = 'bert-base-chinese'

# 对抗样本参数
ADVERSARIAL_RATIO = 0.3     # 对抗样本替换比例
ADVERSARIAL_NUM = 200       # 每类生成对抗样本数
