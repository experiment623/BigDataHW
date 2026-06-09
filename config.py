"""
全局配置文件
"""
import os
import logging

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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
}

NUM_CLASSES = 10  # 0-9 共10个类别

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

# ========== 新增：实验控制参数 ==========

# 子集采样（保证类别分布，数值为最大采样数，None=全量）
SUBSET_TRAIN_SIZE = 30000   # 慢速模型（Word2Vec/Doc2Vec/GCA-Net）的训练子集
SUBSET_BERT_TRAIN = 20000   # BERT 训练子集
SUBSET_BERT_VAL = 5000      # BERT 验证子集
SUBSET_BERT_TEST = 5000     # BERT 测试子集

# 交叉验证参数（对关键模型做 K-Fold）
N_FOLDS = 5                 # K-Fold 折数
CV_SAMPLE_SIZE = 20000      # 交叉验证用子集大小（全量太慢时）
CV_MODELS = ['TF-IDF + LogisticRegression', 'TF-IDF + LinearSVC']  # 对哪些模型做CV

# GCA-Net 参数
GLYPH_DIM = 128             # 字形嵌入维度
GLYPH_WEIGHT = 0.3          # 字形特征在融合时的权重（TF-IDF 权重 = 1 - GLYPH_WEIGHT）
CONTRAST_WEIGHT = 0.1       # 对比学习损失权重
GCA_EPOCHS = 15             # GCA-Net 训练轮数

# 日志配置
LOG_FILE = os.path.join(LOG_DIR, 'experiment.log')
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
LOG_LEVEL = logging.INFO

def setup_logging():
    """配置全局日志（同时输出到文件和控制台）"""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
