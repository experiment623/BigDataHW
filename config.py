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

SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

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
MAX_TEXT_LEN = 256
MAX_VOCAB_SIZE = 30000
NGRAM_RANGE = (1, 2)

# 训练参数
RANDOM_SEED = 42

# BERT 模型
BERT_MODEL_NAME = 'hfl/chinese-macbert-base'

# 对抗样本参数
ADVERSARIAL_RATIO = 0.3
ADVERSARIAL_NUM = 200

# 慢速模型（Word2Vec/Doc2Vec）的训练子集
SUBSET_BASELINE_TRAIN = 30000
SUBSET_BERT_TRAIN = 20000
SUBSET_BERT_VAL = 5000
SUBSET_BERT_TEST = 5000

# 交叉验证参数
N_FOLDS = 5
CV_SAMPLE_SIZE = 20000

# 日志
LOG_FILE = os.path.join(LOG_DIR, 'experiment.log')
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
LOG_LEVEL = logging.INFO


def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
