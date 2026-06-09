"""
模型基类
"""
import pickle
import numpy as np


class BaseModel:
    """所有模型的基类

    input_type 定义模型的期望输入格式:
        'tfidf' - 接受 TF-IDF 稀疏/稠密矩阵
        'text'  - 接受原始文本字符串列表
    """
    def __init__(self, name: str, input_type: str = 'tfidf'):
        self.name = name
        self.model = None
        self.input_type = input_type  # 'tfidf' 或 'text'

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
