"""
模型基类 — 定义训练、预测、序列化的统一接口
"""
import pickle
import numpy as np


class BaseModel:
    """所有模型的基类，提供 fit / predict / predict_proba / save / load 接口。

    input_type 指定模型接受的输入格式：
        'tfidf' — TF-IDF 稀疏或稠密矩阵
        'text'  — 原始文本字符串列表
    """
    def __init__(self, name: str, input_type: str = 'tfidf'):
        self.name = name
        self.model = None
        self.input_type = input_type

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
