"""
模型基类
"""
import pickle
import numpy as np


class BaseModel:
    """所有模型的基类"""
    def __init__(self, name: str):
        self.name = name
        self.model = None

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
