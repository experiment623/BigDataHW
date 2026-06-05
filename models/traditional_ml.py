"""
TF-IDF + 传统机器学习 Baseline 模型
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from config import RANDOM_SEED
from .base import BaseModel


class TfidfLogisticRegression(BaseModel):
    """TF-IDF + 逻辑回归"""
    def __init__(self):
        super().__init__('TF-IDF + LogisticRegression')
        self.model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced',
            random_state=RANDOM_SEED, n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class TfidfLinearSVC(BaseModel):
    """TF-IDF + 线性 SVM"""
    def __init__(self):
        super().__init__('TF-IDF + LinearSVC')
        self.model = LinearSVC(
            C=1.0, max_iter=2000, class_weight='balanced',
            random_state=RANDOM_SEED, dual=False
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        from scipy.special import softmax
        decision = self.model.decision_function(X)
        return softmax(decision, axis=1)


class TfidfRandomForest(BaseModel):
    """TF-IDF + 随机森林"""
    def __init__(self):
        super().__init__('TF-IDF + RandomForest')
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class TfidfNaiveBayes(BaseModel):
    """TF-IDF + 朴素贝叶斯"""
    def __init__(self):
        super().__init__('TF-IDF + MultinomialNB')
        self.model = MultinomialNB(alpha=0.5)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
