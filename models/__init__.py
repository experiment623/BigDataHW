"""
模型模块
包含所有 baseline 和创新模型
"""
from .base import BaseModel
from .traditional_ml import (
    TfidfLogisticRegression, TfidfLinearSVC,
    TfidfRandomForest, TfidfNaiveBayes
)
from .neural_net import SimpleMLP
from .bert_model import BERTClassifier
from .pdf_baselines import (
    Word2VecWordLR,      # Word2Vec-w+LR
    Word2VecCharLR,      # Word2Vec-c+LR
    Word2VecCharGBDT,    # Word2Vec-c+GBDT
    Doc2VecCharGBDT,     # Doc2Vec-c+GBDT
    GAS,                 # GAS (GCN-based)
)
from .gca_net import GCANet
from .evaluation import evaluate_model, compare_models, evaluate_with_confidence_threshold, cross_validate

__all__ = [
    'BaseModel',
    'TfidfLogisticRegression', 'TfidfLinearSVC',
    'TfidfRandomForest', 'TfidfNaiveBayes',
    'SimpleMLP',
    'BERTClassifier',
    'Word2VecWordLR', 'Word2VecCharLR',
    'Word2VecCharGBDT', 'Doc2VecCharGBDT',
    'GAS',
    'GCANet',
    'evaluate_model', 'compare_models', 'evaluate_with_confidence_threshold',
    'cross_validate',
]
