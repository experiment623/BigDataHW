"""
模型模块 — Baseline 模型与评估工具
"""
from .base import BaseModel
from .baselines import (
    Word2VecWordLR,
    Word2VecCharLR,
    Word2VecCharGBDT,
    Doc2VecCharGBDT,
    GAS,
)
from .evaluation import (
    evaluate_model,
    compare_models,
    evaluate_with_confidence_threshold,
    cross_validate,
    compute_metrics_full,
    save_predictions_csv,
    save_metrics_csv,
)

__all__ = [
    'BaseModel',
    'Word2VecWordLR', 'Word2VecCharLR',
    'Word2VecCharGBDT', 'Doc2VecCharGBDT',
    'GAS',
    'evaluate_model', 'compare_models',
    'evaluate_with_confidence_threshold', 'cross_validate',
    'compute_metrics_full', 'save_predictions_csv', 'save_metrics_csv',
]
