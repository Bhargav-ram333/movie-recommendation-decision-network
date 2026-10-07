"""
Root proxy module for evaluation.
Re-exports all symbols from src.evaluation.
"""
from src.evaluation import *
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage,
    novelty_score,
    evaluate_recommender,
    print_evaluation_report,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "catalog_coverage",
    "novelty_score",
    "evaluate_recommender",
    "print_evaluation_report",
]
