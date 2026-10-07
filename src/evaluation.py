"""
evaluation.py
-------------
Standard recommendation system evaluation metrics:
  - RMSE / MAE          (rating prediction accuracy)
  - Precision@K         (fraction of recommended items that are relevant)
  - Recall@K            (fraction of relevant items that are recommended)
  - NDCG@K              (ranking quality)
  - Coverage            (catalogue coverage)
  - Hit Rate@K          (at least one relevant item in top-K)
  - AUC-ROC             (binary like/dislike classification)
  - Novelty             (average popularity rank of recommendations)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error


# ──────────────────────────────────────────────────────────────────────────────
# Rating Accuracy
# ──────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ──────────────────────────────────────────────────────────────────────────────
# Ranking Metrics (per-user, then macro-averaged)
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    top_k = recommended[:k]
    dcg   = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: all relevant items at the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """1 if at least one relevant item is in top-K, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


# ──────────────────────────────────────────────────────────────────────────────
# System-Level Metrics
# ──────────────────────────────────────────────────────────────────────────────

def catalogue_coverage(all_recommendations: list[list[int]],
                       total_items: int) -> float:
    """Fraction of the item catalogue that appears in any recommendation list."""
    recommended_items = set(
        item for recs in all_recommendations for item in recs
    )
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def novelty(recommendations: list[int],
            item_popularity: dict[int, int],
            n_users: int) -> float:
    """
    Average self-information (surprisal) of recommended items.
    Higher novelty → less popular items recommended (more novel).
    """
    scores = []
    for item_id in recommendations:
        pop = item_popularity.get(item_id, 1) / n_users
        scores.append(-np.log2(pop + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Full Evaluation Run
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_recommender(
    recommender,
    test_ratings: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 200,
) -> dict:
    """
    Evaluate the recommender on a held-out test set.

    Parameters
    ----------
    recommender         : DecisionNetworkRecommender instance (already fitted)
    test_ratings        : Test split DataFrame with user_id, item_id, rating
    k                   : Cut-off for ranking metrics
    relevance_threshold : Ratings >= this are considered "relevant"
    max_users           : Max number of users to evaluate (for speed)

    Returns
    -------
    dict of metric_name → value
    """
    # Build ground-truth relevant set per user
    relevant_per_user = (
        test_ratings[test_ratings["rating"] >= relevance_threshold]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # Select evaluation users (must be in both test set and user_profiles)
    eval_users = [
        uid for uid in relevant_per_user
        if uid in recommender.user_profiles.index
    ][:max_users]

    prec_scores, rec_scores, ndcg_scores, hr_scores = [], [], [], []
    all_recs = []

    for uid in eval_users:
        try:
            recs_df  = recommender.recommend(user_id=uid, k=k, only_recommend=False)
            rec_list = recs_df["item_id"].tolist()
        except Exception:
            continue

        relevant = relevant_per_user.get(uid, set())
        if not relevant:
            continue

        prec_scores.append(precision_at_k(rec_list, relevant, k))
        rec_scores.append(recall_at_k(rec_list, relevant, k))
        ndcg_scores.append(ndcg_at_k(rec_list, relevant, k))
        hr_scores.append(hit_rate_at_k(rec_list, relevant, k))
        all_recs.append(rec_list)

    # AUC on the binary prob_like predictions
    batch_pairs = list(zip(test_ratings["user_id"], test_ratings["item_id"]))[:2000]
    scored = recommender.batch_score(batch_pairs)
    merged_eval = scored.merge(
        test_ratings[["user_id", "item_id", "rating"]],
        on=["user_id", "item_id"],
    )
    merged_eval["label"] = (merged_eval["rating"] >= relevance_threshold).astype(int)

    try:
        auc = roc_auc_score(merged_eval["label"], merged_eval["prob_like"])
    except Exception:
        auc = float("nan")

    # Predicted ratings (prob_like → scale back to 1–5)
    merged_eval["pred_rating"] = 1 + merged_eval["prob_like"] * 4
    rmse_val = rmse(merged_eval["rating"], merged_eval["pred_rating"])
    mae_val  = mae(merged_eval["rating"],  merged_eval["pred_rating"])

    # Coverage
    n_items = len(recommender.item_profiles)
    coverage = catalogue_coverage(all_recs, n_items)

    results = {
        f"precision@{k}": float(np.mean(prec_scores)) if prec_scores else 0.0,
        f"recall@{k}":    float(np.mean(rec_scores))  if rec_scores  else 0.0,
        f"ndcg@{k}":      float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"hit_rate@{k}":  float(np.mean(hr_scores))   if hr_scores   else 0.0,
        "auc_roc":   round(auc, 4),
        "rmse":      round(rmse_val, 4),
        "mae":       round(mae_val, 4),
        "coverage":  round(coverage, 4),
        "n_users_evaluated": len(prec_scores),
    }

    # Round ranking metrics
    for key in [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"]:
        results[key] = round(results[key], 4)

    return results


def print_evaluation_report(results: dict):
    """Pretty-print the evaluation results."""
    print("\n" + "═" * 50)
    print("  EVALUATION RESULTS")
    print("═" * 50)
    for metric, value in results.items():
        if metric == "n_users_evaluated":
            print(f"  {'Users Evaluated':<28} {value}")
        else:
            label = metric.upper().replace("_", " ").replace("@", "@")
            print(f"  {label:<28} {value:.4f}" if isinstance(value, float)
                  else f"  {label:<28} {value}")
    print("═" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from src.data_loader import load_all, train_test_split_temporal
    from src.feature_engineering import (
        build_user_profiles, build_item_profiles, build_dn_training_data
    )
    from src.decision_network import DecisionNetwork
    from src.recommender import DecisionNetworkRecommender

    ratings, users, items, merged = load_all()
    train_r, test_r = train_test_split_temporal(ratings)
    train_merged = merged[merged.index.isin(train_r.index)]

    user_profiles = build_user_profiles(train_merged)
    item_profiles = build_item_profiles(train_merged)
    dn_data       = build_dn_training_data(train_merged, user_profiles, item_profiles)

    dn  = DecisionNetwork(utility_threshold=0.3)
    dn.fit(dn_data, method="bayes")

    rec = DecisionNetworkRecommender(dn, user_profiles, item_profiles, train_r)
    results = evaluate_recommender(rec, test_r, k=10, max_users=100)
    print_evaluation_report(results)
