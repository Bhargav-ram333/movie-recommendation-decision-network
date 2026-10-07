"""
recommender.py
--------------
Recommendation engine that wraps DecisionNetwork to:
  1. Score all un-seen items for a given user.
  2. Return Top-K recommendations with explanations.
  3. Support cold-start via genre/demographic priors.
"""

from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.decision_network import DecisionNetwork


# ──────────────────────────────────────────────────────────────────────────────
# Main Recommender
# ──────────────────────────────────────────────────────────────────────────────

class DecisionNetworkRecommender:
    """
    Wraps a fitted DecisionNetwork to generate top-K recommendations
    for any user in the training set, with per-recommendation explanations.
    """

    def __init__(self, dn: Any,
                 user_profiles: pd.DataFrame,
                 item_profiles: pd.DataFrame,
                 ratings: pd.DataFrame):
        """
        Parameters
        ----------
        dn             : Fitted DecisionNetwork
        user_profiles  : Output of build_user_profiles()
        item_profiles  : Output of build_item_profiles()
        ratings        : Raw ratings DataFrame (user_id, item_id, rating)
        """
        self.dn: DecisionNetwork = dn
        self.user_profiles = user_profiles.set_index("user_id")
        self.item_profiles = item_profiles.set_index("item_id")
        self.ratings       = ratings

        # Lookup: which items has each user already rated?
        self._seen = (
            ratings.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

    # ── Evidence Builder ──────────────────────────────────────────────────────

    def _build_evidence(self, user_id: int, item_id: int) -> dict:
        """
        Construct the evidence dict for the Decision Network query
        from user + item profile features.
        """
        up = self.user_profiles.loc[user_id]
        ip = self.item_profiles.loc[item_id]

        return {
            "user_activity":      str(up.get("activity",      "medium")),
            "user_rating_bucket": str(up.get("rating_bucket", "neutral")),
            "user_gender":        str(up.get("gender",        "M")),
            "item_genre":         str(ip.get("primary_genre", "Drama")),
            "item_popularity":    str(ip.get("popularity",    "moderate")),
            "item_quality":       str(ip.get("quality",       "medium")),
        }

    # ── Score All Items ───────────────────────────────────────────────────────

    def score_items(self, user_id: int,
                    candidate_item_ids: list[int] | None = None,
                    filter_seen: bool = True) -> pd.DataFrame:
        """
        Score every candidate item for a given user.

        Parameters
        ----------
        user_id           : Target user
        candidate_item_ids: Items to score (default: all items in profiles)
        filter_seen       : If True, exclude already-rated items

        Returns
        -------
        DataFrame with [item_id, title, prob_like, eu, decision, genre, ...]
        """
        if user_id not in self.user_profiles.index:
            raise ValueError(f"User {user_id} not found in profiles.")

        all_items = (
            candidate_item_ids
            if candidate_item_ids is not None
            else self.item_profiles.index.tolist()
        )

        seen = self._seen.get(user_id, set()) if filter_seen else set()
        candidates = [i for i in all_items if i not in seen]

        records = []
        for item_id in candidates:
            if item_id not in self.item_profiles.index:
                continue
            evidence = self._build_evidence(user_id, item_id)
            decision, eu, prob = self.dn.decide(evidence)
            ip = self.item_profiles.loc[item_id]
            records.append({
                "item_id":    item_id,
                "title":      ip.get("title", f"Movie {item_id}"),
                "genre":      ip.get("primary_genre", "Unknown"),
                "popularity": ip.get("popularity", "?"),
                "quality":    ip.get("quality", "?"),
                "prob_like":  round(prob, 4),
                "eu":         round(eu, 4),
                "decision":   decision,
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.sort_values("eu", ascending=False).reset_index(drop=True)

    # ── Top-K Recommendations ────────────────────────────────────────────────

    def recommend(self, user_id: int,
                  k: int = 10,
                  only_recommend: bool = True) -> pd.DataFrame:
        """
        Return Top-K recommendations for a user.

        Parameters
        ----------
        user_id        : Target user
        k              : Number of recommendations
        only_recommend : If True, only include items where decision='RECOMMEND'

        Returns
        -------
        DataFrame ranked by expected utility (descending)
        """
        scored = self.score_items(user_id)
        if only_recommend:
            scored = scored[scored["decision"] == "RECOMMEND"]
        return scored.head(k)

    # ── Explanation ──────────────────────────────────────────────────────────

    def explain(self, user_id: int, item_id: int) -> dict:
        """
        Return a human-readable explanation for why an item is/isn't recommended.

        Returns
        -------
        dict with keys: evidence, prob_like, eu, decision, reasoning
        """
        evidence         = self._build_evidence(user_id, item_id)
        decision, eu, prob = self.dn.decide(evidence)

        reasoning_parts = []

        # ── User signals ─────────────────────────────────────────────────────
        if evidence["user_activity"] == "high":
            reasoning_parts.append("You are an active rater (strong signal reliability).")
        if evidence["user_rating_bucket"] == "generous":
            reasoning_parts.append("You tend to rate generously — we expect a high rating.")
        elif evidence["user_rating_bucket"] == "harsh":
            reasoning_parts.append("You are a selective rater — we need high confidence to recommend.")

        # ── Item signals ──────────────────────────────────────────────────────
        ip = self.item_profiles.loc[item_id]
        reasoning_parts.append(f"This is a {evidence['item_quality']}-quality "
                               f"{evidence['item_popularity']} {evidence['item_genre']} film.")

        # ── Probability signal ────────────────────────────────────────────────
        reasoning_parts.append(
            f"Estimated probability you will like it: {prob:.1%}."
        )
        reasoning_parts.append(
            f"Expected Utility: {eu:+.3f} "
            f"({'above' if eu >= self.dn.utility_threshold else 'below'} "
            f"threshold of {self.dn.utility_threshold})."
        )

        return {
            "item_id":  item_id,
            "title":    ip.get("title", f"Movie {item_id}"),
            "evidence": evidence,
            "prob_like": round(prob, 4),
            "eu":        round(eu, 4),
            "decision":  decision,
            "reasoning": " ".join(reasoning_parts),
        }

    # ── Batch Scoring (for evaluation) ───────────────────────────────────────

    def batch_score(self, user_item_pairs: list[tuple[int, int]]) -> pd.DataFrame:
        """
        Score a list of (user_id, item_id) pairs — used for test-set evaluation.
        """
        records = []
        for uid, iid in tqdm(user_item_pairs, desc="Batch scoring"):
            if uid not in self.user_profiles.index:
                continue
            if iid not in self.item_profiles.index:
                continue
            evidence         = self._build_evidence(uid, iid)
            decision, eu, prob = self.dn.decide(evidence)
            records.append({
                "user_id":   uid,
                "item_id":   iid,
                "prob_like": prob,
                "eu":        eu,
                "decision":  decision,
            })
        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Cold-Start Fallback (Genre-Based)
# ──────────────────────────────────────────────────────────────────────────────

def cold_start_recommend(
    item_profiles: pd.DataFrame,
    preferred_genres: list[str],
    k: int = 10,
) -> pd.DataFrame:
    """
    For new users with no history: recommend top-rated popular movies
    in their preferred genres.
    """
    mask = item_profiles["primary_genre"].isin(preferred_genres)
    candidates = item_profiles[mask].copy()
    candidates = candidates[candidates["quality"] == "high"]
    candidates = candidates.sort_values("avg_rating", ascending=False)
    return candidates.head(k)[["item_id", "title", "primary_genre",
                                "avg_rating", "rating_count"]]


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

    ratings, users, items, merged = load_all()
    train_r, _ = train_test_split_temporal(ratings)
    train_merged = merged[merged.index.isin(train_r.index)]

    user_profiles = build_user_profiles(train_merged)
    item_profiles = build_item_profiles(train_merged)
    dn_data       = build_dn_training_data(train_merged, user_profiles, item_profiles)

    dn = DecisionNetwork(utility_threshold=0.3)
    dn.fit(dn_data, method="bayes")

    rec = DecisionNetworkRecommender(dn, user_profiles, item_profiles, train_r)

    # Recommend for user 1
    print("\n── Top-10 Recommendations for User 1 ──────────────────")
    top10 = rec.recommend(user_id=1, k=10)
    print(top10[["title", "genre", "prob_like", "eu"]].to_string(index=False))

    # Explain a recommendation
    if not top10.empty:
        item = top10.iloc[0]["item_id"]
        exp = rec.explain(user_id=1, item_id=int(item))
        print(f"\n── Explanation for '{exp['title']}' ────────────────────")
        print(exp["reasoning"])
