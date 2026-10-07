"""
feature_engineering.py
-----------------------
Builds user profiles, item profiles, and interaction features
used to condition the Bayesian Network nodes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer


# ──────────────────────────────────────────────────────────────────────────────
# User Profile Features
# ──────────────────────────────────────────────────────────────────────────────

def build_user_profiles(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-user statistics:
      - avg_rating, rating_count, genre preferences, activity level
    """
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    agg = merged.groupby("user_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        pct_positive=("rating_positive", "mean"),
        age=("age", "first"),
        gender=("gender", "first"),
        occupation=("occupation", "first"),
    ).reset_index()

    # favourite genre (most-rated)
    genre_counts = (
        merged.groupby("user_id")["primary_genre"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("fav_genre")
        .reset_index()
    )
    agg = agg.merge(genre_counts, on="user_id")

    # discretize activity (low / medium / high rater)
    agg["activity"] = pd.qcut(
        agg["rating_count"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )

    # discretize avg_rating bucket
    agg["rating_bucket"] = pd.cut(
        agg["avg_rating"],
        bins=[0, 2.5, 3.5, 5.0],
        labels=["harsh", "neutral", "generous"],
    )

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Item Profile Features
# ──────────────────────────────────────────────────────────────────────────────

def build_item_profiles(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-item statistics:
      - avg_rating, popularity, genre, release_decade
    """
    agg = merged.groupby("item_id").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        pct_positive=("rating_positive", "mean"),
        primary_genre=("primary_genre", "first"),
        release_year=("release_year", "first"),
        title=("title", "first"),
    ).reset_index()

    agg["popularity"] = pd.qcut(
        agg["rating_count"],
        q=3,
        labels=["niche", "moderate", "popular"],
        duplicates="drop",
    )

    agg["quality"] = pd.cut(
        agg["avg_rating"],
        bins=[0, 2.5, 3.5, 5.0],
        labels=["low", "medium", "high"],
    )

    agg["release_decade"] = (agg["release_year"] // 10 * 10).astype("Int32")

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Interaction Matrix (User-Item)
# ──────────────────────────────────────────────────────────────────────────────

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return sparse user×item pivot table of ratings."""
    matrix = ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0,
    )
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# Training Examples for Decision Network
# ──────────────────────────────────────────────────────────────────────────────

def build_dn_training_data(
    merged: pd.DataFrame,
    user_profiles: pd.DataFrame,
    item_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join merged ratings with discretised user + item profile features to
    produce one row per (user, item) interaction — ready for CPT estimation.
    
    Each row represents: given context → was the rating positive?

    Columns fed into the Decision Network:
        user_activity      : low / medium / high
        user_rating_bucket : harsh / neutral / generous
        user_gender        : M / F
        item_genre         : genre string
        item_popularity    : niche / moderate / popular
        item_quality       : low / medium / high
        rating_positive    : 0 / 1  (target / utility-linked variable)
    """
    up = user_profiles[[
        "user_id", "activity", "rating_bucket", "gender"
    ]].copy()
    up.columns = ["user_id", "user_activity", "user_rating_bucket", "user_gender"]

    ip = item_profiles[[
        "item_id", "primary_genre", "popularity", "quality"
    ]].copy()
    ip.columns = ["item_id", "item_genre", "item_popularity", "item_quality"]

    df = merged[["user_id", "item_id", "rating", "rating_positive"]].copy()
    df = df.merge(up, on="user_id").merge(ip, on="item_id")

    # Drop rows with NaN categories (items rated < 3 times may not be binned)
    df = df.dropna(subset=[
        "user_activity", "user_rating_bucket",
        "item_popularity", "item_quality"
    ])

    # Convert categoricals to plain strings for pgmpy
    for col in ["user_activity", "user_rating_bucket",
                "item_popularity", "item_quality"]:
        df[col] = df[col].astype(str)

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utility Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_utility(rating: float,
                    popularity_penalty: float = 0.1,
                    novelty_bonus: float = 0.05) -> float:
    """
    U(rating, item) = normalised_rating - popularity_penalty + novelty_bonus

    Encodes the idea that:
      - Higher ratings → higher utility
      - Very popular items have slight penalty (reduce over-concentration)
      - Novel items get a small bonus (diversity)
    """
    normalised = (rating - 1) / 4.0   # maps [1,5] → [0,1]
    return max(0.0, normalised - popularity_penalty + novelty_bonus)


if __name__ == "__main__":
    from src.data_loader import load_all
    ratings, users, items, merged = load_all()

    user_profiles = build_user_profiles(merged)
    item_profiles = build_item_profiles(merged)
    dn_data       = build_dn_training_data(merged, user_profiles, item_profiles)

    print("\n── User Profiles ──────────────────────────────────────")
    print(user_profiles.head())
    print("\n── Item Profiles ──────────────────────────────────────")
    print(item_profiles.head())
    print("\n── DN Training Data ───────────────────────────────────")
    print(dn_data.head())
    print(f"\nTraining rows: {len(dn_data):,}")
