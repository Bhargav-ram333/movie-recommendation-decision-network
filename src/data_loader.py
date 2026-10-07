"""
data_loader.py
--------------
Downloads MovieLens 100K dataset and preprocesses it into
pandas DataFrames ready for feature engineering and analysis.
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).parent.parent / "data"

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

# ──────────────────────────────────────────────────────────────────────────────
# Download & Extract
# ──────────────────────────────────────────────────────────────────────────────

def download_movielens_100k(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract MovieLens 100K if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / "ml-100k"

    if extracted_dir.exists():
        print(f"[✓] Dataset already downloaded at {extracted_dir}")
        return extracted_dir

    print(f"[↓] Downloading MovieLens 100K from {MOVIELENS_100K_URL}...")
    response = requests.get(MOVIELENS_100K_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    chunks = []
    with tqdm(total=total, unit="iB", unit_scale=True, desc="Downloading") as bar:
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            bar.update(len(chunk))

    print("[✓] Extracting zip...")
    with zipfile.ZipFile(io.BytesIO(b"".join(chunks))) as zf:
        zf.extractall(data_dir)

    print(f"[✓] Extracted to {extracted_dir}")
    return extracted_dir


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_ratings(ml_dir: Path) -> pd.DataFrame:
    """Load u.data → user_id, item_id, rating, timestamp."""
    path = ml_dir / "u.data"
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": np.int32, "item_id": np.int32,
               "rating": np.float32, "timestamp": np.int64},
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_users(ml_dir: Path) -> pd.DataFrame:
    """Load u.user → demographic info."""
    path = ml_dir / "u.user"
    df = pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"user_id": np.int32, "age": np.int32},
    )
    return df


def load_items(ml_dir: Path) -> pd.DataFrame:
    """Load u.item → movie metadata + genre binary flags."""
    path = ml_dir / "u.item"
    cols = ["item_id", "title", "release_date", "video_release_date",
            "imdb_url"] + GENRE_COLUMNS
    df = pd.read_csv(
        path,
        sep="|",
        names=cols,
        encoding="latin-1",
        dtype={"item_id": np.int32},
    )
    df["release_year"] = pd.DatetimeIndex(
        pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce")
    ).year.astype("Int32")
    df["genres"] = df[GENRE_COLUMNS].apply(
        lambda row: [g for g, v in zip(GENRE_COLUMNS, row) if v == 1], axis=1
    )
    df["primary_genre"] = df["genres"].apply(
        lambda gs: gs[0] if gs else "Unknown"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Unified Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_all(data_dir: Path = DATA_DIR):
    """
    Download + load everything, return (ratings_df, users_df, items_df, merged_df).
    """
    ml_dir = download_movielens_100k(data_dir)
    ratings = load_ratings(ml_dir)
    users   = load_users(ml_dir)
    items   = load_items(ml_dir)

    merged = (
        ratings
        .merge(users, on="user_id")
        .merge(items[["item_id", "title", "primary_genre",
                       "genres", "release_year"]], on="item_id")
    )

    # ── Derived columns ──────────────────────────────────────────────────────
    merged["rating_positive"] = (merged["rating"] >= 4).astype(np.int8)
    merged["age_group"] = pd.cut(
        merged["age"],
        bins=[0, 18, 25, 35, 50, 100],
        labels=["<18", "18-25", "25-35", "35-50", "50+"],
    )
    merged["month"] = pd.DatetimeIndex(merged["datetime"]).month
    merged["weekday"] = pd.DatetimeIndex(merged["datetime"]).dayofweek

    print(f"\n[✓] Dataset loaded:")
    print(f"    Ratings : {len(ratings):,}")
    print(f"    Users   : {users['user_id'].nunique():,}")
    print(f"    Movies  : {items['item_id'].nunique():,}")
    print(f"    Genres  : {len(GENRE_COLUMNS)}")

    return ratings, users, items, merged


# ──────────────────────────────────────────────────────────────────────────────
# Train / Test Split
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split_temporal(ratings: pd.DataFrame,
                               test_ratio: float = 0.2,
                               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split: for each user the most recent (test_ratio) of their
    ratings go into the test set — mimics real deployment.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for user_id, group in ratings.groupby("user_id"):
        group_sorted = group.sort_values("timestamp")
        n = len(group_sorted)
        n_test = max(1, int(n * test_ratio))
        test_rows  = group_sorted.tail(n_test).index.tolist()
        train_rows = group_sorted.head(n - n_test).index.tolist()
        train_idx.extend(train_rows)
        test_idx.extend(test_rows)

    return ratings.loc[train_idx].copy(), ratings.loc[test_idx].copy()


if __name__ == "__main__":
    ratings, users, items, merged = load_all()
    train, test = train_test_split_temporal(ratings)
    print(f"\n    Train : {len(train):,}  |  Test : {len(test):,}")
