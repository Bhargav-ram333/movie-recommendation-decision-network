"""
Root proxy module for data_loader.
Re-exports all symbols from src.data_loader for direct import compatibility.
"""
from src.data_loader import *
from src.data_loader import (
    load_all,
    download_movielens_100k,
    load_ratings,
    load_users,
    load_items,
    train_test_split_temporal,
    DATA_DIR,
    GENRE_COLUMNS,
)

__all__ = [
    "load_all",
    "download_movielens_100k",
    "load_ratings",
    "load_users",
    "load_items",
    "train_test_split_temporal",
    "DATA_DIR",
    "GENRE_COLUMNS",
]
