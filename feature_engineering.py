"""
Root proxy module for feature_engineering.
Re-exports all symbols from src.feature_engineering.
"""
from src.feature_engineering import *
from src.feature_engineering import (
    build_user_profiles,
    build_item_profiles,
    build_user_item_matrix,
    build_dn_training_data,
    compute_content_utility,
)

__all__ = [
    "build_user_profiles",
    "build_item_profiles",
    "build_user_item_matrix",
    "build_dn_training_data",
    "compute_content_utility",
]
