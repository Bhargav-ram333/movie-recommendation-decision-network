"""
Root proxy module for recommender.
Re-exports all symbols from src.recommender.
"""
from src.recommender import *
from src.recommender import DecisionNetworkRecommender

__all__ = ["DecisionNetworkRecommender"]
