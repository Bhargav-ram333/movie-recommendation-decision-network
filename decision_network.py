"""
Root proxy module for decision_network.
Re-exports all symbols from src.decision_network.
"""
from src.decision_network import *
from src.decision_network import (
    DecisionNetwork,
    EDGES,
    UTILITY_TABLE,
    EU_THRESHOLD,
)

__all__ = [
    "DecisionNetwork",
    "EDGES",
    "UTILITY_TABLE",
    "EU_THRESHOLD",
]
