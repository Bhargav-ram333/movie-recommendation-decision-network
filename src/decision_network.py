"""
decision_network.py
-------------------
Implements a Decision Network (Bayesian Network + Utility + Decision nodes)
for movie recommendation using pgmpy.

Architecture
============

Chance Nodes (Bayesian Network):
    user_activity     → rating_positive
    user_rating_bucket→ rating_positive
    user_gender       → item_genre
    item_genre        → rating_positive
    item_popularity   → rating_positive
    item_quality      → rating_positive

Utility Node:
    rating_positive → Utility(recommend vs skip)

Decision Node:
    Recommend / Skip
    (chosen by argmax Expected Utility over recommendations)
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
else:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork   # older pgmpy < 0.1.26
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Network Structure
# ──────────────────────────────────────────────────────────────────────────────

EDGES = [
    # User characteristics → rating outcome
    ("user_activity",      "rating_positive"),
    ("user_rating_bucket", "rating_positive"),
    ("item_genre",         "rating_positive"),
    ("item_popularity",    "rating_positive"),
    ("item_quality",       "rating_positive"),
    # Cross-user/item influence
    ("user_gender",        "item_genre"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Utility Function
# ──────────────────────────────────────────────────────────────────────────────

UTILITY_TABLE = {
    # rating_positive → utility value
    "1": +1.0,   # positive rating → high utility for recommending
    "0": -0.5,   # negative rating → cost for a bad recommendation
}

EU_THRESHOLD = 0.3  # Default expected utility recommendation threshold


# ──────────────────────────────────────────────────────────────────────────────
# DecisionNetwork Class
# ──────────────────────────────────────────────────────────────────────────────

class DecisionNetwork:
    """
    A Decision Network built on top of pgmpy's BayesianNetwork.

    Decision Rule (Maximum Expected Utility):
        EU(recommend | evidence) = Σ_r  P(rating_positive=r | evidence) · U(r)
        → Recommend if EU(recommend) > threshold
    """

    def __init__(self, utility_threshold: float = 0.4):
        """
        Parameters
        ----------
        utility_threshold : float
            Minimum expected utility to issue a recommendation (default 0.4).
        """
        self.model = BayesianNetwork(EDGES)
        self.inference: VariableElimination | None = None
        self.utility_threshold = utility_threshold
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame, method: str = "bayes") -> "DecisionNetwork":
        """
        Estimate Conditional Probability Tables from training data.

        Parameters
        ----------
        data   : DataFrame with columns matching EDGES nodes + 'rating_positive'
        method : 'mle' for Maximum Likelihood, 'bayes' for Bayesian (Dirichlet)
        """
        required = {
            "user_activity", "user_rating_bucket", "user_gender",
            "item_genre", "item_popularity", "item_quality", "rating_positive"
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in training data: {missing}")

        # pgmpy needs string types
        train = data[list(required)].astype(str)

        if method == "bayes":
            # pgmpy ≥0.1.26: instantiate estimator separately, then get CPDs
            estimator = BayesianEstimator(self.model, train)
            cpds = [
                estimator.estimate_cpd(
                    node,
                    prior_type="BDeu",
                    equivalent_sample_size=5,
                )
                for node in self.model.nodes()
            ]
            self.model.add_cpds(*cpds)
        else:
            estimator = MaximumLikelihoodEstimator(self.model, train)
            cpds = [estimator.estimate_cpd(node) for node in self.model.nodes()]
            self.model.add_cpds(*cpds)

        self.inference = VariableElimination(self.model)
        self._fitted = True
        print("[✓] Decision Network fitted successfully.")
        print(f"    Nodes : {list(self.model.nodes())}")
        print(f"    Edges : {list(self.model.edges())}")
        return self

    # ── Posterior P(rating_positive | evidence) ──────────────────────────────

    def predict_proba(self, evidence: dict) -> float:
        """
        Return P(rating_positive=1 | evidence).

        Parameters
        ----------
        evidence : dict of {node_name: state_value}  (all strings)
        """
        if not self._fitted or self.inference is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Remove keys not in the network
        valid_evidence = {
            k: str(v) for k, v in evidence.items()
            if k in self.model.nodes() and k != "rating_positive"
        }

        try:
            result = self.inference.query(
                variables=["rating_positive"],
                evidence=valid_evidence,
                show_progress=False,
            )
            # result.values[i] maps to sorted states
            states = result.state_names["rating_positive"]
            state_map = {s: i for i, s in enumerate(states)}
            prob_positive = result.values[state_map.get("1", 1)]
            return float(prob_positive)
        except Exception as e:
            # Fallback: base rate
            return 0.5

    # ── Expected Utility ──────────────────────────────────────────────────────

    def expected_utility(self, evidence: dict) -> float:
        """
        EU(recommend | evidence) = P(pos=1|e)·U(1) + P(pos=0|e)·U(0)
        """
        p_pos = self.predict_proba(evidence)
        p_neg = 1.0 - p_pos
        eu = p_pos * UTILITY_TABLE["1"] + p_neg * UTILITY_TABLE["0"]
        return eu

    # ── Decision ──────────────────────────────────────────────────────────────

    def decide(self, evidence: dict) -> tuple[str, float, float]:
        """
        Apply the MEU decision rule.

        Returns
        -------
        decision : 'RECOMMEND' or 'SKIP'
        eu       : expected utility score
        prob     : P(rating_positive=1 | evidence)
        """
        eu   = self.expected_utility(evidence)
        prob = self.predict_proba(evidence)
        decision = "RECOMMEND" if eu >= self.utility_threshold else "SKIP"
        return decision, eu, prob

    # ── CPT Summary ──────────────────────────────────────────────────────────

    def print_cpds(self):
        """Print all Conditional Probability Distributions."""
        for cpd in self.model.cpds:
            print(cpd)
            print()

    # ── Network Graph ─────────────────────────────────────────────────────────

    def get_network_graph(self) -> nx.DiGraph:
        """Return a NetworkX DiGraph of the Bayesian Network structure."""
        return nx.DiGraph(EDGES)

    # ── Markov Blanket ────────────────────────────────────────────────────────

    def markov_blanket(self, node: str) -> list[str]:
        """Return the Markov blanket of a node (parents ∪ children ∪ co-parents)."""
        return list(self.model.get_markov_blanket(node))

    # ── d-Separation Test ─────────────────────────────────────────────────────

    def is_d_separated(self, x: str, y: str, given: list[str]) -> bool:
        """Check if X ⊥ Y | given using d-separation."""
        return self.model.local_independencies(x).contains(y, given)

    # ── Serialise / Deserialise ───────────────────────────────────────────────

    def save(self, path: str):
        """Save the fitted network (CPDs) to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[✓] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DecisionNetwork":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[✓] Model loaded from {path}")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import load_all, train_test_split_temporal
    from src.feature_engineering import (
        build_user_profiles, build_item_profiles, build_dn_training_data
    )

    ratings, users, items, merged = load_all()
    train_r, test_r = train_test_split_temporal(ratings)
    train_merged = merged[merged.index.isin(train_r.index)]

    user_profiles = build_user_profiles(train_merged)
    item_profiles = build_item_profiles(train_merged)
    dn_data       = build_dn_training_data(train_merged, user_profiles, item_profiles)

    dn = DecisionNetwork(utility_threshold=0.3)
    dn.fit(dn_data, method="bayes")

    # Example query
    evidence = {
        "user_activity":      "high",
        "user_rating_bucket": "generous",
        "user_gender":        "M",
        "item_genre":         "Action",
        "item_popularity":    "popular",
        "item_quality":       "high",
    }

    decision, eu, prob = dn.decide(evidence)
    print(f"\nEvidence : {evidence}")
    print(f"P(like)  : {prob:.3f}")
    print(f"E[U]     : {eu:.3f}")
    print(f"Decision : {decision}")
