"""
run_pipeline.py
---------------
One-shot script to run the complete pipeline:
    1. Download MovieLens 100K
    2. Feature engineering
    3. Fit Decision Network
    4. Evaluate on test set
    5. Print Top-10 recommendations for 3 sample users

Usage:
    cd "MATHS PROJECT"
    python run_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_all, train_test_split_temporal
from src.feature_engineering import (
    build_user_profiles, build_item_profiles, build_dn_training_data
)
from src.decision_network import DecisionNetwork
from src.recommender import DecisionNetworkRecommender
from src.evaluation import evaluate_recommender, print_evaluation_report


def divider(title=""):
    width = 65
    if title:
        print(f"\n{'═'*width}")
        print(f"  {title}")
        print(f"{'═'*width}")
    else:
        print(f"{'─'*width}")


def main():
    print("""
  ██████╗██╗███╗   ██╗███████╗███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗
 ██╔════╝██║████╗  ██║██╔════╝████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║
 ██║     ██║██╔██╗ ██║█████╗  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║
 ██║     ██║██║╚██╗██║██╔══╝  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║
 ╚██████╗██║██║ ╚████║███████╗██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗
  ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
  Decision Network Recommendation System · MovieLens 100K · Project 18
    """)

    # ── STEP 1: Load Data ─────────────────────────────────────
    divider("STEP 1 · Loading Data")
    ratings, users, items, merged = load_all()
    train_r, test_r = train_test_split_temporal(ratings, test_ratio=0.20)
    train_merged = merged[merged.index.isin(train_r.index)]
    print(f"  Train: {len(train_r):,} ratings | Test: {len(test_r):,} ratings")

    # ── STEP 2: Feature Engineering ───────────────────────────
    divider("STEP 2 · Feature Engineering")
    user_profiles = build_user_profiles(train_merged)
    item_profiles  = build_item_profiles(train_merged)
    dn_data        = build_dn_training_data(train_merged, user_profiles, item_profiles)
    print(f"  User profiles : {len(user_profiles):,}")
    print(f"  Item profiles : {len(item_profiles):,}")
    print(f"  DN training rows: {len(dn_data):,}")
    print(f"  Positive rate : {dn_data['rating_positive'].astype(int).mean():.3f}")

    # ── STEP 3: Fit Decision Network ──────────────────────────
    divider("STEP 3 · Fitting Decision Network (Bayesian Estimation)")
    dn = DecisionNetwork(utility_threshold=0.30)
    dn.fit(dn_data, method="bayes")

    # ── STEP 4: Evaluation ─────────────────────────────────────
    divider("STEP 4 · Evaluation on Test Set")
    rec = DecisionNetworkRecommender(dn, user_profiles, item_profiles, train_r)
    print("  Running evaluation on 150 users (patience ~60s)...")
    results = evaluate_recommender(rec, test_r, k=10, max_users=150)
    print_evaluation_report(results)

    # ── STEP 5: Sample Recommendations ────────────────────────
    divider("STEP 5 · Sample Recommendations & Explanations")
    sample_users = [uid for uid in [1, 50, 200] if uid in rec.user_profiles.index]

    for uid in sample_users:
        up   = rec.user_profiles.loc[uid]
        top5 = rec.recommend(user_id=uid, k=5)

        print(f"\n  ┌── User {uid} ──────────────────────────────────────────")
        print(f"  │  Activity : {up.get('activity','?')}")
        print(f"  │  Style    : {up.get('rating_bucket','?')}")
        print(f"  │  Gender   : {up.get('gender','?')}")
        print(f"  │  Fav genre: {up.get('fav_genre','?')}")
        print(f"  └───────────────────────────────────────────────────────")

        if top5.empty:
            print("     (No items above EU threshold for this user)")
        else:
            print(f"  {'Rank':<5} {'Title':<40} {'Genre':<12} {'P(like)':>8} {'EU':>8}")
            print(f"  {'─'*5} {'─'*40} {'─'*12} {'─'*8} {'─'*8}")
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                print(f"  {rank:<5} {str(row['title']):<40} {str(row['genre']):<12} "
                      f"{row['prob_like']:>8.3f} {row['eu']:>8.3f}")

        # Explain the top item
        if not top5.empty:
            top_item = int(top5.iloc[0]["item_id"])
            exp = rec.explain(user_id=uid, item_id=top_item)
            print(f"\n  💡 Why recommend \"{exp['title']}\"?")
            print(f"     {exp['reasoning']}")

    # ── Done ──────────────────────────────────────────────────
    divider()
    print("\n  ✓ Pipeline complete!")
    print("  → Open demo/index.html for the interactive web dashboard")
    print("  → Run notebooks/03_decision_network_model.ipynb for full analysis\n")

    # Save model
    os.makedirs("data", exist_ok=True)
    dn.save("data/decision_network_model.pkl")


if __name__ == "__main__":
    main()
