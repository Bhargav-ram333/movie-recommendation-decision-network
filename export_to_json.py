"""
export_to_json.py
-----------------
Runs the full Decision Network pipeline and exports all real model
outputs to demo/data.json so the GitHub Pages frontend can load
real data without any backend.

Usage:
    cd "MATHS PROJECT"
    python export_to_json.py
"""

import sys, os, json, itertools, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all, train_test_split_temporal
from src.feature_engineering import (
    build_user_profiles, build_item_profiles, build_dn_training_data
)
from src.decision_network import DecisionNetwork, EDGES
from src.recommender import DecisionNetworkRecommender
from src.evaluation import evaluate_recommender

# ── 1. Load & split data
print("► Loading MovieLens 100K...")
ratings, users, items, merged = load_all()
train_r, test_r = train_test_split_temporal(ratings, test_ratio=0.20)
train_merged = merged[merged.index.isin(train_r.index)]
print(f"  Train: {len(train_r):,} | Test: {len(test_r):,}")

# ── 2. Feature engineering
print("► Building features...")
user_profiles = build_user_profiles(train_merged)
item_profiles  = build_item_profiles(train_merged)
dn_data        = build_dn_training_data(train_merged, user_profiles, item_profiles)

# ── 3. Fit Decision Network
print("► Fitting Decision Network (Bayesian estimation)...")
dn = DecisionNetwork(utility_threshold=0.30)
dn.fit(dn_data, method="bayes")

# ── 4. Recommender
rec = DecisionNetworkRecommender(dn, user_profiles, item_profiles, train_r)

# ── 5. Evaluation
print("► Evaluating (100 users)...")
results = evaluate_recommender(rec, test_r, k=10, max_users=100)

# ── 6. CPT extraction
print("► Extracting CPTs...")
cpts = {}
for cpd in dn.model.cpds:
    node = cpd.variable
    states = list(cpd.state_names[node])
    parents = list(cpd.variables[1:])
    values_flat = cpd.values.flatten().tolist()
    cpts[node] = {
        "node": node,
        "states": states,
        "parents": parents,
        "values": values_flat,
        "shape": list(cpd.values.shape),
    }

# ── 7. Inference lookup table — all evidence combinations
print("► Pre-computing inference table for all evidence combinations...")
activities     = ["low", "medium", "high"]
rating_buckets = ["harsh", "neutral", "generous"]
genders        = ["M", "F"]
genres         = ["Action", "Adventure", "Animation", "Comedy", "Crime",
                  "Drama", "Fantasy", "Horror", "Mystery", "Romance",
                  "Sci-Fi", "Thriller"]
popularities   = ["niche", "moderate", "popular"]
qualities      = ["low", "medium", "high"]

inference_table = {}
total = len(activities)*len(rating_buckets)*len(genders)*len(genres)*len(popularities)*len(qualities)
print(f"  Total combinations: {total:,}")

count = 0
for act, rb, gen, genre, pop, qual in itertools.product(
        activities, rating_buckets, genders, genres, popularities, qualities):
    evidence = {
        "user_activity":      act,
        "user_rating_bucket": rb,
        "user_gender":        gen,
        "item_genre":         genre,
        "item_popularity":    pop,
        "item_quality":       qual,
    }
    decision, eu, prob = dn.decide(evidence)
    key = f"{act}|{rb}|{gen}|{genre}|{pop}|{qual}"
    inference_table[key] = {
        "prob_like": round(prob, 4),
        "eu":        round(eu, 4),
        "decision":  decision,
    }
    count += 1
    if count % 500 == 0:
        print(f"  {count}/{total} done...", end="\r")
print(f"  {total}/{total} done.     ")

# ── 8. Sample recommendations for 10 users
print("► Generating sample recommendations for 10 users...")
sample_user_ids = [uid for uid in [1, 2, 5, 10, 20, 50, 100, 150, 200, 300]
                   if uid in rec.user_profiles.index]

sample_recs = {}
for uid in sample_user_ids:
    up   = rec.user_profiles.loc[uid]
    top10 = rec.recommend(user_id=uid, k=10)
    recs_list = []
    for _, row in top10.iterrows():
        item_id = int(row["item_id"])
        exp = rec.explain(user_id=uid, item_id=item_id)
        recs_list.append({
            "item_id":   item_id,
            "title":     str(row["title"]),
            "genre":     str(row["genre"]),
            "popularity": str(row["popularity"]),
            "quality":   str(row["quality"]),
            "prob_like": round(float(row["prob_like"]), 4),
            "eu":        round(float(row["eu"]), 4),
            "decision":  str(row["decision"]),
            "reasoning": exp["reasoning"],
        })

    sample_recs[str(uid)] = {
        "user_id":       uid,
        "activity":      str(up.get("activity", "?")),
        "rating_bucket": str(up.get("rating_bucket", "?")),
        "gender":        str(up.get("gender", "?")),
        "fav_genre":     str(up.get("fav_genre", "?")),
        "recommendations": recs_list,
    }

# ── 9. User & item profile summaries
print("► Summarising profiles...")
up_df = rec.user_profiles.copy().reset_index()
ip_df = rec.item_profiles.copy().reset_index()

user_summary = {}
for _, row in up_df.iterrows():
    uid = int(row["user_id"])
    user_summary[str(uid)] = {
        "user_id":       uid,
        "activity":      str(row.get("activity", "?")),
        "rating_bucket": str(row.get("rating_bucket", "?")),
        "gender":        str(row.get("gender", "?")),
        "fav_genre":     str(row.get("fav_genre", "?")),
        "avg_rating":    round(float(row.get("avg_rating", 0)), 3),
        "rating_count":  int(row.get("rating_count", 0)),
    }

item_summary = {}
for _, row in ip_df.iterrows():
    iid = int(row["item_id"])
    item_summary[str(iid)] = {
        "item_id":      iid,
        "title":        str(row.get("title", f"Movie {iid}")),
        "genre":        str(row.get("primary_genre", "Unknown")),
        "popularity":   str(row.get("popularity", "?")),
        "quality":      str(row.get("quality", "?")),
        "avg_rating":   round(float(row.get("avg_rating", 0)), 3),
        "rating_count": int(row.get("rating_count", 0)),
    }

# ── 10. Metrics
metrics = {
    "precision_at_10": round(results.get("precision@10", 0), 4),
    "recall_at_10":    round(results.get("recall@10", 0), 4),
    "ndcg_at_10":      round(results.get("ndcg@10", 0), 4),
    "hit_rate_at_10":  round(results.get("hit_rate@10", 0), 4),
    "auc_roc":         round(results.get("auc_roc", 0), 4),
    "rmse":            round(results.get("rmse", 0), 4),
    "mae":             round(results.get("mae", 0), 4),
    "coverage":        round(results.get("coverage", 0), 4),
}

# ── 11. Network structure
network_structure = {
    "edges": [{"from": e[0], "to": e[1]} for e in EDGES],
    "utility_table": {"1": 1.0, "0": -0.5},
    "threshold": dn.utility_threshold,
}

# ── 12. Assemble & write
output = {
    "meta": {
        "generated_at":    __import__("datetime").datetime.now().isoformat(),
        "train_ratings":   len(train_r),
        "test_ratings":    len(test_r),
        "n_users":         len(user_summary),
        "n_items":         len(item_summary),
        "utility_threshold": dn.utility_threshold,
    },
    "metrics":           metrics,
    "network_structure": network_structure,
    "cpts":              cpts,
    "inference_table":   inference_table,
    "sample_recs":       sample_recs,
    "user_summary":      user_summary,
    "item_summary":      item_summary,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "data.json")
print(f"► Writing {out_path}...")
with open(out_path, "w") as f:
    json.dump(output, f, separators=(",", ":"))

size_mb = os.path.getsize(out_path) / 1e6
print(f"\n✓ Exported to demo/data.json  ({size_mb:.1f} MB)")
