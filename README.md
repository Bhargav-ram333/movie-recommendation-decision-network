# 🏆 Decision Network for Intelligent Recommendation Systems

> **Project 18 · Core AI Track · MovieLens 100K**  
> An explainable movie recommendation system combining **Bayesian Networks** with **Utility Theory**.

---

## 🧠 What Makes This Different?

Most recommendation systems (collaborative filtering, SVD, neural nets) are **black boxes** — they tell you *what* to recommend, but never *why*.

This project uses a **Decision Network** (also called an Influence Diagram), which:

| Feature | Black-box CF | Decision Network |
|---------|:---:|:---:|
| Accuracy | ✓ High | ✓ Competitive |
| Explainability | ✗ None | ✓ **Full causal reasoning** |
| Cold-start | ✗ Fails | ✓ **Handles gracefully** |
| Catalogue diversity | ✗ Biased | ✓ **64% coverage** |
| Theory foundation | Heuristic | ✓ **Probability + Utility theory** |

---

## 🏗️ Architecture

```
  USER FEATURES              ITEM FEATURES
  ┌──────────────┐           ┌──────────────┐
  │ user_activity│           │item_popularity│
  │ rating_style │    ──►    │  item_quality │
  │  user_gender │    ──►    │  item_genre   │
  └──────┬───────┘           └──────┬────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
           ┌─────────────────┐
           │ rating_positive │  ← Chance Node (BN)
           └────────┬────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  ┌──────────────┐    ┌──────────────────┐
  │ Utility Node │    │  Decision Node   │
  │  U(1) = +1.0 │    │  MEU: EU ≥ θ?   │
  │  U(0) = -0.5 │    │  → RECOMMEND     │
  └──────────────┘    └──────────────────┘
```

**Decision Rule (Maximum Expected Utility):**
```
EU(recommend | evidence) = P(like|e)·(+1.0) + P(dislike|e)·(−0.5)
Recommend if EU ≥ θ = 0.30
```

---

## 📁 Project Structure

```
MATHS PROJECT/
├── data/                       # Auto-created: MovieLens 100K + saved figures
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA: distributions, demographics, genres
│   ├── 02_feature_engineering.ipynb  # Build user/item profiles, discretise features
│   ├── 03_decision_network_model.ipynb  # ⭐ Core: fit BN, CPTs, MEU inference
│   ├── 04_evaluation_metrics.ipynb   # Precision@K, NDCG, ROC, baselines
│   └── 05_visualization.ipynb        # CPT heatmaps, explanations, dashboard
├── src/
│   ├── data_loader.py          # Download + parse MovieLens 100K
│   ├── feature_engineering.py  # User/item profiles, DN training data
│   ├── decision_network.py     # ⭐ DecisionNetwork class (pgmpy + MEU)
│   ├── recommender.py          # Top-K recommendation engine + explanations
│   └── evaluation.py           # All evaluation metrics
├── demo/
│   ├── index.html              # 🌐 Interactive web dashboard (open in browser)
│   ├── style.css
│   └── app.js
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Notebooks (in order)
```bash
cd notebooks
jupyter notebook
```
Open notebooks `01` → `02` → `03` → `04` → `05` in order. The dataset downloads automatically.

### 3. Open the Interactive Demo
```bash
# Simply open in your browser — no server needed
open demo/index.html
```
Or just double-click `demo/index.html`.

### 4. Run the Python Pipeline Directly
```bash
cd src
python decision_network.py   # Fits model + runs test cases
python recommender.py        # Generates Top-10 recommendations
python evaluation.py         # Full evaluation report
```

---

## 📊 Results

| Metric | Decision Network | SVD (best baseline) |
|--------|:-:|:-:|
| **Precision@10** | 0.312 | 0.348 |
| **NDCG@10** | 0.387 | 0.421 |
| **AUC-ROC** | 0.724 | 0.761 |
| **Coverage** | **0.641** ★ | 0.497 |
| **Explainability** | **✓ Full** ★ | ✗ None |
| **Cold-start** | **✓ Yes** ★ | ✗ No |

> 💡 The Decision Network is ~10% behind SVD in accuracy but is the **only model** with full explainability, cold-start support, and superior catalogue diversity.

---

## 🔍 Sample Explanation Output

```
User 1: High activity, Generous rater, Male
Movie: "Star Wars (1977)" | Genre: Sci-Fi | Quality: High | Popularity: Popular

Evidence:
  user_activity      = high
  user_rating_bucket = generous
  item_quality       = high
  item_genre         = Sci-Fi

Inference:
  P(rating_positive=1 | evidence) = 0.784
  EU(recommend) = 0.784×(+1.0) + 0.216×(-0.5) = 0.676

Decision: RECOMMEND ✓  [EU=0.676 ≥ θ=0.30]
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| **Bayesian Network** | DAG encoding conditional independence: P(X₁,...,Xₙ) = ∏ P(Xᵢ \| Pa(Xᵢ)) |
| **Variable Elimination** | Exact inference algorithm for P(query \| evidence) |
| **CPT Estimation** | Bayesian (BDeu prior, α=5) to prevent zero-frequency cells |
| **Utility Node** | Deterministic mapping: outcome → real-valued utility |
| **MEU Principle** | Choose action maximising expected utility under uncertainty |
| **Markov Blanket** | Minimal set of nodes making target independent of all others |

---

## 🛠️ Tech Stack

- **Python 3.10+** — core language
- **pgmpy** — Bayesian Network construction, CPT estimation, Variable Elimination
- **pandas / numpy** — data processing
- **scikit-learn** — evaluation metrics
- **matplotlib / seaborn / plotly** — visualisation
- **HTML / CSS / JS** — interactive web demo (zero dependencies)

---

## 📖 References

1. Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th ed.), Ch. 16 (Decision Theory)
2. Koller & Friedman, *Probabilistic Graphical Models*, MIT Press (2009)
3. Harper & Konstan, *The MovieLens Datasets*, ACM TIIS (2015)
4. pgmpy documentation: https://pgmpy.org
