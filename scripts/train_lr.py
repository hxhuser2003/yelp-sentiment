# scripts/train_lr.py
# Logistic Regression with TF-IDF (+ GloVe + handcrafted) features
# Includes ablation study across 3 feature combinations

import pandas as pd
import numpy as np
import os, time, json, re, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# ── 0. Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed"
RESULTS_DIR = "results"
GLOVE_PATH  = "data/raw/glove.6B.100d.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
val   = pd.read_csv(f"{DATA_DIR}/val.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

X_train, y_train = train["text"].tolist(), train["label"].tolist()
X_val,   y_val   = val["text"].tolist(),   val["label"].tolist()
X_test,  y_test  = test["text"].tolist(),  test["label"].tolist()

# ── 2. Feature Block A: TF-IDF ─────────────────────────────────────────────────
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100_000,
    sublinear_tf=True,
    min_df=3
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)
print(f"  TF-IDF shape: {X_train_tfidf.shape}")

# ── 3. Feature Block B: GloVe Embeddings ──────────────────────────────────────
def load_glove(path):
    """Load GloVe vectors into a dict."""
    glove = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(glove):,} GloVe vectors")
    return glove

def text_to_glove(texts, glove, dim=100):
    """Average GloVe vectors for each text."""
    vecs = []
    for text in texts:
        tokens     = text.lower().split()
        token_vecs = [glove[t] for t in tokens if t in glove]
        vecs.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(dim))
    return np.array(vecs)

glove_available = os.path.exists(GLOVE_PATH)
if glove_available:
    print("Building GloVe features...")
    glove         = load_glove(GLOVE_PATH)
    X_train_glove = csr_matrix(text_to_glove(X_train, glove))
    X_val_glove   = csr_matrix(text_to_glove(X_val,   glove))
    X_test_glove  = csr_matrix(text_to_glove(X_test,  glove))
else:
    print("  GloVe not found — skipping")

# ── 4. Feature Block C: Handcrafted Features (with scaling) ───────────────────
def handcrafted_features_raw(texts):
    """Extract 4 surface-level features per review."""
    negation_words = {"not", "no", "never", "neither", "nobody",
                      "nothing", "nowhere", "nor", "hardly", "barely"}
    feats = []
    for text in texts:
        length     = len(text)
        excl_count = text.count("!")
        cap_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        tokens     = re.findall(r'\b\w+\b', text.lower())
        neg_freq   = sum(1 for t in tokens if t in negation_words) / max(len(tokens), 1)
        feats.append([length, excl_count, cap_ratio, neg_freq])
    return np.array(feats, dtype=np.float32)

print("Building handcrafted features...")
hc_scaler  = StandardScaler()
X_train_hc = csr_matrix(hc_scaler.fit_transform(handcrafted_features_raw(X_train)))
X_val_hc   = csr_matrix(hc_scaler.transform(handcrafted_features_raw(X_val)))
X_test_hc  = csr_matrix(hc_scaler.transform(handcrafted_features_raw(X_test)))

# ── 5. Assemble Feature Combinations for Ablation ─────────────────────────────
feature_sets = {
    "tfidf_only": (X_train_tfidf, X_val_tfidf, X_test_tfidf),
}
if glove_available:
    feature_sets["tfidf_glove"] = (
        hstack([X_train_tfidf, X_train_glove]),
        hstack([X_val_tfidf,   X_val_glove]),
        hstack([X_test_tfidf,  X_test_glove]),
    )
    feature_sets["tfidf_glove_hc"] = (
        hstack([X_train_tfidf, X_train_glove, X_train_hc]),
        hstack([X_val_tfidf,   X_val_glove,   X_val_hc]),
        hstack([X_test_tfidf,  X_test_glove,  X_test_hc]),
    )
else:
    feature_sets["tfidf_hc"] = (
        hstack([X_train_tfidf, X_train_hc]),
        hstack([X_val_tfidf,   X_val_hc]),
        hstack([X_test_tfidf,  X_test_hc]),
    )

# ── 6. Train + Evaluate ────────────────────────────────────────────────────────
def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy":         round(accuracy_score(y, preds), 4),
        "macro_f1":         round(f1_score(y, preds, average="macro"), 4),
        "confusion_matrix": confusion_matrix(y, preds).tolist()
    }

all_results = {}

for feat_name, (Xtr, Xvl, Xte) in feature_sets.items():
    print(f"\n── Training LR  [{feat_name}] ──")

    best_val_f1, best_C, best_model = -1, None, None

    for C in [0.01, 0.1, 1, 10]:
        model = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        t0 = time.time()
        model.fit(Xtr, y_train)
        elapsed = time.time() - t0

        val_metrics = evaluate(model, Xvl, y_val)
        print(f"  C={C:<5}  val_acc={val_metrics['accuracy']}  "
              f"val_f1={val_metrics['macro_f1']}  ({elapsed:.1f}s)")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_C      = C
            best_model  = model

    print(f"  Best C={best_C}  →  running on test set...")
    test_metrics = evaluate(best_model, Xte, y_test)
    print(f"  TEST  acc={test_metrics['accuracy']}  f1={test_metrics['macro_f1']}")

    all_results[feat_name] = {
        "best_C":           best_C,
        "val_f1":           best_val_f1,
        "test_acc":         test_metrics["accuracy"],
        "test_f1":          test_metrics["macro_f1"],
        "confusion_matrix": test_metrics["confusion_matrix"]
    }

    # save tfidf_only model immediately after it finishes
    if feat_name == "tfidf_only":
        with open("models/lr_tfidf_only.pkl", "wb") as f:
            pickle.dump({"tfidf": tfidf, "model": best_model}, f)
        print("  Model saved → models/lr_tfidf_only.pkl")

# ── 7. Save Results ────────────────────────────────────────────────────────────
out_path = f"{RESULTS_DIR}/lr_ablation.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved → {out_path}")

# ── 8. Print Ablation Summary Table ───────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Feature Set':<22} {'Best C':<8} {'Val F1':<10} {'Test Acc':<10} {'Test F1'}")
print("="*55)
for name, r in all_results.items():
    print(f"{name:<22} {r['best_C']:<8} {r['val_f1']:<10} {r['test_acc']:<10} {r['test_f1']}")
print("="*55)