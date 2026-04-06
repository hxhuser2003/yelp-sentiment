# scripts/visualize_dataset.py
# Generate dataset visualization figures for the final report:
#   1. Class distribution bar chart (train/val/test)
#   2. Review length distribution by star rating
#   3. Word clouds for 1-star and 5-star reviews

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import re

# ── 0. Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/processed"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
val   = pd.read_csv(f"{DATA_DIR}/val.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# ── 2. Class Distribution ──────────────────────────────────────────────────────
print("Plotting class distribution...")

fig, ax = plt.subplots(figsize=(8, 4))
x      = np.arange(1, 6)
width  = 0.25
colors = ["#4C72B0", "#55A868", "#C44E52"]

for i, (df, label, color) in enumerate(zip(
        [train, val, test],
        ["Train (35,000)", "Val (7,500)", "Test (7,500)"],
        colors)):
    counts = [len(df[df["label"] == star]) for star in range(1, 6)]
    bars = ax.bar(x + (i - 1) * width, counts, width,
                  label=label, color=color, alpha=0.85)

ax.set_xlabel("Star Rating", fontsize=11)
ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_title("Class Distribution Across Data Splits", fontsize=13,
             fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["1★", "2★", "3★", "4★", "5★"])
ax.legend(fontsize=9)
ax.set_ylim(0, 9000)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=7000, color=colors[0], linestyle="--",
           linewidth=1, alpha=0.5, label="Train target")

plt.tight_layout()
save_path = f"{FIGURES_DIR}/class_distribution.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {save_path}")

# ── 3. Review Length Distribution ─────────────────────────────────────────────
print("Plotting review length distribution...")

fig, ax = plt.subplots(figsize=(8, 4))
colors_star = ["#C44E52", "#DD8452", "#8B8B00", "#55A868", "#4C72B0"]
labels_star = ["1★", "2★", "3★", "4★", "5★"]

for star, color, label in zip(range(1, 6), colors_star, labels_star):
    lengths = train[train["label"] == star]["text"].str.len()
    # clip at 2000 for readability
    lengths = lengths.clip(upper=2000)
    ax.hist(lengths, bins=50, alpha=0.5, color=color,
            label=f"{label} (mean={lengths.mean():.0f})",
            density=True)

ax.set_xlabel("Review Length (characters)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Review Length Distribution by Star Rating",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
save_path = f"{FIGURES_DIR}/review_length_dist.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {save_path}")

# ── 4. Word Cloud ──────────────────────────────────────────────────────────────
print("Generating word clouds...")

# try to import wordcloud, install if missing
try:
    from wordcloud import WordCloud, STOPWORDS
    wordcloud_available = True
except ImportError:
    print("  wordcloud not installed — run: pip install wordcloud")
    wordcloud_available = False

if wordcloud_available:
    stopwords = set(STOPWORDS)
    stopwords.update(["place", "food", "restaurant", "one", "get",
                      "got", "also", "would", "will", "us", "said",
                      "went", "back", "time", "really", "made", "go"])

    def get_text(df, star):
        """Concatenate all reviews for a given star rating."""
        texts = df[df["label"] == star]["text"].tolist()
        return " ".join(texts[:2000])  # limit for speed

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, star, title, colormap in zip(
            axes,
            [1, 5],
            ["1-Star Reviews", "5-Star Reviews"],
            ["Reds", "Blues"]):
        text = get_text(train, star)
        wc   = WordCloud(
            width=600, height=400,
            background_color="white",
            stopwords=stopwords,
            colormap=colormap,
            max_words=80,
            collocations=False,
            random_state=42
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.suptitle("Word Clouds: Extreme Sentiment Reviews",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = f"{FIGURES_DIR}/wordcloud.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ── 5. Summary ─────────────────────────────────────────────────────────────────
print("\nAll dataset figures saved:")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith(".png"):
        print(f"  {FIGURES_DIR}/{f}")