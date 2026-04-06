# scripts/visualize_loss_curves.py
# Generate training vs validation loss/accuracy curves for all trainable models
# LR and SVM: validation F1 vs C (grid search curve)
# RoBERTa: train loss vs val loss per epoch

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. LR: Val F1 vs C (grid search curve) ────────────────────────────────────
print("Plotting LR grid search curve...")

# hardcoded from your actual results
lr_grid = {
    "C":       [0.01, 0.1, 1, 10],
    "val_f1":  [0.4894, 0.5566, 0.5812, 0.5652],
    "val_acc": [0.5057, 0.5639, 0.5839, 0.5664],
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, label, color in zip(
        axes,
        ["val_f1", "val_acc"],
        ["Validation Macro-F1", "Validation Accuracy"],
        ["#4C72B0", "#55A868"]):

    values = lr_grid[metric]
    best_idx = int(np.argmax(values))

    ax.plot(range(len(lr_grid["C"])), values,
            marker="o", linewidth=2, color=color, markersize=8)

    # highlight best
    ax.scatter([best_idx], [values[best_idx]],
               color="red", s=120, zorder=5, label=f"Best C={lr_grid['C'][best_idx]}")

    # annotate each point
    for i, (c, v) in enumerate(zip(lr_grid["C"], values)):
        ax.annotate(f"{v:.4f}", (i, v),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.set_xticks(range(len(lr_grid["C"])))
    ax.set_xticklabels([f"C={c}" for c in lr_grid["C"]])
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel("Regularization Strength C", fontsize=11)
    ax.set_title(f"LR — {label} vs C", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # annotate bias/variance regions
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="red", label="High Bias")
    ax.axvspan(2.5, 3.5, alpha=0.08, color="orange", label="High Variance")
    ax.text(0, min(values) + 0.005, "High\nBias", ha="center",
            fontsize=8, color="red", alpha=0.7)
    ax.text(3, min(values) + 0.005, "High\nVariance", ha="center",
            fontsize=8, color="orange", alpha=0.7)

plt.suptitle("Logistic Regression: Hyperparameter Tuning (Validation Set)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_path = f"{FIGURES_DIR}/lr_grid_search.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {save_path}")

# ── 2. SVM: Val F1 vs C ────────────────────────────────────────────────────────
print("Plotting SVM grid search curve...")

svm_grid = {
    "C":       [0.01, 0.1, 1, 10],
    "val_f1":  [0.5408, 0.5673, 0.5494, 0.5267],
    "val_acc": [0.5553, 0.5761, 0.5536, 0.5283],
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, label, color in zip(
        axes,
        ["val_f1", "val_acc"],
        ["Validation Macro-F1", "Validation Accuracy"],
        ["#DD8452", "#C44E52"]):

    values = svm_grid[metric]
    best_idx = int(np.argmax(values))

    ax.plot(range(len(svm_grid["C"])), values,
            marker="s", linewidth=2, color=color, markersize=8)

    ax.scatter([best_idx], [values[best_idx]],
               color="red", s=120, zorder=5,
               label=f"Best C={svm_grid['C'][best_idx]}")

    for i, (c, v) in enumerate(zip(svm_grid["C"], values)):
        ax.annotate(f"{v:.4f}", (i, v),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.set_xticks(range(len(svm_grid["C"])))
    ax.set_xticklabels([f"C={c}" for c in svm_grid["C"]])
    ax.set_ylabel(label, fontsize=11)
    ax.set_xlabel("Soft-Margin Parameter C", fontsize=11)
    ax.set_title(f"SVM — {label} vs C", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax.axvspan(-0.5, 0.5, alpha=0.08, color="red")
    ax.axvspan(2.5, 3.5, alpha=0.08, color="orange")
    ax.text(0, min(values) + 0.003, "High\nBias", ha="center",
            fontsize=8, color="red", alpha=0.7)
    ax.text(3, min(values) + 0.003, "High\nVariance", ha="center",
            fontsize=8, color="orange", alpha=0.7)

plt.suptitle("Linear SVM: Hyperparameter Tuning (Validation Set)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_path = f"{FIGURES_DIR}/svm_grid_search.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {save_path}")

# ── 3. RoBERTa: Train vs Val Loss + Val F1 per epoch ──────────────────────────
print("Plotting RoBERTa loss curves...")

roberta_data = {
    "epoch":      [1, 2, 3],
    "train_loss": [1.0412, 0.8079, 0.6879],
    "val_loss":   [0.9153, 0.8637, 0.8785],
    "val_acc":    [0.6008, 0.6321, 0.6377],
    "val_f1":     [0.5960, 0.6283, 0.6372],
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# left: train loss vs val loss
ax = axes[0]
ax.plot(roberta_data["epoch"], roberta_data["train_loss"],
        marker="o", linewidth=2, color="#4C72B0",
        markersize=8, label="Train Loss")
ax.plot(roberta_data["epoch"], roberta_data["val_loss"],
        marker="s", linewidth=2, color="#C44E52",
        markersize=8, label="Val Loss")

# annotate values
for e, tl, vl in zip(roberta_data["epoch"],
                     roberta_data["train_loss"],
                     roberta_data["val_loss"]):
    ax.annotate(f"{tl:.4f}", (e, tl),
                textcoords="offset points", xytext=(-18, 6),
                fontsize=8, color="#4C72B0")
    ax.annotate(f"{vl:.4f}", (e, vl),
                textcoords="offset points", xytext=(4, 6),
                fontsize=8, color="#C44E52")

# warmup annotation
ax.axvline(x=1, color="#FFC107", linestyle="--",
           linewidth=1.5, alpha=0.8, label="Warmup end")
ax.text(1.05, 1.02, "warmup\nend", color="#FFC107",
        fontsize=8, va="top")

# best epoch
ax.axvline(x=3, color="#55A868", linestyle=":",
           linewidth=1.5, alpha=0.8, label="Best val F1")

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("RoBERTa: Train vs Val Loss", fontsize=12, fontweight="bold")
ax.set_xticks([1, 2, 3])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# right: val acc + val f1 per epoch
ax = axes[1]
ax.plot(roberta_data["epoch"], roberta_data["val_acc"],
        marker="o", linewidth=2, color="#55A868",
        markersize=8, label="Val Accuracy")
ax.plot(roberta_data["epoch"], roberta_data["val_f1"],
        marker="s", linewidth=2, color="#8172B2",
        markersize=8, label="Val Macro-F1")

for e, va, vf in zip(roberta_data["epoch"],
                     roberta_data["val_acc"],
                     roberta_data["val_f1"]):
    ax.annotate(f"{va:.4f}", (e, va),
                textcoords="offset points", xytext=(-18, 6),
                fontsize=8, color="#55A868")
    ax.annotate(f"{vf:.4f}", (e, vf),
                textcoords="offset points", xytext=(4, 6),
                fontsize=8, color="#8172B2")

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("RoBERTa: Val Accuracy & Macro-F1", fontsize=12, fontweight="bold")
ax.set_xticks([1, 2, 3])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle("RoBERTa Fine-tuning: Training Dynamics (lr=2e-5, bs=16)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_path = f"{FIGURES_DIR}/roberta_training_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {save_path}")

print("\nAll curves saved:")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith(".png"):
        print(f"  {FIGURES_DIR}/{f}")