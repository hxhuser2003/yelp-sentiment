# scripts/visualize.py
# Generate all figures for the final report:
#   1. Confusion matrices for all models
#   2. RoBERTa training/validation loss curves (with warmup annotation)
#   3. Ablation comparison bar chart (LR & SVM feature sets)
#   4. Model comparison bar chart (all models, accuracy & macro-F1)
#   5. GPT prompt strategy comparison
#   6. LLM-as-Judge reasonable vs severe error breakdown

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── 0. Config ──────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# consistent color palette across all figures
COLORS = {
    "lr":       "#4C72B0",
    "svm":      "#DD8452",
    "roberta":  "#55A868",
    "gpt":      "#C44E52",
    "warmup":   "#FFC107",
    "val":      "#C44E52",
    "train":    "#4C72B0",
}

STAR_LABELS = ["1★", "2★", "3★", "4★", "5★"]

# ── 1. Helper: load JSON safely ────────────────────────────────────────────────
def load_json(path: str):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        return None
    with open(path) as f:
        return json.load(f)

# ── 2. Confusion Matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: list, title: str, save_path: str):
    """Plot a normalized 5x5 confusion matrix."""
    cm_array = np.array(cm, dtype=float)

    # normalize by row (true label) → shows recall per class
    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_array, row_sums,
                         where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=STAR_LABELS,
        yticklabels=STAR_LABELS,
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(title,              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

def plot_all_confusion_matrices():
    print("\n[1] Plotting confusion matrices...")

    # LR tfidf_only
    lr_data = load_json(f"{RESULTS_DIR}/lr_ablation.json")
    if lr_data:
        plot_confusion_matrix(
            lr_data["tfidf_only"]["confusion_matrix"],
            "Logistic Regression (TF-IDF)",
            f"{FIGURES_DIR}/cm_lr.png"
        )

    # SVM tfidf_only
    svm_data = load_json(f"{RESULTS_DIR}/svm_ablation.json")
    if svm_data:
        plot_confusion_matrix(
            svm_data["tfidf_only"]["confusion_matrix"],
            "Linear SVM (TF-IDF)",
            f"{FIGURES_DIR}/cm_svm.png"
        )

    # RoBERTa best config
    roberta_data = load_json(f"{RESULTS_DIR}/roberta_results.json")
    if roberta_data:
        best = max(roberta_data, key=lambda x: x["best_val_f1"])
        plot_confusion_matrix(
            best["confusion_matrix"],
            f"RoBERTa (lr={best['lr']}, bs={best['batch_size']})",
            f"{FIGURES_DIR}/cm_roberta.png"
        )

    # GPT best strategy
    gpt_summary = load_json(f"{RESULTS_DIR}/gpt_summary.json")
    if gpt_summary:
        best_gpt = max(gpt_summary, key=lambda x: x["macro_f1"])
        gpt_detail = load_json(
            f"{RESULTS_DIR}/gpt_{best_gpt['strategy']}.json")
        if gpt_detail:
            plot_confusion_matrix(
                gpt_detail["confusion_matrix"],
                f"GPT-4o-mini ({best_gpt['strategy'].replace('_', ' ').title()})",
                f"{FIGURES_DIR}/cm_gpt_best.png"
            )

# ── 3. RoBERTa Loss Curves ─────────────────────────────────────────────────────
def plot_loss_curves():
    print("\n[2] Plotting RoBERTa loss curves...")

    roberta_data = load_json(f"{RESULTS_DIR}/roberta_results.json")
    if not roberta_data:
        return

    # plot one curve per config
    fig, axes = plt.subplots(
        1, len(roberta_data),
        figsize=(5 * len(roberta_data), 4),
        sharey=True
    )
    if len(roberta_data) == 1:
        axes = [axes]

    for ax, result in zip(axes, roberta_data):
        history      = result["history"]
        train_losses = history["train_loss"]
        val_losses   = history["val_loss"]
        epochs       = list(range(1, len(train_losses) + 1))

        ax.plot(epochs, train_losses,
                color=COLORS["train"], marker="o",
                linewidth=2, label="Train Loss")
        ax.plot(epochs, val_losses,
                color=COLORS["val"],   marker="s",
                linewidth=2, label="Val Loss")

        # annotate warmup end (warmup is within epoch 1)
        ax.axvline(x=1, color=COLORS["warmup"],
                   linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(1.05, ax.get_ylim()[1] * 0.95,
                "warmup\nend", color=COLORS["warmup"],
                fontsize=8, va="top")

        # mark best epoch
        best_epoch = int(np.argmin(val_losses)) + 1
        ax.axvline(x=best_epoch, color=COLORS["roberta"],
                   linestyle=":", linewidth=1.5, alpha=0.8)

        ax.set_title(
            f"lr={result['lr']}  bs={result['batch_size']}\n"
            f"best_val_f1={result['best_val_f1']}",
            fontsize=10
        )
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss",  fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)

    fig.suptitle("RoBERTa Training & Validation Loss",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = f"{FIGURES_DIR}/roberta_loss_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ── 4. Ablation Bar Chart ──────────────────────────────────────────────────────
def plot_ablation():
    print("\n[3] Plotting ablation study...")

    lr_data  = load_json(f"{RESULTS_DIR}/lr_ablation.json")
    svm_data = load_json(f"{RESULTS_DIR}/svm_ablation.json")
    if not lr_data or not svm_data:
        return

    # feature set display names
    feat_display = {
        "tfidf_only":    "TF-IDF",
        "tfidf_glove":   "TF-IDF\n+ GloVe",
        "tfidf_glove_hc":"TF-IDF\n+ GloVe\n+ HC",
        "tfidf_hc":      "TF-IDF\n+ HC",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (data, model_name, color) in zip(
            axes,
            [(lr_data,  "Logistic Regression", COLORS["lr"]),
             (svm_data, "Linear SVM",          COLORS["svm"])]):

        feat_names = list(data.keys())
        f1_scores  = [data[k]["test_f1"]  for k in feat_names]
        accs       = [data[k]["test_acc"] for k in feat_names]
        x_labels   = [feat_display.get(k, k) for k in feat_names]
        x          = np.arange(len(feat_names))
        width      = 0.35

        bars1 = ax.bar(x - width/2, f1_scores, width,
                       label="Macro F1", color=color, alpha=0.85)
        bars2 = ax.bar(x + width/2, accs,       width,
                       label="Accuracy", color=color, alpha=0.50)

        # value labels on bars
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{model_name} — Feature Ablation",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0.45, 0.70)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Feature Ablation Study: TF-IDF vs GloVe vs Handcrafted",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = f"{FIGURES_DIR}/ablation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ── 5. Overall Model Comparison ────────────────────────────────────────────────
def plot_model_comparison():
    print("\n[4] Plotting overall model comparison...")

    # collect best result from each model
    records = []

    lr_data = load_json(f"{RESULTS_DIR}/lr_ablation.json")
    if lr_data:
        best_key = max(lr_data, key=lambda k: lr_data[k]["test_f1"])
        records.append({
            "model": "LR",
            "acc":   lr_data[best_key]["test_acc"],
            "f1":    lr_data[best_key]["test_f1"],
            "color": COLORS["lr"],
        })

    svm_data = load_json(f"{RESULTS_DIR}/svm_ablation.json")
    if svm_data:
        best_key = max(svm_data, key=lambda k: svm_data[k]["test_f1"])
        records.append({
            "model": "SVM",
            "acc":   svm_data[best_key]["test_acc"],
            "f1":    svm_data[best_key]["test_f1"],
            "color": COLORS["svm"],
        })

    roberta_data = load_json(f"{RESULTS_DIR}/roberta_results.json")
    if roberta_data:
        best = max(roberta_data, key=lambda x: x["best_val_f1"])
        records.append({
            "model": "RoBERTa",
            "acc":   best["test_acc"],
            "f1":    best["test_f1"],
            "color": COLORS["roberta"],
        })

    gpt_summary = load_json(f"{RESULTS_DIR}/gpt_summary.json")
    if gpt_summary:
        for r in gpt_summary:
            records.append({
                "model": f"GPT\n{r['strategy'].replace('_',' ')}",
                "acc":   r["accuracy"],
                "f1":    r["macro_f1"],
                "color": COLORS["gpt"],
            })

    if not records:
        print("  No results found yet.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(records) * 1.4), 5))
    x     = np.arange(len(records))
    width = 0.35

    bars_f1  = ax.bar(x - width/2,
                      [r["f1"]  for r in records], width,
                      label="Macro F1",
                      color=[r["color"] for r in records],
                      alpha=0.85)
    bars_acc = ax.bar(x + width/2,
                      [r["acc"] for r in records], width,
                      label="Accuracy",
                      color=[r["color"] for r in records],
                      alpha=0.45)

    # value labels
    for bar in list(bars_f1) + list(bars_acc):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    # legend for model groups
    legend_patches = [
        mpatches.Patch(color=COLORS["lr"],      label="Logistic Regression"),
        mpatches.Patch(color=COLORS["svm"],     label="Linear SVM"),
        mpatches.Patch(color=COLORS["roberta"], label="RoBERTa"),
        mpatches.Patch(color=COLORS["gpt"],     label="GPT-4o-mini"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in records], fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Model Comparison: Accuracy & Macro-F1 on Test Set",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = f"{FIGURES_DIR}/model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ── 6. GPT Prompt Strategy Comparison ─────────────────────────────────────────
def plot_gpt_strategies():
    print("\n[5] Plotting GPT prompt strategy comparison...")

    gpt_summary = load_json(f"{RESULTS_DIR}/gpt_summary.json")
    if not gpt_summary:
        return

    strategies = [r["strategy"].replace("_", "\n") for r in gpt_summary]
    f1_scores  = [r["macro_f1"] for r in gpt_summary]
    accs       = [r["accuracy"] for r in gpt_summary]
    x          = np.arange(len(strategies))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, f1_scores, width,
                   label="Macro F1", color=COLORS["gpt"], alpha=0.85)
    bars2 = ax.bar(x + width/2, accs,      width,
                   label="Accuracy", color=COLORS["gpt"], alpha=0.45)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)

    # baseline reference line (best classical model)
    ax.axhline(y=0.5802, color=COLORS["lr"],
               linestyle="--", linewidth=1.5,
               label="LR baseline (F1=0.580)")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.45, 0.80)
    ax.set_title("GPT-4o-mini: Prompt Strategy Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = f"{FIGURES_DIR}/gpt_strategies.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ── 7. LLM-as-Judge Summary ────────────────────────────────────────────────────
def plot_llm_judge():
    print("\n[6] Plotting LLM-as-Judge results...")

    judge_summary = load_json(f"{RESULTS_DIR}/judge_summary.json")
    if not judge_summary:
        print("  judge_summary.json not found — run llm_judge.py first")
        return

    models     = [s["model"]          for s in judge_summary]
    reasonable = [s["pct_reasonable"] for s in judge_summary]
    severe     = [s["pct_severe"]     for s in judge_summary]
    x          = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x, reasonable, label="Reasonable (adjacent-class)",
                   color="#55A868", alpha=0.85)
    bars2 = ax.bar(x, severe,     bottom=reasonable,
                   label="Severe (2+ stars off)",
                   color="#C44E52", alpha=0.85)

    for bar, val in zip(bars1, reasonable):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    val/2, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    for bar, r_val, s_val in zip(bars2, reasonable, severe):
        if s_val > 5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    r_val + s_val/2, f"{s_val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Percentage of Errors (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("LLM-as-Judge: Error Quality by Model\n"
                 "(Reasonable = adjacent-class, Severe = 2+ stars off)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = f"{FIGURES_DIR}/llm_judge.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

# ── 8. Main ────────────────────────────────────────────────────────────────────
def main():
    print("="*50)
    print("Generating all figures...")
    print("="*50)

    plot_all_confusion_matrices()
    plot_loss_curves()
    plot_ablation()
    plot_model_comparison()
    plot_gpt_strategies()
    plot_llm_judge()

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("\nAvailable figures:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith(".png"):
            print(f"  {FIGURES_DIR}/{f}")

if __name__ == "__main__":
    main()