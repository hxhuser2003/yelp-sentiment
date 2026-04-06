# scripts/train_roberta.py
# RoBERTa full fine-tuning for 5-class sentiment classification
# MPS acceleration for Apple M2

import os, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── 0. Config ──────────────────────────────────────────────────────────────────
CFG = {
    # data
    "data_dir":       "data/processed",
    "results_dir":    "results",
    "model_dir":      "models/roberta",

    # model
    "model_name":     "roberta-base",
    "num_labels":     5,
    "max_length":     128,        

    # training grid (single config — full grid requires more memory)
    "learning_rates": [2e-5],
    "batch_sizes":    [16],
    "epochs":         3,          

    # optimization
    "warmup_ratio":   0.1,        #  10% steps linear warmup
    "weight_decay":   0.01,
    "grad_accum":     2,          # simulate 2x batch size
    "early_stop_patience": 2,     

    # reproducibility
    "seed":           42,
}

os.makedirs(CFG["results_dir"], exist_ok=True)
os.makedirs(CFG["model_dir"],   exist_ok=True)

# ── 1. Device Setup ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple MPS (M2 GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("Device: CPU")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ── 2. Dataset ─────────────────────────────────────────────────────────────────
class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        self.labels = torch.tensor([l - 1 for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

# ── 3. Evaluation Helper ───────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss  += outputs.loss.item()
            preds        = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    cm       = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, f1, cm

# ── 4. Training Loop ───────────────────────────────────────────────────────────
def train_one_config(lr, batch_size, train_dataset, val_dataset, test_dataset):
    run_name = f"lr{lr}_bs{batch_size}"
    print(f"\n{'='*60}")
    print(f"Config: lr={lr}  batch_size={batch_size}  epochs={CFG['epochs']}")
    print(f"{'='*60}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = RobertaForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=CFG["num_labels"]
    ).to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": CFG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped, lr=lr)

    # Scheduler（linear warmup + linear decay）
    total_steps   = (len(train_loader) // CFG["grad_accum"]) * CFG["epochs"]
    warmup_steps  = int(total_steps * CFG["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Total steps: {total_steps}  |  Warmup steps: {warmup_steps}")

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc":    [], "val_f1":   [],
        "warmup_end_step": warmup_steps
    }

    best_val_f1    = -1
    best_model_state = None
    patience_count = 0
    global_step    = 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Gradient accumulation
            loss = outputs.loss / CFG["grad_accum"]
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % CFG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Progress（200 step print）
            if (step + 1) % 200 == 0:
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                      f"| loss={outputs.loss.item():.4f} "
                      f"| lr={scheduler.get_last_lr()[0]:.2e}")

        # Epoch-end validation
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\n  Epoch {epoch}/{CFG['epochs']}  ({elapsed:.0f}s)")
        print(f"  train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            patience_count   = 0
            print(f"  ✓ New best val_f1={best_val_f1:.4f} — model saved")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{CFG['early_stop_patience']})")
            if patience_count >= CFG["early_stop_patience"]:
                print(f"  Early stopping triggered at epoch {epoch}")
                break

    # Load best model → test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, device)
    print(f"\n  TEST  acc={test_acc:.4f}  f1={test_f1:.4f}")

    # Save best model weights
    model_save_path = f"{CFG['model_dir']}/{run_name}_best.pt"
    torch.save(best_model_state, model_save_path)

    return {
        "lr":         lr,
        "batch_size": batch_size,
        "best_val_f1":  round(best_val_f1, 4),
        "test_acc":     round(test_acc, 4),
        "test_f1":      round(test_f1, 4),
        "confusion_matrix": test_cm.tolist(),
        "history":      history,
    }

# ── 5. Main ────────────────────────────────────────────────────────────────────
def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(f"{CFG['data_dir']}/train.csv")
    val_df   = pd.read_csv(f"{CFG['data_dir']}/val.csv")
    test_df  = pd.read_csv(f"{CFG['data_dir']}/test.csv")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(CFG["model_name"])

    # Tokenize
    print("Tokenizing datasets (this takes ~2 min)...")
    t0 = time.time()
    train_dataset = YelpDataset(
        train_df["text"].tolist(), train_df["label"].tolist(),
        tokenizer, CFG["max_length"]
    )
    val_dataset = YelpDataset(
        val_df["text"].tolist(), val_df["label"].tolist(),
        tokenizer, CFG["max_length"]
    )
    test_dataset = YelpDataset(
        test_df["text"].tolist(), test_df["label"].tolist(),
        tokenizer, CFG["max_length"]
    )
    print(f"Tokenization done in {time.time()-t0:.0f}s")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  "
          f"Test: {len(test_dataset)}")

    # Grid search over lr x batch_size
    all_results = []
    for lr in CFG["learning_rates"]:
        for bs in CFG["batch_sizes"]:
            result = train_one_config(
                lr, bs, train_dataset, val_dataset, test_dataset
            )
            all_results.append(result)

    # Save all results
    save_results = []
    for r in all_results:
        save_results.append({k: v for k, v in r.items()})

    out_path = f"{CFG['results_dir']}/roberta_results.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nAll results saved → {out_path}")

    # Final summary table
    print("\n" + "="*65)
    print(f"{'LR':<10} {'BS':<6} {'Val F1':<10} {'Test Acc':<12} {'Test F1'}")
    print("="*65)
    for r in sorted(all_results, key=lambda x: x["best_val_f1"], reverse=True):
        print(f"{r['lr']:<10} {r['batch_size']:<6} "
              f"{r['best_val_f1']:<10} {r['test_acc']:<12} {r['test_f1']}")
    print("="*65)

    # Best config vs classical models
    best = max(all_results, key=lambda x: x["best_val_f1"])
    print(f"\nBest config: lr={best['lr']}  bs={best['batch_size']}")
    print(f"RoBERTa test_f1 = {best['test_f1']}")
    print(f"LR      test_f1 = 0.5802  (Δ = {best['test_f1']-0.5802:+.4f})")
    print(f"SVM     test_f1 = 0.5770  (Δ = {best['test_f1']-0.5770:+.4f})")

if __name__ == "__main__":
    main()