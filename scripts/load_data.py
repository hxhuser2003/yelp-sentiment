# Data Preparation & Preprocessing Pipeline

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# ── Specification ──────────────────────────────────────────────────────
TRAIN_SAMPLE = 50_000
TEST_SAMPLE  = 10_000
RANDOM_STATE = 42
SAVE_DIR     = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Download datasets ──────────────────────────────────────────────
print("Downloading Yelp Review Full...")
dataset = load_dataset("yelp_review_full")

train_full = dataset["train"].to_pandas()   # 650,000 rows
test_full  = dataset["test"].to_pandas()    # 50,000 rows

train_full["label"] = train_full["label"] + 1
test_full["label"]  = test_full["label"]  + 1

# ── 2. Stratified sampling ─────────────────────────────────────
print("Sampling...")
train_sampled, _ = train_test_split(
    train_full,
    train_size=TRAIN_SAMPLE,
    stratify=train_full["label"],
    random_state=RANDOM_STATE
)
test_sampled, _ = train_test_split(
    test_full,
    train_size=TEST_SAMPLE,
    stratify=test_full["label"],
    random_state=RANDOM_STATE
)

# ── 3. train → train / val / test ────────────────────────
# 70% train, 15% val, 15% test
train_data, temp = train_test_split(
    train_sampled,
    test_size=0.30,
    stratify=train_sampled["label"],
    random_state=RANDOM_STATE
)
val_data, test_data = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=RANDOM_STATE
)

# ── 4. save ──────────────────────────────────────────────
train_data.to_csv(f"{SAVE_DIR}/train.csv", index=False)
val_data.to_csv(f"{SAVE_DIR}/val.csv",   index=False)
test_data.to_csv(f"{SAVE_DIR}/test.csv",  index=False)

print(f"Train: {len(train_data):,}  |  Val: {len(val_data):,}  |  Test: {len(test_data):,}")
print("Saved to data/processed/")