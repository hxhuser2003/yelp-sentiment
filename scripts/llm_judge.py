# scripts/llm_judge.py
# LLM-as-Judge error analysis
# For each model, sample 50-100 misclassified reviews and ask GPT-4o-mini
# to classify each error as "reasonable" (adjacent-class) or "severe"

import os, json, time, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI

# ── 0. Config ──────────────────────────────────────────────────────────────────
CFG = {
    "results_dir":   "results",
    "data_dir":      "data/processed",
    "model_dir":     "models",
    "model":         "gpt-4o-mini",
    "temperature":   0,
    "max_tokens":    150,
    "n_errors":      50,
    "sleep_sec":     22,
    "seed":          42,
    # RoBERTa config (must match train_roberta.py)
    "roberta_name":  "roberta-base",
    "max_length":    128,
    "num_labels":    5,
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
random.seed(CFG["seed"])

# ── 1. Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── 2. Judge Prompt ────────────────────────────────────────────────────────────
def build_judge_prompt(review: str, true_rating: int, pred_rating: int) -> list:
    """Build prompt asking GPT to classify error as reasonable or severe."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator of sentiment analysis models. "
                "Your job is to judge whether a model's prediction error is "
                "reasonable or severe. "
                "Always respond with valid JSON only."
            )
        },
        {
            "role": "user",
            "content": (
                f"A sentiment model predicted the star rating of a Yelp review.\n\n"
                f'Review: """{review[:500]}"""\n\n'
                f"True star rating:      {true_rating}\n"
                f"Predicted star rating: {pred_rating}\n\n"
                f"Is this a reasonable error or a severe error?\n"
                f"- Reasonable: prediction is only 1 star away from the truth "
                f"(e.g. predicting 4 stars for a true 5-star review)\n"
                f"- Severe: prediction is 2+ stars away from the truth "
                f"(e.g. predicting 1 star for a true 5-star review)\n\n"
                f'Respond with JSON only: '
                f'{{"reasonable": true/false, '
                f'"severity": "reasonable" or "severe", '
                f'"reason": "<one sentence explanation>"}}'
            )
        }
    ]

# ── 3. API Call ────────────────────────────────────────────────────────────────
def call_judge(messages: list, retries=3):
    """Call GPT judge API with retry and rate limiting."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=CFG["model"],
                messages=messages,
                temperature=CFG["temperature"],
                max_tokens=CFG["max_tokens"],
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed  = json.loads(content)

            assert "reasonable" in parsed
            assert "severity"   in parsed

            time.sleep(CFG["sleep_sec"])
            return parsed

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            print(f"    Parse error (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(25)

    return None

# ── 4. Error Loaders ───────────────────────────────────────────────────────────
def _get_classical_errors(model_name: str, test_df: pd.DataFrame):
    """Re-run LR or SVM inference using saved pkl to get per-sample predictions."""
    import pickle

    model_path = f"{CFG['model_dir']}/{model_name}_tfidf_only.pkl"
    if not os.path.exists(model_path):
        print(f"  No saved model at {model_path} — skipping {model_name}")
        return []

    print(f"  Loading {model_path}...")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    tfidf = saved["tfidf"]
    model = saved["model"]

    X_test = tfidf.transform(test_df["text"].tolist())
    preds  = model.predict(X_test)
    trues  = test_df["label"].tolist()

    errors = []
    for i, (pred, true) in enumerate(zip(preds, trues)):
        if int(pred) != int(true):
            errors.append({
                "model":       model_name,
                "review":      test_df.iloc[i]["text"],
                "true_rating": int(true),
                "pred_rating": int(pred),
                "gap":         abs(int(pred) - int(true)),
            })
    print(f"  Found {len(errors)} errors out of {len(trues)} samples")
    return errors


def _get_roberta_errors(test_df: pd.DataFrame):
    """
    Re-run RoBERTa inference using saved best model weights
    to get per-sample predictions.
    """
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    # find best config from results
    results_path = f"{CFG['results_dir']}/roberta_results.json"
    if not os.path.exists(results_path):
        print("  roberta_results.json not found — skipping roberta")
        return []

    with open(results_path) as f:
        results = json.load(f)
    best = max(results, key=lambda x: x["best_val_f1"])
    lr   = best["lr"]
    bs   = best["batch_size"]

    model_path = f"{CFG['model_dir']}/roberta/lr{lr}_bs{bs}_best.pt"
    if not os.path.exists(model_path):
        print(f"  No saved model at {model_path} — skipping roberta")
        return []

    print(f"  Loading RoBERTa weights from {model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(CFG["roberta_name"])
    model     = RobertaForSequenceClassification.from_pretrained(
        CFG["roberta_name"], num_labels=CFG["num_labels"]
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # tokenize test set
    print("  Tokenizing test set...")
    texts  = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=CFG["max_length"],
        return_tensors="pt"
    )

    # run inference in batches
    all_preds = []
    batch_size = 32
    n = len(texts)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end        = min(start + batch_size, n)
            input_ids  = encodings["input_ids"][start:end].to(device)
            attn_mask  = encodings["attention_mask"][start:end].to(device)
            outputs    = model(input_ids=input_ids, attention_mask=attn_mask)
            preds      = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

            if (start // batch_size + 1) % 10 == 0:
                print(f"  Inference: {end}/{n}")

    # convert predictions: model outputs 0-4, labels are 1-5
    errors = []
    for i, (pred, true) in enumerate(zip(all_preds, labels)):
        pred_star = int(pred) + 1   # 0-4 → 1-5
        true_star = int(true)       # already 1-5
        if pred_star != true_star:
            errors.append({
                "model":       "roberta",
                "review":      texts[i],
                "true_rating": true_star,
                "pred_rating": pred_star,
                "gap":         abs(pred_star - true_star),
            })

    print(f"  Found {len(errors)} errors out of {n} samples")
    return errors


def _get_gpt_errors(strategy: str, test_df: pd.DataFrame):
    """Load GPT prediction errors from saved JSON results."""
    path = f"{CFG['results_dir']}/gpt_{strategy}.json"
    if not os.path.exists(path):
        print(f"  {path} not found — skipping gpt_{strategy}")
        return []

    with open(path) as f:
        data = json.load(f)

    errors = []
    for item in data["predictions"]:
        if item["pred"] != item["true"]:
            idx = item["idx"]
            errors.append({
                "model":       f"gpt_{strategy}",
                "review":      test_df.iloc[idx]["text"] if idx < len(test_df) else "",
                "true_rating": item["true"],
                "pred_rating": item["pred"],
                "gap":         abs(item["pred"] - item["true"]),
            })

    print(f"  Found {len(errors)} errors")
    return errors


def load_model_errors(model_name: str, test_df: pd.DataFrame):
    """Route to correct error loader based on model name."""
    if model_name in ["lr", "svm"]:
        return _get_classical_errors(model_name, test_df)
    elif model_name == "roberta":
        return _get_roberta_errors(test_df)
    elif model_name.startswith("gpt_"):
        return _get_gpt_errors(model_name[4:], test_df)
    else:
        print(f"  Unknown model: {model_name}")
        return []

# ── 5. Run Judge on One Model ──────────────────────────────────────────────────
def judge_model(model_name: str, errors: list) -> dict:
    """Sample up to n_errors mistakes and run LLM judge on each."""
    print(f"\n── Judging: {model_name} ({len(errors)} total errors) ──")

    if not errors:
        print("  No errors to judge.")
        return {}

    # stratified sample: ~70% gap-1, ~30% gap-2+
    gap1 = [e for e in errors if e["gap"] == 1]
    gap2 = [e for e in errors if e["gap"] >= 2]

    n_gap1 = min(len(gap1), int(CFG["n_errors"] * 0.7))
    n_gap2 = min(len(gap2), CFG["n_errors"] - n_gap1)

    sampled = (random.sample(gap1, n_gap1) +
               random.sample(gap2, n_gap2))
    random.shuffle(sampled)

    print(f"  Sampled {len(sampled)} errors "
          f"(gap-1: {n_gap1}, gap-2+: {n_gap2})")
    print(f"  Estimated time: ~{len(sampled) * CFG['sleep_sec'] // 60} min")

    judgments = []
    n_reasonable, n_severe, n_failed = 0, 0, 0

    for i, error in enumerate(sampled):
        messages = build_judge_prompt(
            error["review"],
            error["true_rating"],
            error["pred_rating"]
        )
        result = call_judge(messages)

        if result is None:
            n_failed += 1
            judgment = {**error, "judgment": None, "reasonable": None}
        else:
            is_reasonable = bool(result.get("reasonable", False))
            if is_reasonable:
                n_reasonable += 1
            else:
                n_severe += 1
            judgment = {**error, "judgment": result, "reasonable": is_reasonable}

        judgments.append(judgment)

        if (i + 1) % 10 == 0:
            total_judged = n_reasonable + n_severe
            pct = n_reasonable / total_judged * 100 if total_judged > 0 else 0
            print(f"  [{i+1}/{len(sampled)}]  "
                  f"reasonable={n_reasonable}  "
                  f"severe={n_severe}  "
                  f"reasonable%={pct:.1f}%")

    total_judged   = n_reasonable + n_severe
    pct_reasonable = n_reasonable / total_judged * 100 if total_judged > 0 else 0
    pct_severe     = n_severe     / total_judged * 100 if total_judged > 0 else 0

    print(f"\n  FINAL  reasonable={n_reasonable} ({pct_reasonable:.1f}%)  "
          f"severe={n_severe} ({pct_severe:.1f}%)  "
          f"failed={n_failed}")

    return {
        "model":          model_name,
        "total_errors":   len(errors),
        "sampled":        len(sampled),
        "n_reasonable":   n_reasonable,
        "n_severe":       n_severe,
        "n_failed":       n_failed,
        "pct_reasonable": round(pct_reasonable, 1),
        "pct_severe":     round(pct_severe, 1),
        "judgments":      judgments,
    }

# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='sk-...'")
        return

    print("Loading test set...")
    test_df = pd.read_csv(f"{CFG['data_dir']}/test.csv")

    models_to_judge = [
        "lr",
        "svm",
        "roberta",
        "gpt_zero_shot",
        "gpt_few_shot",
        "gpt_cot",
        "gpt_cot_few_shot",
    ]

    all_summaries = []

    for model_name in models_to_judge:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        errors  = load_model_errors(model_name, test_df)
        summary = judge_model(model_name, errors)

        if summary:
            all_summaries.append(summary)

            # save immediately after each model
            out_path = f"{CFG['results_dir']}/judge_{model_name}.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved → {out_path}")

    # final summary table
    print("\n" + "="*65)
    print(f"{'Model':<20} {'Reasonable%':<14} {'Severe%':<12} {'N judged'}")
    print("="*65)
    for s in sorted(all_summaries,
                    key=lambda x: x["pct_reasonable"], reverse=True):
        print(f"{s['model']:<20} {s['pct_reasonable']:<14} "
              f"{s['pct_severe']:<12} {s['sampled']}")
    print("="*65)

    # save summary without full judgments
    summary_path = f"{CFG['results_dir']}/judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump([{k: v for k, v in s.items()
                    if k != "judgments"} for s in all_summaries],
                  f, indent=2)
    print(f"\nSummary saved → {summary_path}")

if __name__ == "__main__":
    main()