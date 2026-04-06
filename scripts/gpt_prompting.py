# scripts/gpt_prompting.py
# GPT-4o-mini prompting: 4 strategies x 500 test samples
# Zero-shot / Few-shot / CoT / CoT+Few-shot

import os, json, time, random
import pandas as pd
from openai import OpenAI

# ── 0. Config ──────────────────────────────────────────────────────────────────
CFG = {
    "data_dir":      "data/processed",
    "results_dir":   "results",
    "model":         "gpt-4o-mini",
    "temperature":   0,            # fixed at 0 for reproducibility
    "max_tokens":    200,
    "n_samples":     50,          # number of test samples per strategy
    "n_few_shot":    3,            # number of examples for few-shot strategies
    "seed":          42,
}
os.makedirs(CFG["results_dir"], exist_ok=True)

# ── 1. OpenAI Client ───────────────────────────────────────────────────────────
# read API key from environment variable (never hardcode)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 2. Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
test_df  = pd.read_csv(f"{CFG['data_dir']}/test.csv")
train_df = pd.read_csv(f"{CFG['data_dir']}/train.csv")

# fixed random sample of 500 test reviews
random.seed(CFG["seed"])
sample_idx  = random.sample(range(len(test_df)), CFG["n_samples"])
test_sample = test_df.iloc[sample_idx].reset_index(drop=True)

# build few-shot candidate pool: several examples per star rating
few_shot_pool = {}
for star in range(1, 6):
    pool = train_df[train_df["label"] == star]["text"].tolist()
    few_shot_pool[star] = pool

def get_few_shot_examples(n=3):
    """Sample n examples from training set, covering different star ratings."""
    examples = []
    stars = [1, 2, 4, 5, 3]   # lead with extreme examples for clearer demonstration
    for i in range(n):
        star = stars[i % len(stars)]
        text = few_shot_pool[star][i][:300]  # truncate to avoid overly long prompts
        examples.append({"text": text, "rating": star})
    return examples

FEW_SHOT_EXAMPLES = get_few_shot_examples(CFG["n_few_shot"])

# ── 3. Prompt Builders ─────────────────────────────────────────────────────────

SYSTEM_BASE = """You are a sentiment analysis expert. 
Your task is to predict the star rating (1-5) of a Yelp business review.
1 star = very negative, 2 stars = negative, 3 stars = neutral, 
4 stars = positive, 5 stars = very positive.
Always respond with valid JSON only."""

def build_zero_shot(review: str) -> list:
    """Strategy 1: Zero-shot — direct instruction with no examples."""
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content":
            f'Predict the star rating for this Yelp review.\n\n'
            f'Review: """{review}"""\n\n'
            f'Respond with JSON only: {{"rating": <integer 1-5>}}'}
    ]

def build_few_shot(review: str) -> list:
    """Strategy 2: Few-shot — 3 labeled examples prepended before target review."""
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += (
            f'Example {i}:\n'
            f'Review: """{ex["text"]}"""\n'
            f'Output: {{"rating": {ex["rating"]}}}\n\n'
        )
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content":
            f'Predict the star rating for a Yelp review. '
            f'Here are {CFG["n_few_shot"]} examples:\n\n'
            f'{examples_text}'
            f'Now predict for this review:\n'
            f'Review: """{review}"""\n\n'
            f'Respond with JSON only: {{"rating": <integer 1-5>}}'}
    ]

def build_cot(review: str) -> list:
    """Strategy 3: Chain-of-Thought — model reasons step by step before rating."""
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content":
            f'Predict the star rating for this Yelp review. '
            f'Think step by step:\n'
            f'1. Identify positive aspects mentioned\n'
            f'2. Identify negative aspects mentioned\n'
            f'3. Consider the overall tone\n'
            f'4. Assign a rating 1-5\n\n'
            f'Review: """{review}"""\n\n'
            f'Respond with JSON only: '
            f'{{"reasoning": "<your step-by-step analysis>", '
            f'"rating": <integer 1-5>}}'}
    ]

def build_cot_few_shot(review: str) -> list:
    """Strategy 4: CoT + Few-shot — labeled examples with reasoning chains."""
    examples_text = ""
    cot_reasons = [
        "The reviewer mentions long wait and poor service. Overall very dissatisfied.",
        "Mixed experience - food was decent but service slow. Slightly below average.",
        "Generally positive with minor complaints about price. Solid experience.",
    ]
    for i, (ex, reason) in enumerate(
            zip(FEW_SHOT_EXAMPLES, cot_reasons), 1):
        examples_text += (
            f'Example {i}:\n'
            f'Review: """{ex["text"]}"""\n'
            f'Output: {{"reasoning": "{reason}", '
            f'"rating": {ex["rating"]}}}\n\n'
        )
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content":
            f'Predict the star rating for a Yelp review using step-by-step '
            f'reasoning. Here are {CFG["n_few_shot"]} examples:\n\n'
            f'{examples_text}'
            f'Now predict for this review:\n'
            f'Review: """{review}"""\n\n'
            f'Respond with JSON only: '
            f'{{"reasoning": "<your analysis>", '
            f'"rating": <integer 1-5>}}'}
    ]

STRATEGIES = {
    "zero_shot":    build_zero_shot,
    "few_shot":     build_few_shot,
    "cot":          build_cot,
    "cot_few_shot": build_cot_few_shot,
}

# ── 4. API Call with Retry ─────────────────────────────────────────────────────
def call_gpt(messages: list, retries=3):
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
            rating  = int(parsed.get("rating", -1))
            if rating not in range(1, 6):
                raise ValueError(f"Invalid rating: {rating}")

            time.sleep(22)   # ← add this: wait 22s to stay under 3 RPM limit

            return parsed

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    Parse error (attempt {attempt+1}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(25)   # ← longer wait on error

    return None

# ── 5. Run All Strategies ──────────────────────────────────────────────────────
def run_strategy(strategy_name: str, build_fn) -> dict:
    print(f"\n── Strategy: {strategy_name} ({'─'*30})")
    preds, trues, raw_outputs = [], [], []
    errors = 0
    t0     = time.time()

    for i, row in test_sample.iterrows():
        review     = row["text"][:1000]   # truncate very long reviews
        true_label = int(row["label"])

        messages = build_fn(review)
        result   = call_gpt(messages)

        if result is None:
            errors += 1
            pred = 3   # fallback to neutral on total failure
        else:
            pred = int(result["rating"])

        preds.append(pred)
        trues.append(true_label)
        raw_outputs.append({
            "idx":        int(i),
            "true":       true_label,
            "pred":       pred,
            "raw_output": result,
        })

        # print progress every 50 samples
        if (len(preds)) % 50 == 0:
            so_far_acc = sum(p == t for p, t in zip(preds, trues)) / len(preds)
            elapsed    = time.time() - t0
            print(f"  [{len(preds)}/500]  running_acc={so_far_acc:.3f}  "
                  f"elapsed={elapsed:.0f}s  errors={errors}")

    # compute final metrics
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro")
    cm  = confusion_matrix(trues, preds, labels=[1,2,3,4,5])

    elapsed = time.time() - t0
    print(f"  DONE  acc={acc:.4f}  f1={f1:.4f}  "
          f"time={elapsed:.0f}s  errors={errors}")

    return {
        "strategy":         strategy_name,
        "accuracy":         round(acc, 4),
        "macro_f1":         round(f1, 4),
        "errors":           errors,
        "elapsed_sec":      round(elapsed, 1),
        "confusion_matrix": cm.tolist(),
        "predictions":      raw_outputs,
    }

# ── 6. Main ────────────────────────────────────────────────────────────────────
def main():
    # verify API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='sk-...'")
        return

    print(f"Model:                {CFG['model']}")
    print(f"Samples per strategy: {CFG['n_samples']}")
    print(f"Total API calls:      {CFG['n_samples'] * len(STRATEGIES)}")

    all_results = []
    for name, build_fn in STRATEGIES.items():
        result = run_strategy(name, build_fn)
        all_results.append(result)

        # save after each strategy in case of interruption
        out_path = f"{CFG['results_dir']}/gpt_{name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")

    # print summary table
    print("\n" + "="*60)
    print(f"{'Strategy':<16} {'Accuracy':<12} {'Macro F1':<12} {'Time(s)'}")
    print("="*60)
    for r in all_results:
        print(f"{r['strategy']:<16} {r['accuracy']:<12} "
              f"{r['macro_f1']:<12} {r['elapsed_sec']}")
    print("="*60)

    # compare against classical models
    print("\n── vs Classical Models ──────────────────────────────────")
    print(f"  LR  (tfidf_only)  f1=0.5802")
    print(f"  SVM (tfidf_only)  f1=0.5770")
    for r in sorted(all_results, key=lambda x: x["macro_f1"], reverse=True):
        delta = r["macro_f1"] - 0.5802
        print(f"  GPT {r['strategy']:<14} f1={r['macro_f1']}  "
              f"(delta vs LR: {delta:+.4f})")

    # save summary without full predictions to keep file small
    summary_path = f"{CFG['results_dir']}/gpt_summary.json"
    with open(summary_path, "w") as f:
        json.dump([{k: v for k, v in r.items()
                    if k != "predictions"} for r in all_results],
                  f, indent=2)
    print(f"\nSummary saved → {summary_path}")

if __name__ == "__main__":
    main()