# ============================================================
# groq_baseline.py
# ============================================================
# WHAT  : Sends 1000 MedHallu test rows to Groq's free Llama
#         API and measures hallucination detection F1 score
# WHY   : Establishes a baseline — the score Person 2's
#         fine-tuned DeBERTa model needs to beat
# HOW   : Loop each row → build a prompt → call Groq API →
#         collect prediction → compute F1 per difficulty
# ============================================================

from groq import Groq
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import time
import os
import json


# --- Configuration -------------------------------------------

API_KEY    = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.1-8b-instant"
NUM_ROWS   = 1000
SLEEP_SEC  = 0.25
OUTPUT_CSV = "outputs/groq_results.csv"

# WHY only "hallucinated" in VALID_LABELS:
# pqa_artificial has ONLY hallucinated rows — every answer is
# wrong by construction. The model should predict "hallucinated"
# for all of them. We still allow other labels so we can measure
# how often Groq incorrectly says "grounded".
VALID_LABELS = {"hallucinated", "grounded", "partially_hallucinated", "not_sure"}


# --- Prompt --------------------------------------------------

PROMPT_TEMPLATE = """You are a medical fact-checking assistant.

Given a medical question, an answer provided by an AI, and the source text the answer should be grounded in, decide if the answer is hallucinated.

Question: {question}

Answer given by AI: {answer}

Source (ground truth): {source}

Is the answer hallucinated, grounded, or partially hallucinated?
Reply with exactly one of these four words only, nothing else:
hallucinated / grounded / partially_hallucinated / not_sure"""


# --- Helper functions ----------------------------------------

def load_and_prepare_data():
    """
    Loads MedHallu pqa_artificial, renames columns to match
    our project's naming convention, adds the label column,
    and returns the test slice (last 1000 rows).

    COLUMN MAPPING (actual → our name):
      'Question'                  → 'question'
      'Knowledge'                 → 'source'
      'Ground Truth'              → 'ground_truth'
      'Hallucinated Answer'       → 'answer'
      'Difficulty Level'          → 'difficulty'
      'Category of Hallucination' → 'hallucination_type'

    WHY add label manually:
    pqa_artificial has no label column because every row is
    hallucinated by definition. We add label='hallucinated'
    so our F1 computation code works consistently.
    """
    print("Loading MedHallu (pqa_artificial)...")
    ds      = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
    full_df = ds["train"].to_pandas()

    print(f"Raw columns  : {full_df.columns.tolist()}")
    print(f"Total rows   : {len(full_df)}")

    # Rename columns
    full_df = full_df.rename(columns={
        "Question":                  "question",
        "Knowledge":                 "source",
        "Ground Truth":              "ground_truth",
        "Hallucinated Answer":       "answer",
        "Difficulty Level":          "difficulty",
        "Category of Hallucination": "hallucination_type",
    })

    # Add label column — all rows are hallucinated
    full_df["label"] = "hallucinated"

    # Normalise difficulty values to lowercase
    # WHY: dataset may store "Easy" or "EASY" — we want "easy"
    full_df["difficulty"] = full_df["difficulty"].str.lower().str.strip()

    # Normalise hallucination_type to lowercase
    full_df["hallucination_type"] = (
        full_df["hallucination_type"]
        .fillna("none")
        .str.lower()
        .str.strip()
    )

    print(f"Renamed columns : {full_df.columns.tolist()}")
    print(f"Difficulty values : {full_df['difficulty'].unique()}")
    print(f"Hallucination types : {full_df['hallucination_type'].unique()}")

    # Use last 1000 rows as test set
    # WHY last 1000: train_deberta.py uses first 7000 for training
    # and rows 7000-8000 for validation — so 8000-9000 is unseen
    test_df = full_df.iloc[8000:].reset_index(drop=True)
    print(f"\nTest rows : {len(test_df)}")

    return test_df


def call_groq(client, question, answer, source):
    """
    Sends one row to Groq and returns the model's prediction.
    Returns 'not_sure' on any error so the loop never crashes.
    """
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        source=source[:800]
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()
        raw = raw.replace(".", "").replace(",", "").replace("'", "")

        if raw not in VALID_LABELS:
            print(f"  Unexpected response: '{raw}' → defaulting to not_sure")
            return "not_sure"

        return raw

    except Exception as e:
        print(f"  API error: {e} → returning not_sure")
        return "not_sure"


def compute_f1_splits(df):
    """
    Computes macro F1 for overall + each difficulty level.

    WHY macro F1:
    Since all true labels are "hallucinated", a model that
    always predicts "hallucinated" gets F1=1.0 on that class
    but 0.0 on all others. Macro F1 averages across all classes
    equally — penalising the model for never predicting "grounded".
    This matches the MedHallu paper's evaluation methodology.
    """
    results = {}

    results["overall"] = f1_score(
        df["label"], df["groq_pred"],
        average="macro",
        zero_division=0
    )

    for diff in ["easy", "medium", "hard"]:
        subset = df[df["difficulty"] == diff]
        if len(subset) == 0:
            print(f"  WARNING: no rows found for difficulty='{diff}'")
            print(f"  Available difficulty values: {df['difficulty'].unique()}")
            results[diff] = None
            continue
        results[diff] = round(f1_score(
            subset["label"], subset["groq_pred"],
            average="macro",
            zero_division=0
        ), 4)

    return results


# --- Main pipeline -------------------------------------------

def main():

    # Validate API key
    if not API_KEY:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Windows : set GROQ_API_KEY=your_key_here")
        print("Mac/Linux: export GROQ_API_KEY=your_key_here")
        print("\nGet a free key at: console.groq.com")
        return

    # Load data
    test_df = load_and_prepare_data()

    # Trim to NUM_ROWS
    test_df = test_df.head(NUM_ROWS).copy()
    print(f"Running predictions on {len(test_df)} rows")

    # Create Groq client
    client = Groq(api_key=API_KEY)
    print(f"Groq client ready. Model: {MODEL_NAME}\n")

    # Run predictions
    print(f"Running {NUM_ROWS} predictions...")
    print("Expected time: ~5 minutes\n")
    predictions = []

    for i, row in test_df.iterrows():
        pred = call_groq(
            client,
            question=str(row["question"]),
            answer=str(row["answer"]),
            source=str(row["source"])
        )
        predictions.append(pred)

        completed = len(predictions)
        if completed % 100 == 0:
            print(f"  {completed}/{NUM_ROWS} done...")

        time.sleep(SLEEP_SEC)

    # Attach predictions
    test_df["groq_pred"] = predictions

    # Save results CSV
    os.makedirs("outputs", exist_ok=True)
    test_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to {OUTPUT_CSV}")

    # Compute F1
    f1_results = compute_f1_splits(test_df)

    print("\n" + "="*50)
    print("GROQ BASELINE RESULTS")
    print("="*50)
    print(f"Overall F1  : {f1_results['overall']:.4f}")
    print(f"Easy F1     : {f1_results.get('easy',  'N/A')}")
    print(f"Medium F1   : {f1_results.get('medium','N/A')}")
    print(f"Hard F1     : {f1_results.get('hard',  'N/A')}")
    print("="*50)

    # Detailed report
    print("\nDetailed classification report:")
    print(classification_report(
        test_df["label"],
        test_df["groq_pred"],
        zero_division=0
    ))

    # Prediction distribution
    # WHY check this: if Groq says "hallucinated" for 100% of rows
    # it's not detecting anything — it's just always guessing wrong
    # because every true label IS hallucinated. A good model should
    # also sometimes say "grounded" for partial credit on nuanced rows.
    print("Groq prediction distribution:")
    print(test_df["groq_pred"].value_counts())
    print("\nActual label distribution:")
    print(test_df["label"].value_counts())

    # Hallucination type breakdown
    # WHY: shows which types of hallucinations Groq struggles with most
    print("\nGroq accuracy by hallucination type:")
    for htype in test_df["hallucination_type"].unique():
        subset = test_df[test_df["hallucination_type"] == htype]
        correct = (subset["groq_pred"] == subset["label"]).sum()
        print(f"  {htype:<20} : {correct}/{len(subset)} correct "
              f"({correct/len(subset)*100:.1f}%)")

    # Save summary JSON for Person 2
    summary = {
        "model":          "Groq Llama-3.1-8B (zero-shot)",
        "overall_f1":     round(f1_results["overall"], 4),
        "easy_f1":        f1_results.get("easy"),
        "medium_f1":      f1_results.get("medium"),
        "hard_f1":        f1_results.get("hard"),
        "rows_evaluated": NUM_ROWS,
        "note":           "All true labels are hallucinated (pqa_artificial)"
    }

    with open("outputs/groq_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary saved to outputs/groq_baseline_summary.json")
    print("Hand this file to Person 2 — they fill in their model's numbers.")


if __name__ == "__main__":
    main()