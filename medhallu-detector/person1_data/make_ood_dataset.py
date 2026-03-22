# ============================================================
# make_ood_dataset.py
# ============================================================
# WHAT  : Builds 200 out-of-distribution (OOD) test rows from
#         MedQA (USMLE-style questions) by taking correct
#         answers and introducing deliberate medical errors
#
# WHY   : Person 2's DeBERTa is trained only on MedHallu.
#         Running it on MedQA — a completely different dataset
#         with different question styles — tests whether the
#         model truly learned to detect hallucinations or just
#         memorised MedHallu patterns.
#
# HOW   : Load MedQA → sample 200 rows → take the correct
#         answer → swap one key medical term with a wrong one
#         → format exactly like a MedHallu row → save CSV
#
# CHANGES FROM ORIGINAL:
#   - Tries 3 different MedQA dataset sources in order
#   - Handles MedQA's specific column structure automatically
#   - Prints full column names after loading so you can see
#     exactly what the dataset looks like before processing
#   - extract_correct_answer() handles all known MedQA formats
# ============================================================


from datasets import load_dataset
import pandas as pd
import random
import re
import json
import os


# --- Configuration -------------------------------------------

SAMPLE_SIZE = 200
RANDOM_SEED = 42
OUTPUT_CSV  = "outputs/medqa_ood_200.csv"


# --- Medical term swap dictionary ----------------------------
# Each entry: "correct term" → "wrong term"
# Paired by medical similarity so swaps look like real LLM mistakes

MEDICAL_SWAPS = {
    # Enzymes and metabolic
    "phenylalanine hydroxylase": "tyrosine hydroxylase",
    "tyrosine hydroxylase":      "phenylalanine hydroxylase",
    "adenylyl cyclase":          "guanylyl cyclase",
    "guanylyl cyclase":          "adenylyl cyclase",

    # Hormones
    "insulin":            "glucagon",
    "glucagon":           "insulin",
    "cortisol":           "aldosterone",
    "aldosterone":        "cortisol",
    "epinephrine":        "norepinephrine",
    "norepinephrine":     "epinephrine",
    "thyroxine":          "triiodothyronine",
    "triiodothyronine":   "thyroxine",
    "estrogen":           "progesterone",
    "progesterone":       "estrogen",

    # Neurotransmitters
    "dopamine":       "serotonin",
    "serotonin":      "dopamine",
    "acetylcholine":  "norepinephrine",
    "gaba":           "glutamate",
    "glutamate":      "gaba",

    # Blood and cardiovascular
    "systolic":    "diastolic",
    "diastolic":   "systolic",
    "artery":      "vein",
    "vein":        "artery",
    "hemoglobin":  "myoglobin",
    "myoglobin":   "hemoglobin",
    "platelet":    "erythrocyte",
    "erythrocyte": "platelet",

    # Drug classes
    "ace inhibitor": "beta blocker",
    "beta blocker":  "ace inhibitor",
    "aspirin":       "ibuprofen",
    "ibuprofen":     "aspirin",
    "metformin":     "glipizide",
    "glipizide":     "metformin",
    "warfarin":      "heparin",
    "heparin":       "warfarin",
    "amoxicillin":   "ampicillin",
    "ampicillin":    "amoxicillin",

    # Conditions
    "hypertension":    "hypotension",
    "hypotension":     "hypertension",
    "hyperglycemia":   "hypoglycemia",
    "hypoglycemia":    "hyperglycemia",
    "tachycardia":     "bradycardia",
    "bradycardia":     "tachycardia",
    "hyperthyroidism": "hypothyroidism",
    "hypothyroidism":  "hyperthyroidism",

    # Organs and anatomy
    "liver":           "kidney",
    "kidney":          "liver",
    "pancreas":        "spleen",
    "spleen":          "pancreas",
    "mitral valve":    "aortic valve",
    "aortic valve":    "mitral valve",
    "left ventricle":  "right ventricle",
    "right ventricle": "left ventricle",

    # Directional / quantitative
    "increases":  "decreases",
    "decreases":  "increases",
    "elevated":   "reduced",
    "reduced":    "elevated",
    "inhibits":   "activates",
    "activates":  "inhibits",
    "deficiency": "excess",
    "excess":     "deficiency",
}


# --- Step 1: Load MedQA --------------------------------------

def load_medqa() -> tuple[pd.DataFrame, str]:
    """
    Tries three known MedQA sources in order.
    Returns (dataframe, source_name) for whichever works first.

    WHY three sources:
    MedQA is available under different HuggingFace repo names
    depending on what version of the datasets library you have.
    We try the most common ones so the script works regardless.
    """

    # Source 1: GBaker — simplest, most reliable, no trust_remote_code
    print("Trying GBaker/MedQA-USMLE-4-options...")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options")
        split = "test" if "test" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        print(f"Loaded {len(df)} rows from GBaker/MedQA-USMLE-4-options ({split} split)")
        print(f"Columns: {df.columns.tolist()}")
        return df, "GBaker"
    except Exception as e:
        print(f"  Failed: {e}")

    # Source 2: bigbio version
    print("\nTrying bigbio/med_qa (med_qa_en_source)...")
    try:
        ds = load_dataset(
            "bigbio/med_qa",
            "med_qa_en_source",
            trust_remote_code=True
        )
        split = "test" if "test" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        print(f"Loaded {len(df)} rows from bigbio/med_qa ({split} split)")
        print(f"Columns: {df.columns.tolist()}")
        return df, "bigbio"
    except Exception as e:
        print(f"  Failed: {e}")

    # Source 3: original MedQA repo
    print("\nTryingmed_qa...")
    try:
        ds = load_dataset("med_qa", trust_remote_code=True)
        split = "test" if "test" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        print(f"Loaded {len(df)} rows from med_qa ({split} split)")
        print(f"Columns: {df.columns.tolist()}")
        return df, "med_qa"
    except Exception as e:
        print(f"  Failed: {e}")

    return None, None


# --- Step 2: Extract correct answer --------------------------

def extract_correct_answer(row: dict, source: str) -> str:
    """
    Extracts the correct answer text from a MedQA row.

    WHY different logic per source:
    Each MedQA version stores answers differently:

    GBaker format:
      row["options"] = {"A": "text", "B": "text", ...}
      row["answer_idx"] = "A"
      → correct = options[answer_idx]

    bigbio format:
      row["answer"] = [{"text": "...", "id": "A"}]
      → correct = answer[0]["text"]

    med_qa format:
      row["options"] = {"A": "text", ...}
      row["answer_idx"] = "A"
      → same as GBaker
    """

    # --- GBaker / med_qa format ---
    if source in ("GBaker", "med_qa"):
        options   = row.get("options", {})
        answer_idx = row.get("answer_idx", row.get("answer", ""))

        # options can be a dict or a JSON string
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                options = {}

        if isinstance(options, dict) and answer_idx in options:
            return str(options[answer_idx])

        # Sometimes answer_idx is the full answer text
        if isinstance(answer_idx, str) and len(answer_idx) > 3:
            return answer_idx

    # --- bigbio format ---
    if source == "bigbio":
        answer_field = row.get("answer", [])

        # List of dicts: [{"text": "...", "id": "A"}]
        if isinstance(answer_field, list) and len(answer_field) > 0:
            first = answer_field[0]
            if isinstance(first, dict):
                return str(first.get("text", ""))
            return str(first)

        # Sometimes it's a plain string
        if isinstance(answer_field, str) and len(answer_field) > 3:
            return answer_field

        # Try choices field
        choices = row.get("choices", row.get("options", []))
        correct_idx = row.get("answerKey", row.get("answer_idx", ""))
        if isinstance(choices, list) and isinstance(correct_idx, int):
            if correct_idx < len(choices):
                return str(choices[correct_idx])

    # Fallback — use question text
    return str(row.get("question", ""))


# --- Step 3: Swap medical term -------------------------------

def swap_medical_term(text: str) -> tuple[str, bool]:
    """
    Finds the first swappable medical term in text and
    replaces it with its wrong counterpart.

    Returns (modified_text, was_swapped).
    Longer phrases are tried first to avoid partial matches.
    """
    text_lower = text.lower()

    # Sort longest first so "left ventricle" matches before "ventricle"
    sorted_terms = sorted(MEDICAL_SWAPS.keys(), key=len, reverse=True)

    for term in sorted_terms:
        if term in text_lower:
            replacement = MEDICAL_SWAPS[term]
            swapped = re.sub(
                re.escape(term),
                replacement,
                text,
                count=1,
                flags=re.IGNORECASE
            )
            return swapped, True

    return text, False


# --- Step 4: Build one OOD row -------------------------------

def build_ood_row(row: dict, source: str) -> dict | None:
    """
    Takes one MedQA row and returns a MedHallu-formatted dict
    with a deliberately hallucinated answer.
    Returns None if no medical term swap was possible.
    """
    question       = str(row.get("question", "")).strip()
    correct_answer = extract_correct_answer(row, source)

    # Try swapping in the correct answer first
    hallucinated_answer, was_swapped = swap_medical_term(correct_answer)

    if not was_swapped:
        # Fall back to swapping in the question text
        hallucinated_answer, was_swapped = swap_medical_term(question)
        if not was_swapped:
            return None

    return {
        "question":           question,
        "answer":             hallucinated_answer,
        "correct_answer":     correct_answer,
        "source":             question,        # MedQA has no separate passage
        "label":              "hallucinated",
        "difficulty":         "medium",
        "hallucination_type": "fabrication",
        "dataset_origin":     "medqa_ood",
    }


# --- Step 5: Main pipeline -----------------------------------

def main():

    # 5a. Load MedQA
    raw_df, source_name = load_medqa()

    if raw_df is None:
        print("\nERROR: Could not load any MedQA source.")
        print("Try running: pip install datasets --upgrade")
        print("Then try again.")
        return

    print(f"\nUsing source: {source_name}")
    print(f"Total rows available: {len(raw_df)}")
    print(f"Columns: {raw_df.columns.tolist()}")

    # Print one raw row so you can verify the format
    print("\n--- First raw row (to verify column structure) ---")
    first = raw_df.iloc[0].to_dict()
    for k, v in first.items():
        print(f"  {k}: {str(v)[:120]}")

    # 5b. Oversample before filtering
    # WHY 3x: many rows won't have swappable terms — we
    # start with 3x the target and trim after building rows
    random.seed(RANDOM_SEED)
    n_sample = min(SAMPLE_SIZE * 3, len(raw_df))
    sample_df = raw_df.sample(n=n_sample, random_state=RANDOM_SEED)
    sample_df = sample_df.reset_index(drop=True)
    print(f"\nSampled {len(sample_df)} rows to process")

    # 5c. Build OOD rows
    print("Building hallucinated rows...")
    ood_rows    = []
    skipped     = 0
    swap_counts = {}

    for _, row in sample_df.iterrows():
        result = build_ood_row(row.to_dict(), source_name)

        if result is None:
            skipped += 1
            continue

        ood_rows.append(result)

        # Track which terms were swapped most
        for term in MEDICAL_SWAPS:
            if term in result["answer"].lower():
                swap_counts[term] = swap_counts.get(term, 0) + 1

        if len(ood_rows) >= SAMPLE_SIZE:
            break

    print(f"Built   : {len(ood_rows)} OOD rows")
    print(f"Skipped : {skipped} rows (no swappable term found)")

    # 5d. Warn if under target
    if len(ood_rows) < SAMPLE_SIZE:
        print(f"\nWARNING: Only built {len(ood_rows)} / {SAMPLE_SIZE} rows")
        print("To fix: add more medical term pairs to MEDICAL_SWAPS above")

    if len(ood_rows) == 0:
        print("\nERROR: Zero rows built.")
        print("The correct answer extraction may have failed.")
        print("Check the 'First raw row' output above and update")
        print("extract_correct_answer() to match your dataset's format.")
        return

    # 5e. Convert to DataFrame
    ood_df = pd.DataFrame(ood_rows)

    # 5f. Verify required columns exist
    required = ["question", "answer", "source", "label", "difficulty"]
    for col in required:
        if col not in ood_df.columns:
            print(f"ERROR: missing column '{col}'")
            return
    print(f"\nColumn check passed: {required}")

    # 5g. Save
    os.makedirs("outputs", exist_ok=True)
    ood_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(ood_df)} rows to {OUTPUT_CSV}")

    # 5h. Preview rows — verify quality manually
    print("\n--- Sample rows (read these carefully) ---")
    for i, row in ood_df.head(5).iterrows():
        print(f"\nRow {i+1}")
        print(f"  Question : {row['question'][:120]}")
        print(f"  Correct  : {row['correct_answer'][:100]}")
        print(f"  HALLUC   : {row['answer'][:100]}")

    # 5i. Swap frequency report
    if swap_counts:
        print("\n--- Most swapped terms ---")
        top = sorted(swap_counts.items(), key=lambda x: -x[1])[:10]
        for term, count in top:
            print(f"  '{term}' → '{MEDICAL_SWAPS[term]}' : {count} rows")

    # 5j. Final summary
    print("\n" + "="*50)
    print("OOD DATASET SUMMARY")
    print("="*50)
    print(f"Source dataset : {source_name}")
    print(f"Total rows     : {len(ood_df)}")
    print(f"Label dist     : {ood_df['label'].value_counts().to_dict()}")
    print(f"Output file    : {OUTPUT_CSV}")
    print("="*50)
    print("\nHand outputs/medqa_ood_200.csv to Person 2.")
    print("They run their trained model on it without retraining.")


if __name__ == "__main__":
    main()