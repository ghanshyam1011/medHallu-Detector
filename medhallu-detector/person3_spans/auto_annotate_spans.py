# ============================================================
# auto_annotate_spans.py
# ============================================================
# WHAT  : Uses Groq free API (Llama) to automatically find
#         the wrong phrase in each hallucinated answer.
#         Saves result as spans.json in Label Studio format
#         so train_span_extractor.py can use it directly.
#
# WHY faster than manual:
#         300 API calls at 0.3s each = ~2 minutes total
#         vs 2.5 hours of manual annotation
#
# RUN   : python auto_annotate_spans.py
# ============================================================

from groq import Groq
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import time


# --- safe_str helper -----------------------------------------

def safe_str(val) -> str:
    """
    Converts Arrow scalars and numpy values to plain strings.
    WHY: pandas with Arrow backend returns Arrow scalars from
    row access — calling str() on them raises ValueError.
    """
    if val is None:
        return ""
    if hasattr(val, "item"):
        try:
            val = val.item()
        except Exception:
            pass
    if hasattr(val, "__len__") and not isinstance(val, str):
        try:
            if len(val) == 0:
                return ""
            val = val[0]
        except Exception:
            pass
    try:
        if val != val:
            return ""
    except Exception:
        pass
    return str(val).strip()


# --- Configuration -------------------------------------------

API_KEY   = os.environ.get("GROQ_API_KEY", "")
MODEL     = "llama-3.1-8b-instant"
NUM_ROWS  = 300
SLEEP_SEC = 0.3
OUTPUT    = "annotation_export/spans.json"


# --- Prompt --------------------------------------------------

PROMPT = """You are a medical fact-checker.

Question: {question}

Source (ground truth): {source}

Hallucinated answer: {answer}

The hallucinated answer contains exactly ONE wrong phrase that contradicts the source.
Reply with ONLY that wrong phrase — nothing else, no explanation, no punctuation at the end.
The phrase must be an exact substring of the hallucinated answer.
Keep it as short as possible — usually 1-4 words."""


# --- Helper functions ----------------------------------------

def find_span(answer: str, phrase: str):
    """
    Finds character start and end positions of phrase in answer.
    Returns (start, end) or None if not found.

    WHY case-insensitive:
    Groq may return the phrase with different casing than
    what appears in the answer text.
    """
    answer_lower = answer.lower()
    phrase_lower = phrase.lower().strip()

    if not phrase_lower:
        return None

    idx = answer_lower.find(phrase_lower)
    if idx == -1:
        # Try partial match using first 3 words
        words = phrase_lower.split()
        if len(words) > 2:
            partial = " ".join(words[:3])
            idx = answer_lower.find(partial)
            if idx != -1:
                return idx, idx + len(partial)
        return None

    return idx, idx + len(phrase_lower)


def call_groq(client, question: str, answer: str, source: str) -> str:
    """
    Calls Groq API and returns the identified wrong phrase.
    Returns empty string on any error so loop never crashes.
    """
    prompt = PROMPT.format(
        question=question[:300],
        source=source[:500],
        answer=answer[:400],
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.0,
        )
        phrase = response.choices[0].message.content.strip()
        phrase = phrase.strip('"\'')
        return phrase
    except Exception as e:
        return ""


# --- Main pipeline -------------------------------------------

def main():

    # Validate API key
    if not API_KEY:
        print("ERROR: GROQ_API_KEY not set.")
        print("Windows : set GROQ_API_KEY=your_key_here")
        print("Mac/Linux: export GROQ_API_KEY=your_key_here")
        print("Get free key at: console.groq.com")
        return

    # Load MedHallu
    print("Loading MedHallu...")
    ds      = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
    full_df = ds["train"].to_pandas()

    full_df = full_df.rename(columns={
        "Question":            "question",
        "Knowledge":           "source",
        "Ground Truth":        "ground_truth",
        "Hallucinated Answer": "answer",
    })

    rows = full_df.head(NUM_ROWS)
    print(f"Processing {len(rows)} rows...")

    # Create Groq client
    client = Groq(api_key=API_KEY)

    # Run auto-annotation
    annotations  = []
    found_count  = 0
    failed_count = 0

    for i, row in tqdm(
        rows.iterrows(),
        total=len(rows),
        desc="Auto-annotating",
        unit="row",
        dynamic_ncols=True,
    ):
        # WHY safe_str: Arrow-backed DataFrame columns return
        # Arrow scalars — str(arrow_scalar or "") raises ValueError
        # because the 'or' triggers a boolean check on the scalar
        question = safe_str(row["question"])
        answer   = safe_str(row["answer"])
        source   = safe_str(row["source"])

        # Skip empty rows
        if not answer:
            failed_count += 1
            annotations.append({"id": i, "text": "", "label": []})
            continue

        # Ask Groq to identify the wrong phrase
        phrase = call_groq(client, question, answer, source)

        # Find the phrase's character position in the answer
        span = find_span(answer, phrase) if phrase else None

        if span:
            found_count += 1
            start, end = span
            annotations.append({
                "id":    i,
                "text":  answer,
                "label": [
                    {
                        "start":  start,
                        "end":    end,
                        "text":   answer[start:end],
                        "labels": ["wrong_phrase"],
                    }
                ],
                "question":     question,
                "source":       source[:300],
                "ground_truth": safe_str(row.get("ground_truth", "")),
            })
        else:
            failed_count += 1
            # Row saved without annotation — skipped by trainer
            annotations.append({
                "id":    i,
                "text":  answer,
                "label": [],
            })

        time.sleep(SLEEP_SEC)

    # Save output
    os.makedirs("annotation_export", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*50}")
    print("AUTO-ANNOTATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total rows       : {len(rows)}")
    print(f"Spans found      : {found_count}")
    print(f"Failed (no match): {failed_count}")
    print(f"Success rate     : {found_count/len(rows)*100:.1f}%")
    print(f"Output saved     : {OUTPUT}")
    print(f"{'='*50}")

    # Verify output
    print("\nVerifying output...")
    with open(OUTPUT) as f:
        check = json.load(f)

    valid = [r for r in check if r.get("label")]
    print(f"Rows with valid spans : {len(valid)}")

    print("\nSample annotations:")
    count = 0
    for item in check:
        if item.get("label") and count < 3:
            span = item["label"][0]
            print(f"  Answer : {item['text'][:80]}")
            print(f"  Phrase : '{span['text']}' (pos {span['start']}-{span['end']})")
            print()
            count += 1

    print("Now run: python train_span_extractor.py")


if __name__ == "__main__":
    main()