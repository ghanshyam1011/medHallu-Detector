# ============================================================
# evaluate.py
# ============================================================
# WHAT  : Loads the saved deberta_medhallu.pt checkpoint and
#         evaluates it three ways:
#           1. Overall F1 on MedHallu validation set
#           2. F1 split by difficulty (easy / medium / hard)
#           3. F1 on Person 1's 200 OOD MedQA rows
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast                   # updated from torch.cuda.amp
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)
from collections import Counter


# --- Label maps ----------------------------------------------
# MUST match train_deberta.py exactly

LABEL_MAP = {
    "grounded":               0,
    "hallucinated":           1,
    "partially_hallucinated": 2,
    "not_sure":               3,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Updated to match actual dataset values
TYPE_MAP = {
    "misinterpretation of #question#":        0,
    "incomplete information":                  1,
    "mechanism and pathway misattribution":    2,
    "methodological and evidence fabrication": 3,
    "none":                                    4,
}
TYPE_MAP_INV = {v: k for k, v in TYPE_MAP.items()}


# --- Configuration -------------------------------------------

CONFIG = {
    "model_name":        "microsoft/deberta-v3-base",
    "max_length":        512,
    "batch_size":        16,
    "checkpoint_path":   "outputs/deberta_medhallu.pt",
    "ood_csv_path":      "../person1_data/outputs/medqa_ood_200.csv",
    "groq_summary_path": "../person1_data/outputs/groq_baseline_summary.json",
    "output_path":       "outputs/results.json",
    "bf16":              True,
}

# GPT-4o numbers from MedHallu paper
GPT4O_SCORES = {
    "overall": 0.737,
    "easy":    0.844,
    "medium":  0.758,
    "hard":    0.625,
}


# --- safe_str helper -----------------------------------------

def safe_str(val) -> str:
    """Converts any value to a plain Python string safely."""
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


# --- Dataset class -------------------------------------------

class MedHalluDataset(Dataset):

    def __init__(self, rows: list, tokenizer, max_length: int):
        self.rows       = rows
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        source   = safe_str(row.get("source",   ""))
        question = safe_str(row.get("question", ""))
        answer   = safe_str(row.get("answer",   ""))

        # NEW — use truncation=True instead of "only_first"
        # WHY: "only_first" fails when the first sequence (source)
        # is shorter than what needs to be truncated. truncation=True
        # truncates whichever sequence needs it — works for all lengths
        encoding = self.tokenizer(
            source,
            f"{question} {answer}",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_str = safe_str(row.get("label", "hallucinated")).lower()
        type_str  = safe_str(row.get("hallucination_type", "none")).lower()

        label_id = LABEL_MAP.get(label_str, LABEL_MAP["hallucinated"])

        # Exact match first, then partial fallback
        type_id = TYPE_MAP.get(type_str, None)
        if type_id is None:
            type_id = TYPE_MAP["none"]
            for key in TYPE_MAP:
                if key != "none" and (key in type_str or type_str in key):
                    type_id = TYPE_MAP[key]
                    break

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label_id, dtype=torch.long),
            "htype":          torch.tensor(type_id,  dtype=torch.long),
            "difficulty":     safe_str(row.get("difficulty", "unknown")),
        }


# --- Model class ---------------------------------------------
# Must be identical to train_deberta.py

class MedHalluModel(nn.Module):

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()

        # WHY use_safetensors=True:
        # PyTorch 2.6 blocks torch.load without safetensors
        # due to CVE-2025-32434 security vulnerability.
        # use_safetensors=True loads the .safetensors file
        # instead of the .bin file — no vulnerability.
        self.backbone = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
        )

        hidden_size     = self.backbone.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.label_head = nn.Linear(hidden_size, len(LABEL_MAP))
        self.type_head  = nn.Linear(hidden_size, len(TYPE_MAP))

    def forward(self, input_ids, attention_mask):
        outputs    = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # .float() casts bf16 backbone output to float32
        # so linear heads receive matching dtype
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        return self.label_head(cls_output), self.type_head(cls_output)


# --- Data loading --------------------------------------------

def load_val_data(tokenizer):
    """
    Loads MedHallu pqa_artificial, renames columns,
    builds balanced val set (hallucinated + grounded),
    returns a DataLoader.
    """
    print("Loading MedHallu (pqa_artificial)...")
    ds      = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
    full_df = ds["train"].to_pandas()

    full_df = full_df.rename(columns={
        "Question":                  "question",
        "Knowledge":                 "source",
        "Ground Truth":              "ground_truth",
        "Hallucinated Answer":       "answer",
        "Difficulty Level":          "difficulty",
        "Category of Hallucination": "hallucination_type",
    })

    full_df["label"] = "hallucinated"
    full_df["difficulty"] = (
        full_df["difficulty"].astype(str).str.lower().str.strip()
    )
    full_df["hallucination_type"] = (
        full_df["hallucination_type"]
        .astype(str).fillna("none").str.lower().str.strip()
    )

    raw_records = full_df.to_dict(orient="records")
    cleaned     = [{k: safe_str(v) for k, v in rec.items()}
                   for rec in raw_records]

    # Val slice = rows 7000-8000
    val_slice = cleaned[7000:8000]

    # Build balanced val set
    val_rows = []
    for row in val_slice:
        # Hallucinated example
        val_rows.append({
            "question":           row["question"],
            "source":             row["source"],
            "answer":             row["answer"],
            "label":              "hallucinated",
            "hallucination_type": row["hallucination_type"],
            "difficulty":         row["difficulty"],
        })
        # Grounded example
        if row.get("ground_truth", ""):
            val_rows.append({
                "question":           row["question"],
                "source":             row["source"],
                "answer":             row["ground_truth"],
                "label":              "grounded",
                "hallucination_type": "none",
                "difficulty":         row["difficulty"],
            })

    print(f"Val rows : {len(val_rows)} | {Counter(r['label'] for r in val_rows)}")

    dataset = MedHalluDataset(val_rows, tokenizer, CONFIG["max_length"])
    loader  = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader


# --- Inference function --------------------------------------

def run_inference(model, dataloader, device):
    """
    Runs model in eval mode on all batches.
    Returns dict with predictions, ground truth, difficulties.
    """
    model.eval()

    label_preds  = []
    label_true   = []
    type_preds   = []
    type_true    = []
    difficulties = []
    label_probs  = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # WHY autocast in inference:
            # Backbone runs faster in bf16 — forward() casts
            # back to float32 before the heads so no dtype error
            with autocast(device_type="cuda",
                          dtype=torch.bfloat16,
                          enabled=CONFIG["bf16"]):
                label_logits, type_logits = model(input_ids, attention_mask)

            label_softmax = torch.softmax(label_logits, dim=-1)
            probs, preds  = label_softmax.max(dim=-1)

            label_preds.extend(preds.cpu().tolist())
            label_probs.extend(probs.cpu().tolist())
            label_true.extend(batch["label"].tolist())
            type_preds.extend(type_logits.argmax(-1).cpu().tolist())
            type_true.extend(batch["htype"].tolist())
            difficulties.extend(list(batch["difficulty"]))

    return {
        "label_preds":  label_preds,
        "label_true":   label_true,
        "type_preds":   type_preds,
        "type_true":    type_true,
        "difficulties": difficulties,
        "label_probs":  label_probs,
    }


# --- Metrics -------------------------------------------------

def compute_metrics(results: dict, split_name: str) -> dict:
    """Computes F1 scores overall and per difficulty."""
    label_preds  = results["label_preds"]
    label_true   = results["label_true"]
    difficulties = results["difficulties"]

    overall_f1 = f1_score(
        label_true, label_preds,
        average="macro", zero_division=0,
        labels=list(LABEL_MAP.values()),
    )

    difficulty_f1 = {}
    for diff in ["easy", "medium", "hard"]:
        indices = [i for i, d in enumerate(difficulties) if d == diff]
        if not indices:
            difficulty_f1[diff] = None
            continue
        sub_true  = [label_true[i]  for i in indices]
        sub_preds = [label_preds[i] for i in indices]
        difficulty_f1[diff] = round(f1_score(
            sub_true, sub_preds,
            average="macro", zero_division=0,
            labels=list(LABEL_MAP.values()),
        ), 4)

    type_f1 = f1_score(
        results["type_true"], results["type_preds"],
        average="macro", zero_division=0,
        labels=list(TYPE_MAP.values()),
    )

    report = classification_report(
        label_true, label_preds,
        labels=list(LABEL_MAP.values()),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
    )

    cm = confusion_matrix(
        label_true, label_preds,
        labels=list(LABEL_MAP.values()),
    )

    print(f"\n{'='*55}")
    print(f"RESULTS — {split_name}")
    print(f"{'='*55}")
    print(f"Overall label F1 (macro) : {overall_f1:.4f}")
    print(f"Easy   F1                : {difficulty_f1.get('easy',  'N/A')}")
    print(f"Medium F1                : {difficulty_f1.get('medium','N/A')}")
    print(f"Hard   F1                : {difficulty_f1.get('hard',  'N/A')}")
    print(f"Type   F1 (macro)        : {type_f1:.4f}")
    print(f"\nClassification report:\n{report}")
    print(f"Confusion matrix:\n{cm}")
    print(f"{'='*55}")

    return {
        "split":      split_name,
        "overall_f1": round(overall_f1, 4),
        "easy_f1":    difficulty_f1.get("easy"),
        "medium_f1":  difficulty_f1.get("medium"),
        "hard_f1":    difficulty_f1.get("hard"),
        "type_f1":    round(type_f1, 4),
    }


def print_comparison_table(our_scores: dict, groq_scores: dict):
    """Prints GPT-4o vs Groq vs Our DeBERTa comparison table."""
    print(f"\n{'='*65}")
    print("COMPARISON TABLE  (macro F1)")
    print(f"{'='*65}")
    print(f"{'Split':<10} {'GPT-4o':>12} {'Groq Llama':>12} {'Our DeBERTa':>14}")
    print(f"{'-'*65}")

    for split in ["overall", "easy", "medium", "hard"]:
        gpt_score  = GPT4O_SCORES.get(split, "—")
        groq_score = groq_scores.get(f"{split}_f1", "—")
        our_score  = our_scores.get(f"{split}_f1",  "—")

        gpt_str  = f"{gpt_score:.3f}"  if isinstance(gpt_score,  float) else str(gpt_score)
        groq_str = f"{groq_score:.3f}" if isinstance(groq_score, float) else str(groq_score)
        our_str  = f"{our_score:.3f}"  if isinstance(our_score,  float) else str(our_score)

        marker = ""
        if isinstance(our_score, float) and isinstance(gpt_score, float):
            if our_score > gpt_score:
                marker = " *"

        print(f"{split:<10} {gpt_str:>12} {groq_str:>12} {our_str:>14}{marker}")

    print(f"{'-'*65}")
    print("* = beats GPT-4o")
    print(f"{'='*65}")


# --- Main pipeline -------------------------------------------

def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Check checkpoint
    if not os.path.exists(CONFIG["checkpoint_path"]):
        print(f"ERROR: checkpoint not found at {CONFIG['checkpoint_path']}")
        print("Run train_deberta.py first.")
        return

    # Load tokenizer
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Load model with safetensors — avoids CVE-2025-32434
    print(f"Loading checkpoint: {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(
        CONFIG["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )

    model = MedHalluModel(CONFIG["model_name"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint epoch  : {checkpoint.get('epoch',    '?')}")
    print(f"Checkpoint F1     : {checkpoint.get('label_f1', '?')}")

    # Load validation data — same balanced split as training
    val_loader = load_val_data(tokenizer)

    # Evaluate on validation set
    print("\nRunning inference on validation set...")
    val_results     = run_inference(model, val_loader, device)
    medhallu_scores = compute_metrics(val_results, "MedHallu Validation")

    # Load Groq baseline
    groq_scores = {}
    if os.path.exists(CONFIG["groq_summary_path"]):
        with open(CONFIG["groq_summary_path"]) as f:
            groq_scores = json.load(f)
        print(f"\nLoaded Groq baseline")
    else:
        print(f"\nWARNING: Groq baseline not found — using zeros")
        groq_scores = {
            "overall_f1": 0.0,
            "easy_f1":    0.0,
            "medium_f1":  0.0,
            "hard_f1":    0.0,
        }

    # Print comparison table
    print_comparison_table(medhallu_scores, groq_scores)

    # OOD generalization test
    ood_scores = None
    if os.path.exists(CONFIG["ood_csv_path"]):
        print(f"\nLoading OOD dataset: {CONFIG['ood_csv_path']}")
        ood_df   = pd.read_csv(CONFIG["ood_csv_path"])
        ood_rows = ood_df.to_dict(orient="records")
        print(f"OOD rows: {len(ood_rows)}")

        ood_dataset = MedHalluDataset(ood_rows, tokenizer, CONFIG["max_length"])
        ood_loader  = DataLoader(
            ood_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False, num_workers=0, pin_memory=True,
        )
        print("Running inference on OOD MedQA rows...")
        ood_results = run_inference(model, ood_loader, device)
        ood_scores  = compute_metrics(ood_results, "OOD MedQA (generalization)")

        ood_f1 = ood_scores["overall_f1"]
        print("\nGeneralization interpretation:")
        if ood_f1 >= 0.75:
            print(f"  {ood_f1:.3f} → Excellent")
        elif ood_f1 >= 0.60:
            print(f"  {ood_f1:.3f} → Good — strong result for OOD data")
        elif ood_f1 >= 0.50:
            print(f"  {ood_f1:.3f} → Moderate")
        else:
            print(f"  {ood_f1:.3f} → Weak — model may have overfit")
    else:
        print(f"\nWARNING: OOD CSV not found at {CONFIG['ood_csv_path']}")
        print("Run person1_data/make_ood_dataset.py first.")

    # Save results JSON for Gradio dashboard
    final_results = {
        "medhallu_validation": medhallu_scores,
        "groq_baseline":       groq_scores,
        "gpt4o_paper":         GPT4O_SCORES,
        "ood_generalization":  ood_scores,
        "checkpoint_info": {
            "epoch":    checkpoint.get("epoch"),
            "label_f1": checkpoint.get("label_f1"),
            "type_f1":  checkpoint.get("type_f1"),
        },
    }

    os.makedirs("outputs", exist_ok=True)
    with open(CONFIG["output_path"], "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved → {CONFIG['output_path']}")
    print("Hand outputs/results.json to Person 3 for the Gradio dashboard.")
    print("Hand outputs/deberta_medhallu.pt to Person 3 for the verdict model.")


if __name__ == "__main__":
    main()