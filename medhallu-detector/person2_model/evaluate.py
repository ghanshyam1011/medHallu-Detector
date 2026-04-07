# ============================================================
# evaluate.py
# ============================================================
# WHAT  : Loads the saved deberta_medhallu.pt checkpoint and
#         evaluates it two ways:
#           1. F1 on MedHallu validation set (rows 7000-8000)
#           2. F1 on MedHallu official TEST split (Easy/Med/Hard)
#
# WHY removed HaluEval + MedQA OOD:
#   - HaluEval: different domain, label distribution mismatch
#   - MedQA 200-row: had 0 grounded samples, metrics meaningless
#   - MedHallu test split: correct dataset, gives valid Easy/Med/Hard
#     F1 directly comparable to GPT-4o numbers from the paper
# ============================================================

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter


# --- Label maps (MUST match train_deberta.py exactly) --------

LABEL_MAP = {
    "grounded":     0,
    "hallucinated": 1,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

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
    "groq_summary_path": "../person1_data/outputs/groq_baseline_summary.json",
    "output_path":       "outputs/results.json",
    "bf16":              True,
    "val_rows_start":    7000,   # same slice used during training
    "val_rows_end":      8000,
}

# GPT-4o numbers from the MedHallu paper
GPT4O_SCORES = {
    "overall": 0.737,
    "easy":    0.844,
    "medium":  0.758,
    "hard":    0.625,
}


# --- safe_str helper -----------------------------------------

def safe_str(val) -> str:
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


# --- Dataset -------------------------------------------------

class MedHalluDataset(Dataset):

    def __init__(self, rows, tokenizer, max_length):
        self.rows      = rows
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row      = self.rows[idx]
        source   = safe_str(row.get("source",   ""))
        question = safe_str(row.get("question", ""))
        answer   = safe_str(row.get("answer",   ""))

        encoding = self.tokenizer(
            source,
            f"{question} {answer}",
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_str = safe_str(row.get("label", "hallucinated")).lower()
        type_str  = safe_str(row.get("hallucination_type", "none")).lower()

        label_id = LABEL_MAP.get(label_str, LABEL_MAP["hallucinated"])

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


# --- Model (identical to train_deberta.py) -------------------

class MedHalluModel(nn.Module):

    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=False)
        hidden_size   = self.backbone.config.hidden_size
        self.dropout  = nn.Dropout(dropout)
        self.label_head = nn.Linear(hidden_size, len(LABEL_MAP))
        self.type_head  = nn.Linear(hidden_size, len(TYPE_MAP))

    def forward(self, input_ids, attention_mask):
        out        = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls        = out.last_hidden_state[:, 0, :].float()
        cls        = self.dropout(cls)
        return self.label_head(cls), self.type_head(cls)


# --- Build balanced rows from raw MedHallu records -----------

def build_balanced_rows(raw_rows):
    """
    Each raw MedHallu row has both a hallucinated answer and a
    ground truth answer. We expand each into two rows so the
    dataset is perfectly balanced (50% grounded, 50% hallucinated).
    """
    rows = []
    for r in raw_rows:
        # hallucinated row
        rows.append({
            "source":            safe_str(r.get("source", r.get("Knowledge", ""))),
            "question":          safe_str(r.get("question", r.get("Question", ""))),
            "answer":            safe_str(r.get("answer", r.get("Hallucinated Answer", ""))),
            "label":             "hallucinated",
            "hallucination_type": safe_str(r.get("hallucination_type",
                                                  r.get("Category of Hallucination", "none"))).lower(),
            "difficulty":        safe_str(r.get("difficulty",
                                                 r.get("Difficulty Level", "unknown"))).lower(),
        })
        # grounded row
        ground_truth = safe_str(r.get("ground_truth", r.get("Ground Truth", "")))
        if ground_truth:
            rows.append({
                "source":            safe_str(r.get("source", r.get("Knowledge", ""))),
                "question":          safe_str(r.get("question", r.get("Question", ""))),
                "answer":            ground_truth,
                "label":             "grounded",
                "hallucination_type": "none",
                "difficulty":        safe_str(r.get("difficulty",
                                                     r.get("Difficulty Level", "unknown"))).lower(),
            })
    return rows


# --- Data loading --------------------------------------------

def load_medhallu_validation(tokenizer):
    """
    Loads rows 7000-8000 from pqa_artificial — same slice used in training.
    """
    print("Loading MedHallu validation set (pqa_artificial rows 7000-8000)...")
    ds      = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
    full_df = ds["train"].to_pandas()
    
    # Rename columns to match expected format
    full_df = full_df.rename(columns={
        "Question":                  "question",
        "Knowledge":                 "source",
        "Ground Truth":              "ground_truth",
        "Hallucinated Answer":       "answer",
        "Difficulty Level":          "difficulty",
        "Category of Hallucination": "hallucination_type",
    }, errors="ignore")
    
    # Convert to plain Python strings to avoid Arrow scalars
    raw_records = full_df.to_dict(orient="records")
    cleaned     = [{k: safe_str(v) for k, v in rec.items()}
                   for rec in raw_records]
    
    print(f"Total cleaned rows : {len(cleaned)}")

    val_slice = cleaned[CONFIG["val_rows_start"]:CONFIG["val_rows_end"]]
    val_rows  = build_balanced_rows(val_slice)
    print(f"Val rows after expansion : {len(val_rows)} | {Counter(r['label'] for r in val_rows)}")

    dataset = MedHalluDataset(val_rows, tokenizer, CONFIG["max_length"])
    return DataLoader(dataset, batch_size=CONFIG["batch_size"],
                      shuffle=False, num_workers=0, pin_memory=True)


def load_medhallu_test(tokenizer):
    """
    Loads the official MedHallu TEST split.
    This gives Easy / Medium / Hard F1 that are directly comparable
    to the GPT-4o numbers reported in the paper.
    """
    print("\nLoading MedHallu official TEST split...")
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")

    # Convert to pandas first to avoid Arrow scalar issues
    if "test" in ds:
        test_df = ds["test"].to_pandas()
        raw_records = test_df.to_dict(orient="records")
        raw = [{k: safe_str(v) for k, v in rec.items()} for rec in raw_records]
        print(f"Test split rows (raw) : {len(raw)}")
    else:
        # Fallback: use rows 8000+ from train if no test split exposed
        print("WARNING: No 'test' split found — using rows 8000-10000 as proxy test set")
        full_df = ds["train"].to_pandas()
        
        # Rename columns to match expected format
        full_df = full_df.rename(columns={
            "Question":                  "question",
            "Knowledge":                 "source",
            "Ground Truth":              "ground_truth",
            "Hallucinated Answer":       "answer",
            "Difficulty Level":          "difficulty",
            "Category of Hallucination": "hallucination_type",
        }, errors="ignore")
        
        raw_records = full_df.to_dict(orient="records")
        cleaned = [{k: safe_str(v) for k, v in rec.items()} for rec in raw_records]
        raw = cleaned[8000:]
        print(f"Proxy test rows (raw) : {len(raw)}")

    test_rows = build_balanced_rows(raw)
    print(f"Test rows after expansion : {len(test_rows)} | {Counter(r['label'] for r in test_rows)}")

    # Verify difficulty distribution
    diff_counts = Counter(r["difficulty"] for r in test_rows)
    print(f"Difficulty counts : {dict(diff_counts)}")

    dataset = MedHalluDataset(test_rows, tokenizer, CONFIG["max_length"])
    return DataLoader(dataset, batch_size=CONFIG["batch_size"],
                      shuffle=False, num_workers=0, pin_memory=True)


# --- Inference -----------------------------------------------

def run_inference(model, dataloader, device):
    model.eval()
    label_preds, label_true = [], []
    type_preds,  type_true  = [], []
    difficulties             = []
    label_probs              = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(device_type=device.type,
                          dtype=torch.bfloat16,
                          enabled=CONFIG["bf16"] and device.type == "cuda"):
                label_logits, type_logits = model(input_ids, attention_mask)

            softmax       = torch.softmax(label_logits, dim=-1)
            probs, preds  = softmax.max(dim=-1)

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

def compute_metrics(results, split_name):
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

    cm = confusion_matrix(label_true, label_preds, labels=list(LABEL_MAP.values()))

    print(f"\n{'='*55}")
    print(f"RESULTS — {split_name}")
    print(f"{'='*55}")
    print(f"Overall label F1 (macro) : {overall_f1:.4f}")
    print(f"Easy   F1                : {difficulty_f1.get('easy')}")
    print(f"Medium F1                : {difficulty_f1.get('medium')}")
    print(f"Hard   F1                : {difficulty_f1.get('hard')}")
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


# --- Comparison table ----------------------------------------

def print_comparison_table(val_scores, test_scores, groq_scores):
    print(f"\n{'='*70}")
    print("COMPARISON TABLE  (macro F1) — MedHallu Test Split")
    print(f"{'='*70}")
    print(f"{'Split':<10} {'GPT-4o (paper)':>16} {'Groq Llama':>12} {'Our DeBERTa':>14}")
    print(f"{'-'*70}")

    for split in ["overall", "easy", "medium", "hard"]:
        gpt_val  = GPT4O_SCORES.get(split)
        groq_val = groq_scores.get(f"{split}_f1")
        our_val  = test_scores.get(f"{split}_f1")

        gpt_str  = f"{gpt_val:.3f}"  if isinstance(gpt_val,  float) else "—"
        groq_str = f"{groq_val:.3f}" if isinstance(groq_val, float) else "—"
        our_str  = f"{our_val:.3f}"  if isinstance(our_val,  float) else "—"

        marker = ""
        if isinstance(our_val, float) and isinstance(gpt_val, float):
            if our_val > gpt_val:
                marker = "  ✓ beats GPT-4o"

        print(f"{split:<10} {gpt_str:>16} {groq_str:>12} {our_str:>14}{marker}")

    print(f"{'-'*70}")
    print(f"{'='*70}")


# --- Main ----------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(CONFIG["checkpoint_path"]):
        print(f"ERROR: checkpoint not found at {CONFIG['checkpoint_path']}")
        print("Run train_deberta.py first.")
        return

    print(f"\nLoading tokenizer : {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    print(f"Loading checkpoint: {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)

    model = MedHalluModel(CONFIG["model_name"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint epoch : {checkpoint.get('epoch',    '?')}")
    print(f"Checkpoint F1    : {checkpoint.get('label_f1', '?')}")

    # --- Evaluation 1: Validation set (same as training) -----
    val_loader    = load_medhallu_validation(tokenizer)
    print("\nRunning inference on validation set...")
    val_results   = run_inference(model, val_loader, device)
    val_scores    = compute_metrics(val_results, "MedHallu Validation")

    # --- Evaluation 2: Official test split -------------------
    test_loader   = load_medhallu_test(tokenizer)
    print("\nRunning inference on test set...")
    test_results  = run_inference(model, test_loader, device)
    test_scores   = compute_metrics(test_results, "MedHallu Test (Easy/Med/Hard)")

    # --- Load Groq baseline ----------------------------------
    groq_scores = {}
    if os.path.exists(CONFIG["groq_summary_path"]):
        with open(CONFIG["groq_summary_path"]) as f:
            groq_scores = json.load(f)
        print("\nLoaded Groq baseline")
    else:
        print(f"\nWARNING: Groq baseline not found at {CONFIG['groq_summary_path']}")
        groq_scores = {"overall_f1": None, "easy_f1": None,
                       "medium_f1": None, "hard_f1": None}

    # --- Print comparison ------------------------------------
    print_comparison_table(val_scores, test_scores, groq_scores)

    # --- Save results.json for Gradio dashboard --------------
    final_results = {
        "medhallu_validation": val_scores,
        "medhallu_test":       test_scores,   # used in dashboard table
        "groq_baseline":       groq_scores,
        "gpt4o_paper":         GPT4O_SCORES,
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
