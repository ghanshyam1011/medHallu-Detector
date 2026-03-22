# ============================================================
# train_deberta.py
# ============================================================
# WHAT  : Fine-tunes microsoft/deberta-v3-base on MedHallu
#         with TWO output heads simultaneously:
#           Head 1 → hallucination label  (4 classes)
#           Head 2 → hallucination type   (5 classes)
# GPU   : NVIDIA RTX 4050/4060 Laptop GPU on Windows
#
# RESUME: If training is interrupted, run the same command
#         again — it automatically resumes from the last
#         saved checkpoint and continues from that epoch.
# ============================================================

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from collections import Counter


# --- Configuration -------------------------------------------

CONFIG = {
    "model_name":          "microsoft/deberta-v3-base",
    "max_length":          256,
    "batch_size":          8,
    "accumulation_steps":  4,
    "epochs":              5,
    "learning_rate":       5e-6,
    "warmup_ratio":        0.1,
    "dropout":             0.1,
    "label_loss_weight":   0.7,
    "type_loss_weight":    0.3,
    "bf16":                True,
    "train_rows":          7000,
    "val_rows_start":      7000,
    "val_rows_end":        8000,
    "checkpoint_dir":      "outputs/",
    "best_model_path":     "outputs/deberta_medhallu.pt",
    "resume_path":         "outputs/resume_checkpoint.pt",
    "results_path":        "outputs/results.json",
}

LABEL_MAP = {
    "grounded":               0,
    "hallucinated":           1,
    "partially_hallucinated": 2,
    "not_sure":               3,
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


# --- Data loading --------------------------------------------

def load_and_prepare_data():
    """
    Loads MedHallu pqa_artificial and builds a balanced dataset
    with both hallucinated AND grounded examples.

    WHY balanced:
    pqa_artificial only has hallucinated rows. Training on one
    class causes model collapse — it predicts class 0 for
    everything and F1 goes to 0. Adding ground truth answers
    as grounded examples gives the model both classes to learn.
    """
    print("Loading MedHallu (pqa_artificial)...")
    ds      = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
    full_df = ds["train"].to_pandas()

    print(f"Raw rows    : {len(full_df)}")
    print(f"Raw columns : {full_df.columns.tolist()}")

    full_df = full_df.rename(columns={
        "Question":                  "question",
        "Knowledge":                 "source",
        "Ground Truth":              "ground_truth",
        "Hallucinated Answer":       "answer",
        "Difficulty Level":          "difficulty",
        "Category of Hallucination": "hallucination_type",
    })

    full_df["difficulty"] = (
        full_df["difficulty"].astype(str).str.lower().str.strip()
    )
    full_df["hallucination_type"] = (
        full_df["hallucination_type"]
        .astype(str).fillna("none").str.lower().str.strip()
    )

    # Convert every value to plain Python string
    # WHY: Arrow-backed columns return Arrow scalars from to_dict()
    raw_records = full_df.to_dict(orient="records")
    cleaned     = [{k: safe_str(v) for k, v in rec.items()}
                   for rec in raw_records]

    train_slice = cleaned[:CONFIG["train_rows"]]
    val_slice   = cleaned[CONFIG["val_rows_start"]:CONFIG["val_rows_end"]]

    def build_balanced(rows):
        """
        For each row, creates two examples:
          1. hallucinated answer  → label = hallucinated
          2. ground truth answer  → label = grounded
        """
        balanced = []
        for row in rows:
            # Hallucinated example
            balanced.append({
                "question":           row["question"],
                "source":             row["source"],
                "answer":             row["answer"],
                "label":              "hallucinated",
                "hallucination_type": row["hallucination_type"],
                "difficulty":         row["difficulty"],
            })
            # Grounded example — use ground truth answer
            if row.get("ground_truth", ""):
                balanced.append({
                    "question":           row["question"],
                    "source":             row["source"],
                    "answer":             row["ground_truth"],
                    "label":              "grounded",
                    "hallucination_type": "none",
                    "difficulty":         row["difficulty"],
                })
        return balanced

    train_rows = build_balanced(train_slice)
    val_rows   = build_balanced(val_slice)

    print(f"Train rows : {len(train_rows)} | {Counter(r['label'] for r in train_rows)}")
    print(f"Val rows   : {len(val_rows)}   | {Counter(r['label'] for r in val_rows)}")

    return train_rows, val_rows


# --- Pre-tokenisation ----------------------------------------

def pretokenise(rows, tokenizer, max_length, desc="Tokenising"):
    """
    Tokenises all rows once before training starts.

    WHY pre-tokenise:
    Tokenising inside __getitem__ re-tokenises every row on
    every epoch. Pre-tokenising once saves hours over 5 epochs.

    WHY truncation=True not "only_first":
    "only_first" crashes when source is shorter than minimum
    required after truncation. truncation=True truncates
    whichever sequence is too long — safe for all input lengths.
    """
    features = []

    for row in tqdm(rows, desc=desc, unit="row", dynamic_ncols=True):

        source   = row.get("source",   "") or ""
        question = row.get("question", "") or ""
        answer   = row.get("answer",   "") or ""

        encoding = tokenizer(
            source,
            f"{question} {answer}",
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_str = (row.get("label", "hallucinated") or "hallucinated").lower().strip()
        type_str  = (row.get("hallucination_type", "none") or "none").lower().strip()

        label_id = LABEL_MAP.get(label_str, LABEL_MAP["hallucinated"])

        # Exact match first, then partial match fallback
        type_id = TYPE_MAP.get(type_str, None)
        if type_id is None:
            type_id = TYPE_MAP["none"]
            for key in TYPE_MAP:
                if key != "none" and (key in type_str or type_str in key):
                    type_id = TYPE_MAP[key]
                    break

        features.append({
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label_id, dtype=torch.long),
            "htype":          torch.tensor(type_id,  dtype=torch.long),
        })

    return features


# --- Dataset class -------------------------------------------

class MedHalluDataset(Dataset):
    """
    Wraps pre-tokenised features.
    __getitem__ is a simple list lookup — no tokenisation cost.
    """
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# --- Model class ---------------------------------------------

class MedHalluModel(nn.Module):

    def __init__(self, model_name, dropout):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,   # avoids CVE-2025-32434 vulnerability
        )
        # gradient_checkpointing disabled — causes nan with bf16
        # on DeBERTa-v3 + Windows + this transformers version
        hidden_size     = self.backbone.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.label_head = nn.Linear(hidden_size, len(LABEL_MAP))
        self.type_head  = nn.Linear(hidden_size, len(TYPE_MAP))

    def forward(self, input_ids, attention_mask):
        outputs    = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # WHY .float():
        # bf16 autocast makes backbone output bf16 tensors.
        # Linear heads are float32 by default — dtype mismatch.
        # .float() casts to float32 so heads receive correct dtype.
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        return self.label_head(cls_output), self.type_head(cls_output)


# --- Evaluation ----------------------------------------------

def evaluate(model, dataloader, device):
    """
    Runs model on validation set.
    Returns label_f1, type_f1, classification report string.
    """
    model.eval()
    all_label_preds, all_label_true = [], []
    all_type_preds,  all_type_true  = [], []

    with torch.no_grad():
        val_bar = tqdm(
            dataloader,
            desc="Validating",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )
        for batch in val_bar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            label_logits, type_logits = model(input_ids, attention_mask)

            all_label_preds.extend(label_logits.argmax(-1).cpu().tolist())
            all_label_true.extend(batch["label"].tolist())
            all_type_preds.extend(type_logits.argmax(-1).cpu().tolist())
            all_type_true.extend(batch["htype"].tolist())

    label_f1 = f1_score(
        all_label_true, all_label_preds,
        average="macro", zero_division=0,
        labels=list(LABEL_MAP.values()),
    )
    type_f1 = f1_score(
        all_type_true, all_type_preds,
        average="macro", zero_division=0,
        labels=list(TYPE_MAP.values()),
    )
    report = classification_report(
        all_label_true, all_label_preds,
        labels=list(LABEL_MAP.values()),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
    )
    return label_f1, type_f1, report


# --- Resume helpers ------------------------------------------

def save_resume_checkpoint(epoch, model, optimizer, scheduler,
                           best_label_f1, results_log, path):
    """
    Saves full training state after every epoch.
    Includes model weights, optimizer state, scheduler state,
    epoch number, best F1 so far, and results log.

    WHY save optimizer + scheduler:
    Optimizer has built-up momentum and adaptive LR state.
    Scheduler tracks how many steps have passed.
    Without these, resuming restarts with a cold optimizer
    and wrong LR — hurting convergence.
    """
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_label_f1":   best_label_f1,
        "results_log":     results_log,
        "label_map":       LABEL_MAP,
        "type_map":        TYPE_MAP,
        "config":          CONFIG,
    }, path)
    print(f"  Resume checkpoint saved → {path}")


def load_resume_checkpoint(path, model, optimizer, scheduler, device):
    """
    Loads resume checkpoint and restores all training state.

    Returns:
        start_epoch   : epoch to start from (completed epoch + 1)
        best_label_f1 : best F1 achieved so far
        results_log   : list of per-epoch result dicts
    """
    print(f"Loading resume checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch   = ckpt["epoch"] + 1
    best_label_f1 = ckpt["best_label_f1"]
    results_log   = ckpt["results_log"]

    print(f"  Last completed epoch : {ckpt['epoch']}")
    print(f"  Resuming from epoch  : {start_epoch} of {CONFIG['epochs']}")
    print(f"  Best F1 so far       : {best_label_f1:.4f}")

    return start_epoch, best_label_f1, results_log


# --- Training loop -------------------------------------------

def train():

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        print("Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    device = torch.device("cuda")
    print(f"Using GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Load and prepare data
    train_rows, val_rows = load_and_prepare_data()

    # Pre-tokenise once — progress bars show during this step
    print("\nPre-tokenising data (runs once, faster every epoch after)...")
    train_features = pretokenise(
        train_rows, tokenizer, CONFIG["max_length"],
        desc="Tokenising train"
    )
    val_features = pretokenise(
        val_rows, tokenizer, CONFIG["max_length"],
        desc="Tokenising val  "
    )
    print(f"Train features : {len(train_features)}")
    print(f"Val features   : {len(val_features)}")

    # Build datasets and loaders
    train_dataset = MedHalluDataset(train_features)
    val_dataset   = MedHalluDataset(val_features)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True,  num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # Build model, optimizer, scheduler
    print(f"\nLoading model: {CONFIG['model_name']}")
    model     = MedHalluModel(CONFIG["model_name"], CONFIG["dropout"])
    model     = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01,
    )

    total_steps  = (len(train_loader) // CONFIG["accumulation_steps"]) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion     = nn.CrossEntropyLoss()
    best_label_f1 = 0.0
    results_log   = []
    start_epoch   = 1

    # --------------------------------------------------------
    # Resume logic — runs AFTER model/optimizer/scheduler are
    # built so we can load their states directly into them
    # --------------------------------------------------------
    if os.path.exists(CONFIG["resume_path"]):
        print("\nResume checkpoint detected.")
        print("  Press Enter → resume from last checkpoint")
        print("  Type '2'    → delete checkpoint and start fresh")
        choice = input("Your choice: ").strip().lower()

        if choice in ("2", "fresh", "f", "no", "n"):
            print("Starting fresh — deleting old checkpoints...")
            for p in [CONFIG["resume_path"],
                      CONFIG["best_model_path"],
                      CONFIG["results_path"]]:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  Deleted: {p}")
            print("Ready to train from epoch 1.")

        else:
            try:
                start_epoch, best_label_f1, results_log = load_resume_checkpoint(
                    CONFIG["resume_path"],
                    model, optimizer, scheduler,
                    device,
                )
                if start_epoch > CONFIG["epochs"]:
                    print(f"\nAll {CONFIG['epochs']} epochs already completed.")
                    print("Type '2' when prompted next run to retrain.")
                    return
            except (EOFError, RuntimeError, Exception) as e:
                print(f"\nWARNING: Resume checkpoint corrupted ({e})")
                print("Deleting corrupted checkpoint and starting fresh...")
                for p in [CONFIG["resume_path"],
                          CONFIG["best_model_path"],
                          CONFIG["results_path"]]:
                    if os.path.exists(p):
                        os.remove(p)
                print("Starting fresh from epoch 1.")
    else:
        print("\nNo resume checkpoint found — starting fresh training.")

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------
    print(f"\nStarting training from epoch {start_epoch}")
    print(f"  Total epochs      : {CONFIG['epochs']}")
    print(f"  Batch size        : {CONFIG['batch_size']}")
    print(f"  Accumulation steps: {CONFIG['accumulation_steps']}")
    print(f"  Effective batch   : {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")
    print(f"  Learning rate     : {CONFIG['learning_rate']}")
    print(f"  Mixed precision   : bf16={CONFIG['bf16']}")
    print(f"  Total steps       : {total_steps}")
    print(f"  Warmup steps      : {warmup_steps}")
    print()

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):

        model.train()
        epoch_start  = time.time()
        running_loss = 0.0
        nan_steps    = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{CONFIG['epochs']}",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in progress_bar:

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            htypes         = batch["htype"].to(device)

            # Forward pass in bf16
            with autocast(device_type="cuda",
                          dtype=torch.bfloat16,
                          enabled=CONFIG["bf16"]):
                label_logits, type_logits = model(input_ids, attention_mask)

            # Loss in float32 — outside autocast for stability
            label_loss = criterion(label_logits, labels)
            type_loss  = criterion(type_logits,  htypes)
            loss = (
                CONFIG["label_loss_weight"] * label_loss +
                CONFIG["type_loss_weight"]  * type_loss
            ) / CONFIG["accumulation_steps"]

            # Skip nan/inf — never corrupt weights
            if torch.isnan(loss) or torch.isinf(loss):
                nan_steps += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            running_loss += loss.item() * CONFIG["accumulation_steps"]

            if (step + 1) % CONFIG["accumulation_steps"] == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    print(f"\n  WARNING: non-finite grad norm at step {step+1} — skipping")
                scheduler.step()
                optimizer.zero_grad()

            # Live progress bar stats
            avg_loss = running_loss / max(step + 1 - nan_steps, 1)
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "nan":  nan_steps,
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Flush any remaining accumulated gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_time     = time.time() - epoch_start
        valid_steps    = max(len(train_loader) - nan_steps, 1)
        avg_train_loss = running_loss / valid_steps

        print(f"\nEpoch {epoch} done in {epoch_time/60:.1f} min")
        print(f"  Avg train loss : {avg_train_loss:.4f}")
        print(f"  Nan steps      : {nan_steps}/{len(train_loader)}")
        print(f"  Running validation...")

        label_f1, type_f1, report = evaluate(model, val_loader, device)

        print(f"  Label F1 (macro): {label_f1:.4f}  ← main metric")
        print(f"  Type  F1 (macro): {type_f1:.4f}")
        print(f"\n{report}")

        results_log.append({
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 4),
            "label_f1":       round(label_f1, 4),
            "type_f1":        round(type_f1, 4),
            "epoch_time_min": round(epoch_time / 60, 1),
            "nan_steps":      nan_steps,
        })

        # Save best model whenever F1 improves
        if label_f1 > best_label_f1:
            best_label_f1 = label_f1
            torch.save({
                "epoch":       epoch,
                "label_f1":    label_f1,
                "type_f1":     type_f1,
                "model_state": model.state_dict(),
                "label_map":   LABEL_MAP,
                "type_map":    TYPE_MAP,
                "config":      CONFIG,
            }, CONFIG["best_model_path"])
            print(f"  Saved new best model → {CONFIG['best_model_path']}")

        # Save resume checkpoint every epoch — enables resuming
        # from the exact epoch where training was interrupted
        save_resume_checkpoint(
            epoch, model, optimizer, scheduler,
            best_label_f1, results_log,
            CONFIG["resume_path"],
        )

        # Save results JSON after every epoch
        with open(CONFIG["results_path"], "w") as f:
            json.dump({
                "config":        CONFIG,
                "epochs":        results_log,
                "best_label_f1": best_label_f1,
                "label_map":     LABEL_MAP,
                "type_map":      TYPE_MAP,
            }, f, indent=2)

        torch.cuda.empty_cache()
        print()

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best label F1 : {best_label_f1:.4f}")
    print(f"Best model    : {CONFIG['best_model_path']}")
    print(f"Results log   : {CONFIG['results_path']}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to see the comparison table")
    print("  2. Hand outputs/deberta_medhallu.pt to Person 3")
    print("  3. Hand outputs/results.json to Person 3")


if __name__ == "__main__":
    train()