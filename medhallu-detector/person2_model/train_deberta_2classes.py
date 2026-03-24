# ============================================================
# train_deberta.py  —  IMPROVED VERSION
# ============================================================
# WHAT  : Fine-tunes microsoft/deberta-v3-base on MedHallu
#         with TWO output heads simultaneously:
#           Head 1 → hallucination label  (2 classes: grounded / hallucinated)
#           Head 2 → hallucination type   (5 classes)
# GPU   : NVIDIA RTX 5050 sm120 — use cu128 PyTorch
#         pip install torch --index-url https://download.pytorch.org/whl/cu128
#
# NEW   : Early stopping, ReduceLROnPlateau, dynamic max epochs,
#         type loss masking on grounded rows, 3 bug fixes.
#
# RESUME: Run the same command again — resumes automatically.
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

    # --- Epoch control ---
    # Training will run up to max_epochs but stop early if F1
    # does not improve for `early_stop_patience` epochs.
    # Raise max_epochs freely — early stopping prevents waste.
    "max_epochs":          15,       # was hardcoded 5; now a ceiling

    # --- Early stopping ---
    # Stop training if label F1 doesn't improve for this many epochs.
    # patience=3 means: if epochs 4,5,6 all fail to beat epoch 3's F1,
    # training halts and the best model (epoch 3) is kept.
    "early_stop_patience": 3,

    # --- LR on plateau ---
    # If F1 doesn't improve for `plateau_patience` epochs,
    # multiply the learning rate by `plateau_factor`.
    # e.g. 5e-6 → 2.5e-6 after 2 stagnant epochs.
    # This lets training "unstick" without stopping entirely.
    "plateau_patience":    2,
    "plateau_factor":      0.5,
    "plateau_min_lr":      1e-7,     # never decay below this

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

# Only 2 classes exist in pqa_artificial — grounded and hallucinated.
# partially_hallucinated and not_sure have zero examples in this dataset.
# Keeping them meant: 2 dead output neurons with no training signal,
# and macro F1 permanently stuck at ~0.45 masking real performance.
# Removed so label head is Linear(768→2) and macro F1 is meaningful.
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

# Class id for "grounded" — used to mask type loss
GROUNDED_ID = LABEL_MAP["grounded"]


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

    raw_records = full_df.to_dict(orient="records")
    cleaned     = [{k: safe_str(v) for k, v in rec.items()}
                   for rec in raw_records]

    train_slice = cleaned[:CONFIG["train_rows"]]
    val_slice   = cleaned[CONFIG["val_rows_start"]:CONFIG["val_rows_end"]]

    def build_balanced(rows):
        balanced = []
        for row in rows:
            balanced.append({
                "question":           row["question"],
                "source":             row["source"],
                "answer":             row["answer"],
                "label":              "hallucinated",
                "hallucination_type": row["hallucination_type"],
                "difficulty":         row["difficulty"],
            })
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
    Also stores is_grounded flag per row — used to mask type loss.
    """
    features      = []
    unknown_types = 0

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

        # FIX: strict lookup only — no partial match fallback.
        # Unknown types go to "none" with a warning counter.
        # Partial substring matching was silently mislabelling rows.
        type_id = TYPE_MAP.get(type_str, None)
        if type_id is None:
            type_id = TYPE_MAP["none"]
            unknown_types += 1

        features.append({
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label_id,              dtype=torch.long),
            "htype":          torch.tensor(type_id,               dtype=torch.long),
            "is_grounded":    torch.tensor(label_id == GROUNDED_ID, dtype=torch.bool),
        })

    if unknown_types > 0:
        print(f"  WARNING: {unknown_types} rows had unknown hallucination_type → mapped to 'none'")

    return features


# --- Dataset class -------------------------------------------

class MedHalluDataset(Dataset):
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
        # .float() — casts bf16 backbone output to fp32 for heads
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        return self.label_head(cls_output), self.type_head(cls_output)


# --- Masked type loss ----------------------------------------

def compute_type_loss(criterion, type_logits, htypes, is_grounded):
    """
    FIX: Only compute type loss on non-grounded rows.

    WHY: Grounded rows always have hallucination_type='none'.
    Training the type head on grounded rows teaches it a shortcut:
    predict 'none' whenever the answer looks clean. At inference
    you won't know the label ahead of time, so this shortcut
    produces wrong type predictions for grounded-looking hallucinations.

    Solution: mask grounded rows out of the type loss entirely.
    The type head only learns from rows that actually have a type.
    """
    non_grounded_mask = ~is_grounded
    if non_grounded_mask.sum() == 0:
        # Entire batch is grounded — skip type loss
        return torch.tensor(0.0, device=type_logits.device, requires_grad=True)
    return criterion(
        type_logits[non_grounded_mask],
        htypes[non_grounded_mask],
    )


# --- Evaluation ----------------------------------------------

def evaluate(model, dataloader, device):
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


# --- Early stopping helper -----------------------------------

class EarlyStopping:
    """
    Stops training when label F1 hasn't improved for `patience` epochs.

    HOW IT WORKS:
    - Every epoch, call .step(label_f1)
    - If label_f1 > best seen so far → reset counter, return False
    - If label_f1 <= best → increment counter
    - If counter reaches patience → return True (stop training)

    The best model is saved by the main loop separately whenever
    F1 improves — early stopping just decides when to halt.
    """
    def __init__(self, patience):
        self.patience  = patience
        self.counter   = 0
        self.best_f1   = 0.0

    def step(self, f1) -> bool:
        """Returns True if training should stop."""
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"  Early stopping: no improvement for {self.counter}/{self.patience} epochs")
            return self.counter >= self.patience

    def state_dict(self):
        return {"counter": self.counter, "best_f1": self.best_f1}

    def load_state_dict(self, d):
        self.counter = d["counter"]
        self.best_f1 = d["best_f1"]


# --- Resume helpers ------------------------------------------

def save_resume_checkpoint(epoch, model, optimizer, scheduler,
                           plateau_scheduler, early_stopper,
                           best_label_f1, results_log, path):
    """
    Saves full training state including early stopper and plateau scheduler.
    """
    torch.save({
        "epoch":                  epoch,
        "model_state":            model.state_dict(),
        "optimizer_state":        optimizer.state_dict(),
        "scheduler_state":        scheduler.state_dict(),
        "plateau_scheduler_state": plateau_scheduler.state_dict(),
        "early_stopper_state":    early_stopper.state_dict(),
        "best_label_f1":          best_label_f1,
        "results_log":            results_log,
        "label_map":              LABEL_MAP,
        "type_map":               TYPE_MAP,
        "config":                 CONFIG,
    }, path)
    print(f"  Resume checkpoint saved → {path}")


def load_resume_checkpoint(path, model, optimizer, scheduler,
                           plateau_scheduler, early_stopper, device):
    print(f"Loading resume checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    if "plateau_scheduler_state" in ckpt:
        plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state"])
    if "early_stopper_state" in ckpt:
        early_stopper.load_state_dict(ckpt["early_stopper_state"])

    start_epoch   = ckpt["epoch"] + 1
    best_label_f1 = ckpt["best_label_f1"]
    results_log   = ckpt["results_log"]

    print(f"  Last completed epoch : {ckpt['epoch']}")
    print(f"  Resuming from epoch  : {start_epoch} of {CONFIG['max_epochs']}")
    print(f"  Best F1 so far       : {best_label_f1:.4f}")
    print(f"  Early stop counter   : {early_stopper.counter}/{CONFIG['early_stop_patience']}")

    return start_epoch, best_label_f1, results_log


# --- Training loop -------------------------------------------

def train():

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        print("Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return

    device = torch.device("cuda")
    print(f"Using GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    train_rows, val_rows = load_and_prepare_data()

    print("\nPre-tokenising data...")
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

    print(f"\nLoading model: {CONFIG['model_name']}")
    model     = MedHalluModel(CONFIG["model_name"], CONFIG["dropout"])
    model     = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01,
    )

    # Warmup + linear decay scheduler (same as before)
    total_steps  = (len(train_loader) // CONFIG["accumulation_steps"]) * CONFIG["max_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ReduceLROnPlateau — secondary scheduler on top of warmup/decay.
    # Monitors label F1 (mode="max"). If F1 doesn't improve for
    # plateau_patience epochs, multiplies LR by plateau_factor.
    # This is additive to the linear decay: both adjust the LR.
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG["plateau_factor"],
        patience=CONFIG["plateau_patience"],
        min_lr=CONFIG["plateau_min_lr"],
    )

    early_stopper = EarlyStopping(patience=CONFIG["early_stop_patience"])
    criterion     = nn.CrossEntropyLoss()
    best_label_f1 = 0.0
    results_log   = []
    start_epoch   = 1

    # --------------------------------------------------------
    # Resume logic
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
                    plateau_scheduler, early_stopper,
                    device,
                )
                if start_epoch > CONFIG["max_epochs"]:
                    print(f"\nAll {CONFIG['max_epochs']} epochs already completed.")
                    print("Type '2' when prompted next run to retrain.")
                    return
            except (EOFError, RuntimeError, Exception) as e:
                print(f"\nWARNING: Resume checkpoint corrupted ({e})")
                print("Deleting and starting fresh...")
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
    print(f"  Max epochs        : {CONFIG['max_epochs']}  (early stop after {CONFIG['early_stop_patience']} non-improving)")
    print(f"  Batch size        : {CONFIG['batch_size']}")
    print(f"  Accumulation steps: {CONFIG['accumulation_steps']}")
    print(f"  Effective batch   : {CONFIG['batch_size'] * CONFIG['accumulation_steps']}")
    print(f"  Learning rate     : {CONFIG['learning_rate']}")
    print(f"  Plateau factor    : {CONFIG['plateau_factor']}x after {CONFIG['plateau_patience']} stagnant epochs")
    print(f"  Mixed precision   : bf16={CONFIG['bf16']}")
    print()

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):

        model.train()
        epoch_start  = time.time()
        running_loss = 0.0
        nan_steps    = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{CONFIG['max_epochs']}",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in progress_bar:

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            htypes         = batch["htype"].to(device)
            is_grounded    = batch["is_grounded"].to(device)

            with autocast(device_type="cuda",
                          dtype=torch.bfloat16,
                          enabled=CONFIG["bf16"]):
                label_logits, type_logits = model(input_ids, attention_mask)

            # FIX: type loss only on non-grounded rows
            label_loss = criterion(label_logits, labels)
            type_loss  = compute_type_loss(criterion, type_logits, htypes, is_grounded)

            loss = (
                CONFIG["label_loss_weight"] * label_loss +
                CONFIG["type_loss_weight"]  * type_loss
            ) / CONFIG["accumulation_steps"]

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

            avg_loss = running_loss / max(step + 1 - nan_steps, 1)
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "nan":  nan_steps,
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # FIX: only flush if there are actually accumulated gradients.
        # Check if step+1 was NOT a multiple of accumulation_steps
        # (meaning the last mini-batch left un-stepped gradients).
        remaining = (step + 1) % CONFIG["accumulation_steps"]
        if remaining != 0:
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

        # Step the plateau scheduler — may reduce LR if F1 stagnates
        plateau_scheduler.step(label_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Current LR after plateau check: {current_lr:.2e}")

        results_log.append({
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 4),
            "label_f1":       round(label_f1, 4),
            "type_f1":        round(type_f1, 4),
            "epoch_time_min": round(epoch_time / 60, 1),
            "nan_steps":      nan_steps,
            "lr":             current_lr,
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

        save_resume_checkpoint(
            epoch, model, optimizer, scheduler,
            plateau_scheduler, early_stopper,
            best_label_f1, results_log,
            CONFIG["resume_path"],
        )

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

        # Check early stopping AFTER saving everything
        # so the resume checkpoint is always up to date
        if early_stopper.step(label_f1):
            print("=" * 50)
            print(f"EARLY STOPPING triggered at epoch {epoch}")
            print(f"No F1 improvement for {CONFIG['early_stop_patience']} epochs.")
            print(f"Best label F1 was : {best_label_f1:.4f}")
            print("=" * 50)
            break

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best label F1 : {best_label_f1:.4f}")
    print(f"Stopped at    : epoch {results_log[-1]['epoch']} of {CONFIG['max_epochs']}")
    print(f"Best model    : {CONFIG['best_model_path']}")
    print(f"Results log   : {CONFIG['results_path']}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to see the comparison table")
    print("  2. Hand outputs/deberta_medhallu.pt to Person 3")
    print("  3. Hand outputs/results.json to Person 3")


if __name__ == "__main__":
    train()