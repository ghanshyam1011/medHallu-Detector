# ============================================================
# train_span_extractor.py
# ============================================================
# WHAT  : Fine-tunes a QA model to find the exact wrong phrase
#         inside a hallucinated medical answer.
#
# WHY different model:
#         microsoft/deberta-v3-base is a masked LM pretrained
#         model — its QA head (qa_outputs) is randomly initialised
#         and training it on only 237 examples causes gradient
#         explosion that poisons the backbone weights.
#
#         deepset/deberta-v3-base-squad2 is ALREADY fine-tuned
#         on SQuAD2 (130k QA pairs). Its qa_outputs weights are
#         already meaningful — we only need a small nudge to
#         specialise it on medical hallucination spans.
#         This gives stable training from step 1.
# ============================================================

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import collections


# --- Configuration -------------------------------------------

CONFIG = {
    # WHY this model instead of microsoft/deberta-v3-base:
    # Already fine-tuned on SQuAD2 — qa_outputs are pretrained
    # not random. Training is stable from step 1.
    "model_name":         "deepset/deberta-v3-base-squad2",

    "max_length":         384,
    "doc_stride":         128,
    "batch_size":         8,
    "accumulation_steps": 2,
    "epochs":             6,

    # WHY same LR for both:
    # Both backbone AND head are pretrained — no need for
    # discriminative LR. Standard fine-tuning LR works.
    "learning_rate":      2e-5,

    "warmup_ratio":       0.1,
    "annotations_path":   "annotation_export/spans.json",
    "test_size":          0.2,
    "random_seed":        42,
    "output_dir":         "outputs/span_extractor_final",
}

SPAN_QUESTION = "Which phrase is not supported by the source?"


# --- Annotation parser ---------------------------------------

def parse_label_studio_export(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed  = []
    skipped = 0

    for item in raw:
        if isinstance(item, list):
            item = item[0]

        context = (
            item.get("text") or
            item.get("data", {}).get("text") or
            item.get("content") or ""
        )
        context = str(context).strip()

        if not context:
            skipped += 1
            continue

        annotations = (
            item.get("label") or
            item.get("annotations", [{}])[0].get("result", []) or []
        )

        if not annotations:
            skipped += 1
            continue

        ann = annotations[0]

        if "start" in ann:
            span_start = ann["start"]
            span_end   = ann["end"]
            span_text  = ann.get("text", context[span_start:span_end])
        elif "value" in ann:
            val        = ann["value"]
            span_start = val.get("start", 0)
            span_end   = val.get("end",   0)
            span_text  = val.get("text",  context[span_start:span_end])
        else:
            skipped += 1
            continue

        if span_end <= span_start:
            skipped += 1
            continue

        if span_text and context[span_start:span_end].lower() != span_text.lower():
            idx = context.lower().find(span_text.lower())
            if idx == -1:
                skipped += 1
                continue
            span_start = idx
            span_end   = idx + len(span_text)

        parsed.append({
            "id":         item.get("id", len(parsed)),
            "context":    context,
            "question":   SPAN_QUESTION,
            "span_text":  context[span_start:span_end],
            "span_start": span_start,
            "span_end":   span_end,
        })

    print(f"Parsed  : {len(parsed)} annotated rows")
    print(f"Skipped : {skipped} rows")
    return parsed


# --- Dataset -------------------------------------------------

class SpanDataset(Dataset):

    def __init__(self, items, tokenizer, max_length, doc_stride):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        print("Tokenising span data...")
        self.features = self._tokenise_all(items)

    def _tokenise_all(self, items):
        features = []
        skipped  = 0

        for item in tqdm(items, desc="Tokenising", unit="row", dynamic_ncols=True):
            question        = item["question"]
            context         = item["context"]
            span_start_char = item["span_start"]
            span_end_char   = item["span_end"]

            encoding = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                stride=self.doc_stride,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            offset_mapping = encoding["offset_mapping"]
            num_chunks     = offset_mapping.shape[0]

            for chunk_idx in range(num_chunks):
                offsets      = offset_mapping[chunk_idx].tolist()
                sequence_ids = encoding.sequence_ids(chunk_idx)

                context_start_idx = None
                context_end_idx   = None
                for i, sid in enumerate(sequence_ids):
                    if sid == 1:
                        if context_start_idx is None:
                            context_start_idx = i
                        context_end_idx = i

                if context_start_idx is None:
                    continue

                start_token = None
                end_token   = None

                for i in range(context_start_idx, context_end_idx + 1):
                    t_start, t_end = offsets[i]
                    if t_start <= span_start_char < t_end:
                        start_token = i
                    if t_start < span_end_char <= t_end:
                        end_token = i

                # Skip chunk if span not found in this window
                # WHY: setting start=0, end=0 poisons training
                # with wrong labels → nan loss → weight corruption
                if start_token is None or end_token is None:
                    skipped += 1
                    continue

                if start_token > end_token:
                    start_token = end_token

                features.append({
                    "input_ids":       encoding["input_ids"][chunk_idx],
                    "attention_mask":  encoding["attention_mask"][chunk_idx],
                    "start_positions": torch.tensor(start_token, dtype=torch.long),
                    "end_positions":   torch.tensor(end_token,   dtype=torch.long),
                    "context":         context,
                    "span_text":       item["span_text"],
                    "offsets":         offsets,
                    "sequence_ids":    sequence_ids,
                })

        print(f"  Valid features : {len(features)}")
        print(f"  Skipped chunks : {skipped} (span not in window — normal)")
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        return {
            "input_ids":       f["input_ids"],
            "attention_mask":  f["attention_mask"],
            "start_positions": f["start_positions"],
            "end_positions":   f["end_positions"],
        }


# --- Token F1 ------------------------------------------------

def compute_token_f1(pred: str, true: str) -> float:
    pt = pred.lower().split()
    tt = true.lower().split()
    if not pt or not tt:
        return 0.0
    pc = collections.Counter(pt)
    tc = collections.Counter(tt)
    common = sum((pc & tc).values())
    if common == 0:
        return 0.0
    prec = common / len(pt)
    rec  = common / len(tt)
    return 2 * prec * rec / (prec + rec)


# --- Span extraction -----------------------------------------

def extract_span(start_logits, end_logits, offsets, sequence_ids,
                 context, max_len=50):
    sl = start_logits.cpu().numpy()
    el = end_logits.cpu().numpy()

    ctx_idx = [i for i, s in enumerate(sequence_ids) if s == 1]
    if not ctx_idx:
        return ""

    ms = np.full_like(sl, -np.inf)
    me = np.full_like(el, -np.inf)
    for i in ctx_idx:
        ms[i] = sl[i]
        me[i] = el[i]

    best, bs, be = -np.inf, ctx_idx[0], ctx_idx[0]
    for s in ctx_idx:
        for e in ctx_idx:
            if e < s or e - s + 1 > max_len:
                continue
            score = ms[s] + me[e]
            if score > best:
                best, bs, be = score, s, e

    cs = offsets[bs][0]
    ce = offsets[be][1]
    if cs >= ce:
        return ""
    return context[cs:ce]


# --- Evaluation ----------------------------------------------

def evaluate(model, tokenizer, test_items, device):
    model.eval()
    f1s, em = [], 0

    with torch.no_grad():
        for item in tqdm(test_items, desc="Evaluating",
                         dynamic_ncols=True, leave=False):
            enc = tokenizer(
                item["question"], item["context"],
                max_length=CONFIG["max_length"],
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets      = enc["offset_mapping"][0].tolist()
            sequence_ids = enc.sequence_ids(0)

            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )

            pred = extract_span(
                out.start_logits[0], out.end_logits[0],
                offsets, sequence_ids, item["context"],
            )
            true = item["span_text"].strip()

            f1 = compute_token_f1(pred, true)
            f1s.append(f1)
            if pred.lower() == true.lower():
                em += 1

    avg_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"  Token F1    : {avg_f1:.4f}")
    print(f"  Exact match : {em/len(test_items):.4f}  ({em}/{len(test_items)})")
    return avg_f1


# --- Training ------------------------------------------------

def train():

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    print(f"Device : {device}")
    if device_str == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(CONFIG["annotations_path"]):
        print("ERROR: spans.json not found. Run auto_annotate_spans.py first.")
        return

    items = parse_label_studio_export(CONFIG["annotations_path"])
    if len(items) < 20:
        print(f"ERROR: only {len(items)} valid items — too few to train.")
        return

    train_items, test_items = train_test_split(
        items, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"],
    )
    print(f"Train items : {len(train_items)}")
    print(f"Test items  : {len(test_items)}")

    print(f"\nLoading tokenizer + model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # WHY no use_safetensors here:
    # deepset/deberta-v3-base-squad2 may not have .safetensors
    # We let transformers decide the best format automatically
    model = AutoModelForQuestionAnswering.from_pretrained(
        CONFIG["model_name"]
    )
    model = model.to(device)

    train_dataset = SpanDataset(
        train_items, tokenizer,
        CONFIG["max_length"], CONFIG["doc_stride"]
    )
    print(f"Training features : {len(train_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: 0 valid training features.")
        print("All spans fell outside the token windows.")
        print("Try reducing max_length or check spans.json quality.")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,
    )

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

    best_f1 = 0.0

    print(f"\nStarting training")
    print(f"  Model        : {CONFIG['model_name']}")
    print(f"  Epochs       : {CONFIG['epochs']}")
    print(f"  Batch size   : {CONFIG['batch_size']}")
    print(f"  LR           : {CONFIG['learning_rate']}")
    print(f"  Total steps  : {total_steps}")
    print()

    for epoch in range(1, CONFIG["epochs"] + 1):

        model.train()
        running_loss = 0.0
        nan_steps    = 0
        valid_steps  = 0
        optimizer.zero_grad()

        bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{CONFIG['epochs']}",
            unit="step", dynamic_ncols=True, leave=True,
        )

        for step, batch in bar:

            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions   = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            raw_loss = outputs.loss
            loss     = raw_loss / CONFIG["accumulation_steps"]

            # Check BEFORE backward
            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                nan_steps += 1
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None
                continue

            loss.backward()
            valid_steps += 1
            running_loss += raw_loss.item()

            if (step + 1) % CONFIG["accumulation_steps"] == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(gn):
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg = running_loss / max(valid_steps, 1)
            bar.set_postfix({
                "loss": f"{avg:.4f}",
                "nan":  nan_steps,
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Flush
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        avg_loss = running_loss / max(valid_steps, 1)
        print(f"\nEpoch {epoch} — loss: {avg_loss:.4f} | nan: {nan_steps}/{len(train_loader)}")
        print("  Evaluating...")
        model.eval()
        f1 = evaluate(model, tokenizer, test_items, device)

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            model.save_pretrained(CONFIG["output_dir"])
            tokenizer.save_pretrained(CONFIG["output_dir"])
            print(f"  Saved → {CONFIG['output_dir']}")

        torch.cuda.empty_cache()
        print()

    print("=" * 50)
    print("SPAN EXTRACTOR TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best token F1 : {best_f1:.4f}")
    print(f"Model saved   : {CONFIG['output_dir']}")

    if best_f1 >= 0.50:
        print("  Good result for this dataset size")
    elif best_f1 >= 0.30:
        print("  Moderate — acceptable for demo")
    elif best_f1 >= 0.15:
        print("  Functional for demo — will highlight wrong phrases")
    else:
        print("  Weak — but span extractor is not the main model")

    print("\nHand outputs/span_extractor_final/ to app.py")


if __name__ == "__main__":
    train()