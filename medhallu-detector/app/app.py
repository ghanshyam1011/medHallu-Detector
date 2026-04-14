import os
import json
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from difflib import SequenceMatcher
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)

# --- Paths ---------------------------------------------------
CLASSIFIER_MODEL_NAME = "microsoft/deberta-v3-base"
CLASSIFIER_CHECKPOINT_REL = "../person2_model/outputs/deberta_medhallu.pt"
SPAN_MODEL_DIR_REL        = "../span_extractor_model"
RESULTS_JSON_REL          = "../person2_model/outputs/results.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_CHECKPOINT = os.path.normpath(os.path.join(BASE_DIR, CLASSIFIER_CHECKPOINT_REL))
SPAN_MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, SPAN_MODEL_DIR_REL))
RESULTS_JSON = os.path.normpath(os.path.join(BASE_DIR, RESULTS_JSON_REL))

# --- Label maps (MUST match training exactly) ---------------
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

TYPE_DESCRIPTIONS = {
    "misinterpretation of #question#":
        "The answer misinterprets what the question is actually asking.",
    "incomplete information":
        "The answer omits critical facts present in the source.",
    "mechanism and pathway misattribution":
        "The answer attributes a biological mechanism to the wrong pathway or molecule.",
    "methodological and evidence fabrication":
        "The answer fabricates study findings or evidence not present in the source.",
    "none":
        "No specific hallucination type identified.",
}

VERDICT_COLORS = {
    "hallucinated":          "#ef4444",
    "partially_hallucinated": "#f59e0b",
    "likely_grounded":        "#22c55e",
    "grounded":               "#10b981",
    "not_sure":               "#6b7280",
}

GPT4O_SCORES = {
    "overall": 0.737,
    "easy":    0.844,
    "medium":  0.758,
    "hard":    0.625,
}

# ============================================================
# Model class
# ============================================================
class MedHalluModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=False)
        hidden_size   = self.backbone.config.hidden_size
        self.dropout  = nn.Dropout(dropout)
        self.label_head = nn.Linear(hidden_size, len(LABEL_MAP))
        self.type_head  = nn.Linear(hidden_size, len(TYPE_MAP))

    def forward(self, input_ids, attention_mask):
        outputs    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        return self.label_head(cls_output), self.type_head(cls_output)


# ============================================================
# Load models
# ============================================================
print("=" * 55)
print("Loading models...")
print("=" * 55)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\n[1/3] Loading hallucination classifier...")
clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
clf_model     = MedHalluModel(CLASSIFIER_MODEL_NAME)
clf_loaded    = False

if os.path.exists(CLASSIFIER_CHECKPOINT):
    try:
        checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=False)
        clf_model.load_state_dict(checkpoint["model_state"])
        clf_loaded = True
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Checkpoint label F1: {checkpoint.get('label_f1', '?')}")
    except Exception as e:
        print(f"  WARNING: Could not load checkpoint: {e}")
else:
    print(f"  WARNING: checkpoint not found at {CLASSIFIER_CHECKPOINT}")

clf_model = clf_model.to(device)
clf_model.eval()
print("  Classifier ready")

print("\n[2/3] Loading span extractor...")
span_tokenizer = None
span_model_hf  = None

if os.path.exists(SPAN_MODEL_DIR):
    local_model_loaded = False
    local_tokenizer_loaded = False

    try:
        span_model_hf = AutoModelForQuestionAnswering.from_pretrained(
            SPAN_MODEL_DIR,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
        local_model_loaded = True
        print("  Local fine-tuned span weights loaded")
    except Exception as e:
        print(f"  WARNING: Could not load local span weights: {e}")

    try:
        span_tokenizer = AutoTokenizer.from_pretrained(
            SPAN_MODEL_DIR,
            local_files_only=True,
        )
        local_tokenizer_loaded = True
        print("  Local span tokenizer loaded")
    except Exception as e:
        print(f"  WARNING: Could not load local span tokenizer: {e}")
        try:
            span_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            print("  Using roberta-base tokenizer fallback with local weights")
        except Exception as e2:
            print(f"  WARNING: Could not load tokenizer fallback: {e2}")

    if local_model_loaded and span_tokenizer is not None:
        span_model_hf = span_model_hf.to(device)
        span_model_hf.eval()
        if local_tokenizer_loaded:
            print("  Span extractor ready (local fine-tuned model + local tokenizer)")
        else:
            print("  Span extractor ready (local fine-tuned model + roberta-base tokenizer)")
    else:
        try:
            print("  Trying fallback: loading base span model from HuggingFace...")
            BASE_SPAN = "deepset/roberta-base-squad2"
            span_tokenizer = AutoTokenizer.from_pretrained(BASE_SPAN)
            span_model_hf  = AutoModelForQuestionAnswering.from_pretrained(BASE_SPAN)
            span_model_hf  = span_model_hf.to(device)
            span_model_hf.eval()
            print("  Span extractor ready (base model — no fine-tuning)")
        except Exception as e2:
            print(f"  WARNING: Fallback also failed: {e2}")
            span_tokenizer = None
            span_model_hf = None
            print("  Span highlighting disabled")

print("\n[3/3] Loading comparison data...")
results_data = None
if os.path.exists(RESULTS_JSON):
    try:
        with open(RESULTS_JSON) as f:
            results_data = json.load(f)
        print("  Results data ready")
    except Exception as e:
        print(f"  WARNING: Could not load results.json: {e}")
else:
    print(f"  WARNING: results.json not found at {RESULTS_JSON}")


def _fallback_to_cpu_if_needed(error: Exception):
    global device
    err_text = str(error).lower()
    if device.type != "cuda":
        return False
    if "cuda" not in err_text and "cublas" not in err_text and "cudnn" not in err_text:
        return False

    print(f"CUDA inference error detected, falling back to CPU: {error}")
    device = torch.device("cpu")
    clf_model.to(device)
    if span_model_hf is not None:
        span_model_hf.to(device)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return True

print("\nAll models loaded. Starting Gradio app...")


def get_4class_verdict(halluc_prob: float) -> str:
    if halluc_prob >= 0.75:
        return "hallucinated"
    if halluc_prob >= 0.55:
        return "partially_hallucinated"
    if halluc_prob >= 0.35:
        return "likely_grounded"
    return "grounded"


def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def is_directly_supported(answer: str, source: str) -> bool:
    ans = _normalize_text(answer)
    src = _normalize_text(source)
    if not ans or not src:
        return False
    if len(ans) < 40:
        return ans == src
    if ans in src:
        return True
    ans_tokens = [token for token in ans.split() if token]
    src_tokens = set(src.split())
    if not ans_tokens:
        return False
    token_overlap = sum(1 for token in ans_tokens if token in src_tokens) / len(ans_tokens)
    sequence_ratio = SequenceMatcher(None, ans, src).ratio()
    return token_overlap >= 0.85 or sequence_ratio >= 0.9


# ============================================================
# Prediction functions
# ============================================================
def run_classifier(question, answer, source):
    question = str(question).strip()
    answer   = str(answer).strip()
    source   = str(source).strip()

    encoding = clf_tokenizer(
        source, f"{question} {answer}",
        max_length=256, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    try:
        with torch.no_grad():
            label_logits, type_logits = clf_model(input_ids, attention_mask)
    except RuntimeError as e:
        if not _fallback_to_cpu_if_needed(e):
            raise
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            label_logits, type_logits = clf_model(input_ids, attention_mask)

    label_probs = torch.softmax(label_logits, dim=-1)[0]
    type_probs  = torch.softmax(type_logits,  dim=-1)[0]

    label_id    = label_probs.argmax().item()
    type_id     = type_probs.argmax().item()
    verdict     = LABEL_MAP_INV[label_id]
    confidence  = label_probs[label_id].item()
    htype       = TYPE_MAP_INV[type_id]
    halluc_prob = label_probs[LABEL_MAP["hallucinated"]].item()
    verdict_4class = get_4class_verdict(halluc_prob)

    all_probs = {
        LABEL_MAP_INV[i]: round(label_probs[i].item(), 4)
        for i in range(len(LABEL_MAP))
    }
    return verdict, confidence, htype, all_probs, halluc_prob, verdict_4class


# ============================================================
# UPDATED: span extractor now receives source too
# ============================================================
def run_span_extractor(answer, source):
    if span_model_hf is None or span_tokenizer is None:
        return "", 0.0

    SPAN_QUESTION = "Which phrase in the answer is not supported by the source?"
    combined_context = f"Source: {source} Answer: {answer}"

    try:
        encoding = span_tokenizer(
            SPAN_QUESTION,
            combined_context,
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets      = encoding["offset_mapping"][0].tolist()
        sequence_ids = encoding.sequence_ids(0)
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        try:
            with torch.no_grad():
                outputs = span_model_hf(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if not _fallback_to_cpu_if_needed(e):
                raise
            input_ids      = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.no_grad():
                outputs = span_model_hf(input_ids=input_ids, attention_mask=attention_mask)

        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits   = outputs.end_logits[0].cpu().numpy()

        # --------------------------------------------------------
        # Find where "Answer: " starts in the combined_context
        # so we only allow the model to predict spans from the
        # answer portion, not from the source portion.
        # --------------------------------------------------------
        answer_prefix    = f"Source: {source} Answer: "
        answer_offset    = len(answer_prefix)

        # Filter ctx_indices to only tokens that fall inside the answer portion
        ctx_indices = []
        for i, (seq_id, (char_s, char_e)) in enumerate(zip(sequence_ids, offsets)):
            if seq_id == 1 and char_s >= answer_offset:
                ctx_indices.append(i)

        # Fallback: if nothing maps to the answer portion (e.g. source was
        # so long it got truncated and ate into the answer), use ALL context
        # tokens so we at least return something.
        if not ctx_indices:
            ctx_indices = [i for i, s in enumerate(sequence_ids) if s == 1]
            answer_offset = 0  # can't adjust, return raw span

        if not ctx_indices:
            return "", 0.0

        masked_start = np.full_like(start_logits, -np.inf)
        masked_end   = np.full_like(end_logits,   -np.inf)
        for i in ctx_indices:
            masked_start[i] = start_logits[i]
            masked_end[i]   = end_logits[i]

        best_score = -np.inf
        best_start = best_end = ctx_indices[0]
        for s in ctx_indices:
            for e in ctx_indices:
                if e < s or e - s + 1 > 50:
                    continue
                score = masked_start[s] + masked_end[e]
                if score > best_score:
                    best_score, best_start, best_end = score, s, e

        char_start = offsets[best_start][0]
        char_end   = offsets[best_end][1]
        if char_start >= char_end:
            return "", 0.0

        # Adjust back to answer-only coordinates
        adj_start = char_start - answer_offset
        adj_end   = char_end   - answer_offset

        if adj_start < 0 or adj_end <= adj_start or adj_end > len(answer):
            return "", 0.0

        span_text  = answer[adj_start:adj_end].strip()
        confidence = float(torch.softmax(outputs.start_logits[0], dim=-1)[best_start].item())

        if not span_text:
            return "", 0.0

        return span_text, confidence

    except Exception as e:
        print(f"Span extractor error: {e}")
        return "", 0.0

def highlight_span(answer, span):
    base_style = "font-size:15px;line-height:2;padding:16px;border-radius:10px;border:1px solid #1e293b;background:#0f172a;color:#e2e8f0;font-family:'IBM Plex Mono',monospace;"
    if not span:
        return f"<div style='{base_style}'>{answer}</div>"

    lower_answer = answer.lower()
    idx = lower_answer.find(span.lower())
    if idx == -1:
        return f"<div style='{base_style}'>{answer}</div>"

    actual_span = answer[idx: idx + len(span)]
    highlighted = (
        answer[:idx] +
        f'<mark style="background:#fca5a5;color:#7f1d1d;padding:2px 6px;'
        f'border-radius:4px;font-weight:700;font-family:inherit">{actual_span}</mark>' +
        answer[idx + len(span):]
    )
    return f"<div style='{base_style}'>{highlighted}</div>"


def predict(question, answer, source):
    err_style = "color:#64748b;padding:16px;font-family:'IBM Plex Mono',monospace;font-size:14px"
    if not question.strip():
        return f"<p style='{err_style}'>⚠ Please enter a question.</p>", "", "", "", ""
    if not answer.strip():
        return f"<p style='{err_style}'>⚠ Please enter an answer to check.</p>", "", "", "", ""
    if not source.strip():
        return f"<p style='{err_style}'>⚠ Please enter a source text.</p>", "", "", "", ""

    _, confidence, htype, all_probs, halluc_prob, verdict = run_classifier(question, answer, source)

    if is_directly_supported(answer, source):
        verdict     = "grounded"
        halluc_prob = min(halluc_prob, 0.05)
        confidence  = max(confidence, 1.0 - halluc_prob)
        htype       = "none"
        all_probs["hallucinated"] = round(halluc_prob, 4)
        all_probs["grounded"]     = round(1.0 - halluc_prob, 4)

    span_text = ""
    if verdict in {"hallucinated", "partially_hallucinated"}:
        # UPDATED: pass source so the extractor can cross-reference
        span_text, _ = run_span_extractor(answer, source)

    if verdict in {"grounded", "likely_grounded"}:
        htype = "none"

    color         = VERDICT_COLORS.get(verdict, "#6b7280")
    verdict_label = verdict.replace("_", " ").upper()
    conf_pct      = f"{confidence * 100:.1f}%"
    halluc_pct    = f"{halluc_prob * 100:.1f}%"
    type_display  = htype.replace("_", " ").title()

    if verdict in {"hallucinated", "partially_hallucinated"}:
        icon = "✗"
    elif verdict == "likely_grounded":
        icon = "~"
    else:
        icon = "✓"

    verdict_html = f"""
    <div style='padding:20px 24px;border-radius:12px;border:1px solid {color}33;
                background:linear-gradient(135deg,{color}11,{color}06);
                font-family:"IBM Plex Mono",monospace;position:relative;overflow:hidden'>
        <div style='position:absolute;right:20px;top:50%;transform:translateY(-50%);
                    font-size:64px;opacity:0.07;font-weight:900;color:{color}'>{icon}</div>
        <div style='font-size:11px;letter-spacing:3px;text-transform:uppercase;
                    color:{color};opacity:0.8;margin-bottom:6px'>Verdict</div>
        <div style='font-size:26px;font-weight:700;color:{color};letter-spacing:1px'>
            {icon} {verdict_label}
        </div>
        <div style='display:flex;gap:20px;margin-top:12px;font-size:13px;color:#94a3b8'>
            <span>Confidence <strong style='color:#e2e8f0'>{conf_pct}</strong></span>
            <span style='opacity:0.4'>|</span>
            <span>P(Hallucinated) <strong style='color:#e2e8f0'>{halluc_pct}</strong></span>
            <span style='opacity:0.4'>|</span>
            <span>Type <strong style='color:#e2e8f0'>{type_display}</strong></span>
        </div>
    </div>
    """

    answer_html  = highlight_span(answer, span_text)
    span_display = span_text if span_text else "— no specific phrase identified"

    bars_html = "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:13px'>"
    bars_html += "<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#475569;margin-bottom:12px'>Class Probabilities</div>"
    for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        pct       = prob * 100
        bar_color = VERDICT_COLORS.get(label, "#6b7280")
        lbl       = label.replace("_", " ").upper()
        bars_html += f"""
        <div style='margin-bottom:10px'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                <span style='color:#94a3b8;font-size:11px;letter-spacing:1px'>{lbl}</span>
                <span style='color:{bar_color};font-weight:700;font-size:12px'>{pct:.1f}%</span>
            </div>
            <div style='background:#1e293b;border-radius:2px;height:6px'>
                <div style='background:{bar_color};border-radius:2px;height:6px;
                            width:{pct:.1f}%;transition:width 0.5s ease'></div>
            </div>
        </div>"""
    bars_html += "</div>"

    type_desc = TYPE_DESCRIPTIONS.get(htype, "")
    type_md   = f"**{type_display}** — {type_desc}" if type_desc else ""

    return verdict_html, answer_html, span_display, bars_html, type_md


# ============================================================
# Comparison table
# ============================================================
def build_comparison_table():
    if results_data is None:
        return "<p style='color:#64748b;font-family:IBM Plex Mono,monospace'>results.json not found. Run evaluate.py first.</p>"

    checkpoint = results_data.get("checkpoint_info", {})
    our  = results_data.get("medhallu_test", results_data.get("medhallu_validation", {}))
    groq = results_data.get("groq_baseline", {})

    def fmt(val):
        if val is None:
            return "—"
        try:
            return f"{float(val):.3f}"
        except:
            return "—"

    def beats(our_val, gpt_val):
        try:
            return float(our_val) > float(gpt_val)
        except:
            return False

    rows_html = ""
    splits = [
        ("Overall", "overall_f1", GPT4O_SCORES["overall"]),
        ("Easy",    "easy_f1",    GPT4O_SCORES["easy"]),
        ("Medium",  "medium_f1",  GPT4O_SCORES["medium"]),
        ("Hard",    "hard_f1",    GPT4O_SCORES["hard"]),
    ]

    for label, key, gpt_val in splits:
        our_val  = fmt(our.get(key))
        groq_val = fmt(groq.get(key))
        gpt_str  = f"{gpt_val:.3f}"
        won      = beats(our_val, gpt_val)
        our_color = "#10b981" if won else "#94a3b8"
        badge     = "<span style='font-size:9px;background:#10b98122;color:#10b981;padding:2px 6px;border-radius:20px;margin-left:6px;letter-spacing:1px'>BEST</span>" if won else ""

        rows_html += f"""
        <tr style='border-bottom:1px solid #1e293b'>
            <td style='padding:14px 18px;color:#e2e8f0;font-weight:600;font-size:13px'>{label}</td>
            <td style='padding:14px 18px;text-align:center;color:#64748b;font-size:13px'>{gpt_str}</td>
            <td style='padding:14px 18px;text-align:center;color:#64748b;font-size:13px'>{groq_val}</td>
            <td style='padding:14px 18px;text-align:center;color:{our_color};font-weight:700;font-size:13px'>{our_val}{badge}</td>
        </tr>"""

    label_f1 = fmt(checkpoint.get("label_f1"))
    type_f1  = fmt(checkpoint.get("type_f1"))
    epoch    = checkpoint.get("epoch", "—")

    html = f"""
    <div style='font-family:"IBM Plex Mono",monospace'>
        <div style='margin-bottom:24px'>
            <div style='font-size:10px;letter-spacing:3px;color:#475569;text-transform:uppercase;margin-bottom:16px'>
                Performance vs Baselines — MedHallu Test Split
            </div>
            <div style='border-radius:12px;overflow:hidden;border:1px solid #1e293b'>
                <table style='width:100%;border-collapse:collapse;font-size:13px;background:#0f172a'>
                    <thead>
                        <tr style='background:#1e293b'>
                            <th style='padding:14px 18px;text-align:left;color:#94a3b8;font-weight:500;font-size:11px;letter-spacing:2px;text-transform:uppercase'>Split</th>
                            <th style='padding:14px 18px;text-align:center;color:#94a3b8;font-weight:500;font-size:11px;letter-spacing:2px;text-transform:uppercase'>GPT-4o</th>
                            <th style='padding:14px 18px;text-align:center;color:#94a3b8;font-weight:500;font-size:11px;letter-spacing:2px;text-transform:uppercase'>Groq Llama</th>
                            <th style='padding:14px 18px;text-align:center;color:#94a3b8;font-weight:500;font-size:11px;letter-spacing:2px;text-transform:uppercase'>Our DeBERTa</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            <div style='font-size:11px;color:#334155;margin-top:8px'>
                ✓ Green = beats GPT-4o &nbsp;|&nbsp; Evaluated on MedHallu rows 8000–10000 (unseen during training)
            </div>
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:24px'>
            <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px'>
                <div style='font-size:10px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:8px'>Best Epoch</div>
                <div style='font-size:28px;font-weight:700;color:#e2e8f0'>{epoch}</div>
            </div>
            <div style='background:#0f172a;border:1px solid #10b98133;border-radius:10px;padding:16px'>
                <div style='font-size:10px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:8px'>Label F1</div>
                <div style='font-size:28px;font-weight:700;color:#10b981'>{label_f1}</div>
            </div>
            <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px'>
                <div style='font-size:10px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:8px'>Type F1</div>
                <div style='font-size:28px;font-weight:700;color:#94a3b8'>{type_f1}</div>
            </div>
        </div>
        <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;font-size:12px;color:#475569;line-height:1.8'>
            <span style='color:#10b981'>●</span> Trained on MedHallu pqa_artificial rows 0–7000 &nbsp;
            <span style='color:#475569'>●</span> Validated on rows 7000–8000 &nbsp;
            <span style='color:#475569'>●</span> Tested on rows 8000–10000
        </div>
    </div>
    """
    return html


# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #020817 !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.tab-nav {
    background: #0f172a !important;
    border-bottom: 1px solid #1e293b !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0 8px !important;
}
.tab-nav button {
    color: #475569 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 14px 20px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #10b981 !important;
    border-bottom: 2px solid #10b981 !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: #e2e8f0 !important;
}

textarea, input[type="text"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px #10b98120 !important;
    outline: none !important;
}

label span {
    color: #64748b !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

button.primary {
    background: #10b981 !important;
    color: #020817 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 24px !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    background: #059669 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #10b98133 !important;
}

.block, .form {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

.output-textbox textarea {
    color: #94a3b8 !important;
    font-size: 13px !important;
}

.examples-holder {
    background: transparent !important;
}
.examples table {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
.examples td {
    color: #64748b !important;
    border-color: #1e293b !important;
}
.examples tr:hover td {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    cursor: pointer !important;
}

.prose, .md {
    color: #94a3b8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}
.prose strong, .md strong {
    color: #e2e8f0 !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020817; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
"""

# ============================================================
# Examples
# ============================================================
EXAMPLES = [
    [
        "What enzyme is deficient in phenylketonuria (PKU)?",
        "Tyrosine hydroxylase is the enzyme that is deficient in phenylketonuria, leading to accumulation of phenylalanine in the blood.",
        "Phenylketonuria (PKU) is an autosomal recessive metabolic disorder caused by a deficiency of phenylalanine hydroxylase, the enzyme that converts phenylalanine to tyrosine. Without this enzyme, phenylalanine accumulates to toxic levels.",
    ],
    [
        "What is the primary mechanism of action of metformin?",
        "Metformin works by stimulating insulin secretion from pancreatic beta cells, which lowers blood glucose in type 2 diabetes.",
        "Metformin's primary mechanism is inhibition of hepatic gluconeogenesis via activation of AMP-activated protein kinase (AMPK). It does not stimulate insulin secretion and carries no risk of hypoglycemia.",
    ],
    [
        "What vitamin deficiency causes scurvy?",
        "Scurvy is caused by a deficiency of vitamin C.",
        "Scurvy is caused by a deficiency of vitamin C (ascorbic acid), which is required for collagen synthesis and wound healing.",
    ],
    [
        "What is the main cause of type 1 diabetes?",
        "Type 1 diabetes is caused by eating too much sugar and consuming excessive carbohydrates, which can be prevented through strict diet control alone.",
        "Type 1 diabetes is an autoimmune disease where the body's immune system attacks insulin-producing beta cells in the pancreas. It is NOT caused by diet and cannot be prevented by dietary changes alone. It requires insulin therapy for management.",
    ],
    [
        "What is the function of hemoglobin in red blood cells?",
        "Hemoglobin is a protein that binds and transports oxygen throughout the body.",
        "Hemoglobin is an iron-containing protein in red blood cells that binds oxygen in the lungs and releases it in tissues, enabling oxygen transport throughout the body.",
    ],
]


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="MedHallu Detector") as app:

    gr.HTML("""
    <div style='padding:40px 0 32px;font-family:"Syne",sans-serif'>
        <div style='font-size:10px;letter-spacing:4px;text-transform:uppercase;
                    color:#10b981;margin-bottom:12px;font-family:"IBM Plex Mono",monospace'>
            Medical AI Safety
        </div>
        <div style='font-size:42px;font-weight:800;color:#f8fafc;line-height:1.1;margin-bottom:12px'>
            Hallucination<br>
            <span style='color:#10b981'>Detector</span>
        </div>
        <div style='font-size:13px;color:#475569;font-family:"IBM Plex Mono",monospace;
                    max-width:520px;line-height:1.7'>
            Fine-tuned DeBERTa-v3 · MedHallu benchmark · 
            <span style='color:#10b981'>0.954 macro F1</span> · beats GPT-4o on all splits
        </div>
    </div>
    """)

    with gr.Tab("◈  Detector"):
        with gr.Row(equal_height=False):

            with gr.Column(scale=1):
                gr.HTML("<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#475569;margin-bottom:16px;font-family:IBM Plex Mono,monospace'>Input</div>")

                q_input = gr.Textbox(
                    label="Question",
                    placeholder="What enzyme is deficient in PKU?",
                    lines=2,
                )
                a_input = gr.Textbox(
                    label="LLM Answer to Verify",
                    placeholder="Paste the AI-generated answer here...",
                    lines=5,
                )
                s_input = gr.Textbox(
                    label="Source / Ground Truth",
                    placeholder="Paste the source context the answer should be grounded in...",
                    lines=5,
                )
                check_btn = gr.Button("▶  Run Detection", variant="primary", size="lg")

                gr.HTML("<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#334155;margin:20px 0 10px;font-family:IBM Plex Mono,monospace'>Examples</div>")
                gr.Examples(examples=EXAMPLES, inputs=[q_input, a_input, s_input], label="")

            with gr.Column(scale=1):
                gr.HTML("<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#475569;margin-bottom:16px;font-family:IBM Plex Mono,monospace'>Analysis</div>")

                verdict_out = gr.HTML(
                    value="<div style='padding:20px;border:1px dashed #1e293b;border-radius:12px;color:#334155;font-family:IBM Plex Mono,monospace;font-size:12px;text-align:center;letter-spacing:2px'>AWAITING INPUT</div>"
                )

                gr.HTML("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#334155;margin:16px 0 8px;font-family:IBM Plex Mono,monospace'>Answer with Highlighted Phrase</div>")
                answer_html_out = gr.HTML()

                span_out = gr.Textbox(
                    label="Flagged Phrase",
                    interactive=False,
                )

                gr.HTML("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#334155;margin:16px 0 8px;font-family:IBM Plex Mono,monospace'>Hallucination Type</div>")
                type_out = gr.Markdown()

                gr.HTML("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#334155;margin:16px 0 8px;font-family:IBM Plex Mono,monospace'>Class Probabilities</div>")
                prob_out = gr.HTML()

        check_btn.click(
            fn=predict,
            inputs=[q_input, a_input, s_input],
            outputs=[verdict_out, answer_html_out, span_out, prob_out, type_out],
        )

    with gr.Tab("◈  Results"):
        gr.HTML("<div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#475569;margin:24px 0 20px;font-family:IBM Plex Mono,monospace'>Model Performance</div>")
        gr.HTML(value=build_comparison_table())
        gr.HTML("""
        <div style='font-family:"IBM Plex Mono",monospace;font-size:12px;color:#334155;
                    line-height:2;margin-top:20px;border-top:1px solid #1e293b;padding-top:16px'>
            <span style='color:#475569'>OVERALL</span> — Macro-averaged F1 across grounded/hallucinated classes<br>
            <span style='color:#475569'>EASY / MEDIUM / HARD</span> — MedHallu difficulty tiers based on semantic similarity to ground truth<br>
            <span style='color:#475569'>LABEL F1</span> — Binary classification performance saved at best checkpoint<br>
            <span style='color:#475569'>TYPE F1</span> — Hallucination type classification (4 categories)
        </div>
        """)

    with gr.Tab("◈  About"):
        gr.HTML("""
        <div style='font-family:"IBM Plex Mono",monospace;max-width:680px;padding:24px 0;line-height:2'>
            <div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#475569;margin-bottom:20px'>About This Project</div>

            <div style='color:#e2e8f0;font-size:14px;margin-bottom:24px'>
                Fine-tuned <strong style='color:#10b981'>DeBERTa-v3-base</strong> for binary medical hallucination detection,
                trained on the MedHallu benchmark — 10,000 QA pairs derived from PubMedQA.
            </div>

            <div style='border-left:2px solid #10b981;padding-left:16px;margin-bottom:24px'>
                <div style='color:#10b981;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Key Results</div>
                <div style='color:#94a3b8;font-size:13px'>
                    0.954 overall macro F1 on test split<br>
                    Beats GPT-4o on Easy / Medium / Hard<br>
                    Binary: grounded vs hallucinated
                </div>
            </div>

            <div style='border-left:2px solid #1e293b;padding-left:16px;margin-bottom:24px'>
                <div style='color:#475569;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Models</div>
                <div style='color:#64748b;font-size:13px'>
                    Classifier · microsoft/deberta-v3-base · binary + type heads<br>
                    Span extractor · deepset/deberta-v3-base-squad2<br>
                    Baseline · llama-3.1-8b-instant via Groq (zero-shot)
                </div>
            </div>

            <div style='border-left:2px solid #1e293b;padding-left:16px'>
                <div style='color:#475569;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Dataset</div>
                <div style='color:#64748b;font-size:13px'>
                    UTAustin-AIHealth/MedHallu · pqa_artificial split<br>
                    Train: rows 0–7000 · Val: 7000–8000 · Test: 8000–10000
                </div>
            </div>
        </div>
        """)

# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )
