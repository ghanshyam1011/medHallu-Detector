# ============================================================
# app.py
# ============================================================
# WHAT  : Gradio web app connecting all 3 models:
#           1. Person 2's DeBERTa  → verdict + type + confidence
#           2. Person 3's span model → which phrase is wrong
#           3. Person 1's results.json → comparison dashboard
#
# RUN   : python app.py
#         Opens at http://127.0.0.1:7860
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
)


# --- Paths ---------------------------------------------------
# Adjust these if your folder structure is different

CLASSIFIER_MODEL_NAME = "microsoft/deberta-v3-base"
CLASSIFIER_CHECKPOINT = "../person2_model/outputs/deberta_medhallu.pt"
SPAN_MODEL_DIR        = "../person3_spans/outputs/span_extractor_final"
RESULTS_JSON          = "../person2_model/outputs/results.json"


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

# Human readable descriptions shown in the UI
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
    "hallucinated":           "#dc2626",
    "partially_hallucinated": "#d97706",
    "grounded":               "#16a34a",
    "not_sure":               "#6b7280",
}

# GPT-4o scores from MedHallu paper — used in comparison table
GPT4O_SCORES = {
    "overall": 0.737,
    "easy":    0.844,
    "medium":  0.758,
    "hard":    0.625,
}


# ============================================================
# Classifier model class
# Must be identical to train_deberta.py
# ============================================================

class MedHalluModel(nn.Module):

    def __init__(self, model_name: str, dropout: float = 0.1):
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
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        return self.label_head(cls_output), self.type_head(cls_output)


# ============================================================
# Load all models at startup
# WHY at startup not inside predict():
# Loading takes 10-30 seconds. Loading once means only the
# first request is slow. All subsequent clicks are fast.
# ============================================================

print("=" * 55)
print("Loading models — this takes ~30 seconds on first run")
print("=" * 55)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# --- Load classifier (Person 2's model) ----------------------
print("\n[1/3] Loading hallucination classifier...")

clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
clf_model     = MedHalluModel(CLASSIFIER_MODEL_NAME)
clf_loaded    = False

if os.path.exists(CLASSIFIER_CHECKPOINT):
    try:
        checkpoint = torch.load(
            CLASSIFIER_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
        clf_model.load_state_dict(checkpoint["model_state"])
        clf_loaded = True
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Checkpoint label F1: {checkpoint.get('label_f1', '?')}")
    except Exception as e:
        print(f"  WARNING: Could not load checkpoint: {e}")
        print("  Using untrained model — predictions will be random")
else:
    print(f"  WARNING: checkpoint not found at {CLASSIFIER_CHECKPOINT}")
    print("  Run person2_model/train_deberta.py first")

clf_model = clf_model.to(device)
clf_model.eval()
print("  Classifier ready")


# --- Load span extractor (Person 3's model) ------------------
print("\n[2/3] Loading span extractor...")

span_tokenizer = None
span_model_hf  = None

if os.path.exists(SPAN_MODEL_DIR):
    try:
        span_tokenizer = AutoTokenizer.from_pretrained(SPAN_MODEL_DIR)
        span_model_hf  = AutoModelForQuestionAnswering.from_pretrained(
            SPAN_MODEL_DIR
        )
        span_model_hf  = span_model_hf.to(device)
        span_model_hf.eval()
        print("  Span extractor ready")
    except Exception as e:
        print(f"  WARNING: Could not load span model: {e}")
        print("  Span highlighting disabled")
else:
    print(f"  WARNING: span model not found at {SPAN_MODEL_DIR}")
    print("  Run person3_spans/train_span_extractor.py first")
    print("  Span highlighting will be disabled")


# --- Load results JSON (Person 1's comparison data) ----------
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
    print("  Run person2_model/evaluate.py first")

print("\nAll models loaded. Starting Gradio app...")


# ============================================================
# Core prediction functions
# ============================================================

def run_classifier(question: str, answer: str, source: str):
    """
    Runs Person 2's DeBERTa classifier.
    Returns verdict, confidence, hallucination type, all probs.
    """
    encoding = clf_tokenizer(
        source,
        f"{question} {answer}",
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        label_logits, type_logits = clf_model(input_ids, attention_mask)

    label_probs = torch.softmax(label_logits, dim=-1)[0]
    type_probs  = torch.softmax(type_logits,  dim=-1)[0]

    label_id   = label_probs.argmax().item()
    type_id    = type_probs.argmax().item()

    verdict    = LABEL_MAP_INV[label_id]
    confidence = label_probs[label_id].item()
    htype      = TYPE_MAP_INV[type_id]

    all_probs = {
        LABEL_MAP_INV[i]: round(label_probs[i].item(), 4)
        for i in range(len(LABEL_MAP))
    }

    return verdict, confidence, htype, all_probs


def run_span_extractor(answer: str) -> tuple[str, float]:
    """
    Runs Person 3's span extractor to find the wrong phrase.
    Returns (span_text, confidence_score).
    """
    if span_model_hf is None or span_tokenizer is None:
        return "", 0.0

    SPAN_QUESTION = "Which phrase is not supported by the source?"

    try:
        encoding = span_tokenizer(
            SPAN_QUESTION,
            answer,
            max_length=384,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets      = encoding["offset_mapping"][0].tolist()
        sequence_ids = encoding.sequence_ids(0)

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = span_model_hf(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Manual span extraction — pipeline is broken in newer transformers
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits   = outputs.end_logits[0].cpu().numpy()

        # Only consider context tokens (sequence_id == 1)
        ctx_indices = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not ctx_indices:
            return "", 0.0

        masked_start = np.full_like(start_logits, -np.inf)
        masked_end   = np.full_like(end_logits,   -np.inf)
        for i in ctx_indices:
            masked_start[i] = start_logits[i]
            masked_end[i]   = end_logits[i]

        best_score = -np.inf
        best_start = ctx_indices[0]
        best_end   = ctx_indices[0]

        for s in ctx_indices:
            for e in ctx_indices:
                if e < s or e - s + 1 > 50:
                    continue
                score = masked_start[s] + masked_end[e]
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end   = e

        char_start = offsets[best_start][0]
        char_end   = offsets[best_end][1]

        if char_start >= char_end:
            return "", 0.0

        span_text = answer[char_start:char_end].strip()

        # Convert logit score to rough confidence
        confidence = float(
            torch.softmax(outputs.start_logits[0], dim=-1)[best_start].item()
        )

        return span_text, confidence

    except Exception as e:
        print(f"Span extractor error: {e}")
        return "", 0.0


def highlight_span(answer: str, span: str) -> str:
    """
    Returns answer as HTML with wrong phrase highlighted in red.
    """
    if not span:
        return f"<div style='font-size:14px;line-height:1.9;padding:12px;border-radius:8px;border:1px solid #e5e7eb'>{answer}</div>"

    # Case-insensitive search
    lower_answer = answer.lower()
    lower_span   = span.lower()
    idx = lower_answer.find(lower_span)

    if idx == -1:
        return f"<div style='font-size:14px;line-height:1.9;padding:12px;border-radius:8px;border:1px solid #e5e7eb'>{answer}</div>"

    actual_span = answer[idx: idx + len(span)]
    highlighted = (
        answer[:idx] +
        f'<mark style="background:#fee2e2;color:#991b1b;'
        f'padding:2px 5px;border-radius:4px;'
        f'border:1px solid #fca5a5;font-weight:600">'
        f'{actual_span}</mark>' +
        answer[idx + len(span):]
    )

    return (
        f"<div style='font-size:14px;line-height:1.9;"
        f"padding:12px;border-radius:8px;"
        f"border:1px solid #e5e7eb'>{highlighted}</div>"
    )


# ============================================================
# Main predict function — called when user clicks Check
# ============================================================

def predict(question: str, answer: str, source: str):
    """
    Full pipeline: classifier → span extractor → HTML outputs.

    Returns 5 values mapping to 5 Gradio output components:
      1. verdict_html     → coloured verdict box
      2. answer_html      → answer with red highlight
      3. span_text        → the wrong phrase as plain text
      4. prob_html        → probability breakdown bars
      5. type_description → type explanation markdown
    """
    # Input validation
    if not question.strip():
        return (
            "<p style='color:#6b7280;padding:10px'>Please enter a question.</p>",
            "", "", "", ""
        )
    if not answer.strip():
        return (
            "<p style='color:#6b7280;padding:10px'>Please enter an answer to check.</p>",
            "", "", "", ""
        )
    if not source.strip():
        return (
            "<p style='color:#6b7280;padding:10px'>Please enter a source text.</p>",
            "", "", "", ""
        )

    # Step 1 — run classifier
    verdict, confidence, htype, all_probs = run_classifier(
        question, answer, source
    )

    # Step 2 — run span extractor only on hallucinated answers
    # WHY skip grounded: no wrong phrase exists in grounded answers
    span_text  = ""
    if verdict in ("hallucinated", "partially_hallucinated"):
        span_text, _ = run_span_extractor(answer)

    # Step 3 — build verdict box HTML
    color         = VERDICT_COLORS.get(verdict, "#6b7280")
    verdict_label = verdict.replace("_", " ").upper()
    conf_pct      = f"{confidence * 100:.1f}%"
    type_display  = htype.replace("_", " ").title()

    verdict_html = f"""
    <div style='padding:16px 20px;border-radius:10px;
                border:2px solid {color};background:{color}18;
                margin-bottom:4px'>
        <div style='font-size:22px;font-weight:700;color:{color}'>
            {verdict_label}
        </div>
        <div style='font-size:14px;color:#374151;margin-top:6px'>
            Confidence: <strong>{conf_pct}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Type: <strong>{type_display}</strong>
        </div>
    </div>
    """

    # Step 4 — highlighted answer HTML
    answer_html = highlight_span(answer, span_text)

    # Step 5 — span text for plain textbox
    span_display = span_text if span_text else "— (no specific phrase identified)"

    # Step 6 — probability breakdown bars
    bars_html = "<div style='font-size:13px'>"
    bars_html += "<div style='font-weight:500;margin-bottom:8px;color:#374151'>Class probabilities</div>"

    for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        pct        = prob * 100
        bar_color  = VERDICT_COLORS.get(label, "#6b7280")
        label_disp = label.replace("_", " ").title()
        bars_html += f"""
        <div style='margin-bottom:7px'>
            <div style='display:flex;justify-content:space-between;
                        margin-bottom:3px;font-size:12px'>
                <span style='color:#374151'>{label_disp}</span>
                <span style='color:#6b7280'>{pct:.1f}%</span>
            </div>
            <div style='background:#f3f4f6;border-radius:4px;height:8px'>
                <div style='background:{bar_color};border-radius:4px;
                            height:8px;width:{pct:.1f}%'></div>
            </div>
        </div>
        """
    bars_html += "</div>"

    # Step 7 — type description
    type_desc = TYPE_DESCRIPTIONS.get(htype, "")
    type_md   = f"**{type_display}** — {type_desc}" if type_desc else ""

    return verdict_html, answer_html, span_display, bars_html, type_md


# ============================================================
# Comparison table builder
# ============================================================

def build_comparison_table() -> str:
    """Builds HTML comparison table from results.json."""
    if results_data is None:
        return "<p style='color:#6b7280'>results.json not found. Run evaluate.py first.</p>"

    our   = results_data.get("medhallu_validation", {})
    groq  = results_data.get("groq_baseline", {})

    def fmt(val):
        if val is None:
            return "—"
        try:
            return f"{float(val):.3f}"
        except (ValueError, TypeError):
            return "—"

    def table_row(split, our_key, groq_key, gpt_val):
        our_val  = fmt(our.get(our_key))
        groq_val = fmt(groq.get(groq_key))
        gpt_str  = f"{gpt_val:.3f}"

        our_style = ""
        try:
            if float(our_val) > gpt_val:
                our_style = "color:#16a34a;font-weight:700"
        except (ValueError, TypeError):
            pass

        return f"""
        <tr>
            <td style='padding:9px 14px;border-bottom:1px solid #e5e7eb'>{split}</td>
            <td style='padding:9px 14px;border-bottom:1px solid #e5e7eb;text-align:center'>{gpt_str}</td>
            <td style='padding:9px 14px;border-bottom:1px solid #e5e7eb;text-align:center'>{groq_val}</td>
            <td style='padding:9px 14px;border-bottom:1px solid #e5e7eb;text-align:center;{our_style}'>{our_val}</td>
        </tr>
        """

    html = """
    <table style='width:100%;border-collapse:collapse;font-size:14px'>
        <thead>
            <tr style='background:#f9fafb'>
                <th style='padding:10px 14px;text-align:left;border-bottom:2px solid #e5e7eb'>Split</th>
                <th style='padding:10px 14px;text-align:center;border-bottom:2px solid #e5e7eb'>GPT-4o (paper)</th>
                <th style='padding:10px 14px;text-align:center;border-bottom:2px solid #e5e7eb'>Groq Llama baseline</th>
                <th style='padding:10px 14px;text-align:center;border-bottom:2px solid #e5e7eb'>Our DeBERTa</th>
            </tr>
        </thead>
        <tbody>
    """

    html += table_row("Overall", "overall_f1", "overall_f1", GPT4O_SCORES["overall"])
    html += table_row("Easy",    "easy_f1",    "easy_f1",    GPT4O_SCORES["easy"])
    html += table_row("Medium",  "medium_f1",  "medium_f1",  GPT4O_SCORES["medium"])
    html += table_row("Hard",    "hard_f1",    "hard_f1",    GPT4O_SCORES["hard"])

    html += "</tbody></table>"
    html += "<p style='font-size:12px;color:#6b7280;margin-top:8px'>Green = beats GPT-4o</p>"

    # OOD generalization row
    ood = results_data.get("ood_generalization")
    if ood and ood.get("overall_f1") is not None:
        ood_f1 = fmt(ood["overall_f1"])
        html += f"""
        <div style='margin-top:16px;padding:12px 16px;
                    background:#f0fdf4;border-radius:8px;
                    border:1px solid #bbf7d0'>
            <div style='font-weight:600;color:#15803d;font-size:14px'>
                Generalization — OOD MedQA test
            </div>
            <div style='font-size:13px;color:#374151;margin-top:4px'>
                F1 on 200 unseen MedQA rows (no retraining):
                <strong style='color:#15803d'>{ood_f1}</strong>
            </div>
        </div>
        """

    return html


# ============================================================
# Example inputs
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
        "Which part of the nephron is responsible for glucose reabsorption?",
        "Glucose reabsorption occurs primarily in the proximal convoluted tubule via sodium-glucose cotransporters (SGLT2).",
        "In the nephron, approximately 90% of filtered glucose is reabsorbed in the proximal convoluted tubule through SGLT2 transporters, with the remaining 10% reabsorbed via SGLT1.",
    ],
]


# ============================================================
# Build Gradio interface
# ============================================================

with gr.Blocks(
    title="MedHallu Detector",
    theme=gr.themes.Default(
        primary_hue="blue",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as app:

    gr.Markdown("""
    # Medical Hallucination Detector
    Detects hallucinations in medical LLM outputs using fine-tuned DeBERTa.
    Shows verdict, hallucination type, confidence, and the exact wrong phrase highlighted in red.
    """)

    # --- Tab 1: Detector ---
    with gr.Tab("Hallucination Detector"):

        with gr.Row():

            # Left — inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                q_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What enzyme is deficient in PKU?",
                    lines=2,
                )
                a_input = gr.Textbox(
                    label="LLM Answer to check",
                    placeholder="Paste the AI-generated answer here...",
                    lines=4,
                )
                s_input = gr.Textbox(
                    label="Source / Ground truth context",
                    placeholder="Paste the source text the answer should be grounded in...",
                    lines=4,
                )
                check_btn = gr.Button(
                    "Check for Hallucination",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown("**Try an example:**")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[q_input, a_input, s_input],
                    label="",
                )

            # Right — outputs
            with gr.Column(scale=1):
                gr.Markdown("### Results")

                verdict_out = gr.HTML(
                    label="Verdict",
                    value="<p style='color:#9ca3af;font-size:14px;padding:10px'>Results will appear here after clicking Check.</p>",
                )

                gr.Markdown("**Answer with highlighted wrong phrase:**")
                answer_html_out = gr.HTML()

                span_out = gr.Textbox(
                    label="Wrong phrase identified",
                    interactive=False,
                )
                type_out    = gr.Markdown()
                prob_out    = gr.HTML(label="Probability breakdown")

        # Wire button
        check_btn.click(
            fn=predict,
            inputs=[q_input, a_input, s_input],
            outputs=[verdict_out, answer_html_out, span_out, prob_out, type_out],
        )

    # --- Tab 2: Comparison Dashboard ---
    with gr.Tab("Model Comparison"):
        gr.Markdown("""
        ### Results comparison
        Macro F1 scores on MedHallu validation set.
        Our fine-tuned DeBERTa vs Groq Llama-3.1 baseline vs GPT-4o (from paper).
        """)
        gr.HTML(value=build_comparison_table())
        gr.Markdown("""
        **Overall F1** — macro-averaged across all label classes.
        **Easy / Medium / Hard** — MedHallu difficulty tiers. Hard rows are semantically closest to ground truth.
        **OOD generalization** — F1 on 200 MedQA rows the model was never trained on.
        """)

    # --- Tab 3: About ---
    with gr.Tab("About"):
        gr.Markdown("""
        ### About this project

        This tool uses a fine-tuned **DeBERTa-v3-base** model trained on the
        [MedHallu benchmark](https://medhallu.github.io/) — 10,000 medical QA pairs
        derived from PubMedQA.

        **Three unique contributions:**
        1. Multi-task learning — one model predicts hallucination label AND type simultaneously
        2. Generalization — evaluated on out-of-distribution MedQA data without retraining
        3. Span-level detection — identifies the exact wrong phrase, not just a binary verdict

        **Models used:**
        - Classifier: `microsoft/deberta-v3-base` + two linear heads (label + type)
        - Span extractor: `deepset/deberta-v3-base-squad2` + QA head fine-tuned on medical spans
        - Baseline: `llama-3.1-8b-instant` via Groq free API (zero-shot)

        **Dataset:** [UTAustin-AIHealth/MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)
        """)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )