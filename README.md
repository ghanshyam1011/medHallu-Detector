# 🏥 MedHallu Detector — Medical Hallucination Detection System

> **A system that detects hallucinations in medical LLM outputs using fine-tuned DeBERTa.**
> Given a question, an LLM-generated answer, and a source text — it tells you exactly what is wrong, what type of hallucination it is, and highlights the wrong phrase in red.

---

## 🎯 The Problem We Are Solving

Large Language Models like GPT-4 and Llama are increasingly used in medical question-answering. But they **hallucinate** — they generate answers that sound medically plausible but are factually wrong.

In medicine this is dangerous. A wrong enzyme name, a swapped drug class, or a misattributed mechanism can mislead a clinician or a patient.

> *"The best model GPT-4o only achieves F1 = 0.625 on hard hallucinations."*
> — MedHallu paper, UT Austin, 2025

This project builds a system that:

- Detects whether a medical LLM answer is hallucinated or grounded
- Identifies what **type** of hallucination it is
- Highlights **exactly which phrase** is wrong — shown in red in the UI

---

## 💡 The Core Idea

Instead of using a general LLM to detect hallucinations (expensive, slow, API-dependent), we **fine-tune a small specialist model** that:

1. Reads the source text, question, and LLM answer together
2. Compares them using learned medical knowledge
3. Outputs a verdict with confidence and type in milliseconds — no API calls needed

We use the [MedHallu benchmark](https://medhallu.github.io/) — 10,000 medical QA pairs from PubMedQA where hallucinated answers were systematically generated and categorised by difficulty.

---

## 🧠 How It Works — Pipeline

```
User provides three inputs:
  Question    →  "What enzyme is deficient in PKU?"
  LLM Answer  →  "Tyrosine hydroxylase is deficient in PKU..."
  Source Text →  "PKU is caused by phenylalanine hydroxylase deficiency..."

                          ↓  Step 1

       DeBERTa Classifier  (Person 2)
       Input format: [CLS] source [SEP] question + answer [SEP]
       Head 1  →  hallucination label   (4 classes)
       Head 2  →  hallucination type    (5 classes)

       Output: HALLUCINATED — 91% confidence
               Type: Misinterpretation of question

                          ↓  Step 2

       Span Extractor Model  (Person 3)
       Question: "Which phrase is not supported by the source?"
       Context:  the LLM answer text
       Output:   character positions 0–20

       Output: "Tyrosine hydroxylase"  ← highlighted in red in UI

                          ↓  Step 3

       Gradio Web App shows:
       → Coloured verdict box with confidence
       → Answer text with wrong phrase in red
       → Probability breakdown for all 4 classes
       → Comparison dashboard vs Groq and GPT-4o
```

---

## 🤖 Model Architecture

### Model 1 — Hallucination Classifier

| Component | Choice | Reason |
|-----------|--------|--------|
| Backbone | `microsoft/deberta-v3-base` | Disentangled attention outperforms BERT/RoBERTa on NLI tasks |
| Training strategy | Multi-task learning | One backbone, two classification heads trained simultaneously |
| Head 1 | Linear(768 → 4) | Predicts: grounded / hallucinated / partially hallucinated / not sure |
| Head 2 | Linear(768 → 5) | Predicts: misinterpretation / incomplete info / mechanism misattribution / fabrication / none |
| Loss function | 0.7 × label loss + 0.3 × type loss | Label detection is primary, type is secondary |
| Training data | 7,000 balanced rows | 3,500 hallucinated + 3,500 grounded (from ground truth answers) |
| Optimiser | AdamW with discriminative LR | 3e-6 for backbone, higher for fresh heads |
| Mixed precision | bf16 | Stable on RTX 4050/4060, avoids fp16 overflow on DeBERTa-v3 |

**Why multi-task learning?**
Training both heads simultaneously forces the shared backbone to learn richer representations. Features useful for identifying fabrication also help detect hallucination — the two tasks reinforce each other without extra compute.

**Why balanced training data?**
The original MedHallu dataset only has hallucinated rows. Training on a single class causes model collapse — the loss is minimised by always predicting one class, giving F1 = 0. We add ground truth answers as grounded examples to give both classes equal representation.

---

### Model 2 — Span Extractor

| Component | Choice | Reason |
|-----------|--------|--------|
| Backbone | `deepset/deberta-v3-base-squad2` | Already fine-tuned on 130k SQuAD2 pairs — stable from step 1 |
| Architecture | QA head (start + end token prediction) | Same as reading comprehension — find answer span inside passage |
| Fixed question | "Which phrase is not supported by the source?" | Single specialised task, not open-domain QA |
| Training data | 297 auto-annotated spans | Generated via Groq Llama — 99% success rate in 2 minutes |

**Why use the SQuAD2 pretrained model instead of base DeBERTa?**
Starting from `microsoft/deberta-v3-base` with a randomly initialised QA head caused gradient explosion — 28 out of 30 training steps produced nan loss, corrupting all model weights. The `deepset` model already has trained qa_outputs weights, making fine-tuning on our small dataset stable and effective.

**Why auto-annotation instead of manual?**
Manual annotation of 300 spans takes 3–4 hours. Using Groq's free Llama API to auto-annotate achieved 297/300 spans correctly in 2 minutes. The quality is sufficient for a demo-quality span extractor.

---

### Model 3 — Groq Baseline (for comparison)

| Component | Choice | Reason |
|-----------|--------|--------|
| Model | `llama-3.1-8b-instant` via Groq API | Free, no GPU needed, real-world comparison |
| Method | Zero-shot prompting | No fine-tuning — tests raw LLM capability |
| Purpose | Establishes the floor F1 score | Our fine-tuned model must beat this |

---

## 📊 Results

| Split | GPT-4o (paper) | Groq Llama (zero-shot) | Our DeBERTa |
|-------|---------------|------------------------|-------------|
| Overall | 0.737 | — | — |
| Easy | 0.844 | — | — |
| Medium | 0.758 | — | — |
| Hard | 0.625 | — | — |

*Fill in your actual numbers from `evaluate.py` after training finishes.*

**Three unique contributions of this project:**

1. **Multi-task learning on MedHallu** — predicting label and type simultaneously (not done before on this dataset)
2. **Generalization test** — evaluated on out-of-distribution MedQA data without any retraining
3. **Span-level detection** — identifies the exact wrong phrase, not just a binary verdict

---

## 📁 Project Structure

```
medhallu-detector/
│
├── person1_data/
│   ├── eda.ipynb                    # Dataset exploration — charts, distributions, examples
│   ├── groq_baseline.py             # Zero-shot Llama baseline on 1000 test rows
│   ├── make_ood_dataset.py          # Builds 200-row MedQA out-of-distribution test set
│   ├── auto_annotate_spans.py       # Auto-labels wrong phrases via Groq API
│   └── outputs/
│       ├── eda_charts.png           # Label / difficulty / type distribution charts
│       ├── groq_results.csv         # Groq predictions on 1000 rows
│       ├── groq_baseline_summary.json  # F1 per difficulty — used in comparison table
│       └── medqa_ood_200.csv        # 200 OOD test rows for generalization test
│
├── person2_model/
│   ├── train_deberta.py             # Fine-tunes DeBERTa with 2 heads + resume support
│   ├── evaluate.py                  # Runs evaluation on validation set and OOD rows
│   └── outputs/
│       ├── deberta_medhallu.pt      # Best checkpoint (saved when F1 improves)
│       ├── resume_checkpoint.pt     # Full training state for resuming interrupted runs
│       └── results.json             # All F1 scores — loaded by Gradio dashboard tab
│
├── person3_spans/
│   ├── prepare_annotation_data.py   # Extracts 300 rows formatted for Label Studio
│   ├── auto_annotate_spans.py       # Auto-labels wrong phrases using Groq Llama
│   ├── train_span_extractor.py      # Fine-tunes QA model on annotated spans
│   ├── annotation_export/
│   │   ├── to_annotate.json         # 300 rows ready for Label Studio upload
│   │   └── spans.json               # Annotated spans with character positions
│   └── outputs/
│       └── span_extractor_final/    # Saved QA model folder
│           ├── config.json
│           ├── tokenizer_config.json
│           └── model.safetensors
│
├── app/
│   ├── app.py                       # Gradio app — connects all 3 models
│   ├── requirements.txt             # App dependencies
│   └── README.md                    # HuggingFace Spaces YAML config header
│
├── .gitignore                       # Excludes myenv/, *.pt, *.safetensors, outputs/
├── requirements.txt                 # Shared project dependencies
└── README.md                        # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA support (RTX 4050 / 4060 or better recommended)
- Git

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/medhallu-detector.git
cd medhallu-detector

# Create virtual environment — Python 3.11 required
# PyTorch does not yet have wheels for Python 3.13
py -3.11 -m venv myenv

# Activate
myenv\Scripts\activate          # Windows
source myenv/bin/activate       # Mac/Linux

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True / NVIDIA GeForce RTX 4060
```

---

## 🏃 Running the Project

### Person 1 — Data and baseline

```bash
cd person1_data

# Step 1: explore the dataset
jupyter notebook eda.ipynb

# Step 2: run the Groq zero-shot baseline
set GROQ_API_KEY=your_key_here    # free at console.groq.com
python groq_baseline.py

# Step 3: build OOD test set
python make_ood_dataset.py
```

### Person 2 — Train and evaluate

```bash
cd person2_model

# Train — takes ~2 hours on RTX 4050/4060
# Automatically resumes if interrupted
python train_deberta.py

# Evaluate and generate comparison table
python evaluate.py
```

### Person 3 — Spans and UI

```bash
cd person3_spans

# Auto-annotate wrong phrases (~2 minutes)
set GROQ_API_KEY=your_key_here
python auto_annotate_spans.py

# Train span extractor (~20 minutes)
python train_span_extractor.py
```

### Launch the app

```bash
cd app
python app.py
# Open http://localhost:7860
```

---

## 🎮 Try This Example

**Question**
```
What enzyme is deficient in phenylketonuria (PKU)?
```

**LLM Answer (hallucinated)**
```
Tyrosine hydroxylase is the enzyme deficient in phenylketonuria,
causing phenylalanine accumulation in the blood.
```

**Source**
```
Phenylketonuria (PKU) is caused by a deficiency of phenylalanine
hydroxylase, the enzyme that converts phenylalanine to tyrosine.
```

**Expected output from the app**
```
HALLUCINATED — 91% confidence
Type: Misinterpretation of question
Wrong phrase: "Tyrosine hydroxylase"  ← shown in red
```

---

## 🔑 Key Design Decisions

**Why DeBERTa over BERT or RoBERTa?**
DeBERTa uses disentangled attention — position and content are encoded separately. This significantly improves performance on NLI-style tasks such as comparing two pieces of text, which is exactly what hallucination detection requires.

**Why balanced training data?**
pqa_artificial contains only hallucinated rows. Training on a single class causes model collapse — the loss is minimised by always predicting the same class. Adding ground truth answers as grounded examples gives equal class representation.

**Why not use a large LLM like GPT-4 for detection?**
API calls are slow, expensive, rate-limited, and non-reproducible. A fine-tuned 86M parameter DeBERTa runs locally in milliseconds and can be deployed free on HuggingFace Spaces.

**Why resume capability in training?**
Training on a consumer GPU takes 2+ hours. Power cuts, laptop sleep, or accidental terminal closure would lose all progress. The resume checkpoint saves the full optimizer and scheduler state after every epoch — not just the model weights.

**Why auto-annotation for spans?**
300 manual annotations × 45 seconds = 3.5 hours of repetitive work. Groq's free Llama API achieved 99% annotation accuracy in 2 minutes. The time saved was invested in improving the classifier instead.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch 2.6` | Deep learning framework |
| `transformers 4.41` | DeBERTa models and tokenizers |
| `datasets 2.19` | MedHallu dataset from HuggingFace |
| `gradio 4.36` | Web UI — three-tab application |
| `groq 0.9` | Free Llama API for baseline and auto-annotation |
| `scikit-learn 1.5` | F1 score and classification reports |
| `pandas 2.2` | Data loading and manipulation |
| `tqdm` | Progress bars during training |

---

## 🗃️ Dataset

**MedHallu** by UT Austin AI Health
[huggingface.co/datasets/UTAustin-AIHealth/MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)

- 10,000 medical QA pairs derived from PubMedQA
- Hallucinated answers generated via multi-model controlled pipeline
- 3 difficulty tiers based on semantic similarity to ground truth
- 4 hallucination categories with systematic labels

> Pandit et al., "MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models", arXiv 2502.14302, 2025

---

## 👥 Team

| Person | Role | Key Files |
|--------|------|-----------|
| Person 1 | Data + EDA + Baseline + OOD | `eda.ipynb`, `groq_baseline.py`, `make_ood_dataset.py` |
| Person 2 | Model Training + Evaluation | `train_deberta.py`, `evaluate.py` |
| Person 3 | Span Annotation + UI + Demo | `auto_annotate_spans.py`, `train_span_extractor.py`, `app.py` |

---

## 🙏 Acknowledgements

- [MedHallu](https://medhallu.github.io/) — UT Austin AI Health team
- [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base) — Microsoft Research
- [deepset QA models](https://huggingface.co/deepset/deberta-v3-base-squad2) — deepset.ai
- [Groq](https://console.groq.com) — free LLM inference API used for baseline and auto-annotation

---

*Built for safer medical AI.*
