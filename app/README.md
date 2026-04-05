# Person 3: Span Extraction & Gradio UI

This module is responsible for isolating the exact hallucinated phrase within an LLM-generated medical answer and displaying it in an interactive web UI.

### 🏆 Evaluation Results (Out-of-Distribution Test Split)
* **Exact Match (EM):** 64.86%
* **F1 Score:** 83.67%

*Note: An F1 score of 83.67% on a 300-row dataset demonstrates strong semantic understanding of the hallucination extraction task.*

### 🔑 Key Engineering Decisions
* **Pivot to RoBERTa:** Initial fine-tuning attempts using `microsoft/deberta-v3-base` resulted in gradient explosion (NaN loss) due to instability on small datasets. We pivoted to `deepset/roberta-base-squad2`, which provided mathematical stability and an excellent F1 score.
* **Context-as-Question Hack:** Instead of passing a generic string, we mapped the true Medical Facts to the QA `question` parameter, and the generated lie to the `context` parameter. This forces the model's cross-attention to strictly compare the two texts.
* **Raw PyTorch Inference:** The Gradio app bypasses the Hugging Face `pipeline` task registry to avoid known versioning bugs, utilizing manual tensor extraction for faster, more reliable inference.

### 🚀 Running the App
```bash
cd app
python app.py