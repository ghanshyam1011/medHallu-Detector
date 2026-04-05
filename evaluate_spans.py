import json
import torch
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

print("Loading Test Data and RoBERTa Model...")

# 1. Load the SQuAD Evaluation Metric
squad_metric = evaluate.load("squad_v2")

# 2. Setup Device (Use RTX GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on: {device}")

# 3. Load the raw architecture directly to the GPU
tokenizer = AutoTokenizer.from_pretrained("./final_span_model")
model = AutoModelForQuestionAnswering.from_pretrained("./final_span_model").to(device)

# 4. Load the data
with open("standard_spans.json", "r", encoding="utf-8") as f:
    ls_data = json.load(f)

references = []
predictions = []

print("Running predictions on validation rows...")
for i, item in enumerate(ls_data):
    if 'annotations' not in item or len(item['annotations']) == 0:
        continue
        
    result = item['annotations'][0].get('result', [])
    if len(result) == 0:
        continue

    # Extract data
    ref_context = item['data']['context']
    gen_answer = item['data']['text']
    true_val = result[0]['value']['text']
    
    # --- RAW PYTORCH INFERENCE ---
    try:
        inputs = tokenizer(
            text=ref_context,
            text_pair=gen_answer,
            return_tensors="pt",
            max_length=512,
            truncation="only_first"
        ).to(device) # Send inputs to GPU
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits)
        
        # Translate tokens back to text
        predict_answer_tokens = inputs.input_ids[0, start_index : end_index + 1]
        pred_text = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        
        if pred_text.strip() == "" or pred_text == "<s>":
            pred_text = ""

    except Exception:
         pred_text = ""

    # Format for the SQuAD Evaluator
    uid = str(i)
    
    # Reference (Ground Truth)
    ans_dict = {
        "text": [true_val], 
        "answer_start": [result[0]['value']['start']]
    }
    references.append({"id": uid, "answers": ans_dict})
    
    # Prediction (Model's Guess)
    predictions.append({"id": uid, "prediction_text": pred_text, "no_answer_probability": 0.0})

# 5. Calculate the Final Scores
print("\nCalculating SQuAD v2 Metrics...")
results = squad_metric.compute(predictions=predictions, references=references)

print("\n" + "="*40)
print("🏆 FINAL EXTRACTOR SCORES 🏆")
print("="*40)
print(f"Exact Match (EM): {results['exact']:.2f}%")
print(f"F1 Score:         {results['f1']:.2f}%")
print("="*40)