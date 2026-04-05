import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer
)

print("Loading Standard data...")
# Make sure your file name matches here!
with open("standard_spans.json", "r", encoding="utf-8") as f:
    ls_data = json.load(f)

formatted_data = {"id": [], "question": [], "context": [], "answers": []}

for item in ls_data:
    ref_context = item['data']['context']
    gen_answer = item['data']['text']
    
    if 'annotations' in item and len(item['annotations']) > 0:
        result = item['annotations'][0].get('result', [])
        
        if len(result) > 0:
            val = result[0]['value']
            
            formatted_data["id"].append(str(item['data'].get('id', len(formatted_data["id"]))))
            formatted_data["question"].append(ref_context)
            formatted_data["context"].append(gen_answer)
            formatted_data["answers"].append({
                "answer_start": [val['start']],
                "text": [val['text']]
            })

hf_dataset = Dataset.from_dict(formatted_data)
print(f"Successfully loaded {len(hf_dataset)} annotated examples.")

print("Downloading DeBERTa model...")
model_name = "microsoft/deberta-v3-base"

# --- THE FIX: use_fast=False stops the local Windows crashing bug ---
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# --- THE BULLETPROOF COLAB FIXES ---
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation="only_first", # Fixed truncation
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Safety Net Loop
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        if context_start >= len(sequence_ids):
            start_positions.append(0)
            end_positions.append(0)
            continue

        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print("Tokenizing data...")
tokenized_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=hf_dataset.column_names)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# --- STABILIZED LOCAL HARDWARE SETTINGS (RTX 3050) ---
training_args = TrainingArguments(
    output_dir="./deberta-span-extractor",
    eval_strategy="epoch",          # Updated Hugging Face naming
    learning_rate=1e-5,             # Safe learning rate
    per_device_train_batch_size=2,  # Keep this at 2 to protect your 4GB VRAM!
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  
    num_train_epochs=3,             
    weight_decay=0.01,
    fp16=True,                      # Must be True for local RTX 3050 memory limits
    save_strategy="epoch",
    logging_steps=10,
    max_grad_norm=1.0,              # Stops math explosions
    warmup_ratio=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    processing_class=tokenizer,     # Updated Hugging Face naming
)

print("Starting training! Let your GPU do the heavy lifting...")
trainer.train()

print("Training complete! Saving your custom model...")
trainer.save_model("./final_span_model")
print("Model saved to the /final_span_model directory! Ready for the UI.")