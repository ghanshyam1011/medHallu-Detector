# prepare_annotation_data.py
# ============================================================
# WHAT : Formats 300 MedHallu rows for Label Studio upload
# FIX  : Uses correct Label Studio JSON format with 'data' key
# ============================================================

from datasets import load_dataset
import json
import os

print("Loading MedHallu...")
ds      = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
full_df = ds["train"].to_pandas()

full_df = full_df.rename(columns={
    "Question":            "question",
    "Knowledge":           "source",
    "Ground Truth":        "ground_truth",
    "Hallucinated Answer": "answer",
})

rows = full_df.head(300)

# WHY 'data' key:
# Label Studio requires each task to have a 'data' dict
# The field inside 'data' must match your labeling template
# We use 'text' as the field name — matches Named Entity Recognition
upload_data = []
for i, row in rows.iterrows():
    upload_data.append({
        "data": {
            "text":     str(row["answer"]),
            "question": str(row["question"]),
            "source":   str(row["source"])[:400],
        }
    })

os.makedirs("annotation_export", exist_ok=True)
output_path = "annotation_export/to_annotate.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(upload_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(upload_data)} rows to {output_path}")

# Verify
with open(output_path) as f:
    check = json.load(f)
print(f"Verification: {len(check)} tasks in file")
print(f"First task structure:")
print(json.dumps(check[0], indent=2)[:300])