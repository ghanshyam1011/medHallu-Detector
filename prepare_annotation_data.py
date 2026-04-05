from datasets import load_dataset
import json

dataset_repo = "UTAustin-AIHealth/MedHallu"
config_name = "pqa_artificial"

print(f"Connecting to Hugging Face to download {dataset_repo} ({config_name})...")

try:
    dataset = load_dataset(dataset_repo, config_name, split="train")
    subset = dataset.select(range(300))
    
    formatted_data = []
    
    for i, row in enumerate(subset):
        # We are now using the EXACT column names from the MedHallu dataset
        context_text = row.get("Knowledge", "")
        hallucinated_text = row.get("Hallucinated Answer", "")
        
        formatted_data.append({
            "id": i + 1,
            "context": context_text,
            "generated_answer": hallucinated_text
        })
        
    with open("medhallu_300.json", "w") as f:
        json.dump(formatted_data, f, indent=4)
        
    print("Success! Downloaded 300 rows and saved to medhallu_300.json")
    print("You can now run: python pre_annotate.py")

except Exception as e:
    print("Error downloading dataset.")
    print(f"Exact error: {e}")