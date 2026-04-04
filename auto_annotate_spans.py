import json
import os
import sys
import time
from groq import Groq
from dotenv import load_dotenv

# 1. Load the current API key from your .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Loading raw data from medhallu_300.json...")
with open("medhallu_300.json", "r", encoding="utf-8") as file:
    raw_data = json.load(file)

label_studio_format = []
output_filename = "pre_annotated_data.json"

# 2. THE RESUME ENGINE: Check if we have saved data
if os.path.exists(output_filename):
    with open(output_filename, "r", encoding="utf-8") as f:
        try:
            label_studio_format = json.load(f)
            print(f"Found saved progress! Automatically resuming safely...")
        except json.JSONDecodeError:
            print("No valid previous data found, starting fresh.")
            label_studio_format = []

# Calculate exactly where we left off based on how many items are saved
start_index = len(label_studio_format)

if start_index >= len(raw_data):
    print("All 300 rows are already processed! You are done.")
    sys.exit(0)

print(f"Starting LLM Annotation at Row {start_index + 1} (Using 70B Model)...")

# 3. MAIN LOOP: Start from where we left off
for i in range(start_index, len(raw_data)):
    item = raw_data[i]
    
    # Format the context correctly
    if isinstance(item['context'], list):
        clean_context = " ".join(item['context'])
    else:
        clean_context = item['context']
        
    answer = item['generated_answer']

    prompt = f"""
    You are a strict medical data annotator. 
    Context: {clean_context}
    Generated Answer: {answer}
    
    Task: The Generated Answer contains a mix of true facts and fabricated hallucinations. 
    Identify ONLY the specific sentence or phrase that is factually incorrect based on the Context.
    
    CRITICAL INSTRUCTIONS:
    1. Return the SHORTEST possible exact substring from the Generated Answer.
    2. Do NOT include any sentences that are actually true or supported by the context. 
    3. Return ONLY the exact substring. Do not add quotes, introductions, or explanations.
    """
    
    success = False
    while not success:
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            hallucinated_substring = completion.choices[0].message.content.strip()
            success = True 
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print("\n" + "="*60)
                print("🛑 RATE LIMIT HIT! 🛑")
                print("Your progress is 100% saved.")
                print("1. Go to your .env file and paste a new API key.")
                print("2. Run `python pre_annotate.py` again to resume instantly.")
                print("="*60)
                sys.exit(0) # Safely exit the script so you can swap keys
            else:
                print(f"\n[ERROR] API error on Row {i+1}: {e}")
                # We save a blank response so it doesn't get stuck in an infinite loop
                hallucinated_substring = ""
                success = True 
    
    start_idx = answer.find(hallucinated_substring)
    
    if start_idx != -1 and hallucinated_substring != "":
        end_idx = start_idx + len(hallucinated_substring)
        
        ls_entry = {
            "data": {
                "id": item['id'],
                "text": answer,
                "context": clean_context
            },
            "predictions": [{
                "model_version": "llama-3.3-70b",
                "result": [{
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": start_idx,
                        "end": end_idx,
                        "text": hallucinated_substring,
                        "labels": ["Hallucination_Span"]
                    }
                }]
            }]
        }
    else:
        # If the LLM failed to find a match, still save the structure so Label Studio can read it
        ls_entry = {
            "data": {"id": item['id'], "text": answer, "context": clean_context},
            "predictions": []
        }
        print(f"  -> Failed exact match for Row {i+1}, but row saved. Needs manual review.")

    label_studio_format.append(ls_entry)
    print(f"Successfully processed Row {i+1}/300")

    # 4. THE AUTO-SAVE: Write the file to your hard drive immediately
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(label_studio_format, f, indent=4)

    # 5. Delay to prevent hitting the Requests Per Minute limit
    time.sleep(4) 

print("\n🎉 Finished processing all 300 rows! Ready for Label Studio.")