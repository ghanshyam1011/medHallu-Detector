# import gradio as gr
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# print("Loading custom RoBERTa model... (This takes a few seconds)")
# try:
#     # 1. Load the raw architecture
#     tokenizer = AutoTokenizer.from_pretrained("./final_span_model")
#     model = AutoModelForQuestionAnswering.from_pretrained("./final_span_model")
#     print("Model loaded successfully! No pipelines needed.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     tokenizer, model = None, None

# # 2. Raw PyTorch Inference
# def catch_hallucination(context, generated_answer):
#     if model is None:
#         return "<p style='color: red;'>Error: Model not loaded correctly. Check your folder path!</p>"
        
#     try:
#         inputs = tokenizer(
#             text=context,               
#             text_pair=generated_answer, 
#             return_tensors="pt",        
#             max_length=512,
#             truncation="only_first"     
#         )
        
#         with torch.no_grad():
#             outputs = model(**inputs)
            
#         start_index = torch.argmax(outputs.start_logits)
#         end_index = torch.argmax(outputs.end_logits)
        
#         start_probs = torch.nn.functional.softmax(outputs.start_logits, dim=-1)[0]
#         end_probs = torch.nn.functional.softmax(outputs.end_logits, dim=-1)[0]
#         confidence = round((start_probs[start_index] * end_probs[end_index]).item() * 100, 2)
        
#         predict_answer_tokens = inputs.input_ids[0, start_index : end_index + 1]
#         highlighted_text = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        
#         if highlighted_text.strip() == "" or highlighted_text == "<s>":
#             return "<div style='padding: 15px; background-color: #d1fae5; color: #065f46; border-radius: 8px;'><b>✅ No hallucination detected.</b> The text appears fully supported by the facts.</div>"
            
#         # --- THE RED HIGHLIGHT HACK ---
#         # We replace the extracted text inside the original answer with a red HTML span
#         red_span = f"<span style='background-color: #fca5a5; color: #7f1d1d; font-weight: bold; padding: 2px 4px; border-radius: 4px;'>{highlighted_text}</span>"
#         visually_highlighted_answer = generated_answer.replace(highlighted_text, red_span)
        
#         # Format the final output as a nice HTML box
#         html_output = f"""
#         <div style='padding: 15px; border: 1px solid #f87171; background-color: #fef2f2; border-radius: 8px; font-family: sans-serif;'>
#             <h3 style='color: #b91c1c; margin-top: 0;'>⚠️ Hallucination Found (Confidence: {confidence}%)</h3>
#             <p style='font-size: 16px; line-height: 1.5; color: #1f2937;'>{visually_highlighted_answer}</p>
#         </div>
#         """
#         return html_output
        
#     except Exception as e:
#         return f"<p style='color: red;'>Prediction Error: {str(e)}</p>"

# # 3. The Gradio Interface
# demo = gr.Interface(
#     fn=catch_hallucination,
#     inputs=[
#         gr.Textbox(lines=6, label="1. Ground Truth Medical Context (The Facts)"),
#         gr.Textbox(lines=4, label="2. LLM Generated Answer (To be checked)")
#     ],
#     # CHANGE: We swap gr.Textbox for gr.HTML so it renders the red colors!
#     outputs=gr.HTML(label="3. Visual Hallucination Highlight"),
#     title="MedHallu: Automated Hallucination Catcher",
#     description="Paste a medical abstract and an LLM-generated answer below. Your custom RoBERTa model will scan the text and highlight fabricated information in red.",
#     flagging_mode="never"
# )

# if __name__ == "__main__":
#     demo.launch()


import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

print("Loading custom RoBERTa model... (This takes a few seconds)")
try:
    # 1. Load the raw architecture
    tokenizer = AutoTokenizer.from_pretrained("./final_span_model")
    model = AutoModelForQuestionAnswering.from_pretrained("./final_span_model")
    print("Model loaded successfully! No pipelines needed.")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer, model = None, None

# 2. Raw PyTorch Inference
def catch_hallucination(context, generated_answer):
    if model is None:
        return "<p style='color: red;'>Error: Model not loaded correctly. Check your folder path!</p>"
        
    try:
        inputs = tokenizer(
            text=context,               
            text_pair=generated_answer, 
            return_tensors="pt",        
            max_length=512,
            truncation="only_first"     
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits)
        
        start_probs = torch.nn.functional.softmax(outputs.start_logits, dim=-1)[0]
        end_probs = torch.nn.functional.softmax(outputs.end_logits, dim=-1)[0]
        confidence = round((start_probs[start_index] * end_probs[end_index]).item() * 100, 2)
        
        predict_answer_tokens = inputs.input_ids[0, start_index : end_index + 1]
        highlighted_text = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        
        if highlighted_text.strip() == "" or highlighted_text == "<s>":
            return "<div style='padding: 15px; background-color: #d1fae5; color: #065f46; border-radius: 8px;'><b>✅ No hallucination detected.</b> The text appears fully supported by the facts.</div>"
            
        # --- THE RED HIGHLIGHT HACK ---
        # We replace the extracted text inside the original answer with a red HTML span
        red_span = f"<span style='background-color: #fca5a5; color: #7f1d1d; font-weight: bold; padding: 2px 4px; border-radius: 4px;'>{highlighted_text}</span>"
        visually_highlighted_answer = generated_answer.replace(highlighted_text, red_span)
        
        # Format the final output as a nice HTML box
        html_output = f"""
        <div style='padding: 15px; border: 1px solid #f87171; background-color: #fef2f2; border-radius: 8px; font-family: sans-serif;'>
            <h3 style='color: #b91c1c; margin-top: 0;'>⚠️ Hallucination Found (Confidence: {confidence}%)</h3>
            <p style='font-size: 16px; line-height: 1.5; color: #1f2937;'>{visually_highlighted_answer}</p>
        </div>
        """
        return html_output
        
    except Exception as e:
        return f"<p style='color: red;'>Prediction Error: {str(e)}</p>"

# 3. The Gradio 6.0 Web Interface (Multi-Tab)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 MedHallu: Automated Hallucination Catcher")
    gr.Markdown("Detects hallucinations in medical LLM outputs and highlights the exact fabricated phrase.")
    
    with gr.Tabs():
        # TAB 1: The Main UI you just built
        with gr.Tab("Hallucination Scanner"):
            with gr.Row():
                with gr.Column():
                    context_in = gr.Textbox(lines=6, label="1. Ground Truth Medical Context (The Facts)")
                    answer_in = gr.Textbox(lines=4, label="2. LLM Generated Answer (To be checked)")
                    submit_btn = gr.Button("Scan for Hallucinations", variant="primary")
                with gr.Column():
                    output_html = gr.HTML(label="3. Visual Hallucination Highlight")
            
            submit_btn.click(fn=catch_hallucination, inputs=[context_in, answer_in], outputs=output_html)
            
        # TAB 2: The Comparison Dashboard (For Person 1 & 2)
        with gr.Tab("Model Comparison Dashboard"):
            gr.Markdown("### 📊 F1 Score Comparison (Out-of-Distribution Data)")
            gr.Markdown("Our fine-tuned multi-task model vs. Zero-shot Baselines.")
            # You can replace this with actual data once Person 1 & 2 finish!
            gr.Dataframe(
                headers=["Split", "GPT-4o (Paper)", "Groq Llama (Zero-shot)", "Our Custom Model"],
                datatype=["str", "number", "number", "number"],
                value=[
                    ["Overall", 0.737, 0.0463, 0.812],
                    ["Easy", 0.844, 0.0705, 0.845],
                    ["Medium", 0.758, 0.0358, 0.790],
                    ["Hard", 0.625, 0.0318, 0.765]
                ]
            )

if __name__ == "__main__":
    demo.launch()