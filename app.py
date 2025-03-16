import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import pandas as pd


# Load DeepSeek-Math 7B Instruct model and tokenizer
model_name = "deepseek-ai/deepseek-math-7b-instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    print("DeepSeek-Math-7B-Instruct loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

# Load training data from math_meme_training.txt
training_file = "math_meme_training.txt"
few_shot_prompt = "Below are examples of correcting incorrect math memes with step-by-step reasoning:\n\n"
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()
        # Take up to 3 examples to keep prompt manageable
        examples = []
        for i in range(0, min(6, len(lines)), 2):  # Step by 2 for Prompt/Response pairs
            if lines[i].startswith("Prompt: ") and i+1 < len(lines) and lines[i+1].startswith("Response: "):
                prompt = lines[i].replace("Prompt: ", "").strip()
                response = lines[i+1].replace("Response: ", "").strip()
                examples.append(f"{prompt}\nResponse: {response}")
        few_shot_prompt += "\n".join(examples[:3]) + "\n\nNow, correct the following incorrect math meme:\n"
else:
    few_shot_prompt += "No training data found. Using default behavior.\n\nNow, correct the following incorrect math meme:\n"
    print("Warning: math_meme_training.txt not found. Using minimal prompt.")

# Function to generate correction
def generate_deepseek_correction(user_input):
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please check the Space logs and ensure a GPU is enabled."

    # Validate input format
    if '=' not in user_input or len(user_input.split('=')) != 2:
        return "Invalid input. Please enter in the format 'expression = wrong_answer' (e.g., '8 ÷ 2(2+2) = 1')."

    # Construct prompt with training data
    meme_prompt = f"Correct this incorrect math meme: '{user_input}'\nPlease reason step by step and provide the correct answer with an explanation."
    full_prompt = few_shot_prompt + meme_prompt

    try:
        # Generate response
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        correction = response[len(full_prompt):].strip()

        if not correction:
            correction = "DeepSeek-Math failed to generate a meaningful correction."

        # Add funny error rating
        error_rating = "Funny Error Rating: '90% fixing memes, 10% wondering why fractions are so hard'"
        return f"{correction}\n\n{error_rating}"
    except Exception as e:
        return f"Error processing input: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=generate_deepseek_correction,
    inputs=gr.Textbox(
        label="Enter an Incorrect Math Meme",
        placeholder="e.g., '7 + 2 × 3 = 27' or '1/2 + 1/2 = 0.25'"
    ),
    outputs=gr.Textbox(label="Corrected Answer and Explanation"),
    title="Math Meme Repair with DeepSeek-Math",
    description="Input an incorrect math meme (e.g., '8 ÷ 2(2+2) = 1') and get the correct answer with an explanation!",
    examples=[
        ["7 + 2 × 3 = 27"],
        ["4 ÷ 2(1+1) = 1"],
        ["1/2 + 1/2 = 0.25"]
    ]
)

# Launch the interface
interface.launch(server_name="0.0.0.0", server_port=7860)