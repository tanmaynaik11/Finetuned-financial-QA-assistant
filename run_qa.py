from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, "tinyllama_lora_adapter")

# Move to CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def ask(question: str):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
while True:
    q = input("Ask a finance question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print(ask(q))
