# tinyllama_lora_finetune.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
import os
import requests
from bs4 import BeautifulSoup
import mlflow
import pandas as pd

# === Step 1: Load Base Model & Tokenizer ===
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    
    torch_dtype=torch.float16,
    device_map={"": "cpu"}
)

# === Step 2: PEFT Configuration for LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)


df = pd.read_csv("finance_qa_dataset.csv")
dataset = Dataset.from_pandas(df)  # 'text' column will be used automatically


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

# Split and tokenize dataset
dataset = dataset.train_test_split(test_size=0.1)
tokenized_train = dataset["train"].map(tokenize)
tokenized_val = dataset["test"].map(tokenize)

# === Step 4: Training Config ===
training_args = TrainingArguments(
    output_dir="./tinyllama-lora-finance",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-4,
    fp16=False,
    logging_steps=1,
    save_strategy="no",
    save_steps=50,
    evaluation_strategy="no",
    report_to="none"
)

# === Step 5: Data Collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Step 6: Train the LoRA Model with MLflow Tracking ===
mlflow.set_experiment("tinyllama-finance-lora")
with mlflow.start_run(run_name="lora_finetune_run_01"):
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("epochs", training_args.num_train_epochs)
    mlflow.log_param("learning_rate", training_args.learning_rate)
    mlflow.log_param("dataset_size", len(dataset["train"]))
    mlflow.log_param("lora_r", peft_config.r)
    mlflow.log_param("lora_alpha", peft_config.lora_alpha)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )

    print("🚀 Training started...")
    train_result = trainer.train()
    print("✅ Training finished.")
    eval_result = trainer.evaluate()

    mlflow.log_metrics({
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"]
    })

    # Save Adapter
    adapter_path = "./tinyllama_lora_adapter"
    model.save_pretrained(adapter_path)
    mlflow.log_artifact(adapter_path)

print("✅ LoRA fine-tuning complete, adapter saved, and run tracked in MLflow!")
