import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import pandas as pd

# Check GPU availability
#print("CUDA Available:", torch.cuda.is_available())
#print("Current Device:", torch.cuda.current_device())
#print("Device Name:", torch.cuda.get_device_name(0))

# Load and prepare dataset
def load_custom_dataset(file_path):
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Ensure 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    
    # Convert to Hugging Face dataset
    dataset = load_dataset('csv', data_files=file_path, split='train')
    return dataset

# Model and Tokenizer Setup
model_name = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    device_map="auto"  # Automatic device mapping
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=1024)

# Prepare dataset
dataset = load_custom_dataset('instructions.csv')
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./ansible-review-model",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    fp16=True,  # Use mixed precision
    logging_dir='./logs',
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # For causal language modeling
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start Training
trainer.train()

# Save Model and Tokenizer
trainer.save_model("./ansible-review-model")
tokenizer.save_pretrained("./ansible-review-model")

print("Training Complete!")