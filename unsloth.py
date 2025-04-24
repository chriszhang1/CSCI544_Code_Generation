import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# Check transformers version for debugging
import transformers
print(f"Transformers version: {transformers.__version__}")

# 1. Load and preprocess the data
data = pd.read_csv('leet.csv')

# Clean data - extract problem descriptions and their corresponding categories
def extract_main_categories(topics_str):
    if pd.isna(topics_str):
        return []
    # Parse the related_topics string and extract category names
    categories = []
    for topic in topics_str.split(','):
        # Extract just the category name, removing any extra info
        category = topic.strip().split(',')[0]
        categories.append(category)
    return categories

# Extract relevant columns and preprocess
problems = []
for _, row in data.iterrows():
    if pd.isna(row['description']) or pd.isna(row['related_topics']):
        continue
        
    categories = extract_main_categories(row['related_topics'])
    if not categories:
        continue
        
    problems.append({
        'description': row['description'],
        'categories': categories
    })

# 2. Prepare dataset
class ProblemDataset(Dataset):
    def __init__(self, problems, tokenizer, max_length=512):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.problems)
        
    def __getitem__(self, idx):
        problem = self.problems[idx]
        description = problem['description']
        categories = problem['categories']
        
        # Create prompt
        prompt = f"""Given the following programming problem, classify it into one of these categories:
Array, Dynamic Programming, String, Math, Tree, Depth-first Search, Greedy, Hash Table, Binary Search, Breadth-first Search, Sort, Two Pointers, Backtracking, Stack, Design

Problem description:
{description}

The problem type is: {categories[0]}"""
        
        encodings = self.tokenizer(prompt, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # Create labels (same as input_ids for causal language modeling)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 3. Split data
train_problems, val_problems = train_test_split(problems, test_size=0.2, random_state=42)

# 4. Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", torch_dtype=torch.bfloat16)

# 5. Set up LoRA configuration for efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Typically for Llama models
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# 6. Create datasets
train_dataset = ProblemDataset(train_problems, tokenizer)
val_dataset = ProblemDataset(val_problems, tokenizer)

# 7. Set up training arguments - FIXED VERSION
# Using a version-agnostic approach to evaluation settings
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    # Removed problematic evaluation_strategy parameter
    save_total_limit=3,
    save_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    # Removed load_best_model_at_end parameter
)

# 8. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 9. Train the model
trainer.train()

# 10. Save the fine-tuned model
model.save_pretrained("./leetcode-classifier-model")
tokenizer.save_pretrained("./leetcode-classifier-model")

# 11. Function to use the model for inference
def classify_problem(problem_description, model, tokenizer, categories):
    prompt = f"""Given the following programming problem, classify it into one of these categories:
Array, Dynamic Programming, String, Math, Tree, Depth-first Search, Greedy, Hash Table, Binary Search, Breadth-first Search, Sort, Two Pointers, Backtracking, Stack, Design

Problem description:
{problem_description}

The problem type is:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_category = generated_text.split("The problem type is:")[-1].strip()
    
    # Match to closest category
    for category in categories:
        if category.lower() in predicted_category.lower():
            return category
    
    return predicted_category

# Example usage
categories = ["Array", "Dynamic Programming", "String", "Math", "Tree", 
              "Depth-first Search", "Greedy", "Hash Table", "Binary Search", 
              "Breadth-first Search", "Sort", "Two Pointers", "Backtracking", 
              "Stack", "Design"]

# Test on a sample problem
test_problem = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
prediction = classify_problem(test_problem, model, tokenizer, categories)
print(f"Predicted category: {prediction}")