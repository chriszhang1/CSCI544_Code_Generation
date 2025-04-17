# -*- coding: utf-8 -*-
"""roberta_leetcode_classification.py

Script to fine-tune RoBERTa for multi-label classification of LeetCode problems.
"""

# --- Installation and Setup ---
# Ensure these libraries are installed in your environment:
# pip install transformers datasets pandas torch scikit-learn scikit-multilearn joblib

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score # Include accuracy for reference
# For stratified splitting in multi-label scenarios
from skmultilearn.model_selection import iterative_train_test_split
import re
import logging
import joblib # For saving/loading the MultiLabelBinarizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATASET_PATH = "filtered_leetcode_dataset.csv" # Make sure this file is accessible
MODEL_NAME = "FacebookAI/roberta-base"
TEXT_COLUMN = "description"
LABEL_COLUMN = "related_topics"
ID_COLUMN = "id"
TITLE_COLUMN = "title" # Keep title for potential future use or inspection

TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_LENGTH = 512 # Max sequence length for RoBERTa
BATCH_SIZE = 16 # Adjust based on GPU memory (A100 can likely handle more)
EPOCHS = 4 # Adjust as needed
LEARNING_RATE = 2e-5
OUTPUT_DIR = './results'
LOGGING_DIR = './logs'
FINAL_MODEL_PATH = f"{OUTPUT_DIR}/final_model"

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

# --- Data Loading and Preprocessing ---

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_topics(topics_str):
    """Parses comma-separated topics into a list of strings."""
    if pd.isna(topics_str) or not topics_str.strip():
        return []
    # Split by comma, strip whitespace, remove empty strings
    topics = [topic.strip() for topic in topics_str.split(',') if topic.strip()]
    return topics

def load_and_preprocess_data(file_path):
    """Loads data, performs cleaning, parsing, and filtering."""
    logger.info(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Error: Dataset file not found at {file_path}")
        logger.error("Please ensure the file is available in the specified path.")
        raise # Reraise the exception to halt execution

    # Select and potentially rename columns if needed
    if not all(col in df.columns for col in [ID_COLUMN, TITLE_COLUMN, TEXT_COLUMN, LABEL_COLUMN]):
        logger.error("Error: Required columns missing from the dataset.")
        raise ValueError("Dataset missing required columns")

    df = df[[ID_COLUMN, TITLE_COLUMN, TEXT_COLUMN, LABEL_COLUMN]]

    # Clean the description text
    logger.info("Cleaning text descriptions...")
    df['cleaned_description'] = df[TEXT_COLUMN].apply(clean_text)

    # Parse the topics string into lists
    logger.info("Parsing topics...")
    df['topic_list'] = df[LABEL_COLUMN].apply(parse_topics)

    # Filter out rows with no description or no topics after parsing
    initial_rows = len(df)
    df = df[df['cleaned_description'].str.len() > 0]
    df = df[df['topic_list'].apply(len) > 0]
    filtered_rows = len(df)
    logger.info(f"Filtered out {initial_rows - filtered_rows} rows with empty description or topics.")
    logger.info(f"Dataset shape after cleaning and filtering: {df.shape}")

    if filtered_rows == 0:
        logger.error("Error: No valid data remaining after filtering.")
        raise ValueError("No data left after preprocessing")

    return df

df = load_and_preprocess_data(DATASET_PATH)

# --- Topic Extraction and Encoding ---
logger.info("Extracting and encoding topics...")
# Use MultiLabelBinarizer to handle topic encoding and create the mapping
mlb = MultiLabelBinarizer()

# Fit on the training data topic lists AFTER splitting might be safer
# to avoid data leakage, but fitting on all data is common practice
# if the label space is assumed to be fixed.
# Let's fit on all data for now to ensure all topics are known.
all_labels_encoded = mlb.fit_transform(df['topic_list'])

num_labels = len(mlb.classes_)
logger.info(f"Found {num_labels} unique topics.")
# Uncomment to see all topics:
# logger.info(f"Topics: {mlb.classes_}")

# --- Stratified Train/Test Split ---
logger.info(f"Performing stratified train/test split (Test size: {TEST_SIZE})...")

# We need X (features) and y (labels) for the splitter.
# X can be just indices if we split the dataframe later.
y = all_labels_encoded
X_indices = np.arange(len(df)).reshape(-1, 1) # Need 2D array for splitter

# Perform the split using indices and the encoded labels
try:
    # iterative_train_test_split returns (X_train, y_train, X_test, y_test)
    train_indices, y_train_split, test_indices, y_test_split = iterative_train_test_split(
        X_indices, y, test_size=TEST_SIZE
    )
except ValueError as e:
    logger.error(f"Error during stratified split: {e}")
    logger.error("This can happen if a label combination is too rare for the split.")
    logger.error("Consider reducing TEST_SIZE or handling rare labels differently.")
    # Fallback to standard split (might lose stratification guarantee)
    logger.warning("Falling back to standard train_test_split (no stratification guarantee)...")
    from sklearn.model_selection import train_test_split
    train_indices, test_indices, y_train_split, y_test_split = train_test_split(
        X_indices, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    # Ensure indices are flattened if using fallback
    train_indices = train_indices.flatten()
    test_indices = test_indices.flatten()

# Flatten the index arrays if they came from iterative_train_test_split
if isinstance(train_indices, np.ndarray) and train_indices.ndim > 1:
    train_indices = train_indices.flatten()
if isinstance(test_indices, np.ndarray) and test_indices.ndim > 1:
     test_indices = test_indices.flatten()

# Create the train and test dataframes using the indices
df_train = df.iloc[train_indices].reset_index(drop=True)
df_test = df.iloc[test_indices].reset_index(drop=True)

# Get the corresponding labels for train/test sets (already split as y_train_split, y_test_split)
train_labels = y_train_split.astype(np.float32)
test_labels = y_test_split.astype(np.float32)

# Verification
logger.info(f"Training set size: {len(df_train)}")
logger.info(f"Testing set size: {len(df_test)}")

# Verify label presence (optional detailed check)
train_labels_present = train_labels.sum(axis=0) > 0
test_labels_present = test_labels.sum(axis=0) > 0
if np.all(train_labels_present) and np.all(test_labels_present):
    logger.info("Verified: All individual labels seem present in both training and testing sets.")
else:
    missing_train_idx = np.where(~train_labels_present)[0]
    missing_test_idx = np.where(~test_labels_present)[0]
    if len(missing_train_idx) > 0:
         logger.warning(f"Labels missing in train: {mlb.classes_[missing_train_idx]}")
    if len(missing_test_idx) > 0:
         logger.warning(f"Labels missing in test: {mlb.classes_[missing_test_idx]}")
    logger.warning("This can happen with very rare labels, even with stratified splits.")


# --- Tokenization and Dataset Creation ---
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

logger.info("Tokenizing training data...")
train_encodings = tokenizer(
    df_train['cleaned_description'].tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)
logger.info("Tokenizing test data...")
test_encodings = tokenizer(
    df_test['cleaned_description'].tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

class LeetCodeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Optimization: convert lists to tensors beforehand if possible,
        # but HuggingFace tokenizers often return lists of lists/ints.
        # The dict comprehension approach is generally fine.
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        # The number of items is the length of the first list in encodings (e.g., 'input_ids')
        # or simply the number of labels
        return len(self.labels)

train_dataset = LeetCodeDataset(train_encodings, train_labels)
test_dataset = LeetCodeDataset(test_encodings, test_labels)

# --- Model Loading ---
logger.info(f"Loading model: {MODEL_NAME} for multi-label sequence classification")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification", # Crucial for multi-label tasks
    # Consider ignoring mismatch size errors if you modify the classifier head later,
    # but usually not needed when using problem_type correctly.
    # ignore_mismatched_sizes=True
)
model.to(device) # Move model to GPU if available

# --- Metrics Calculation ---
def multi_label_metrics(predictions, labels, threshold=0.5):
    # First, apply sigmoid on predictions which are raw logits
    sigmoid = torch.nn.Sigmoid()
    # Ensure predictions is a tensor and on CPU for numpy conversion later
    # Convert raw predictions (logits) to probabilities
    probs = sigmoid(torch.Tensor(predictions).float()) # Ensure float type for sigmoid
    
    # Move probabilities to CPU before converting to numpy
    probs_np = probs.cpu().numpy()

    # Next, use threshold to turn probabilities into binary predictions
    y_pred = np.zeros(probs_np.shape)
    y_pred[np.where(probs_np >= threshold)] = 1

    # Finally, compute metrics
    y_true = labels # labels are already numpy arrays from EvalPrediction
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    # Use the numpy probabilities for ROC AUC
    roc_auc = roc_auc_score(y_true=y_true, y_score=probs_np, average='micro')
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred) # Exact match accuracy
    # Return as dictionary
    metrics = {
        'f1_micro': f1_micro,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }
    return metrics

def compute_metrics(p: EvalPrediction):
    # p.predictions contains the raw logits from the model
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # p.label_ids contains the actual labels
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

# --- Training ---
logger.info("Setting up training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,             # Directory for TensorBoard logs
    logging_strategy="steps",            # Log metrics at intervals
    logging_steps=50,                  # Log metrics every 50 steps
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
    save_strategy="epoch",               # Save model checkpoint at the end of each epoch
    save_total_limit=2,                  # Only keep the last 2 checkpoints
    load_best_model_at_end=True,         # Load the best model based on evaluation metric
    metric_for_best_model="f1_micro",    # Optimize for F1 micro score
    greater_is_better=True,              # Higher F1 is better
    fp16=torch.cuda.is_available(),      # Enable mixed precision training if GPU is available
    report_to="tensorboard",             # Log to TensorBoard (can add "wandb" if needed)
    seed=RANDOM_SEED,
    # Gradient accumulation can help if batches don't fit in memory
    # gradient_accumulation_steps=2, # Example: effective batch size = BATCH_SIZE * 2
    # Use 'no' for push_to_hub if not using Hugging Face Hub
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer, # Good practice to pass tokenizer
    # Default data collator should work fine for sequence classification
)

logger.info("Starting training...")
trainer.train()

logger.info("Training finished.")

# --- Evaluation ---
logger.info("Evaluating the best model on the test set...")
eval_results = trainer.evaluate(eval_dataset=test_dataset) # Explicitly pass test set

logger.info("Final Evaluation Results on Test Set:")
for key, value in eval_results.items():
    # Map internal names to more readable ones if needed
    metric_name = key.replace("eval_", "")
    logger.info(f"  {metric_name}: {value:.4f}")

# --- Save the final model, tokenizer, and label binarizer ---
logger.info(f"Saving the final (best) model, tokenizer, and label binarizer to {FINAL_MODEL_PATH}...")
trainer.save_model(FINAL_MODEL_PATH) # Saves the best model automatically due to load_best_model_at_end=True
tokenizer.save_pretrained(FINAL_MODEL_PATH)
joblib.dump(mlb, f"{FINAL_MODEL_PATH}/mlb.joblib")

logger.info(f"Model saved. Training script finished successfully.")

# --- Example Inference (Commented out by default) ---
# def predict_topics(text, model_path=FINAL_MODEL_PATH, threshold=0.5):
#     logger.info("\n--- Loading model for inference ---")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         mlb = joblib.load(f"{model_path}/mlb.joblib")
#     except Exception as e:
#         logger.error(f"Error loading model/tokenizer/mlb from {model_path}: {e}")
#         return None
# 
#     model.to(device) # Ensure model is on the correct device
#     model.eval()
# 
#     logger.info(f"Input Text: {text[:200]}...")
# 
#     # Clean and tokenize the input text
#     cleaned_text = clean_text(text)
#     inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
# 
#     with torch.no_grad():
#         logits = model(**inputs).logits
# 
#     # Process logits
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(logits.squeeze().cpu()) # Move logits to CPU
#     predictions = np.zeros(probs.shape, dtype=int)
#     predictions[np.where(probs >= threshold)] = 1
# 
#     # Get predicted label names
#     predicted_indices = np.where(predictions == 1)[0]
#     predicted_labels = mlb.classes_[predicted_indices]
# 
#     logger.info(f"Predicted Labels (Threshold={threshold}): {predicted_labels.tolist()}")
#     # logger.info(f"Predicted Probabilities: {probs.numpy()}")
#     return predicted_labels.tolist(), probs.numpy()

# # Example usage:
# if len(df_test) > 0:
#     sample_idx = 0 # Predict for the first item in the test set
#     sample_text = df_test['cleaned_description'].iloc[sample_idx]
#     sample_true_labels = df_test['topic_list'].iloc[sample_idx]
#     logger.info(f"\n--- Example Inference on Test Sample {sample_idx} ---")
#     logger.info(f"True Labels: {sample_true_labels}")
#     predicted_labels, _ = predict_topics(sample_text)
# else:
#     logger.info("Skipping example inference as test set is empty.") 