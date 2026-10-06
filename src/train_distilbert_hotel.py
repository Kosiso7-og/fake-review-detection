import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

TRAIN_PATH = (
    "data/processed/"
    "maide_english_hotel_transformer_train.csv"
)

VALIDATION_PATH = (
    "data/processed/"
    "maide_english_hotel_transformer_validation.csv"
)

OUTPUT_DIR = (
    "models/distilbert-maide-hotel"
)

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------
# Device information
# ---------------------------------------------------------

print("=" * 70)
print("DISTILBERT HOTEL-DISJOINT TRAINING")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Device: CPU")

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

print(f"Training reviews: {len(train_df)}")
print(f"Validation reviews: {len(validation_df)}")

print(
    f"Training hotels: "
    f"{train_df['Hotel Name'].nunique()}"
)

print(
    f"Validation hotels: "
    f"{validation_df['Hotel Name'].nunique()}"
)

# ---------------------------------------------------------
# Verify hotel separation
# ---------------------------------------------------------

train_hotels = set(
    train_df["Hotel Name"].unique()
)

validation_hotels = set(
    validation_df["Hotel Name"].unique()
)

overlap = train_hotels.intersection(
    validation_hotels
)

print(f"Train/validation hotel overlap: {len(overlap)}")

assert len(overlap) == 0, (
    "Hotel leakage detected between "
    "training and validation!"
)

# ---------------------------------------------------------
# Keep only model columns
# ---------------------------------------------------------

train_model_df = train_df[
    ["review_text", "label"]
].copy()

validation_model_df = validation_df[
    ["review_text", "label"]
].copy()

# Rename text column for convenience

train_model_df = train_model_df.rename(
    columns={"review_text": "text"}
)

validation_model_df = validation_model_df.rename(
    columns={"review_text": "text"}
)

# ---------------------------------------------------------
# Convert to Hugging Face datasets
# ---------------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_model_df,
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_model_df,
    preserve_index=False
)

# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

print("Tokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenizing validation data...")

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

# Remove raw text after tokenization

train_dataset = train_dataset.remove_columns(
    ["text"]
)

validation_dataset = validation_dataset.remove_columns(
    ["text"]
)

# ---------------------------------------------------------
# Load DistilBERT
# ---------------------------------------------------------

print("\nLoading DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "Human",
        1: "AI"
    },
    label2id={
        "Human": 0,
        "AI": 1
    }
)

# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ---------------------------------------------------------
# Training arguments
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=2,

    logging_strategy="steps",
    logging_steps=50,

    report_to="none",

    seed=SEED,

    dataloader_pin_memory=False
)

# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nStarting training...")

trainer.train()

# ---------------------------------------------------------
# Final validation evaluation
# ---------------------------------------------------------

print("\nEvaluating best model on validation set...")

validation_results = trainer.evaluate()

print("\n" + "=" * 70)
print("BEST MODEL VALIDATION RESULTS")
print("=" * 70)

for key, value in validation_results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

# ---------------------------------------------------------
# Save best model
# ---------------------------------------------------------

print("\nSaving best model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(f"\nModel saved to: {OUTPUT_DIR}")

print("\nIMPORTANT:")
print(
    "The final 20-hotel test set has NOT been "
    "used during training or model selection."
)

print("\n" + "=" * 70)
print("HOTEL-DISJOINT TRAINING COMPLETE")
print("=" * 70)