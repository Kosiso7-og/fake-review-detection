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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# -----------------------------
# Paths
# -----------------------------

TRAIN_PATH = "data/processed/maide_english_transformer_train.csv"
VALIDATION_PATH = "data/processed/maide_english_validation.csv"
TEST_PATH = "data/processed/maide_english_test.csv"

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert-maide"


# -----------------------------
# Configuration
# -----------------------------

MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -----------------------------
# Main
# -----------------------------

def main():

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training reviews: {len(train_df)}")
    print(f"Validation reviews: {len(validation_df)}")
    print(f"Test reviews: {len(test_df)}")

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation label distribution:")
    print(validation_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    # Convert pandas DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(
        train_df[["review_text", "label"]],
        preserve_index=False
    )

    validation_dataset = Dataset.from_pandas(
        validation_df[["review_text", "label"]],
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df[["review_text", "label"]],
        preserve_index=False
    )

    # -----------------------------
    # Tokenizer
    # -----------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["review_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    print("Tokenizing datasets...")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    # Keep only the columns required by the model
    columns = ["input_ids", "attention_mask", "label"]

    train_dataset = train_dataset.remove_columns(
        [column for column in train_dataset.column_names if column not in columns]
    )

    validation_dataset = validation_dataset.remove_columns(
        [column for column in validation_dataset.column_names if column not in columns]
    )

    test_dataset = test_dataset.remove_columns(
        [column for column in test_dataset.column_names if column not in columns]
    )

    # -----------------------------
    # Model
    # -----------------------------

    print("\nLoading DistilBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # -----------------------------
    # Training arguments
    # -----------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_strategy="epoch",

        save_total_limit=1,

        report_to="none",

        seed=RANDOM_SEED,
    )

    # -----------------------------
    # Trainer
    # -----------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # Train
    # -----------------------------

    print("\nStarting DistilBERT training...")
    print("This may take a while on CPU.\n")

    trainer.train()

    # -----------------------------
    # Validation
    # -----------------------------

    print("\nFinal validation results:")

    validation_results = trainer.evaluate(
        eval_dataset=validation_dataset
    )

    for key, value in validation_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # -----------------------------
    # Test
    # -----------------------------

    print("\nEvaluating on untouched test set...")

    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test"
    )

    print("\nFinal test results:")

    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # -----------------------------
    # Save model
    # -----------------------------

    print("\nSaving final model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()