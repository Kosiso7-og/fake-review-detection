import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_PATH = "models/distilbert-maide-hotel"

TEST_PATH = (
    "data/processed/maide_english_hotel_test.csv"
)

REPORT_PATH = (
    "results/distilbert_hotel_report.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/distilbert_hotel_confusion_matrix.png"
)

PREDICTIONS_PATH = (
    "results/distilbert_hotel_predictions.csv"
)

ERRORS_PATH = (
    "results/distilbert_hotel_errors.csv"
)

# ---------------------------------------------------------
# Device information
# ---------------------------------------------------------

print("=" * 70)
print("DISTILBERT UNSEEN-HOTEL FINAL EVALUATION")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Device: CPU")

# ---------------------------------------------------------
# Load final test set
# ---------------------------------------------------------

print("\nLoading final unseen-hotel test set...")

test_df = pd.read_csv(TEST_PATH)

print(f"Test reviews: {len(test_df)}")

print(
    f"Unique test hotels: "
    f"{test_df['Hotel Name'].nunique()}"
)

print("\nClass distribution:")

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------

assert len(test_df) == 400, (
    "Expected 400 final test reviews."
)

assert test_df["Hotel Name"].nunique() == 20, (
    "Expected 20 unseen test hotels."
)

assert (test_df["label"] == 0).sum() == 200, (
    "Expected 200 Human reviews."
)

assert (test_df["label"] == 1).sum() == 200, (
    "Expected 200 AI reviews."
)

# ---------------------------------------------------------
# Load tokenizer and trained model
# ---------------------------------------------------------

print("\nLoading saved hotel-disjoint DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# ---------------------------------------------------------
# Prepare dataset
# ---------------------------------------------------------

model_df = test_df[
    ["review_text", "label"]
].copy()

model_df = model_df.rename(
    columns={"review_text": "text"}
)

test_dataset = Dataset.from_pandas(
    model_df,
    preserve_index=False
)

def tokenize_function(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

print("Tokenizing final test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.remove_columns(
    ["text"]
)

# ---------------------------------------------------------
# Create evaluation Trainer
# ---------------------------------------------------------

evaluation_args = TrainingArguments(
    output_dir="models/distilbert-maide-hotel-eval-temp",
    per_device_eval_batch_size=8,
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=evaluation_args
)

# ---------------------------------------------------------
# Final predictions
# ---------------------------------------------------------

print("\nRunning FINAL evaluation...")

prediction_output = trainer.predict(
    test_dataset
)

logits = prediction_output.predictions

predictions = np.argmax(
    logits,
    axis=-1
)

true_labels = test_df["label"].to_numpy()

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predictions
)

report = classification_report(
    true_labels,
    predictions,
    target_names=["Human", "AI"],
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    true_labels,
    predictions
)

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL UNSEEN-HOTEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        target_names=["Human", "AI"],
        zero_division=0
    )
)

print("Confusion Matrix:")

print(matrix)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    REPORT_PATH
)

# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Human", "AI"]
)

display.plot()

plt.title(
    "DistilBERT - Unseen-Hotel MAiDE-up Test Set"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# Save individual predictions
# ---------------------------------------------------------

prediction_df = test_df.copy()

prediction_df["prediction"] = predictions

prediction_df["correct"] = (
    prediction_df["label"]
    == prediction_df["prediction"]
)

prediction_df["actual_class"] = (
    prediction_df["label"]
    .map({
        0: "Human",
        1: "AI"
    })
)

prediction_df["predicted_class"] = (
    prediction_df["prediction"]
    .map({
        0: "Human",
        1: "AI"
    })
)

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

# ---------------------------------------------------------
# Save errors only
# ---------------------------------------------------------

errors_df = prediction_df[
    prediction_df["correct"] == False
].copy()

errors_df.to_csv(
    ERRORS_PATH,
    index=False
)

# ---------------------------------------------------------
# Error summary
# ---------------------------------------------------------

human_to_ai = (
    (true_labels == 0)
    & (predictions == 1)
).sum()

ai_to_human = (
    (true_labels == 1)
    & (predictions == 0)
).sum()

correct = (
    true_labels == predictions
).sum()

print("\n" + "-" * 70)
print("ERROR SUMMARY")
print("-" * 70)

print(f"Correct predictions: {correct}/400")
print(f"Incorrect predictions: {len(errors_df)}/400")

print(
    f"Human predicted as AI: "
    f"{human_to_ai}"
)

print(
    f"AI predicted as Human: "
    f"{ai_to_human}"
)

# ---------------------------------------------------------
# Comparison with original DistilBERT
# ---------------------------------------------------------

ORIGINAL_RANDOM_ACCURACY = 0.935

difference = (
    accuracy
    - ORIGINAL_RANDOM_ACCURACY
) * 100

print("\n" + "-" * 70)
print("COMPARISON WITH ORIGINAL RANDOM SPLIT")
print("-" * 70)

print(
    "Original DistilBERT random-split accuracy: "
    f"{ORIGINAL_RANDOM_ACCURACY * 100:.2f}%"
)

print(
    "Unseen-hotel DistilBERT accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "Difference: "
    f"{difference:+.2f} percentage points"
)

# ---------------------------------------------------------
# Saved files
# ---------------------------------------------------------

print("\nFiles saved:")

print(REPORT_PATH)
print(CONFUSION_MATRIX_PATH)
print(PREDICTIONS_PATH)
print(ERRORS_PATH)

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)