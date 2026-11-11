
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "models/distilbert-maide-hotel"
EXTERNAL_PATH = "data/processed/llama_external_test.csv"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Device information
# --------------------------------------------------

print("=" * 65)
print("DISTILBERT EXTERNAL OTT/LLAMA EVALUATION")
print("=" * 65)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")

# --------------------------------------------------
# Load external dataset
# --------------------------------------------------

external_df = pd.read_csv(EXTERNAL_PATH)

print("\nExternal reviews:", len(external_df))
print("\nClass distribution:")
print(external_df["label"].value_counts().sort_index())

assert len(external_df) == 1586, (
    "Unexpected external dataset size."
)

assert set(external_df["label"].unique()) == {0, 1}, (
    "Expected Human=0 and AI=1 labels."
)

assert external_df["review_text"].notna().all(), (
    "Missing review text detected."
)

# --------------------------------------------------
# Load saved model and tokenizer
# --------------------------------------------------

print("\nLoading saved hotel-disjoint DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# --------------------------------------------------
# Prepare dataset
# --------------------------------------------------

model_df = external_df[
    ["review_text", "label"]
].copy()

model_df = model_df.rename(
    columns={"review_text": "text"}
)

external_dataset = Dataset.from_pandas(
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

print("\nTokenizing external reviews...")

external_dataset = external_dataset.map(
    tokenize_function,
    batched=True
)

external_dataset = external_dataset.remove_columns(
    ["text"]
)

# --------------------------------------------------
# Evaluation Trainer
# --------------------------------------------------

evaluation_args = TrainingArguments(
    output_dir="models/distilbert-llama-eval-temp",
    per_device_eval_batch_size=8,
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=evaluation_args
)

# --------------------------------------------------
# Run inference
# --------------------------------------------------

print("\nRunning external evaluation...")
print("No model training will occur.\n")

prediction_output = trainer.predict(
    external_dataset
)

logits = prediction_output.predictions

predictions = np.argmax(
    logits,
    axis=-1
)

true_labels = external_df["label"].to_numpy()

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    true_labels, predictions
)

precision = precision_score(
    true_labels, predictions, zero_division=0
)

recall = recall_score(
    true_labels, predictions, zero_division=0
)

f1 = f1_score(
    true_labels, predictions, zero_division=0
)

matrix = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

print("\n" + "=" * 65)
print("EXTERNAL LLAMA EVALUATION — DISTILBERT")
print("=" * 65)

print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(matrix)

# --------------------------------------------------
# Save predictions
# --------------------------------------------------

prediction_df = external_df.copy()

prediction_df["distilbert_prediction"] = predictions

prediction_df["correct"] = (
    prediction_df["label"] ==
    prediction_df["distilbert_prediction"]
)

prediction_path = (
    RESULTS_DIR / "llama_external_distilbert_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_df = pd.DataFrame([{
    "model": "DistilBERT",
    "dataset": "Ott/Llama External",
    "accuracy": accuracy,
    "precision_ai": precision,
    "recall_ai": recall,
    "f1_ai": f1,
    "tn": int(matrix[0, 0]),
    "fp": int(matrix[0, 1]),
    "fn": int(matrix[1, 0]),
    "tp": int(matrix[1, 1])
}])

metrics_path = (
    RESULTS_DIR / "llama_external_distilbert_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)

# --------------------------------------------------
# Save classification report
# --------------------------------------------------

report = classification_report(
    true_labels,
    predictions,
    labels=[0, 1],
    target_names=["Human", "AI"],
    output_dict=True,
    zero_division=0
)

report_path = (
    RESULTS_DIR / "llama_external_distilbert_report.csv"
)

pd.DataFrame(report).transpose().to_csv(
    report_path
)

# --------------------------------------------------
# Save confusion matrix
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Human", "AI"]
)

display.plot(cmap="Blues")

plt.title(
    "DistilBERT — External Ott/Llama Test"
)

plt.tight_layout()

figure_path = (
    RESULTS_DIR /
    "llama_external_distilbert_confusion_matrix.png"
)

plt.savefig(
    figure_path,
    dpi=300
)

plt.close()

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\nSaved files:")
print(prediction_path)
print(metrics_path)
print(report_path)
print(figure_path)

print("\nExternal DistilBERT evaluation complete.")
