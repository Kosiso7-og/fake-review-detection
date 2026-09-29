import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# -----------------------------
# Paths
# -----------------------------

TEST_PATH = "data/processed/maide_english_test.csv"

MODEL_PATH = "models/distilbert-maide"

RESULTS_DIR = "results"

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "distilbert_report.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "distilbert_confusion_matrix.png"
)


# -----------------------------
# Configuration
# -----------------------------

MAX_LENGTH = 256
BATCH_SIZE = 8


# -----------------------------
# Main
# -----------------------------

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    print(f"Test reviews: {len(test_df)}")

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())


    # -----------------------------
    # Convert to Hugging Face dataset
    # -----------------------------

    test_dataset = Dataset.from_pandas(
        test_df[["review_text", "label"]],
        preserve_index=False
    )


    # -----------------------------
    # Load tokenizer
    # -----------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )


    def tokenize_function(examples):

        return tokenizer(
            examples["review_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )


    print("Tokenizing test dataset...")

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )


    # -----------------------------
    # Keep model columns
    # -----------------------------

    columns = [
        "input_ids",
        "attention_mask",
        "label"
    ]

    test_dataset = test_dataset.remove_columns(
        [
            column
            for column in test_dataset.column_names
            if column not in columns
        ]
    )


    # -----------------------------
    # Load trained model
    # -----------------------------

    print("\nLoading trained DistilBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )


    # -----------------------------
    # Create Trainer
    # -----------------------------

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
    )


    # -----------------------------
    # Generate predictions
    # -----------------------------

    print("\nGenerating predictions...")

    predictions_output = trainer.predict(
        test_dataset
    )

    predictions = np.argmax(
        predictions_output.predictions,
        axis=-1
    )

    labels = np.array(
        test_df["label"]
    )


    # -----------------------------
    # Classification report
    # -----------------------------

    print("\nClassification Report:")

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "Human",
            "AI"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    print(report_df)

    report_df.to_csv(
        REPORT_PATH
    )


    # -----------------------------
    # Confusion matrix
    # -----------------------------

    cm = confusion_matrix(
        labels,
        predictions
    )

    print("\nConfusion Matrix:")

    print(cm)


    # -----------------------------
    # Create confusion matrix image
    # -----------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        "DistilBERT Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Human", "AI"]
    )

    plt.yticks(
        [0, 1],
        ["Human", "AI"]
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )


    # Add numbers to cells

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300
    )

    plt.close()


    # -----------------------------
    # Overall metrics
    # -----------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )


    print("\nFinal Test Results:")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )


    # -----------------------------
    # Save paths
    # -----------------------------

    print("\nFiles saved:")

    print(
        f"Classification report: "
        f"{REPORT_PATH}"
    )

    print(
        f"Confusion matrix: "
        f"{CONFUSION_MATRIX_PATH}"
    )


if __name__ == "__main__":
    main()