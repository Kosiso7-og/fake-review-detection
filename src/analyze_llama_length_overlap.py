
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

INPUT_PATH = Path(
    "results/llama_external_error_analysis.csv"
)

OUTPUT_PATH = Path(
    "results/llama_length_overlap_metrics.csv"
)

df = pd.read_csv(INPUT_PATH)

# Restrict to the observed overlapping length range.
subset = df[
    df["word_count"].between(30, 80)
].copy()

print("=" * 65)
print("LENGTH-OVERLAP EXTERNAL EVALUATION")
print("=" * 65)

print("\nReviews in subset:", len(subset))

print("\nClass distribution:")
print(subset["label"].value_counts().sort_index())

print("\nSentiment by class:")
print(pd.crosstab(subset["label"], subset["sentiment"]))

results = []

for model_name, prediction_column in [
    ("Linear SVM", "svm_prediction"),
    ("DistilBERT", "distilbert_prediction"),
]:
    y_true = subset["label"]
    y_pred = subset[prediction_column]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    print("\n" + "-" * 65)
    print(model_name)
    print("-" * 65)

    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    results.append({
        "model": model_name,
        "subset": "30-80 words",
        "reviews": len(subset),
        "human_reviews": int((y_true == 0).sum()),
        "ai_reviews": int((y_true == 1).sum()),
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
    })

pd.DataFrame(results).to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:", OUTPUT_PATH)
print("\nLength-overlap evaluation complete.")
