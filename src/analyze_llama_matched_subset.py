
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

RESULTS_DIR = Path("results")

df = pd.read_csv(INPUT_PATH)

# Restrict to the overlapping length range
df = df[
    df["word_count"].between(30, 80)
].copy()

# Five-word bins
df["length_bin"] = (
    (df["word_count"] - 30) // 5
)

matched_parts = []

# Match by sentiment and length bin
for _, group in df.groupby(
    ["sentiment", "length_bin"]
):
    human = group[group["label"] == 0]
    ai = group[group["label"] == 1]

    n = min(len(human), len(ai))

    if n == 0:
        continue

    matched_parts.append(
        human.sample(n=n, random_state=42)
    )

    matched_parts.append(
        ai.sample(n=n, random_state=42)
    )

if not matched_parts:
    raise ValueError(
        "No matched human/AI review groups found."
    )

matched = pd.concat(
    matched_parts,
    ignore_index=True
)

print("=" * 65)
print("MATCHED EXTERNAL REVIEW EVALUATION")
print("=" * 65)

print("\nTotal matched reviews:", len(matched))

print("\nClass distribution:")
print(matched["label"].value_counts().sort_index())

print("\nSentiment distribution:")
print(pd.crosstab(
    matched["label"],
    matched["sentiment"]
))

print("\nWord-count summary:")
print(
    matched.groupby("label")["word_count"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

results = []

for name, column in [
    ("Linear SVM", "svm_prediction"),
    ("DistilBERT", "distilbert_prediction")
]:
    y_true = matched["label"]
    y_pred = matched[column]

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
    print(name)
    print("-" * 65)

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"AI Precision: {precision:.4f}")
    print(f"AI Recall: {recall:.4f}")
    print(f"AI F1-score: {f1:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    results.append({
        "model": name,
        "reviews": len(matched),
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1
    })

matched.to_csv(
    RESULTS_DIR / "llama_matched_reviews.csv",
    index=False
)

pd.DataFrame(results).to_csv(
    RESULTS_DIR / "llama_matched_metrics.csv",
    index=False
)

print("\nSaved matched evaluation files.")
print("\nMatched analysis complete.")
