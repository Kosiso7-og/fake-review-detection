
import pandas as pd
from pathlib import Path

INPUT_PATH = Path(
    "results/llama_external_error_analysis.csv"
)

RESULTS_DIR = Path("results")

df = pd.read_csv(INPUT_PATH)

# --------------------------------------------------
# Class labels
# --------------------------------------------------

df["actual_class"] = df["label"].map({
    0: "Human",
    1: "AI"
})

# --------------------------------------------------
# Review length by actual class
# --------------------------------------------------

print("=" * 65)
print("REVIEW LENGTH ANALYSIS BY ACTUAL CLASS")
print("=" * 65)

class_length = (
    df.groupby("actual_class")["word_count"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\nREVIEW LENGTH BY CLASS")
print("-" * 65)
print(class_length)

# --------------------------------------------------
# Define review length categories
# --------------------------------------------------

df["length_category"] = pd.cut(
    df["word_count"],
    bins=[0, 50, 100, 200, float("inf")],
    labels=[
        "1-50 words",
        "51-100 words",
        "101-200 words",
        "201+ words"
    ],
    include_lowest=True
)

# --------------------------------------------------
# Class distribution by review length
# --------------------------------------------------

print("\nCLASS DISTRIBUTION BY LENGTH")
print("-" * 65)

distribution = pd.crosstab(
    df["length_category"],
    df["actual_class"]
)

print(distribution)

# --------------------------------------------------
# Accuracy by length and actual class
# --------------------------------------------------

print("\nMODEL ACCURACY BY LENGTH AND CLASS")
print("-" * 65)

accuracy_table = (
    df.groupby(
        ["length_category", "actual_class"],
        observed=True
    )
    .agg(
        reviews=("label", "size"),
        svm_accuracy=("svm_correct", "mean"),
        distilbert_accuracy=("distilbert_correct", "mean")
    )
    .reset_index()
)

accuracy_table["svm_accuracy"] *= 100
accuracy_table["distilbert_accuracy"] *= 100

accuracy_table = accuracy_table.round(2)

print(accuracy_table.to_string(index=False))

# --------------------------------------------------
# Save results
# --------------------------------------------------

class_length.to_csv(
    RESULTS_DIR / "llama_review_length_by_class.csv"
)

distribution.to_csv(
    RESULTS_DIR / "llama_review_length_distribution.csv"
)

accuracy_table.to_csv(
    RESULTS_DIR / "llama_accuracy_by_length_and_class.csv",
    index=False
)

print("\nSaved analysis files to results/")
print("\nReview length analysis complete.")
