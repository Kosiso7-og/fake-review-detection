
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "results/llama_external_model_comparison.csv"
)

RESULTS_DIR = Path("results")

# --------------------------------------------------
# Load paired predictions
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 65)
print("COMPARATIVE ERROR ANALYSIS — OTT/LLAMA")
print("=" * 65)

print("\nTotal reviews:", len(df))

# --------------------------------------------------
# Classify prediction outcomes
# --------------------------------------------------

def classify_outcome(row):
    svm_correct = row["svm_correct"]
    bert_correct = row["distilbert_correct"]

    if svm_correct and bert_correct:
        return "Both Correct"
    elif not svm_correct and not bert_correct:
        return "Both Wrong"
    elif svm_correct:
        return "SVM Only Correct"
    else:
        return "DistilBERT Only Correct"

df["outcome"] = df.apply(
    classify_outcome,
    axis=1
)

# --------------------------------------------------
# Summary of outcomes
# --------------------------------------------------

print("\nPREDICTION OUTCOMES")
print("-" * 65)

print(
    df["outcome"].value_counts()
)

# --------------------------------------------------
# Outcomes by actual class
# --------------------------------------------------

print("\nOUTCOMES BY ACTUAL CLASS")
print("-" * 65)

class_table = pd.crosstab(
    df["outcome"],
    df["label"]
)

class_table = class_table.rename(
    columns={0: "Human", 1: "AI"}
)

print(class_table)

# --------------------------------------------------
# Outcomes by sentiment
# --------------------------------------------------

print("\nOUTCOMES BY SENTIMENT")
print("-" * 65)

sentiment_table = pd.crosstab(
    df["outcome"],
    df["sentiment"]
)

print(sentiment_table)

# --------------------------------------------------
# Review length analysis
# --------------------------------------------------

df["word_count"] = (
    df["review_text"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\nAVERAGE REVIEW LENGTH BY OUTCOME")
print("-" * 65)

length_summary = (
    df.groupby("outcome")["word_count"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print(length_summary)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------

df.to_csv(
    RESULTS_DIR / "llama_external_error_analysis.csv",
    index=False
)

class_table.to_csv(
    RESULTS_DIR / "llama_external_errors_by_class.csv"
)

sentiment_table.to_csv(
    RESULTS_DIR / "llama_external_errors_by_sentiment.csv"
)

length_summary.to_csv(
    RESULTS_DIR / "llama_external_errors_by_length.csv"
)

print("\nSaved analysis files to results/")

print("\nComparative error analysis complete.")
