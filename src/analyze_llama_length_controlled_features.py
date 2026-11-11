
import pandas as pd
from pathlib import Path

INPUT_PATH = Path(
    "results/llama_linguistic_features.csv"
)

OUTPUT_PATH = Path(
    "results/llama_length_controlled_features.csv"
)

df = pd.read_csv(INPUT_PATH)

# Focus on cases where both models agreed
# and were either both correct or both wrong.
df = df[
    df["outcome"].isin([
        "Both Correct",
        "Both Wrong"
    ])
].copy()

# Restrict to similar-length reviews
df = df[
    df["word_count"].between(40, 60)
].copy()

features = [
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "lexical_diversity",
    "exclamation_count",
    "question_count",
    "comma_count",
]

print("=" * 65)
print("LENGTH-CONTROLLED LINGUISTIC FEATURE ANALYSIS")
print("=" * 65)

print("\nReview counts by outcome:")
print(df["outcome"].value_counts())

summary = (
    df.groupby("outcome")[features]
    .agg(["mean", "median"])
    .round(3)
)

print("\nLINGUISTIC FEATURE COMPARISON")
print("-" * 65)
print(summary.to_string())

summary.to_csv(OUTPUT_PATH)

print("\nSaved:", OUTPUT_PATH)
print("\nLength-controlled analysis complete.")
