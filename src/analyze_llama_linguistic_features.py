
import pandas as pd
import re
from pathlib import Path

INPUT_PATH = Path(
    "results/llama_external_error_analysis.csv"
)

OUTPUT_PATH = Path(
    "results/llama_linguistic_features.csv"
)

df = pd.read_csv(INPUT_PATH)

# Focus on AI-generated reviews only.
df = df[df["label"] == 1].copy()

# --------------------------------------------------
# Linguistic feature extraction
# --------------------------------------------------

def extract_features(text):
    text = str(text)

    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)
    sentences = [
        sentence for sentence in sentences
        if sentence.strip()
    ]

    word_count = len(words)
    unique_words = len(set(words))
    sentence_count = len(sentences)

    return pd.Series({
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": (
            sum(len(word) for word in words) / word_count
            if word_count else 0
        ),
        "avg_sentence_length": (
            word_count / sentence_count
            if sentence_count else 0
        ),
        "lexical_diversity": (
            unique_words / word_count
            if word_count else 0
        ),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "comma_count": text.count(","),
    })

# The linguistic analysis uses regex-based token counting.
df = df.drop(columns=["word_count"])

features = df["review_text"].apply(
    extract_features
)

df = pd.concat(
    [df, features],
    axis=1
)

# --------------------------------------------------
# Group by model prediction outcomes
# --------------------------------------------------

feature_columns = [
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "lexical_diversity",
    "exclamation_count",
    "question_count",
    "comma_count",
]

summary = (
    df.groupby("outcome")[feature_columns]
    .mean()
    .round(3)
)

print("=" * 65)
print("LINGUISTIC FEATURE ANALYSIS — LLAMA REVIEWS")
print("=" * 65)

print("\nAI review outcomes:")
print(df["outcome"].value_counts())

print("\nAVERAGE LINGUISTIC FEATURES BY OUTCOME")
print("-" * 65)

print(summary.to_string())

# --------------------------------------------------
# Save results
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

summary.to_csv(
    "results/llama_linguistic_feature_summary.csv"
)

print("\nSaved:")
print(OUTPUT_PATH)
print("results/llama_linguistic_feature_summary.csv")

print("\nLinguistic feature analysis complete.")
