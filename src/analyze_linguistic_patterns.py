import pandas as pd
import re

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TEST_PATH = "data/processed/maide_english_test.csv"
ERROR_PATH = "results/distilbert_errors.csv"

OUTPUT_PATH = "results/distilbert_linguistic_analysis.csv"

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

test_df = pd.read_csv(TEST_PATH)
error_df = pd.read_csv(ERROR_PATH)

# ---------------------------------------------------------
# Create prediction status
# ---------------------------------------------------------

test_df["prediction_status"] = "Correct"

for _, error in error_df.iterrows():

    mask = (
        (test_df["review_text"] == error["review_text"])
        & (test_df["label"] == error["label"])
    )

    test_df.loc[mask, "prediction_status"] = error["error_type"]

# ---------------------------------------------------------
# Linguistic analysis functions
# ---------------------------------------------------------

positive_words = {
    "excellent",
    "great",
    "good",
    "wonderful",
    "amazing",
    "lovely",
    "perfect",
    "comfortable",
    "friendly",
    "helpful",
    "clean",
    "beautiful",
    "fantastic",
    "delicious",
    "recommend",
    "recommended",
    "convenient",
    "exceptional",
    "nice",
    "pleasant"
}

negative_words = {
    "bad",
    "poor",
    "dirty",
    "unclean",
    "uncomfortable",
    "terrible",
    "awful",
    "noisy",
    "noise",
    "small",
    "limited",
    "disappointing",
    "disappointed",
    "unhelpful",
    "unfriendly",
    "problem",
    "problems",
    "issue",
    "issues",
    "worst",
    "underwhelming",
    "outdated",
    "shabby",
    "smell",
    "smelly"
}

first_person_words = {
    "i",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "ours"
}

hotel_terms = {
    "hotel",
    "room",
    "rooms",
    "staff",
    "breakfast",
    "location",
    "bathroom",
    "bed",
    "wifi",
    "internet",
    "clean",
    "restaurant",
    "service",
    "view",
    "pool",
    "floor"
}


def analyze_text(text):

    text = str(text)

    # Word tokens
    words = re.findall(r"\b[\w']+\b", text.lower())

    word_count = len(words)

    # Sentence estimation
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_count = len(sentences)

    if sentence_count > 0:
        average_sentence_length = word_count / sentence_count
    else:
        average_sentence_length = 0

    # First-person language
    first_person_count = sum(
        1 for word in words if word in first_person_words
    )

    # Positive / negative language
    positive_count = sum(
        1 for word in words if word in positive_words
    )

    negative_count = sum(
        1 for word in words if word in negative_words
    )

    # Punctuation
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Hotel-specific vocabulary
    hotel_term_count = sum(
        1 for word in words if word in hotel_terms
    )

    # Capitalization
    uppercase_words = sum(
        1 for word in text.split()
        if len(word) > 1 and word.isupper()
    )

    return pd.Series({
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "first_person_count": first_person_count,
        "positive_word_count": positive_count,
        "negative_word_count": negative_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "hotel_term_count": hotel_term_count,
        "uppercase_word_count": uppercase_words
    })


# ---------------------------------------------------------
# Apply analysis
# ---------------------------------------------------------

linguistic_features = test_df["review_text"].apply(analyze_text)

test_df = pd.concat(
    [test_df, linguistic_features],
    axis=1
)

# ---------------------------------------------------------
# Create summary
# ---------------------------------------------------------

feature_columns = [
    "sentence_count",
    "average_sentence_length",
    "first_person_count",
    "positive_word_count",
    "negative_word_count",
    "exclamation_count",
    "question_count",
    "hotel_term_count",
    "uppercase_word_count"
]

summary = (
    test_df
    .groupby("prediction_status")[feature_columns]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DISTILBERT LINGUISTIC ERROR ANALYSIS")
print("=" * 80)

print("\nLinguistic characteristics by prediction status:")

print(summary)

print("\n" + "-" * 80)
print("Average linguistic characteristics")
print("-" * 80)

average_summary = (
    test_df
    .groupby("prediction_status")[feature_columns]
    .mean()
    .round(2)
)

print(average_summary)

# ---------------------------------------------------------
# Error-only analysis
# ---------------------------------------------------------

errors_only = test_df[
    test_df["prediction_status"] != "Correct"
]

print("\n" + "-" * 80)
print("Linguistic characteristics of errors only")
print("-" * 80)

error_summary = (
    errors_only
    .groupby("prediction_status")[feature_columns]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print(error_summary)

# ---------------------------------------------------------
# Save detailed results
# ---------------------------------------------------------

test_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDetailed linguistic analysis saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 80)
print("LINGUISTIC ANALYSIS COMPLETE")
print("=" * 80)