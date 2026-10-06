import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TEST_PATH = "data/processed/maide_english_test.csv"
ERROR_PATH = "results/distilbert_errors.csv"

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

test_df = pd.read_csv(TEST_PATH)
error_df = pd.read_csv(ERROR_PATH)

# ---------------------------------------------------------
# Calculate text length
# ---------------------------------------------------------

test_df["character_count"] = test_df["review_text"].astype(str).str.len()

test_df["word_count"] = (
    test_df["review_text"]
    .astype(str)
    .str.split()
    .str.len()
)

# ---------------------------------------------------------
# Determine correct / incorrect predictions
# ---------------------------------------------------------

# Add predictions from error file where available.
# Start by assuming every review is correct.

test_df["prediction_status"] = "Correct"

# Match each error review back to the test dataset
for _, error in error_df.iterrows():

    mask = (
        (test_df["review_text"] == error["review_text"])
        & (test_df["label"] == error["label"])
    )

    test_df.loc[mask, "prediction_status"] = error["error_type"]

# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DISTILBERT ERROR PATTERN ANALYSIS")
print("=" * 70)

print("\nReview length by prediction status:")
print(
    test_df.groupby("prediction_status")[["character_count", "word_count"]]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# ---------------------------------------------------------
# Length by actual class
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("Review length by actual class")
print("-" * 70)

print(
    test_df.groupby("label")[["character_count", "word_count"]]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# ---------------------------------------------------------
# Error-specific statistics
# ---------------------------------------------------------

errors_only = test_df[test_df["prediction_status"] != "Correct"]

print("\n" + "-" * 70)
print("Error-specific length statistics")
print("-" * 70)

print(
    errors_only.groupby("prediction_status")[["character_count", "word_count"]]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# ---------------------------------------------------------
# Save enhanced error file
# ---------------------------------------------------------

enhanced_errors = test_df[
    test_df["prediction_status"] != "Correct"
].copy()

enhanced_errors.to_csv(
    "results/distilbert_errors_with_lengths.csv",
    index=False
)

print("\nEnhanced error file saved to:")
print("results/distilbert_errors_with_lengths.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)