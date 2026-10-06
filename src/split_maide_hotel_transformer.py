import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DEVELOPMENT_PATH = (
    "data/processed/maide_english_hotel_train.csv"
)

FINAL_TEST_PATH = (
    "data/processed/maide_english_hotel_test.csv"
)

TRAIN_OUTPUT = (
    "data/processed/"
    "maide_english_hotel_transformer_train.csv"
)

VALIDATION_OUTPUT = (
    "data/processed/"
    "maide_english_hotel_transformer_validation.csv"
)

SUMMARY_OUTPUT = (
    "results/"
    "maide_hotel_transformer_split_summary.csv"
)

RANDOM_STATE = 42

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading hotel-disjoint datasets...")

development_df = pd.read_csv(
    DEVELOPMENT_PATH
)

final_test_df = pd.read_csv(
    FINAL_TEST_PATH
)

print(
    f"Development reviews: {len(development_df)}"
)

print(
    f"Final test reviews: {len(final_test_df)}"
)

# ---------------------------------------------------------
# Get hotel names
# ---------------------------------------------------------

development_hotels = sorted(
    development_df["Hotel Name"]
    .dropna()
    .unique()
)

final_test_hotels = set(
    final_test_df["Hotel Name"]
    .dropna()
    .unique()
)

print(
    f"Development hotels: {len(development_hotels)}"
)

print(
    f"Final test hotels: {len(final_test_hotels)}"
)

# ---------------------------------------------------------
# Verify development/test separation
# ---------------------------------------------------------

initial_overlap = (
    set(development_hotels)
    .intersection(final_test_hotels)
)

assert len(initial_overlap) == 0, (
    "ERROR: Development and final test hotels overlap!"
)

# ---------------------------------------------------------
# Split DEVELOPMENT HOTELS into train/validation
#
# 80 development hotels:
# 64 training hotels
# 16 validation hotels
#
# Final 20 test hotels remain untouched.
# ---------------------------------------------------------

train_hotels, validation_hotels = train_test_split(
    development_hotels,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_hotels = set(train_hotels)
validation_hotels = set(validation_hotels)

# ---------------------------------------------------------
# Create datasets
# ---------------------------------------------------------

train_df = development_df[
    development_df["Hotel Name"].isin(train_hotels)
].copy()

validation_df = development_df[
    development_df["Hotel Name"].isin(
        validation_hotels
    )
].copy()

# ---------------------------------------------------------
# Check ALL hotel overlap
# ---------------------------------------------------------

train_validation_overlap = (
    train_hotels.intersection(
        validation_hotels
    )
)

train_test_overlap = (
    train_hotels.intersection(
        final_test_hotels
    )
)

validation_test_overlap = (
    validation_hotels.intersection(
        final_test_hotels
    )
)

# ---------------------------------------------------------
# Print split information
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DISTILBERT HOTEL-DISJOINT SPLIT")
print("=" * 70)

print("\nTRAINING SET")

print(
    f"Reviews: {len(train_df)}"
)

print(
    f"Hotels: "
    f"{train_df['Hotel Name'].nunique()}"
)

print("\nClass distribution:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print("\nVALIDATION SET")

print(
    f"Reviews: {len(validation_df)}"
)

print(
    f"Hotels: "
    f"{validation_df['Hotel Name'].nunique()}"
)

print("\nClass distribution:")

print(
    validation_df["label"]
    .value_counts()
    .sort_index()
)

print("\nFINAL TEST SET")

print(
    f"Reviews: {len(final_test_df)}"
)

print(
    f"Hotels: "
    f"{final_test_df['Hotel Name'].nunique()}"
)

print("\nClass distribution:")

print(
    final_test_df["label"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("HOTEL LEAKAGE CHECK")
print("-" * 70)

print(
    "Train/Validation hotel overlap:",
    len(train_validation_overlap)
)

print(
    "Train/Test hotel overlap:",
    len(train_test_overlap)
)

print(
    "Validation/Test hotel overlap:",
    len(validation_test_overlap)
)

# Stop execution if ANY leakage exists.

assert len(train_validation_overlap) == 0, (
    "Train/validation hotel leakage detected!"
)

assert len(train_test_overlap) == 0, (
    "Train/test hotel leakage detected!"
)

assert len(validation_test_overlap) == 0, (
    "Validation/test hotel leakage detected!"
)

# ---------------------------------------------------------
# Review-count checks
# ---------------------------------------------------------

assert (
    len(train_df)
    + len(validation_df)
    == len(development_df)
), "Reviews were lost during train/validation split."

assert (
    len(train_df)
    + len(validation_df)
    + len(final_test_df)
    == 2000
), "Expected 2,000 total English reviews."

# ---------------------------------------------------------
# Save train and validation
# ---------------------------------------------------------

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)

validation_df.to_csv(
    VALIDATION_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# Save hotel-level summary
# ---------------------------------------------------------

summary_rows = []

for hotel in sorted(train_hotels):

    hotel_data = train_df[
        train_df["Hotel Name"] == hotel
    ]

    summary_rows.append({
        "hotel_name": hotel,
        "split": "train",
        "reviews": len(hotel_data),
        "human_reviews": (
            hotel_data["label"] == 0
        ).sum(),
        "ai_reviews": (
            hotel_data["label"] == 1
        ).sum()
    })

for hotel in sorted(validation_hotels):

    hotel_data = validation_df[
        validation_df["Hotel Name"] == hotel
    ]

    summary_rows.append({
        "hotel_name": hotel,
        "split": "validation",
        "reviews": len(hotel_data),
        "human_reviews": (
            hotel_data["label"] == 0
        ).sum(),
        "ai_reviews": (
            hotel_data["label"] == 1
        ).sum()
    })

for hotel in sorted(final_test_hotels):

    hotel_data = final_test_df[
        final_test_df["Hotel Name"] == hotel
    ]

    summary_rows.append({
        "hotel_name": hotel,
        "split": "test",
        "reviews": len(hotel_data),
        "human_reviews": (
            hotel_data["label"] == 0
        ).sum(),
        "ai_reviews": (
            hotel_data["label"] == 1
        ).sum()
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# Final success message
# ---------------------------------------------------------

print("\nFiles saved:")

print(TRAIN_OUTPUT)
print(VALIDATION_OUTPUT)
print(SUMMARY_OUTPUT)

print("\nSUCCESS:")

print(
    "Training, validation, and final test hotels "
    "are completely disjoint."
)

print("\n" + "=" * 70)
print("TRANSFORMER SPLIT COMPLETE")
print("=" * 70)