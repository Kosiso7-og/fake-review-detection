import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_PATH = "data/raw/all_data.csv"
TRAIN_PATH = "data/processed/maide_english_train.csv"
TEST_PATH = "data/processed/maide_english_test.csv"

OUTPUT_PATH = "results/maide_hotel_audit.csv"

# ---------------------------------------------------------
# Load original MAiDE-up dataset
# ---------------------------------------------------------

print("Loading MAiDE-up dataset...")

raw_df = pd.read_csv(RAW_PATH)

# Keep English reviews only
english_df = raw_df[
    raw_df["Review_Language"] == "English"
].copy()

print(f"\nEnglish reviews: {len(english_df)}")

# ---------------------------------------------------------
# Basic hotel information
# ---------------------------------------------------------

unique_hotels = english_df["Hotel Name"].nunique()

print(f"Unique hotels: {unique_hotels}")

# Count reviews and labels for each hotel
hotel_summary = (
    english_df
    .groupby(["City Name", "Hotel Name", "source"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Make sure both label columns exist
if 0 not in hotel_summary.columns:
    hotel_summary[0] = 0

if 1 not in hotel_summary.columns:
    hotel_summary[1] = 0

hotel_summary = hotel_summary.rename(
    columns={
        0: "human_reviews",
        1: "ai_reviews"
    }
)

hotel_summary["total_reviews"] = (
    hotel_summary["human_reviews"]
    + hotel_summary["ai_reviews"]
)

hotel_summary["contains_both_classes"] = (
    (hotel_summary["human_reviews"] > 0)
    & (hotel_summary["ai_reviews"] > 0)
)

# ---------------------------------------------------------
# Hotel-level summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MAIDE-UP ENGLISH HOTEL AUDIT")
print("=" * 70)

print(f"\nTotal English reviews: {len(english_df)}")
print(f"Unique hotels: {unique_hotels}")

print(
    f"Hotels containing both Human and AI reviews: "
    f"{hotel_summary['contains_both_classes'].sum()}"
)

print(
    f"Hotels missing one of the two classes: "
    f"{(~hotel_summary['contains_both_classes']).sum()}"
)

print("\nReviews per hotel:")
print(
    hotel_summary["total_reviews"]
    .describe()
    .round(2)
)

print("\nHuman reviews per hotel:")
print(
    hotel_summary["human_reviews"]
    .describe()
    .round(2)
)

print("\nAI reviews per hotel:")
print(
    hotel_summary["ai_reviews"]
    .describe()
    .round(2)
)

# ---------------------------------------------------------
# Save hotel summary
# ---------------------------------------------------------

hotel_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nHotel-level audit saved to: {OUTPUT_PATH}")

# ---------------------------------------------------------
# Audit CURRENT random train/test split
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CURRENT RANDOM SPLIT HOTEL OVERLAP")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# The processed files only contain review_text + label,
# so we reconstruct the combined review text in the raw dataset.

english_df["Upside_Review"] = (
    english_df["Upside_Review"].fillna("")
)

english_df["Downside_Review"] = (
    english_df["Downside_Review"].fillna("")
)

english_df["review_text"] = (
    english_df["Upside_Review"].astype(str).str.strip()
    + " "
    + english_df["Downside_Review"].astype(str).str.strip()
).str.strip()

# Match processed reviews back to their hotel names.
train_with_hotels = train_df.merge(
    english_df[
        ["review_text", "source", "Hotel Name", "City Name"]
    ],
    left_on=["review_text", "label"],
    right_on=["review_text", "source"],
    how="left"
)

test_with_hotels = test_df.merge(
    english_df[
        ["review_text", "source", "Hotel Name", "City Name"]
    ],
    left_on=["review_text", "label"],
    right_on=["review_text", "source"],
    how="left"
)

# ---------------------------------------------------------
# Check matching success
# ---------------------------------------------------------

train_missing = train_with_hotels["Hotel Name"].isna().sum()
test_missing = test_with_hotels["Hotel Name"].isna().sum()

print(f"\nTraining reviews without matched hotel: {train_missing}")
print(f"Testing reviews without matched hotel: {test_missing}")

# ---------------------------------------------------------
# Calculate hotel overlap
# ---------------------------------------------------------

train_hotels = set(
    train_with_hotels["Hotel Name"].dropna().unique()
)

test_hotels = set(
    test_with_hotels["Hotel Name"].dropna().unique()
)

overlapping_hotels = train_hotels.intersection(test_hotels)

train_only_hotels = train_hotels - test_hotels
test_only_hotels = test_hotels - train_hotels

print(f"\nUnique hotels in training set: {len(train_hotels)}")
print(f"Unique hotels in testing set: {len(test_hotels)}")

print(
    f"Hotels appearing in BOTH training and testing: "
    f"{len(overlapping_hotels)}"
)

print(
    f"Hotels appearing ONLY in training: "
    f"{len(train_only_hotels)}"
)

print(
    f"Hotels appearing ONLY in testing: "
    f"{len(test_only_hotels)}"
)

if len(test_hotels) > 0:

    overlap_percentage = (
        len(overlapping_hotels)
        / len(test_hotels)
        * 100
    )

    print(
        f"\nPercentage of test hotels also seen during training: "
        f"{overlap_percentage:.2f}%"
    )

# ---------------------------------------------------------
# Count TEST REVIEWS from seen vs unseen hotels
# ---------------------------------------------------------

test_with_hotels["hotel_seen_in_training"] = (
    test_with_hotels["Hotel Name"].isin(train_hotels)
)

seen_test_reviews = (
    test_with_hotels["hotel_seen_in_training"].sum()
)

unseen_test_reviews = (
    (~test_with_hotels["hotel_seen_in_training"]).sum()
)

print(
    f"\nTest reviews from hotels seen during training: "
    f"{seen_test_reviews}"
)

print(
    f"Test reviews from completely unseen hotels: "
    f"{unseen_test_reviews}"
)

if len(test_with_hotels) > 0:

    seen_percentage = (
        seen_test_reviews
        / len(test_with_hotels)
        * 100
    )

    print(
        f"Percentage of test reviews from seen hotels: "
        f"{seen_percentage:.2f}%"
    )

# ---------------------------------------------------------
# Show overlapping hotels
# ---------------------------------------------------------

print("\nExample hotels appearing in BOTH sets:")

for hotel in sorted(overlapping_hotels)[:15]:
    print(f"  - {hotel}")

print("\n" + "=" * 70)
print("HOTEL AUDIT COMPLETE")
print("=" * 70)