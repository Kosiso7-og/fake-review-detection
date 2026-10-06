import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_PATH = "data/raw/all_data.csv"

TRAIN_OUTPUT = (
    "data/processed/maide_english_hotel_train.csv"
)

TEST_OUTPUT = (
    "data/processed/maide_english_hotel_test.csv"
)

SPLIT_SUMMARY_OUTPUT = (
    "results/maide_hotel_disjoint_split.csv"
)

RANDOM_STATE = 42

# ---------------------------------------------------------
# Load MAiDE-up
# ---------------------------------------------------------

print("Loading MAiDE-up dataset...")

df = pd.read_csv(RAW_PATH)

# ---------------------------------------------------------
# Keep English reviews only
# ---------------------------------------------------------

df = df[
    df["Review_Language"] == "English"
].copy()

print(f"English reviews: {len(df)}")

# ---------------------------------------------------------
# Construct review text
# ---------------------------------------------------------

df["Upside_Review"] = (
    df["Upside_Review"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Downside_Review"] = (
    df["Downside_Review"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["review_text"] = (
    df["Upside_Review"]
    + " "
    + df["Downside_Review"]
).str.strip()

# Rename source to label:
# 0 = Human
# 1 = AI-generated

df = df.rename(
    columns={"source": "label"}
)

# Keep only fields needed for the experiment

df = df[
    [
        "review_text",
        "label",
        "Hotel Name",
        "City Name"
    ]
].copy()

# Remove empty review text, if any

df = df[
    df["review_text"].str.len() > 0
].copy()

# ---------------------------------------------------------
# Get unique hotels
# ---------------------------------------------------------

hotels = sorted(
    df["Hotel Name"]
    .dropna()
    .unique()
)

print(f"Unique hotels: {len(hotels)}")

# ---------------------------------------------------------
# Split HOTEL NAMES, not individual reviews
# ---------------------------------------------------------

train_hotels, test_hotels = train_test_split(
    hotels,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_hotels = set(train_hotels)
test_hotels = set(test_hotels)

# ---------------------------------------------------------
# Create datasets
# ---------------------------------------------------------

train_df = df[
    df["Hotel Name"].isin(train_hotels)
].copy()

test_df = df[
    df["Hotel Name"].isin(test_hotels)
].copy()

# ---------------------------------------------------------
# Verify there is NO hotel overlap
# ---------------------------------------------------------

actual_train_hotels = set(
    train_df["Hotel Name"].unique()
)

actual_test_hotels = set(
    test_df["Hotel Name"].unique()
)

overlap = (
    actual_train_hotels
    .intersection(actual_test_hotels)
)

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("HOTEL-DISJOINT MAIDE-UP SPLIT")
print("=" * 70)

print("\nTRAINING SET")

print(f"Reviews: {len(train_df)}")
print(f"Hotels: {train_df['Hotel Name'].nunique()}")

print("\nClass distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nTEST SET")

print(f"Reviews: {len(test_df)}")
print(f"Hotels: {test_df['Hotel Name'].nunique()}")

print("\nClass distribution:")
print(test_df["label"].value_counts().sort_index())

print("\n" + "-" * 70)
print("GENERALIZATION CHECK")
print("-" * 70)

print(
    f"Hotels appearing in BOTH train and test: "
    f"{len(overlap)}"
)

if len(overlap) == 0:
    print(
        "SUCCESS: Test hotels are completely unseen "
        "during training."
    )
else:
    print("WARNING: Hotel overlap detected.")

# ---------------------------------------------------------
# Additional validation
# ---------------------------------------------------------

assert len(overlap) == 0, (
    "Hotel leakage detected!"
)

assert len(train_df) + len(test_df) == len(df), (
    "Some reviews were lost during splitting."
)

# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False
)

test_df.to_csv(
    TEST_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# Save split summary
# ---------------------------------------------------------

summary_rows = []

for hotel in sorted(actual_train_hotels):

    hotel_data = train_df[
        train_df["Hotel Name"] == hotel
    ]

    summary_rows.append({
        "hotel_name": hotel,
        "split": "train",
        "total_reviews": len(hotel_data),
        "human_reviews": (
            hotel_data["label"] == 0
        ).sum(),
        "ai_reviews": (
            hotel_data["label"] == 1
        ).sum()
    })

for hotel in sorted(actual_test_hotels):

    hotel_data = test_df[
        test_df["Hotel Name"] == hotel
    ]

    summary_rows.append({
        "hotel_name": hotel,
        "split": "test",
        "total_reviews": len(hotel_data),
        "human_reviews": (
            hotel_data["label"] == 0
        ).sum(),
        "ai_reviews": (
            hotel_data["label"] == 1
        ).sum()
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SPLIT_SUMMARY_OUTPUT,
    index=False
)

print("\nFiles saved:")

print(TRAIN_OUTPUT)
print(TEST_OUTPUT)
print(SPLIT_SUMMARY_OUTPUT)

print("\n" + "=" * 70)
print("HOTEL-DISJOINT SPLIT COMPLETE")
print("=" * 70)