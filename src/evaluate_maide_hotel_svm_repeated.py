import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RAW_PATH = "data/raw/all_data.csv"

OUTPUT_PATH = (
    "results/maide_hotel_svm_repeated_results.csv"
)

SUMMARY_PATH = (
    "results/maide_hotel_svm_repeated_summary.csv"
)

# Different random seeds create different groups
# of 20 unseen test hotels.

SEEDS = [42, 7, 21, 84, 100]

# ---------------------------------------------------------
# Load MAiDE-up
# ---------------------------------------------------------

print("Loading MAiDE-up dataset...")

df = pd.read_csv(RAW_PATH)

# Keep English reviews only

df = df[
    df["Review_Language"] == "English"
].copy()

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

# source:
# 0 = Human
# 1 = AI-generated

df = df.rename(
    columns={"source": "label"}
)

df = df[
    [
        "review_text",
        "label",
        "Hotel Name",
        "City Name"
    ]
].copy()

df = df[
    df["review_text"].str.len() > 0
].copy()

# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

hotels = sorted(
    df["Hotel Name"]
    .dropna()
    .unique()
)

print(f"English reviews: {len(df)}")
print(f"Unique hotels: {len(hotels)}")
print(f"Repeated splits: {len(SEEDS)}")

# ---------------------------------------------------------
# Store results
# ---------------------------------------------------------

results = []

# ---------------------------------------------------------
# Repeated hotel-disjoint evaluation
# ---------------------------------------------------------

for seed in SEEDS:

    print("\n" + "=" * 70)
    print(f"RANDOM SEED: {seed}")
    print("=" * 70)

    # -----------------------------------------------------
    # Split HOTEL NAMES
    # -----------------------------------------------------

    train_hotels, test_hotels = train_test_split(
        hotels,
        test_size=0.20,
        random_state=seed
    )

    train_hotels = set(train_hotels)
    test_hotels = set(test_hotels)

    # -----------------------------------------------------
    # Create review-level datasets
    # -----------------------------------------------------

    train_df = df[
        df["Hotel Name"].isin(train_hotels)
    ].copy()

    test_df = df[
        df["Hotel Name"].isin(test_hotels)
    ].copy()

    # -----------------------------------------------------
    # Verify hotel separation
    # -----------------------------------------------------

    overlap = (
        train_hotels.intersection(test_hotels)
    )

    assert len(overlap) == 0, (
        "Hotel leakage detected!"
    )

    # Because each hotel should contain
    # 10 Human + 10 AI reviews

    train_counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    test_counts = (
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Train: {len(train_df)} reviews, "
        f"{len(train_hotels)} hotels"
    )

    print(
        f"Test:  {len(test_df)} reviews, "
        f"{len(test_hotels)} hotels"
    )

    print(
        f"Hotel overlap: {len(overlap)}"
    )

    print(
        "Train labels:",
        train_counts.to_dict()
    )

    print(
        "Test labels:",
        test_counts.to_dict()
    )

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    X_train = vectorizer.fit_transform(
        train_df["review_text"]
    )

    X_test = vectorizer.transform(
        test_df["review_text"]
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    # -----------------------------------------------------
    # Train Linear SVM
    # -----------------------------------------------------

    model = LinearSVC()

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Treat AI-generated review (1) as positive class.

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = matrix.ravel()

    print(
        f"\nAccuracy:  {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(f"AI Precision: {precision:.4f}")
    print(f"AI Recall:    {recall:.4f}")
    print(f"AI F1:        {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    results.append({
        "seed": seed,
        "train_hotels": len(train_hotels),
        "test_hotels": len(test_hotels),
        "train_reviews": len(train_df),
        "test_reviews": len(test_df),
        "hotel_overlap": len(overlap),
        "accuracy": accuracy,
        "ai_precision": precision,
        "ai_recall": recall,
        "ai_f1": f1,
        "true_human": tn,
        "human_predicted_ai": fp,
        "ai_predicted_human": fn,
        "true_ai": tp
    })

# ---------------------------------------------------------
# Convert results to DataFrame
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

metric_columns = [
    "accuracy",
    "ai_precision",
    "ai_recall",
    "ai_f1"
]

summary_df = (
    results_df[metric_columns]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    )
)

# ---------------------------------------------------------
# Display all results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ALL HOTEL-DISJOINT RESULTS")
print("=" * 70)

display_columns = [
    "seed",
    "accuracy",
    "ai_precision",
    "ai_recall",
    "ai_f1"
]

print(
    results_df[
        display_columns
    ].round(4).to_string(index=False)
)

# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY ACROSS ALL SPLITS")
print("=" * 70)

print(
    summary_df
    .round(4)
    .to_string()
)

# ---------------------------------------------------------
# Accuracy in percentages
# ---------------------------------------------------------

mean_accuracy = (
    results_df["accuracy"].mean() * 100
)

std_accuracy = (
    results_df["accuracy"].std() * 100
)

min_accuracy = (
    results_df["accuracy"].min() * 100
)

max_accuracy = (
    results_df["accuracy"].max() * 100
)

print("\nAccuracy summary:")

print(
    f"Mean accuracy: {mean_accuracy:.2f}%"
)

print(
    f"Standard deviation: {std_accuracy:.2f} "
    "percentage points"
)

print(
    f"Minimum accuracy: {min_accuracy:.2f}%"
)

print(
    f"Maximum accuracy: {max_accuracy:.2f}%"
)

# ---------------------------------------------------------
# Compare with original random review split
# ---------------------------------------------------------

ORIGINAL_RANDOM_ACCURACY = 0.9275

difference = (
    results_df["accuracy"].mean()
    - ORIGINAL_RANDOM_ACCURACY
) * 100

print("\nComparison with original random split:")

print(
    "Original random-split SVM accuracy: "
    f"{ORIGINAL_RANDOM_ACCURACY * 100:.2f}%"
)

print(
    "Mean hotel-disjoint SVM accuracy: "
    f"{mean_accuracy:.2f}%"
)

print(
    "Difference: "
    f"{difference:+.2f} percentage points"
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

summary_df.to_csv(
    SUMMARY_PATH
)

print("\nFiles saved:")

print(OUTPUT_PATH)
print(SUMMARY_PATH)

print("\n" + "=" * 70)
print("REPEATED HOTEL EVALUATION COMPLETE")
print("=" * 70)