import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TRAIN_PATH = (
    "data/processed/maide_english_hotel_train.csv"
)

TEST_PATH = (
    "data/processed/maide_english_hotel_test.csv"
)

DISTILBERT_PREDICTIONS_PATH = (
    "results/distilbert_hotel_predictions.csv"
)

COMPARISON_OUTPUT = (
    "results/svm_distilbert_comparison.csv"
)

SUMMARY_OUTPUT = (
    "results/svm_distilbert_mcnemar_summary.csv"
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 70)
print("SVM VS DISTILBERT PAIRED COMPARISON")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

distilbert_df = pd.read_csv(
    DISTILBERT_PREDICTIONS_PATH
)

print(f"SVM training reviews: {len(train_df)}")
print(f"Shared test reviews: {len(test_df)}")
print(
    f"DistilBERT predictions: "
    f"{len(distilbert_df)}"
)

# ---------------------------------------------------------
# Verify datasets align
# ---------------------------------------------------------

assert len(test_df) == 400, (
    "Expected 400 test reviews."
)

assert len(distilbert_df) == 400, (
    "Expected 400 DistilBERT predictions."
)

assert test_df["review_text"].tolist() == (
    distilbert_df["review_text"].tolist()
), (
    "ERROR: Test reviews and DistilBERT "
    "predictions are not in the same order."
)

assert test_df["label"].tolist() == (
    distilbert_df["label"].tolist()
), (
    "ERROR: Labels do not match."
)

print(
    "Dataset alignment check: PASSED"
)

# ---------------------------------------------------------
# Verify hotel separation for SVM
# ---------------------------------------------------------

train_hotels = set(
    train_df["Hotel Name"].unique()
)

test_hotels = set(
    test_df["Hotel Name"].unique()
)

hotel_overlap = (
    train_hotels.intersection(test_hotels)
)

print(
    f"Train/test hotel overlap: "
    f"{len(hotel_overlap)}"
)

assert len(hotel_overlap) == 0, (
    "Hotel leakage detected!"
)

# ---------------------------------------------------------
# Train SVM
# ---------------------------------------------------------

print("\nTraining Linear SVM...")

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

svm_model = LinearSVC()

svm_model.fit(
    X_train,
    train_df["label"]
)

svm_predictions = svm_model.predict(
    X_test
)

# ---------------------------------------------------------
# DistilBERT predictions
# ---------------------------------------------------------

distilbert_predictions = (
    distilbert_df["prediction"]
    .to_numpy()
)

true_labels = (
    test_df["label"]
    .to_numpy()
)

# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

svm_accuracy = accuracy_score(
    true_labels,
    svm_predictions
)

distilbert_accuracy = accuracy_score(
    true_labels,
    distilbert_predictions
)

print("\n" + "-" * 70)
print("MODEL ACCURACY")
print("-" * 70)

print(
    f"SVM accuracy: "
    f"{svm_accuracy * 100:.2f}%"
)

print(
    f"DistilBERT accuracy: "
    f"{distilbert_accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# Correct / incorrect status
# ---------------------------------------------------------

svm_correct = (
    svm_predictions == true_labels
)

distilbert_correct = (
    distilbert_predictions == true_labels
)

# ---------------------------------------------------------
# Paired comparison counts
# ---------------------------------------------------------

both_correct = np.sum(
    svm_correct & distilbert_correct
)

both_wrong = np.sum(
    (~svm_correct) & (~distilbert_correct)
)

svm_only_correct = np.sum(
    svm_correct & (~distilbert_correct)
)

distilbert_only_correct = np.sum(
    (~svm_correct) & distilbert_correct
)

print("\n" + "-" * 70)
print("PAIRED PREDICTION COMPARISON")
print("-" * 70)

print(
    f"Both models correct: "
    f"{both_correct}"
)

print(
    f"Both models wrong: "
    f"{both_wrong}"
)

print(
    "SVM correct / DistilBERT wrong: "
    f"{svm_only_correct}"
)

print(
    "SVM wrong / DistilBERT correct: "
    f"{distilbert_only_correct}"
)

assert (
    both_correct
    + both_wrong
    + svm_only_correct
    + distilbert_only_correct
    == 400
), "Paired counts do not total 400."

# ---------------------------------------------------------
# Exact McNemar test
#
# Only disagreements matter:
#
# b = SVM correct, DistilBERT wrong
# c = SVM wrong, DistilBERT correct
#
# Under the null hypothesis, each direction of
# disagreement is equally likely.
#
# We calculate the exact two-sided binomial probability.
# This avoids requiring statsmodels.
# ---------------------------------------------------------

b = int(svm_only_correct)
c = int(distilbert_only_correct)

n_disagreements = b + c

print("\n" + "-" * 70)
print("EXACT MCNEMAR TEST")
print("-" * 70)

print(f"b = {b}")
print(f"c = {c}")
print(
    f"Total disagreements: "
    f"{n_disagreements}"
)

# ---------------------------------------------------------
# Exact two-sided binomial calculation
# ---------------------------------------------------------

if n_disagreements == 0:

    p_value = 1.0

else:

    from math import comb

    smaller = min(b, c)

    cumulative_probability = sum(
        comb(n_disagreements, k)
        * (0.5 ** n_disagreements)
        for k in range(smaller + 1)
    )

    p_value = min(
        1.0,
        2 * cumulative_probability
    )

print(
    f"Exact two-sided p-value: "
    f"{p_value:.6f}"
)

# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------

ALPHA = 0.05

print("\nInterpretation:")

if p_value < ALPHA:

    print(
        "The difference is statistically significant "
        "at alpha = 0.05."
    )

    print(
        "The paired results provide evidence that "
        "the models differ in classification accuracy "
        "on this test set."
    )

else:

    print(
        "The difference is NOT statistically significant "
        "at alpha = 0.05."
    )

    print(
        "The paired results do not provide sufficient "
        "evidence that the models differ in classification "
        "accuracy on this test set."
    )

# ---------------------------------------------------------
# Create review-level comparison file
# ---------------------------------------------------------

comparison_df = test_df.copy()

comparison_df["svm_prediction"] = (
    svm_predictions
)

comparison_df["distilbert_prediction"] = (
    distilbert_predictions
)

comparison_df["svm_correct"] = (
    svm_correct
)

comparison_df["distilbert_correct"] = (
    distilbert_correct
)

def comparison_category(row):

    if (
        row["svm_correct"]
        and row["distilbert_correct"]
    ):
        return "Both correct"

    elif (
        not row["svm_correct"]
        and not row["distilbert_correct"]
    ):
        return "Both wrong"

    elif (
        row["svm_correct"]
        and not row["distilbert_correct"]
    ):
        return "SVM only correct"

    else:
        return "DistilBERT only correct"

comparison_df["comparison_category"] = (
    comparison_df.apply(
        comparison_category,
        axis=1
    )
)

comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

summary_df = pd.DataFrame([
    {
        "test_reviews": 400,
        "svm_accuracy": svm_accuracy,
        "distilbert_accuracy": (
            distilbert_accuracy
        ),
        "accuracy_difference": (
            distilbert_accuracy
            - svm_accuracy
        ),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "svm_only_correct": (
            svm_only_correct
        ),
        "distilbert_only_correct": (
            distilbert_only_correct
        ),
        "total_disagreements": (
            n_disagreements
        ),
        "mcnemar_exact_p_value": (
            p_value
        ),
        "alpha": ALPHA,
        "statistically_significant": (
            p_value < ALPHA
        )
    }
])

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print("\nFiles saved:")

print(COMPARISON_OUTPUT)
print(SUMMARY_OUTPUT)

print("\n" + "=" * 70)
print("PAIRED COMPARISON COMPLETE")
print("=" * 70)