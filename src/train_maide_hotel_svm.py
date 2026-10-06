import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TRAIN_PATH = (
    "data/processed/maide_english_hotel_train.csv"
)

TEST_PATH = (
    "data/processed/maide_english_hotel_test.csv"
)

REPORT_PATH = (
    "results/maide_hotel_svm_report.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/maide_hotel_svm_confusion_matrix.png"
)

# ---------------------------------------------------------
# Load hotel-disjoint datasets
# ---------------------------------------------------------

print("Loading hotel-disjoint datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training reviews: {len(train_df)}")
print(f"Testing reviews: {len(test_df)}")

print(
    f"Training hotels: "
    f"{train_df['Hotel Name'].nunique()}"
)

print(
    f"Testing hotels: "
    f"{test_df['Hotel Name'].nunique()}"
)

# ---------------------------------------------------------
# Verify hotel separation
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
    f"Hotel overlap: {len(hotel_overlap)}"
)

assert len(hotel_overlap) == 0, (
    "Hotel leakage detected!"
)

# ---------------------------------------------------------
# Prepare text and labels
# ---------------------------------------------------------

X_train = train_df["review_text"]
y_train = train_df["label"]

X_test = test_df["review_text"]
y_test = test_df["label"]

# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"Number of TF-IDF features: "
    f"{X_train_tfidf.shape[1]}"
)

# ---------------------------------------------------------
# Train Linear SVM
# ---------------------------------------------------------

print("\nTraining Linear SVM...")

model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

predictions = model.predict(
    X_test_tfidf
)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions,
    target_names=["Human", "AI"],
    output_dict=True
)

matrix = confusion_matrix(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("HOTEL-DISJOINT LINEAR SVM RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Human", "AI"]
    )
)

print("Confusion Matrix:")
print(matrix)

# ---------------------------------------------------------
# Save classification report
# ---------------------------------------------------------

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    REPORT_PATH
)

# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Human", "AI"]
)

display.plot()

plt.title(
    "Linear SVM - Hotel-Disjoint MAiDE-up Test Set"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\nFiles saved:")

print(REPORT_PATH)
print(CONFUSION_MATRIX_PATH)

print("\n" + "=" * 70)
print("SVM GENERALIZATION TEST COMPLETE")
print("=" * 70)