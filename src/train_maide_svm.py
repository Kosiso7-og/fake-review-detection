from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_PATH = Path("data/processed/maide_english_train.csv")
TEST_PATH = Path("data/processed/maide_english_test.csv")

REPORT_PATH = Path("results/maide_svm_report.csv")
CONFUSION_MATRIX_PATH = Path(
    "results/maide_svm_confusion_matrix.png"
)


def main():
    print("=" * 60)
    print("MAiDE-UP LINEAR SVM EXPERIMENT")
    print("=" * 60)

    # Load training and testing data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["review_text"]
    y_train = train_df["label"]

    X_test = test_df["review_text"]
    y_test = test_df["label"]

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")

    # Convert review text into TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nNumber of TF-IDF features: {X_train_tfidf.shape[1]}")

    # Train Linear SVM
    model = LinearSVC(
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Human", "AI-Generated"],
        output_dict=True
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Human", "AI-Generated"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Create results directory
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_PATH)

    # Save confusion matrix visualization
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Human", "AI-Generated"],
        yticklabels=["Human", "AI-Generated"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("MAiDE-up Linear SVM Confusion Matrix")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print("\nResults saved:")
    print(REPORT_PATH)
    print(CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()