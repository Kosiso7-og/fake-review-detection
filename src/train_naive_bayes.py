from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


DATA_PATH = Path("data/raw/deceptive-reviews.csv")
RESULTS_DIR = Path("results")

TEXT_COLUMN = "text"
LABEL_COLUMN = "deceptive"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")

    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nMultinomial Naive Bayes Accuracy: {accuracy:.4f}")
    print(f"Multinomial Naive Bayes Accuracy: {accuracy:.2%}")

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = RESULTS_DIR / "naive_bayes_report.csv"
    report_df.to_csv(report_path)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        cmap="Blues",
    )

    ax.set_title("Multinomial Naive Bayes Confusion Matrix")
    plt.tight_layout()

    confusion_matrix_path = (
        RESULTS_DIR / "naive_bayes_confusion_matrix.png"
    )

    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
