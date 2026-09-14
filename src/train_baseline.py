from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/raw/deceptive-reviews.csv")
RESULTS_DIR = Path("results")

TEXT_COLUMN = "text"
LABEL_COLUMN = "deceptive"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Keep only review text and the target label.
    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

    # Remove rows with missing text or labels.
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    # Make sure review text is treated as text.
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    print(f"Dataset shape after cleaning: {df.shape}")

    print("\nClass distribution:")
    print(df[LABEL_COLUMN].value_counts())

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=20_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Making predictions...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n===== BASELINE RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        RESULTS_DIR / "logistic_regression_report.csv"
    )

    matrix = confusion_matrix(y_test, predictions)

    print("\nConfusion matrix:")
    print(matrix)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )

    plt.title("TF-IDF + Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "logistic_regression_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    print("\nResults saved in the results/ folder.")


if __name__ == "__main__":
    main()