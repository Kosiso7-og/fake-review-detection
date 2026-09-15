from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# -----------------------------
# 1. File paths and settings
# -----------------------------

DATA_PATH = Path("data/raw/deceptive-reviews.csv")
RESULTS_DIR = Path("results")

TEXT_COLUMN = "text"
LABEL_COLUMN = "deceptive"

RANDOM_STATE = 42


# -----------------------------
# 2. Load the dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# -----------------------------
# 3. Select required columns
# -----------------------------

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]


# -----------------------------
# 4. Convert labels to numbers
# -----------------------------

label_mapping = {
    "truthful": 0,
    "deceptive": 1,
}

y = y.map(label_mapping)


# -----------------------------
# 5. Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")


# -----------------------------
# 6. Convert text into TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# 7. Train the SVM model
# -----------------------------

model = LinearSVC(
    random_state=RANDOM_STATE,
)

model.fit(X_train_tfidf, y_train)


# -----------------------------
# 8. Make predictions
# -----------------------------

y_pred = model.predict(X_test_tfidf)


# -----------------------------
# 9. Evaluate the model
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Results")
print("-" * 30)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["truthful", "deceptive"],
    )
)


# -----------------------------
# 10. Save classification report
# -----------------------------

RESULTS_DIR.mkdir(exist_ok=True)

report = classification_report(
    y_test,
    y_pred,
    target_names=["truthful", "deceptive"],
    output_dict=True,
)

report_df = pd.DataFrame(report).transpose()

report_path = RESULTS_DIR / "svm_report.csv"
report_df.to_csv(report_path)

print(f"Classification report saved to: {report_path}")


# -----------------------------
# 11. Save confusion matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["truthful", "deceptive"],
)

display.plot()

plt.title("SVM Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = (
    RESULTS_DIR / "svm_confusion_matrix.png"
)

plt.savefig(confusion_matrix_path)
plt.close()

print(
    f"Confusion matrix saved to: "
    f"{confusion_matrix_path}"
)