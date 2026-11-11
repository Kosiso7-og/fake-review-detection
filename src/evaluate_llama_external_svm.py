
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_PATH = Path(
    "data/processed/maide_english_hotel_train.csv"
)

EXTERNAL_PATH = Path(
    "data/processed/llama_external_test.csv"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
external_df = pd.read_csv(EXTERNAL_PATH)

X_train = train_df["review_text"]
y_train = train_df["label"]

X_external = external_df["review_text"]
y_external = external_df["label"]

print("Training reviews:", len(train_df))
print("External reviews:", len(external_df))

# --------------------------------------------------
# Reproduce original TF-IDF configuration
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_external_tfidf = vectorizer.transform(X_external)

# --------------------------------------------------
# Reproduce original SVM
# --------------------------------------------------

print("\nTraining Linear SVM on MAiDE-up...")

model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# --------------------------------------------------
# External evaluation (no training on Llama)
# --------------------------------------------------

print("\nEvaluating on Ott/Llama external dataset...")

predictions = model.predict(X_external_tfidf)

accuracy = accuracy_score(y_external, predictions)
precision = precision_score(
    y_external, predictions, zero_division=0
)
recall = recall_score(
    y_external, predictions, zero_division=0
)
f1 = f1_score(
    y_external, predictions, zero_division=0
)

matrix = confusion_matrix(
    y_external, predictions, labels=[0, 1]
)

print("\n" + "=" * 60)
print("EXTERNAL LLAMA EVALUATION — LINEAR SVM")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_external,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(matrix)

# --------------------------------------------------
# Save predictions for later paired comparison
# --------------------------------------------------

prediction_df = external_df.copy()
prediction_df["svm_prediction"] = predictions

prediction_path = (
    RESULTS_DIR / "llama_external_svm_predictions.csv"
)

prediction_df.to_csv(prediction_path, index=False)

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_df = pd.DataFrame([{
    "model": "Linear SVM",
    "dataset": "Ott/Llama External",
    "accuracy": accuracy,
    "precision_ai": precision,
    "recall_ai": recall,
    "f1_ai": f1,
    "tn": int(matrix[0, 0]),
    "fp": int(matrix[0, 1]),
    "fn": int(matrix[1, 0]),
    "tp": int(matrix[1, 1])
}])

metrics_path = (
    RESULTS_DIR / "llama_external_svm_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)

# --------------------------------------------------
# Save confusion matrix
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Human", "AI"]
)

display.plot(cmap="Blues")

plt.title("Linear SVM — External Ott/Llama Test")
plt.tight_layout()

figure_path = (
    RESULTS_DIR / "llama_external_svm_confusion_matrix.png"
)

plt.savefig(figure_path, dpi=300)
plt.close()

print("\nSaved files:")
print(prediction_path)
print(metrics_path)
print(figure_path)

print("\nExternal SVM evaluation complete.")
