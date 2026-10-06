import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_PATH = "models/distilbert-maide"
TEST_PATH = "data/processed/maide_english_test.csv"
OUTPUT_PATH = "results/distilbert_errors.csv"

# ---------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------

print("Loading DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# Use CPU because this project was trained/evaluated on CPU
device = torch.device("cpu")
model.to(device)

# ---------------------------------------------------------
# Load test dataset
# ---------------------------------------------------------

print("Loading test dataset...")

df = pd.read_csv(TEST_PATH)

print(f"Test dataset size: {len(df)}")

# ---------------------------------------------------------
# Make predictions
# ---------------------------------------------------------

predictions = []

print("Running predictions...")

for index, row in df.iterrows():

    text = str(row["review_text"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    predictions.append(predicted_label)

# ---------------------------------------------------------
# Add predictions to dataframe
# ---------------------------------------------------------

df["predicted_label"] = predictions

# Convert numeric labels to readable names

label_names = {
    0: "Human",
    1: "AI"
}

df["actual_class"] = df["label"].map(label_names)
df["predicted_class"] = df["predicted_label"].map(label_names)

# ---------------------------------------------------------
# Identify errors
# ---------------------------------------------------------

errors = df[df["label"] != df["predicted_label"]].copy()

# Identify the direction of the error

errors["error_type"] = errors.apply(
    lambda row: f"{row['actual_class']} -> {row['predicted_class']}",
    axis=1
)

# ---------------------------------------------------------
# Save errors
# ---------------------------------------------------------

errors.to_csv(OUTPUT_PATH, index=False)

# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DISTILBERT ERROR ANALYSIS")
print("=" * 60)

print(f"\nTotal test reviews: {len(df)}")
print(f"Correct predictions: {(df['label'] == df['predicted_label']).sum()}")
print(f"Incorrect predictions: {len(errors)}")

print("\nError breakdown:")
print(errors["error_type"].value_counts())

print("\nActual class distribution among errors:")
print(errors["actual_class"].value_counts())

print("\nPredicted class distribution among errors:")
print(errors["predicted_class"].value_counts())

print("\nErrors saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)