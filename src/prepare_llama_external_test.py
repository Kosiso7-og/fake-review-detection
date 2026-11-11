
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load the original truthful human reviews
ott = pd.read_csv(RAW_DIR / "deceptive-reviews.csv")

human = ott[
    ott["deceptive"].str.lower() == "truthful"
].copy()

human = human.rename(columns={
    "text": "review_text",
    "polarity": "sentiment"
})

human["label"] = 0
human["generator"] = "human"
human["source_dataset"] = "Ott"

human = human[
    ["review_text", "label", "sentiment",
     "hotel", "generator", "source_dataset"]
]

# Load the previously audited Llama reviews
llama = pd.read_csv("results/llama_dataset_audit.csv")

llama = llama[
    llama["review_text"].notna()
].copy()

llama = llama[
    llama["review_text"].astype(str).str.strip() != ""
].copy()

llama["label"] = 1
llama["hotel"] = llama["filename"].str.extract(
    r"^[a-z]_([a-z]+)_\d+\.txt$"
)[0]

llama["source_dataset"] = "op_spam_AI"

llama = llama[
    ["review_text", "label", "sentiment",
     "hotel", "generator", "source_dataset"]
]

# Combine human and AI-generated reviews
external = pd.concat(
    [human, llama],
    ignore_index=True
)

# Remove exact duplicate texts within each class.
external["normalized_text"] = (
    external["review_text"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

before = len(external)

external = external.drop_duplicates(
    subset=["label", "normalized_text"]
).copy()

removed = before - len(external)

external = external.drop(columns=["normalized_text"])

# Shuffle reproducibly
external = external.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

output = PROCESSED_DIR / "llama_external_test.csv"
external.to_csv(output, index=False)

print("EXTERNAL EVALUATION DATASET")
print("=" * 45)

print("Rows before deduplication:", before)
print("Duplicates removed:", removed)
print("Final reviews:", len(external))

print("\nClass distribution:")
print(external["label"].value_counts().sort_index())

print("\nSentiment by class:")
print(pd.crosstab(
    external["label"],
    external["sentiment"]
))

print("\nMissing hotel identifiers:")
print(external["hotel"].isna().sum())

print("\nSaved to:", output)
