
import pandas as pd
import re
from pathlib import Path

# File locations
MAIDE_PATH = Path("data/processed/maide_english.csv")
LLAMA_PATH = Path("results/llama_dataset_audit.csv")

# Load datasets
maide = pd.read_csv(MAIDE_PATH)
llama = pd.read_csv(LLAMA_PATH)

print("DATASET OVERLAP AUDIT")
print("=" * 50)

print("\nMAiDE-up columns:")
print(maide.columns.tolist())

print("\nLlama columns:")
print(llama.columns.tolist())

# Normalize text to catch differences in capitalization
# and whitespace.
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Compare the actual review text
maide_texts = set(
    maide["review_text"].apply(normalize_text)
)

llama_texts = set(
    llama["review_text"].apply(normalize_text)
)

# Exclude empty strings
maide_texts.discard("")
llama_texts.discard("")

overlap = maide_texts.intersection(llama_texts)

print("\nMAiDE-up unique nonempty texts:", len(maide_texts))
print("Llama unique nonempty texts:", len(llama_texts))
print("Exact normalized text overlaps:", len(overlap))

# Save summary without exposing full review text
output = Path("results/llama_maide_overlap_summary.csv")

pd.DataFrame([{
    "maide_unique_texts": len(maide_texts),
    "llama_unique_texts": len(llama_texts),
    "exact_normalized_overlaps": len(overlap)
}]).to_csv(output, index=False)

print("\nSaved summary to:", output)
