
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAIDE_PATH = Path("data/processed/maide_english.csv")
LLAMA_PATH = Path("results/llama_dataset_audit.csv")

maide = pd.read_csv(MAIDE_PATH)
llama = pd.read_csv(LLAMA_PATH)

maide_texts = maide["review_text"].fillna("").astype(str)
llama = llama[llama["review_text"].notna()].copy()
llama = llama[
    llama["review_text"].astype(str).str.strip() != ""
].copy()

llama_texts = llama["review_text"].astype(str)

print("NEAR-DUPLICATE SIMILARITY AUDIT")
print("=" * 50)

# Fit one shared vocabulary for both datasets.
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english",
    max_features=20000
)

combined = pd.concat(
    [maide_texts, llama_texts],
    ignore_index=True
)

matrix = vectorizer.fit_transform(combined)

maide_matrix = matrix[:len(maide_texts)]
llama_matrix = matrix[len(maide_texts):]

similarities = cosine_similarity(
    llama_matrix,
    maide_matrix
)

max_similarity = similarities.max(axis=1)
closest_match = similarities.argmax(axis=1)

print("Llama reviews checked:", len(llama_texts))
print("Mean maximum similarity:", round(max_similarity.mean(), 4))
print("Highest similarity:", round(max_similarity.max(), 4))

for threshold in [0.70, 0.80, 0.90, 0.95]:
    count = int((max_similarity >= threshold).sum())
    print(f"Reviews with similarity >= {threshold}: {count}")

results = pd.DataFrame({
    "llama_filename": llama["filename"].values,
    "llama_sentiment": llama["sentiment"].values,
    "max_cosine_similarity": max_similarity,
    "closest_maide_row": closest_match
})

output = Path("results/llama_maide_similarity_audit.csv")
results.to_csv(output, index=False)

print("\nTop 5 most similar review pairs:")

top_indices = np.argsort(max_similarity)[-5:][::-1]

for idx in top_indices:
    print(
        f"Llama: {llama.iloc[idx]['filename']} "
        f"({llama.iloc[idx]['sentiment']})"
    )
    print("Similarity:", round(float(max_similarity[idx]), 4))
    print("Closest MAiDE row:", int(closest_match[idx]))
    print("-" * 40)

print("\nSaved results to:", output)
