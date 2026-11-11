
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pathlib import Path

INPUT_PATH = Path(
    "results/llama_linguistic_features.csv"
)

OUTPUT_PATH = Path(
    "results/llama_linguistic_significance.csv"
)

df = pd.read_csv(INPUT_PATH)

# Focus on Llama reviews of similar lengths.
df = df[
    df["word_count"].between(40, 60)
].copy()

correct = df[
    df["outcome"] == "Both Correct"
]

wrong = df[
    df["outcome"] == "Both Wrong"
]

features = [
    "avg_word_length",
    "avg_sentence_length",
    "lexical_diversity",
    "sentence_count",
    "comma_count",
    "exclamation_count",
    "question_count",
]

results = []

print("=" * 65)
print("LINGUISTIC FEATURE SIGNIFICANCE TESTING")
print("=" * 65)

print("\nBoth correct:", len(correct))
print("Both wrong:", len(wrong))

for feature in features:
    x = correct[feature].dropna()
    y = wrong[feature].dropna()

    if len(x) == 0 or len(y) == 0:
        continue

    statistic, p_value = mannwhitneyu(
        x, y, alternative="two-sided"
    )

    # Rank-biserial effect size:
    # Positive means higher values in the correct group.
    effect_size = (
        2 * statistic / (len(x) * len(y)) - 1
    )

    results.append({
        "feature": feature,
        "correct_mean": x.mean(),
        "wrong_mean": y.mean(),
        "p_value": p_value,
        "rank_biserial": effect_size,
    })

results_df = pd.DataFrame(results)

# Benjamini-Hochberg false discovery rate adjustment
results_df["adjusted_p"] = multipletests(
    results_df["p_value"],
    method="fdr_bh"
)[1]

results_df["significant_fdr_05"] = (
    results_df["adjusted_p"] < 0.05
)

results_df = results_df.sort_values(
    "adjusted_p"
)

print("\nRESULTS")
print("-" * 65)

print(
    results_df.round(5).to_string(index=False)
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:", OUTPUT_PATH)
print("\nSignificance testing complete.")
