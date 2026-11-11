
import pandas as pd
from scipy.stats import binomtest

SVM_PATH = "results/llama_external_svm_predictions.csv"
BERT_PATH = "results/llama_external_distilbert_predictions.csv"

svm = pd.read_csv(SVM_PATH)
bert = pd.read_csv(BERT_PATH)

# Verify the two prediction files contain the same
# reviews in the same order.
assert len(svm) == len(bert)
assert svm["review_text"].equals(bert["review_text"])
assert svm["label"].equals(bert["label"])

comparison = svm[
    ["review_text", "label", "sentiment", "hotel"]
].copy()

comparison["svm_prediction"] = svm["svm_prediction"]
comparison["distilbert_prediction"] = bert["distilbert_prediction"]

comparison["svm_correct"] = (
    comparison["svm_prediction"] == comparison["label"]
)

comparison["distilbert_correct"] = (
    comparison["distilbert_prediction"] == comparison["label"]
)

both_correct = (
    comparison["svm_correct"] &
    comparison["distilbert_correct"]
).sum()

both_wrong = (
    ~comparison["svm_correct"] &
    ~comparison["distilbert_correct"]
).sum()

svm_only = (
    comparison["svm_correct"] &
    ~comparison["distilbert_correct"]
).sum()

bert_only = (
    ~comparison["svm_correct"] &
    comparison["distilbert_correct"]
).sum()

print("EXTERNAL MODEL COMPARISON")
print("=" * 50)

print("Total reviews:", len(comparison))
print("Both correct:", both_correct)
print("Both wrong:", both_wrong)
print("Only SVM correct:", svm_only)
print("Only DistilBERT correct:", bert_only)

# Exact McNemar test via binomial distribution
discordant = svm_only + bert_only

if discordant > 0:
    p_value = binomtest(
        min(svm_only, bert_only),
        n=discordant,
        p=0.5,
        alternative="two-sided"
    ).pvalue
else:
    p_value = 1.0

print("\nExact McNemar p-value:", round(p_value, 6))

if p_value < 0.05:
    print("Difference is statistically significant at 0.05.")
else:
    print("Difference is not statistically significant at 0.05.")

comparison.to_csv(
    "results/llama_external_model_comparison.csv",
    index=False
)

print("\nSaved: results/llama_external_model_comparison.csv")
