\# Experiment 001: TF-IDF + Logistic Regression



\## Dataset



\- Dataset name: Deceptive Opinion Spam Corpus

\- Dataset source: Kaggle

\- Number of original records: 1,600

\- Number of records after cleaning: 1,600

\- Text column: `text`

\- Label column: `deceptive`

\- Classes: `truthful` and `deceptive`

\- Class distribution:

&#x20; - Truthful: 800

&#x20; - Deceptive: 800

\- Missing values: None



\## Data Split



\- Training percentage: 80%

\- Testing percentage: 20%

\- Training examples: 1,280

\- Testing examples: 320

\- Random state: 42

\- Stratification: Yes



\## Model



\- Feature extraction: TF-IDF

\- Lowercase: Yes

\- Stop words: English

\- N-grams: 1–2

\- Maximum features: 20,000

\- Classifier: Logistic Regression

\- Maximum iterations: 1,000



\## Results



\- Accuracy: 0.8750, or 87.50%

\- Precision:

&#x20; - Deceptive: 0.88

&#x20; - Truthful: 0.88

&#x20; - Macro average: 0.88

&#x20; - Weighted average: 0.88

\- Recall:

&#x20; - Deceptive: 0.88

&#x20; - Truthful: 0.88

&#x20; - Macro average: 0.88

&#x20; - Weighted average: 0.88

\- F1-score:

&#x20; - Deceptive: 0.88

&#x20; - Truthful: 0.88

&#x20; - Macro average: 0.88

&#x20; - Weighted average: 0.88



\## Confusion Matrix



The confusion matrix was:



&#x20;             Predicted

&#x20;             Truthful  Deceptive



Actual Truthful     140       20

Actual Deceptive     20      140



\- Correctly classified truthful reviews: 140

\- Truthful reviews incorrectly classified as deceptive: 20

\- Deceptive reviews incorrectly classified as truthful: 20

\- Correctly classified deceptive reviews: 140

\- Total correct predictions: 280

\- Total incorrect predictions: 40



\## Observations



\- The TF-IDF + Logistic Regression model achieved an accuracy of 87.50%.

\- The model performed similarly on both classes.

\- Precision, recall, and F1-score were approximately 0.88 for both truthful and deceptive reviews.

\- The dataset was balanced, with 800 truthful and 800 deceptive reviews.

\- The confusion matrix shows that the model correctly classified 280 out of 320 test reviews.

\- The model incorrectly classified 40 test reviews.

\- The model made 20 false-positive predictions and 20 false-negative predictions.

\- The baseline provides a useful reference point for comparing more advanced models.



\## Problems Encountered



\- The dataset filename initially did not match the filename expected by the Python script.

\- The dataset path was corrected to `data/raw/deceptive-reviews.csv`.

\- The dataset was successfully loaded after correcting the file path.

\- No missing values were found in the dataset.

\- No major model-training errors occurred.



\## Next Experiment



\- TF-IDF + Support Vector Machine

\- Use the same dataset split and evaluation metrics.

\- Compare accuracy, precision, recall, F1-score, and confusion matrix with Experiment 001.



\## Reproducibility



The experiment was run using:



```powershell

python src\\train\_baseline.py

