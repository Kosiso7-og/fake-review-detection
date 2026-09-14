# Comparative Evaluation of Machine Learning and Transformer-Based Models for Detecting Deceptive and AI-Generated Online Reviews

## Project Overview

This project investigates the use of Natural Language Processing (NLP) and machine learning techniques to detect deceptive online reviews.

The study compares traditional machine learning approaches with transformer-based NLP models to evaluate their effectiveness in identifying deceptive and AI-generated reviews.

This project is being developed as a **Computer Science Senior Seminar research project**.

### Research Question

> **How effectively can traditional machine learning and transformer-based NLP models detect deceptive and AI-generated online reviews, and how do their performance and limitations compare?**

### Project Objectives

The main objectives are to:

* Develop a baseline machine learning model for deceptive review detection.
* Compare multiple traditional machine learning algorithms.
* Evaluate transformer-based NLP models.
* Investigate the detection of AI-generated review text.
* Compare model performance using consistent evaluation metrics.
* Analyze the strengths and limitations of each approach.
* Develop a web-based application for demonstrating the final detection system.

---

## Dataset

The initial experiments use the **Deceptive Opinion Spam Corpus**, obtained through Kaggle.

The dataset contains **1,600 hotel reviews**.

| Label     | Number of Reviews |
| --------- | ----------------: |
| Truthful  |               800 |
| Deceptive |               800 |
| **Total** |         **1,600** |

The dataset is balanced, containing an equal number of truthful and deceptive reviews.

### Dataset Columns

The original dataset contains the following columns:

| Column      | Description                      |
| ----------- | -------------------------------- |
| `deceptive` | Review classification label      |
| `hotel`     | Hotel associated with the review |
| `polarity`  | Review sentiment/polarity        |
| `source`    | Dataset source information       |
| `text`      | Review text                      |

The current baseline experiment uses:

* **Text:** `text`
* **Target:** `deceptive`

The raw dataset is intentionally excluded from GitHub through `.gitignore`.

For local reproduction, the dataset should be placed at:

```text
data/raw/deceptive-reviews.csv
```

---

## Project Structure

```text
fake-review-detection/
│
├── data/
│   └── raw/
│       └── deceptive-reviews.csv
│
├── results/
│   ├── experiment_log.md
│   ├── logistic_regression_report.csv
│   └── logistic_regression_confusion_matrix.png
│
├── src/
│   ├── inspect_dataset.py
│   └── train_baseline.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** The raw dataset and Python virtual environment are excluded from GitHub.

---

## Environment Setup

This project uses Python and a virtual environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Kosiso7-og/fake-review-detection.git
cd fake-review-detection
```

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Dataset Inspection

Before training a model, the dataset can be inspected using:

```powershell
python src/inspect_dataset.py
```

The inspection script checks:

* Dataset dimensions
* Column names
* Missing values
* Data types
* Label distribution

The current dataset contains **1,600 reviews** with no missing values in the inspected columns.

---

# Experiment 1: Logistic Regression Baseline

The first experiment establishes a traditional machine learning baseline.

## Preprocessing

The review text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The baseline configuration uses:

* Lowercase text
* English stop-word removal
* Unigrams and bigrams
* Maximum of 20,000 TF-IDF features

## Model

The classifier used is **Logistic Regression** with:

```text
max_iter = 1000
random_state = 42
```

## Train/Test Split

The dataset was divided using an **80/20 stratified split**.

| Dataset Split | Examples |
| ------------- | -------: |
| Training      |    1,280 |
| Testing       |      320 |

Stratification preserves the balanced class distribution between the training and testing sets.

---

## Baseline Results

The Logistic Regression model achieved:

# **87.50% Accuracy**

The test set contained **320 reviews**.

| Metric    | Deceptive | Truthful |
| --------- | --------: | -------: |
| Precision |      0.88 |     0.88 |
| Recall    |      0.88 |     0.88 |
| F1-score  |      0.88 |     0.88 |

### Overall Performance

* **Correct predictions:** 280 / 320
* **Incorrect predictions:** 40 / 320
* **Accuracy:** 87.50%

### Confusion Matrix

```text
[[140  20]
 [ 20 140]]
```

The model produced:

* **140** correctly classified deceptive reviews
* **140** correctly classified truthful reviews
* **20** deceptive reviews incorrectly classified as truthful
* **20** truthful reviews incorrectly classified as deceptive

The generated confusion matrix is available at:

```text
results/logistic_regression_confusion_matrix.png
```

The classification report is available at:

```text
results/logistic_regression_report.csv
```

---

## Current Findings

The initial Logistic Regression experiment provides a strong baseline for the project.

The model achieved **87.50% accuracy** using a conventional TF-IDF-based text classification approach.

Because the dataset is balanced, accuracy provides a useful initial performance measure. Precision, recall, and F1-score will also be considered when comparing models.

This baseline will serve as the comparison point for subsequent experiments.

### Important Scope Note

This experiment currently addresses:

> **Deceptive vs. truthful review classification**

It does **not yet determine whether a review was specifically written by an AI system**.

AI-generated review detection will be investigated in later experiments.

---

# Planned Experiments

The project will progressively evaluate different modeling approaches.

## Traditional Machine Learning

Planned traditional machine learning models include:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Naive Bayes
4. Random Forest or another tree-based model

These models will use comparable text representations where appropriate to support a fair comparison.

## Transformer-Based Models

Later experiments will investigate transformer-based NLP models, potentially including:

* BERT
* DistilBERT
* RoBERTa

The transformer experiments will evaluate whether contextual language representations improve deceptive-review detection compared with traditional TF-IDF-based approaches.

## AI-Generated Review Detection

A later phase will extend the project beyond deceptive reviews to investigate whether models can distinguish between:

* Human-written reviews
* AI-generated reviews
* Potentially deceptive AI-generated reviews

This phase will require an appropriate dataset and clearly defined labeling methodology.

---

## Evaluation Metrics

Models will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Additional metrics may be included when appropriate.

The same evaluation framework will be used across models to support a fair comparison.

---

## Results

Current experiments:

| Model               | Representation |   Accuracy |
| ------------------- | -------------- | ---------: |
| Logistic Regression | TF-IDF         | **87.50%** |

Additional models will be added as experiments are completed.

---

## Reproducibility

The project uses fixed random seeds where applicable to make experiments reproducible.

The current baseline uses:

```text
random_state = 42
```

Raw datasets and virtual environments are excluded from version control.

The repository contains the source code, experiment results, documentation, and dependency information needed to reproduce the experiments after obtaining the dataset.

---

## Project Status

**Current Phase:** Traditional Machine Learning Baseline

* [x] Project repository created
* [x] Dataset obtained
* [x] Dataset inspected
* [x] Data cleaning completed
* [x] Logistic Regression baseline completed
* [x] Baseline evaluation completed
* [x] Results saved
* [x] Results uploaded to GitHub
* [ ] SVM experiment
* [ ] Additional traditional ML experiments
* [ ] Transformer experiment
* [ ] AI-generated review detection experiment
* [ ] Model comparison
* [ ] Full-stack application
* [ ] Final analysis and documentation

---

## Author

**Computer Science Senior Seminar Project**

**Project:** Comparative Evaluation of Machine Learning and Transformer-Based Models for Detecting Deceptive and AI-Generated Online Reviews


