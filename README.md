# Wine Classification Using Gaussian Naive Bayes

A beginner-friendly machine learning project that uses **Gaussian Naive Bayes** to classify wine samples into one of three classes based on their chemical properties.

The project demonstrates an end-to-end classification workflow:

**Data Loading → Data Validation → Exploratory Data Analysis → Preprocessing → Train/Test Evaluation → Gaussian Naive Bayes → Error Analysis → Cross-Validation → Prediction**

---

## Overview

The goal of this project is to build a multiclass classification model that predicts the class of a wine sample from its measured chemical properties.

The project uses **Gaussian Naive Bayes**, a probabilistic classification algorithm that is well suited to continuous numerical features.

The workflow includes:

- Dataset inspection and validation
- Exploratory data analysis
- Class distribution analysis
- Feature distribution analysis
- Correlation analysis
- Stratified train/validation splitting
- Feature standardization
- Gaussian Naive Bayes classification
- Accuracy, precision, recall, and F1-score evaluation
- Confusion matrix analysis
- Misclassification/error analysis
- 5-fold stratified cross-validation
- Model persistence with `joblib`
- Prediction of new wine samples
- Prediction on the provided unlabeled test dataset

---

## Project Structure

```text
wine-quality-naive-bayes/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── gaussian_naive_bayes.joblib
│   └── .gitkeep
├── notebooks/
│   └── 01_wine_classification.ipynb
├── reports/
│   └── figures/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── correlation_heatmap.png
│       └── feature_distributions.png
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

### Main files

| File | Purpose |
|---|---|
| `data/train.csv` | Labeled training dataset used for model development and evaluation |
| `data/test.csv` | Unlabeled test dataset used for generating final predictions |
| `notebooks/01_wine_classification.ipynb` | Main end-to-end analysis and modeling workflow |
| `src/preprocessing.py` | Data loading, validation, feature preparation, splitting, and preprocessing |
| `src/train.py` | Model training and model serialization |
| `src/evaluate.py` | Classification metrics, confusion matrix, and error analysis utilities |
| `src/predict.py` | Loads the trained model and predicts new samples |
| `reports/figures/` | EDA and evaluation visualizations |
| `models/` | Saved model artifact |

---

## About the Dataset

The project uses a **Wine Class Classification** dataset containing numerical measurements describing the chemical characteristics of wine samples.

The dataset supplied with this project contains:

| Dataset | Rows | Columns | Target |
|---|---:|---:|---|
| Training | 534 | 15 | Yes |
| Test | 178 | 14 | No |

The training dataset contains:

- **534 labeled observations**
- **13 numerical wine features**
- **1 target column**
- **3 target classes**
- **1 ID column**

The test dataset contains:

- **178 observations**
- The same 13 numerical features
- No target column
- An ID column that can be used to associate predictions with the original rows

### Features

The 13 chemical features are:

1. `alcohol`
2. `malic_acid`
3. `ash`
4. `alcalinity_of_ash`
5. `magnesium`
6. `total_phenols`
7. `flavanoids`
8. `nonflavanoid_phenols`
9. `proanthocyanins`
10. `color_intensity`
11. `hue`
12. `od280/od315_of_diluted_wines`
13. `proline`

The `id` column is treated as an identifier rather than a predictive feature.

### Target Distribution

The labeled training data contains three classes:

| Class | Records |
|---:|---:|
| 0 | 175 |
| 1 | 212 |
| 2 | 147 |
| **Total** | **534** |

There are no missing values in the supplied training or test datasets.

### Dataset Source

The original project documentation identifies the data as the Kaggle **Wine Class Classification** dataset:

Kaggle: https://www.kaggle.com/c/wine-m/data


---

## Methodology

### 1. Data Loading

The training data is loaded from:

```text
data/train.csv
```

The provided test data is loaded from:

```text
data/test.csv
```

The training dataset is used for supervised learning because it contains the target class.

The test dataset does not contain labels, so its role is to generate final predictions after the model has been trained.

### 2. Data Validation

The preprocessing workflow checks that:

- At least 100 observations are available
- At least 5 features are available
- At least 3 target classes are available
- Features are numeric
- Missing values are handled
- A valid target column can be identified

### 3. Feature Preparation

The `id` column is removed because it identifies observations rather than describing wine characteristics.

The remaining 13 numerical measurements are used as model features.

### 4. Exploratory Data Analysis

The notebook examines:

- Class distribution
- Feature distributions
- Summary statistics
- Missing values
- Feature correlations

Generated visualizations are saved under:

```text
reports/figures/
```

### 5. Train/Validation Split

For model evaluation, the labeled training data is divided into:

- **80% training data**
- **20% validation data**

The split is stratified so that the relative class distribution is preserved.

A fixed:

```python
random_state = 42
```

is used for reproducibility.

This produces:

- 427 training observations
- 107 validation observations

### 6. Feature Scaling

A `StandardScaler` is included in the modeling pipeline.

The pipeline is:

```text
StandardScaler → GaussianNB
```

Although scaling is not generally critical for Gaussian Naive Bayes in the same way it is for distance-based algorithms, keeping preprocessing inside a pipeline makes the workflow explicit, consistent, and reusable.

### 7. Gaussian Naive Bayes

The classifier used is:

```python
GaussianNB()
```

Gaussian Naive Bayes estimates the probability of each class based on the observed feature values.

Conceptually:

```text
P(Class | Features)
    ∝
P(Class) × P(Features | Class)
```

The "naive" assumption is that the features are conditionally independent given the class.

Because the dataset contains continuous numerical measurements, Gaussian Naive Bayes is a natural baseline classification algorithm for this project.

---

## Evaluation

The model was evaluated using the reproducible 80/20 stratified validation split described above.

### Holdout Results

| Metric | Score |
|---|---:|
| Accuracy | **93.46%** |
| Weighted Precision | **93.74%** |
| Weighted Recall | **93.46%** |
| Weighted F1-score | **93.44%** |

The validation set contained 107 observations, of which 7 were incorrectly classified.

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 97.06% | 94.29% | 95.65% | 35 |
| 1 | 95.00% | 88.37% | 91.57% | 43 |
| 2 | 87.88% | 100.00% | 93.55% | 29 |

### Confusion Matrix

The validation confusion matrix was:

```text
              Predicted
              0    1    2
Actual  0    33    2    0
        1     1   38    4
        2     0    0   29
```

The main source of validation errors was confusion involving **Class 1 and Class 2**, with four Class 1 observations predicted as Class 2.

The confusion matrix visualization is available at:

```text
reports/figures/confusion_matrix.png
```

### Cross-Validation

To evaluate whether performance is consistent across different subsets of the labeled data, the project also uses **5-fold stratified cross-validation**.

The observed accuracy scores were approximately:

```text
0.9346
0.9346
0.9720
0.9626
0.9434
```

Mean cross-validation accuracy:

```text
94.94%
```

Cross-validation standard deviation:

```text
1.52 percentage points
```

The cross-validation results provide additional evidence about performance consistency beyond a single train/validation split.

> The reported metrics are based on the dataset version included with this project and the specified random seed. Results can change if the data, preprocessing, split, or model configuration changes.

---

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd wine-quality-naive-bayes
```

### 2. Create a virtual environment

#### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/01_wine_classification.ipynb
```

Run the cells from top to bottom.

The notebook performs the complete analysis, including:

1. Loading the data
2. Validating the dataset
3. Exploring the data
4. Preparing features
5. Creating the validation split
6. Training Gaussian Naive Bayes
7. Evaluating predictions
8. Analyzing errors
9. Running cross-validation
10. Saving the model
11. Demonstrating prediction

### 5. Train using the Python module

The training module can also be executed directly:

```bash
python src/train.py
```

This saves the trained model to:

```text
models/gaussian_naive_bayes.joblib
```

### 6. Generate predictions for the provided test dataset

The provided `test.csv` contains no target labels, so it cannot be used to calculate accuracy directly.

After fitting the model on the labeled training data, predictions can be generated for the test dataset using the saved model and the same feature columns.

Example:

```python
import pandas as pd
import joblib

artifact = joblib.load("models/gaussian_naive_bayes.joblib")

model = artifact["model"]
features = artifact["feature_names"]

test = pd.read_csv("data/test.csv")

ids = test["id"]
X_test = test[features]

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "id": ids,
    "prediction": predictions
})

submission.to_csv("test_predictions.csv", index=False)
```

---

## Key Learning Points

### 1. Naive Bayes is a strong baseline for classification

Gaussian Naive Bayes is simple, fast, and probabilistic. It can provide a useful baseline before moving to more complex algorithms.

### 2. Data preprocessing should be part of the modeling workflow

Using a preprocessing pipeline helps ensure that the same transformations are applied consistently during training and prediction.

### 3. Accuracy alone is not enough

The confusion matrix and per-class precision, recall, and F1-score show where classification errors occur.

In this project, overall accuracy is high, but Class 1 has lower recall than the other classes, showing why per-class metrics are useful.

### 4. Cross-validation provides a more robust performance estimate

A single train/validation split can be affected by the particular observations selected for validation. Stratified cross-validation evaluates the model across several different splits.

### 5. Error analysis is important

Instead of only reporting a metric, examining which classes are confused can provide direction for future feature engineering and model selection.

### 6. The test dataset may not have labels

The supplied `test.csv` contains features but no target column. It is therefore suitable for generating predictions, but not for calculating supervised evaluation metrics unless ground-truth labels are available separately.

---

## Limitations

1. **Single model**

   The project focuses on Gaussian Naive Bayes. Other algorithms may perform differently.

2. **Limited dataset size**

   The labeled dataset contains 534 observations. Performance estimates can therefore be sensitive to the specific sample and split.

3. **Naive independence assumption**

   Naive Bayes assumes conditional independence between features given the class. Wine chemistry variables can be correlated, so this assumption may not fully reflect the underlying data.

4. **Limited feature engineering**

   The project primarily uses the original numerical measurements without extensive feature engineering.

5. **No hyperparameter/model comparison study**

   Gaussian Naive Bayes is evaluated independently rather than through a systematic comparison against multiple classification algorithms.

6. **Unlabeled test set**

   The provided test dataset has no target labels, so its predictions cannot be evaluated locally unless the corresponding ground-truth labels are available.

7. **Dataset provenance and licensing**

   The dataset is identified as a Kaggle dataset. Its redistribution terms should be checked before committing the CSV files to a public GitHub repository.

---

## Future Improvements

Possible next steps include:

- Compare Gaussian Naive Bayes with:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
- Perform systematic model comparison using the same cross-validation folds.
- Investigate feature importance or model interpretability.
- Analyze outliers and unusual feature distributions.
- Compare performance with and without feature scaling.
- Perform feature selection.
- Tune model parameters where applicable.
- Add automated tests for preprocessing and prediction functions.
- Add a reproducible training pipeline.
- Add experiment tracking.
- Create a small prediction API.
- Create a lightweight Streamlit application for interactive predictions.
- Add CI/CD checks with GitHub Actions.
- Add a formal data dictionary and dataset license information.

---

## Reproducibility

The main workflow uses:

```text
random_state = 42
test_size = 0.20
5-fold StratifiedKFold
```

The model pipeline is:

```text
Numerical Features
       ↓
Remove ID
       ↓
StandardScaler
       ↓
Gaussian Naive Bayes
       ↓
Class Prediction
```

The reported validation results were produced from the dataset version included with this project.

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Joblib

---

## Author

**Bea Jane Lazona**

This project was created as a machine learning classification portfolio project focused on understanding the complete workflow from data exploration to model evaluation and prediction.
