# 🧠 EEG Seizure Prediction with Machine Learning

> Predicting epileptic seizures from high-dimensional EEG recordings using classical ML algorithms, SMOTE oversampling, and GridSearchCV hyperparameter tuning.

---

## 📌 Overview

Epilepsy affects over 50 million people worldwide. Early and accurate seizure detection from EEG signals can significantly improve patient outcomes. This project builds and evaluates a suite of ML classifiers on the **Epileptic Seizure Recognition dataset** — 11,500 EEG samples × 178 time-series features — to distinguish epileptic from non-epileptic brain activity.

---

## 📊 Dataset

- **Source:** Epileptic Seizure Recognition Dataset (`EpilepticSeizureRecognition.csv`)
- **Shape:** 11,500 rows × 179 columns (178 EEG signal features + 1 target)
- **Target classes:**
  - `1` — Epileptic (Ictal / Preictal EEG signal)
  - `0` — Non-Epileptic (Interictal EEG signal)

---

## ⚙️ Pipeline

```
Raw EEG Data
     │
     ▼
Null Check & Exploratory Data Analysis
     │
     ▼
Normalization (sklearn normalize)
     │
     ▼
Outlier Detection & Handling (IQR method → median imputation)
     │
     ▼
Class Balancing (SMOTE oversampling on minority class)
     │
     ▼
Train / Test Split (80/20, random_state=42)
     │
     ▼
Model Training & Evaluation (6 classifiers)
     │
     ▼
Hyperparameter Tuning (GridSearchCV on best model)
     │
     ▼
Final Evaluation (Confusion Matrix + Classification Report)
```

---

## 🤖 Models Evaluated

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear classifier |
| Support Vector Machine (SVC) | Kernel-based, strong on high-dimensional data |
| Decision Tree | Interpretable, fast |
| **Random Forest** ⭐ | Best performer — tuned with GridSearchCV |
| Gradient Boosting | Ensemble boosting approach |
| K-Nearest Neighbors | Distance-based classifier |

---

## 🔧 Hyperparameter Tuning (Random Forest)

GridSearchCV with 3-Fold Cross Validation over:

```python
n_estimators = [10, 100, 1000]
max_features  = ['sqrt', 'log2']
```

Best parameters selected by accuracy score across all folds.

---

## 📈 Results

After tuning, the final **Random Forest** classifier achieved:

| Metric | Class 0 (Non-Epileptic) | Class 1 (Epileptic) |
|--------|------------------------|---------------------|
| Precision | 81% | 74% |
| Recall    | 77% | 78% |
| F1-Score  | ~79% | ~76% |
| **Overall Accuracy** | **78%** | |

> Confusion matrix and full classification report generated via `sklearn.metrics`.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logoColor=white)

**Libraries:** `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `cufflinks`

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/wasiqvoid/EEG-Seizure-Prediction-with-ML.git
cd EEG-Seizure-Prediction-with-ML
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn plotly cufflinks
```

### 3. Add the dataset
Download `EpilepticSeizureRecognition.csv` from [Kaggle](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition) and place it in the project root.

### 4. Run the notebook
```bash
jupyter notebook EEG_Seizure_Prediction_with_ML_algos.ipynb
```

---

## 📁 Project Structure

```
EEG-Seizure-Prediction-with-ML/
│
├── EEG_Seizure_Prediction_with_ML_algos.ipynb   # Main notebook
├── EpilepticSeizureRecognition.csv               # Dataset (add manually)
└── README.md
```

---

## 🔍 Key Techniques

- **IQR-based outlier detection** across all 178 EEG features → replaced with column median
- **SMOTE** (Synthetic Minority Over-sampling) to fix class imbalance
- **Correlation heatmap** (25×25) for feature relationship analysis
- **GridSearchCV + KFold** for robust hyperparameter selection
- **6 classifiers** benchmarked side-by-side with accuracy scores

---

## 👤 Author

**Wasiq Bakhsh** — MS Data Science @ University at Buffalo

[![LinkedIn](https://img.shields.io/badge/LinkedIn-wasiqbakhsh-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/wasiqbakhsh)
[![GitHub](https://img.shields.io/badge/GitHub-wasiqvoid-181717?style=flat-square&logo=github)](https://github.com/wasiqvoid)
[![Portfolio](https://img.shields.io/badge/Portfolio-wasiqvoid.github.io-E6A817?style=flat-square)](https://wasiqvoid.github.io)
