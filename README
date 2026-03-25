# 🚢 Scikit-learn Pipeline with ColumnTransformer

A complete end-to-end Machine Learning pipeline using **scikit-learn**, demonstrating:

* Data preprocessing
* Feature engineering
* Feature selection
* Model training & evaluation
* Clean production-style pipeline design

---

## 📌 Project Overview

This project builds a structured ML pipeline using a synthetic dataset inspired by the Titanic survival problem.

The pipeline handles:

* Missing values
* Categorical encoding
* Feature scaling
* Feature selection
* Model training

All steps are combined into a **single reusable pipeline**, avoiding data leakage.

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn

---

## 🧱 Pipeline Architecture

```
ColumnTransformer
    ↓
Feature Selection (SelectKBest)
    ↓
Model (Logistic Regression / Random Forest)
```

---

## 🔍 Key Features

* Separate pipelines for:

  * Numerical features
  * Ordinal features
  * Nominal features
  * High-cardinality categorical features

* Automatic handling of:

  * Missing values
  * Unknown categories
  * Rare categories

* Cross-validation for robust evaluation

---

## 📊 Results

### Logistic Regression

* F1 Score: **~0.88**
* Balanced precision and recall

### Random Forest

* F1 Score: **~0.83**
* Slightly lower generalisation

---

## 🧪 Example Prediction

Input:

```
age: 28
fare: 72
class: 1
sex: female
```

Output:

```
Prediction: Survived
Probability: 97.7%
```

---

## ▶️ How to Run

### 1. Clone repository

```
git clone https://github.com/your-username/sklearn-pipeline-demo.git
cd sklearn-pipeline-demo
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run script

```
python sklearn_pipeline_demo.py
```

---

## 💡 Key Learnings

* How to use `ColumnTransformer` for structured preprocessing
* How to build production-ready pipelines in scikit-learn
* How to avoid data leakage
* How to combine preprocessing + model into one object

---

## 🚀 Future Improvements

* Use real Titanic dataset
* Add hyperparameter tuning (GridSearchCV)
* Deploy as an API
* Add visualizations

---

## 👩‍💻 Author

Nazrin Atayeva
AI & iOS Engineer
