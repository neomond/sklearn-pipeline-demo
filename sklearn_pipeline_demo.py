"""
Scikit-learn ColumnTransformer Pipeline Demo
=============================================
Full pipeline: preprocessing → feature engineering → model
Uses the Titanic-style synthetic dataset to demonstrate real-world patterns.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =============================================================================
# 1. CREATE A SYNTHETIC DATASET (mimics Titanic-style survival data)
# =============================================================================

np.random.seed(42)
n = 800

data = pd.DataFrame({
    "age":        np.random.normal(35, 15, n).clip(1, 80),
    "fare":       np.random.exponential(40, n).clip(5, 500),
    "sibsp":      np.random.randint(0, 5, n),          # siblings/spouses aboard
    "pclass":     np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    "sex":        np.random.choice(["male", "female"], n),
    "embarked":   np.random.choice(["S", "C", "Q", None], n, p=[0.6, 0.2, 0.15, 0.05]),
    "cabin_deck": np.random.choice(["A","B","C","D","E", None], n, p=[0.05,0.1,0.15,0.1,0.1,0.5]),
})

# Target: survival (loosely correlated with fare, sex, class)
survival_score = (
    (data["sex"] == "female").astype(float) * 1.5
    + (data["pclass"] == 1).astype(float) * 0.8
    + data["fare"] / 500
    + np.random.normal(0, 0.5, n)
)
data["survived"] = (survival_score > 1.0).astype(int)

# Inject some realistic missing values
data.loc[np.random.choice(n, 50, replace=False), "age"] = np.nan
data.loc[np.random.choice(n, 30, replace=False), "fare"] = np.nan

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {data.shape}")
print(f"\nMissing values:\n{data.isnull().sum()}")
print(f"\nTarget distribution:\n{data['survived'].value_counts()}")
print(f"\nFirst few rows:\n{data.head()}")


# =============================================================================
# 2. DEFINE COLUMN GROUPS
# =============================================================================

# Separate features and target
X = data.drop("survived", axis=1)
y = data["survived"]

# Identify column types — this is the key step for ColumnTransformer
numeric_features   = ["age", "fare", "sibsp"]
ordinal_features   = ["pclass"]                  # has natural order: 1 > 2 > 3
nominal_features   = ["sex", "embarked"]         # no order → one-hot
high_card_features = ["cabin_deck"]              # many missing → special handling

print(f"\nNumeric:  {numeric_features}")
print(f"Ordinal:  {ordinal_features}")
print(f"Nominal:  {nominal_features}")
print(f"High-card: {high_card_features}")


# =============================================================================
# 3. BUILD SUB-PIPELINES FOR EACH COLUMN GROUP
# =============================================================================

# --- Numeric: impute missing → scale ---
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # median is robust to outliers
    ("scaler",  StandardScaler()),                    # z-score normalisation
])

# --- Ordinal: impute → encode with order preserved ---
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[[1, 2, 3]])),  # explicit order
])

# --- Nominal (low cardinality): impute → one-hot ---
nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False,
                              handle_unknown="infrequent_if_exist")),
])

# --- High cardinality / mostly missing: impute → one-hot + handle unknowns ---
highcard_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
    ("encoder", OneHotEncoder(sparse_output=False,
                              handle_unknown="infrequent_if_exist",
                              min_frequency=0.05)),  # group rare categories
])


# =============================================================================
# 4. ASSEMBLE WITH ColumnTransformer
# =============================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num",      numeric_pipeline,  numeric_features),
        ("ord",      ordinal_pipeline,  ordinal_features),
        ("nom",      nominal_pipeline,  nominal_features),
        ("highcard", highcard_pipeline,  high_card_features),
    ],
    remainder="drop",   # drop any columns not listed above
    verbose_feature_names_out=True,
)


# =============================================================================
# 5. FULL PIPELINE: preprocessing → feature selection → model
# =============================================================================

# --- Pipeline with Logistic Regression ---
lr_pipeline = Pipeline([
    ("preprocessor",      preprocessor),
    ("feature_selection",  SelectKBest(mutual_info_classif, k=8)),
    ("classifier",         LogisticRegression(max_iter=1000, random_state=42)),
])

# --- Pipeline with Random Forest (no scaling needed, but pipeline still works) ---
rf_pipeline = Pipeline([
    ("preprocessor",      preprocessor),
    ("feature_selection",  SelectKBest(mutual_info_classif, k=8)),
    ("classifier",         RandomForestClassifier(n_estimators=100, random_state=42)),
])


# =============================================================================
# 6. TRAIN / EVALUATE
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("TRAINING & EVALUATION")
print("=" * 60)

for name, pipe in [("Logistic Regression", lr_pipeline),
                    ("Random Forest", rf_pipeline)]:

    # Cross-validation on training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")

    # Fit on full training set, predict on test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"CV F1 scores:  {cv_scores.round(3)}")
    print(f"CV F1 mean:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"\nTest set classification report:")
    print(classification_report(y_test, y_pred, target_names=["died", "survived"]))
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")


# =============================================================================
# 7. INSPECT WHAT THE PIPELINE BUILT
# =============================================================================

print("\n" + "=" * 60)
print("PIPELINE INTERNALS")
print("=" * 60)

# See the feature names after preprocessing
pipe = lr_pipeline  # use the fitted LR pipeline
feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
print(f"\nAll features after preprocessing ({len(feature_names)}):")
for i, name in enumerate(feature_names):
    print(f"  {i}: {name}")

# See which features were selected
selector = pipe.named_steps["feature_selection"]
selected_mask = selector.get_support()
selected_names = feature_names[selected_mask]
print(f"\nSelected features ({len(selected_names)}):")
for name, score in zip(selected_names, selector.scores_[selected_mask]):
    print(f"  {name:40s}  MI score: {score:.4f}")

# Access individual components
print(f"\nScaler mean (age, fare, sibsp): "
      f"{pipe.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].mean_.round(2)}")


# =============================================================================
# 8. USING THE PIPELINE FOR NEW DATA
# =============================================================================

print("\n" + "=" * 60)
print("PREDICTION ON NEW DATA")
print("=" * 60)

new_passenger = pd.DataFrame([{
    "age": 28, "fare": 72.0, "sibsp": 1, "pclass": 1,
    "sex": "female", "embarked": "C", "cabin_deck": "B",
}])

prediction = lr_pipeline.predict(new_passenger)
probability = lr_pipeline.predict_proba(new_passenger)

print(f"\nNew passenger: {new_passenger.to_dict('records')[0]}")
print(f"Prediction:    {'Survived' if prediction[0] else 'Died'}")
print(f"Probability:   {probability[0].round(3)}  [died, survived]")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. ColumnTransformer routes different column types to different sub-pipelines.
2. Each sub-pipeline handles imputation + encoding/scaling for its column group.
3. The full Pipeline chains: ColumnTransformer → SelectKBest → Model.
4. Everything is fit on training data only — no data leakage.
5. pipe.predict(new_data) handles ALL preprocessing automatically.
6. Use pipe.named_steps['...'] to inspect any component after fitting.
""")
