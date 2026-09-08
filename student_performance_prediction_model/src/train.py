"""
train.py  —  ML Training Script
================================
What this script does (step by step):

  1. Load the dataset from dataset/performance.csv
  2. Inspect the data (shape, columns, missing values)
  3. Separate features (inputs) from targets (what we want to predict)
  4. Split data: 80% training, 20% testing
  5. Build a preprocessing Pipeline (encodes gender, scales numbers)
  6. Train FOUR classification models  →  compare  →  pick best
  7. Train FOUR regression models      →  compare  →  pick best
  8. Save both best models as .pkl files
  9. Save accuracy/metrics to model_metadata.json
     (Flask reads this to display on the UI)

KEY ML CONCEPTS:
  Pipeline     : Chains preprocessing + model so no data leakage occurs.
  ColumnTransformer : Applies different preprocessing to different columns.
  train_test_split  : Keeps test data completely unseen during training.
  Accuracy     : % of test samples the classifier predicted correctly.
  MAE          : Average error in marks (regression).
  R²           : How well the regression model fits (1.0 = perfect).
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Classification models ──────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)

# ── Regression models ──────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor)

# ── Metrics ────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)

# ── Paths ──────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET     = os.path.join(BASE, "dataset", "performance.csv")
MODELS_DIR  = os.path.join(BASE, "models")
CLF_PATH    = os.path.join(MODELS_DIR, "best_classifier.pkl")
REG_PATH    = os.path.join(MODELS_DIR, "best_regressor.pkl")
META_PATH   = os.path.join(MODELS_DIR, "model_metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)


def sep(char="=", w=60):
    print(char * w)


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATASET
# ══════════════════════════════════════════════════════════════════
def load_data():
    if not os.path.exists(DATASET):
        sep()
        print("  ERROR: Dataset not found!")
        print(f"  Expected: {DATASET}")
        print("  Run first:  python generate_dataset.py")
        sep()
        sys.exit(1)

    df = pd.read_csv(DATASET)
    return df


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — INSPECT DATASET
# ══════════════════════════════════════════════════════════════════
def inspect(df):
    sep()
    print("  DATASET INSPECTION")
    sep()
    print(f"  File    : dataset/performance.csv")
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")
    print()
    print(f"  {'Column':<22} {'Type':<12} {'Missing'}")
    print("  " + "-" * 44)
    for col in df.columns:
        missing = df[col].isnull().sum()
        print(f"  {col:<22} {str(df[col].dtype):<12} {missing}")
    print()
    print("  Sample rows (first 3):")
    print(df.head(3).to_string(index=False))
    print()
    print("  Performance distribution (classification target):")
    for label, cnt in df["performance"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {label:<8}: {cnt} ({pct:.1f}%)")
    print()
    print("  Final Score stats (regression target):")
    print(f"    Mean: {df['final_score'].mean():.2f}  "
          f"Std: {df['final_score'].std():.2f}  "
          f"Min: {df['final_score'].min():.1f}  "
          f"Max: {df['final_score'].max():.1f}")
    sep()


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — PREPARE FEATURES & TARGETS
# ══════════════════════════════════════════════════════════════════
def prepare(df):
    """
    We use 6 input features for the ML model.
    'name' and 'student_id' are just identifiers — ignored.
    'final_score' and 'performance' are what we want to predict.
    """
    FEATURE_COLS = [
        "gender",           # categorical
        "attendance",       # numerical
        "study_hours",      # numerical
        "previous_score",   # numerical
        "assignment_score", # numerical
        "midterm_score",    # numerical
    ]
    X = df[FEATURE_COLS]
    y_clf = df["performance"]      # classification target
    y_reg = df["final_score"]      # regression target

    CAT_COLS = ["gender"]
    NUM_COLS = ["attendance", "study_hours", "previous_score",
                "assignment_score", "midterm_score"]

    return X, y_clf, y_reg, CAT_COLS, NUM_COLS


# ══════════════════════════════════════════════════════════════════
#  STEP 4 — BUILD PREPROCESSOR
# ══════════════════════════════════════════════════════════════════
def build_preprocessor(cat_cols, num_cols):
    """
    ColumnTransformer applies different steps to different columns:
      - Categorical (gender): OneHotEncoder converts text → numbers
        e.g., Male → [1,0]   Female → [0,1]
      - Numerical (scores): StandardScaler rescales to mean=0, std=1
        This helps most ML algorithms converge faster.

    IMPORTANT: The preprocessor is fitted ONLY on training data.
    When predicting, it transforms new input using the same fitted values.
    This prevents data leakage.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    return preprocessor


# ══════════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN & COMPARE CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════
def train_classifiers(X_train, X_test, y_train, y_test, preprocessor):
    """
    We train 4 classification models and pick the best one.

    MODELS EXPLAINED:
      Logistic Regression  : Simple, fast, good baseline for classification.
      Decision Tree        : Splits data on rules (like a flowchart).
      Random Forest        : Many decision trees combined → more robust.
      Gradient Boosting    : Trees built sequentially, each fixing errors
                             of the previous → often very accurate.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                       random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100,
                                                           random_state=42),
    }

    sep("-")
    print("  CLASSIFICATION MODELS (predicting Good / Average / Poor)")
    sep("-")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10}"
          f" {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 65)

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted",
                               zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted",
                            zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted",
                        zero_division=0)

        print(f"  {name:<25} {acc*100:>9.2f}% {prec*100:>9.2f}%"
              f" {rec*100:>7.2f}% {f1*100:>7.2f}%")

        results[name] = {
            "pipeline": pipe,
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
        }

    # Pick the model with highest accuracy on the test set
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]

    sep("-")
    print(f"\n  ✔ Best Classifier : {best_name}")
    print(f"    Test Accuracy   : {best['accuracy']*100:.2f}%")
    print(f"    Precision       : {best['precision']*100:.2f}%")
    print(f"    Recall          : {best['recall']*100:.2f}%")
    print(f"    F1-score        : {best['f1']*100:.2f}%")

    # Full classification report
    print()
    print("  Detailed Classification Report:")
    y_pred_best = best["pipeline"].predict(X_test)
    print(classification_report(y_test, y_pred_best, zero_division=0))

    return best_name, best


# ══════════════════════════════════════════════════════════════════
#  STEP 6 — TRAIN & COMPARE REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════
def train_regressors(X_train, X_test, y_train, y_test, preprocessor):
    """
    We train 4 regression models and pick the best one.

    METRICS EXPLAINED:
      MAE  (Mean Absolute Error)  : Average marks off from actual score.
                                    Lower = better.
      RMSE (Root Mean Sq. Error)  : Like MAE but penalises large errors more.
                                    Lower = better.
      R²   (R-squared)            : 1.0 = perfect fit, 0.0 = no better than
                                    predicting the mean every time.
                                    Higher = better.
    """
    models = {
        "Linear Regression":        LinearRegression(),
        "Decision Tree Regressor":  DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor":  RandomForestRegressor(n_estimators=100,
                                                           random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
                                          n_estimators=100, random_state=42),
    }

    sep("-")
    print("  REGRESSION MODELS (predicting Final Score 0–100)")
    sep("-")
    print(f"  {'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 58)

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("reg", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f"  {name:<30} {mae:>7.2f}  {rmse:>7.2f}  {r2:>7.4f}")

        results[name] = {
            "pipeline": pipe,
            "mae": mae, "rmse": rmse, "r2": r2,
        }

    # Pick the model with lowest MAE (most interpretable for a beginner)
    best_name = min(results, key=lambda k: results[k]["mae"])
    best      = results[best_name]

    sep("-")
    print(f"\n  ✔ Best Regressor  : {best_name}")
    print(f"    MAE             : {best['mae']:.2f} marks")
    print(f"    RMSE            : {best['rmse']:.2f} marks")
    print(f"    R²              : {best['r2']:.4f}")

    return best_name, best


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    sep()
    print("  STUDENT PERFORMANCE ML TRAINING")
    sep()

    # 1. Load
    df = load_data()

    # 2. Inspect
    inspect(df)

    # 3. Prepare
    X, y_clf, y_reg, cat_cols, num_cols = prepare(df)

    # 4. Split — 80% train, 20% test
    #    stratify=y_clf ensures each class (Good/Avg/Poor) is
    #    proportionally represented in both splits.
    print("  Splitting data: 80% training / 20% testing ...")
    (X_train, X_test,
     y_clf_train, y_clf_test,
     y_reg_train, y_reg_test) = train_test_split(
        X, y_clf, y_reg, test_size=0.2,
        random_state=42, stratify=y_clf
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    sep()

    # 5. Build preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # 6. Train classifiers
    clf_name, clf_best = train_classifiers(
        X_train, X_test, y_clf_train, y_clf_test, preprocessor
    )

    sep()

    # 7. Train regressors
    reg_name, reg_best = train_regressors(
        X_train, X_test, y_reg_train, y_reg_test, preprocessor
    )

    # 8. Save models
    sep()
    print("  SAVING MODELS ...")
    joblib.dump(clf_best["pipeline"], CLF_PATH)
    joblib.dump(reg_best["pipeline"], REG_PATH)
    print(f"  ✔ Classifier saved : {CLF_PATH}")
    print(f"  ✔ Regressor  saved : {REG_PATH}")

    # 9. Save metadata (Flask reads this to show on UI)
    metadata = {
        "classifier": {
            "name":      clf_name,
            "accuracy":  round(clf_best["accuracy"] * 100, 2),
            "precision": round(clf_best["precision"] * 100, 2),
            "recall":    round(clf_best["recall"] * 100, 2),
            "f1":        round(clf_best["f1"] * 100, 2),
        },
        "regressor": {
            "name": reg_name,
            "mae":  round(reg_best["mae"], 2),
            "rmse": round(reg_best["rmse"], 2),
            "r2":   round(reg_best["r2"], 4),
        },
        "features": [
            "gender", "attendance", "study_hours",
            "previous_score", "assignment_score", "midterm_score"
        ]
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✔ Metadata  saved : {META_PATH}")

    sep()
    print("  TRAINING COMPLETE!")
    print()
    print(f"  Classifier : {clf_name}")
    print(f"  Accuracy   : {metadata['classifier']['accuracy']}%")
    print()
    print(f"  Regressor  : {reg_name}")
    print(f"  MAE        : {metadata['regressor']['mae']} marks")
    print(f"  R²         : {metadata['regressor']['r2']}")
    sep()
    print("  Next step: python app.py")
    sep()


if __name__ == "__main__":
    main()
