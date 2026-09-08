"""
predict.py  —  Prediction Helper
==================================
This module loads the saved ML models and metadata,
then provides a single function predict_student() that
Flask (app.py) calls whenever a user submits the form.

HOW PREDICTION WORKS:
  1. User fills the form in the browser.
  2. Flask calls predict_student(input_dict).
  3. This function converts the dict → DataFrame.
  4. The DataFrame is passed into the saved Pipeline.
  5. Pipeline runs the same preprocessing as during training.
  6. The model returns a prediction.
  7. Flask sends the result back to the browser.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLF_PATH  = os.path.join(BASE, "models", "best_classifier.pkl")
REG_PATH  = os.path.join(BASE, "models", "best_regressor.pkl")
META_PATH = os.path.join(BASE, "models", "model_metadata.json")

# ── Load models once when the module is imported ──────────────────
#    (Flask imports this at startup, so models are loaded only once)
def _load():
    """Load models and metadata. Raises clear errors if files missing."""
    errors = []
    for path, label in [(CLF_PATH, "Classifier"), (REG_PATH, "Regressor"),
                        (META_PATH, "Metadata")]:
        if not os.path.exists(path):
            errors.append(f"{label} file not found: {path}")
    if errors:
        raise FileNotFoundError(
            "Models not found. Run 'python src/train.py' first.\n"
            + "\n".join(errors)
        )
    clf      = joblib.load(CLF_PATH)
    reg      = joblib.load(REG_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return clf, reg, meta

try:
    _classifier, _regressor, _metadata = _load()
    MODELS_READY = True
    LOAD_ERROR   = None
except Exception as e:
    MODELS_READY = False
    LOAD_ERROR   = str(e)
    _classifier = _regressor = _metadata = None


# ── Column order must match training ──────────────────────────────
FEATURE_COLS = [
    "gender", "attendance", "study_hours",
    "previous_score", "assignment_score", "midterm_score"
]


def predict_student(data: dict) -> dict:
    """
    Make a prediction for a single student.

    Parameters
    ----------
    data : dict with keys:
        gender, attendance, study_hours,
        previous_score, assignment_score, midterm_score

    Returns
    -------
    dict with:
        performance   (str)  : "Good" / "Average" / "Poor"
        final_score   (float): predicted score 0–100
        clf_name      (str)  : classifier model name
        accuracy      (float): classifier accuracy %
        precision     (float): precision %
        recall        (float): recall %
        f1            (float): f1 %
        reg_name      (str)  : regressor model name
        mae           (float): regressor MAE
        rmse          (float): regressor RMSE
        r2            (float): regressor R²
    """
    if not MODELS_READY:
        raise RuntimeError(LOAD_ERROR)

    # Build a single-row DataFrame (same structure as training data)
    row = pd.DataFrame([{
        "gender":          str(data["gender"]),
        "attendance":      float(data["attendance"]),
        "study_hours":     float(data["study_hours"]),
        "previous_score":  float(data["previous_score"]),
        "assignment_score":float(data["assignment_score"]),
        "midterm_score":   float(data["midterm_score"]),
    }], columns=FEATURE_COLS)

    # The Pipeline handles preprocessing automatically
    performance  = _classifier.predict(row)[0]            # "Good" etc.
    final_score  = float(round(_regressor.predict(row)[0], 1))
    final_score  = max(0.0, min(100.0, final_score))       # clamp 0–100

    m = _metadata
    return {
        "performance":  performance,
        "final_score":  final_score,
        "clf_name":     m["classifier"]["name"],
        "accuracy":     m["classifier"]["accuracy"],
        "precision":    m["classifier"]["precision"],
        "recall":       m["classifier"]["recall"],
        "f1":           m["classifier"]["f1"],
        "reg_name":     m["regressor"]["name"],
        "mae":          m["regressor"]["mae"],
        "rmse":         m["regressor"]["rmse"],
        "r2":           m["regressor"]["r2"],
    }
