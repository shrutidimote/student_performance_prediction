"""
app.py  —  Flask Web Server
============================
This is the web application backend.

ROUTES:
  GET  /          → Renders index.html (the form)
  POST /predict   → Receives form data, calls predict_student(),
                    returns JSON result to the browser
  GET  /health    → Quick check that the server is running

TERMINAL OUTPUT:
  Every prediction prints the input values and results
  to the terminal so you can see exactly what the model received.
"""

import sys
import os

# Make sure Python can find src/predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request, jsonify
from predict import predict_student, MODELS_READY, LOAD_ERROR

app = Flask(__name__)

# ── Separator helper ───────────────────────────────────────────────
def sep(char="=", w=60):
    print(char * w)


# ══════════════════════════════════════════════════════════════════
#  HOME — Serve the HTML form
# ══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html", models_ready=MODELS_READY,
                           load_error=LOAD_ERROR)


# ══════════════════════════════════════════════════════════════════
#  PREDICT — Receive data, return prediction as JSON
# ══════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Validate: all required fields must be present ──────────
        required = [
            "student_id", "gender",
            "attendance", "study_hours",
            "previous_score", "assignment_score", "midterm_score"
        ]
        missing = [f for f in required if not str(data.get(f, "")).strip()]
        if missing:
            return jsonify({
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        # ── Validate numeric ranges ────────────────────────────────
        numeric_rules = {
            "attendance":      (50, 100),
            "study_hours":     (1, 12),
            "previous_score":  (0, 100),
            "assignment_score":(0, 100),
            "midterm_score":   (0, 100),
        }
        for field, (lo, hi) in numeric_rules.items():
            try:
                val = float(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"'{field}' must be a number."
                }), 400
            if not (lo <= val <= hi):
                return jsonify({
                    "error": f"'{field}' must be between {lo} and {hi}."
                }), 400

        # ── Print INPUT to terminal ────────────────────────────────
        sep()
        print("  PREDICTION REQUEST RECEIVED")
        sep("-")
        print(f"  Student ID       : {data['student_id']}")
        print(f"  Gender           : {data['gender']}")
        print(f"  Attendance (%)   : {data['attendance']}")
        print(f"  Study Hours/Day  : {data['study_hours']}")
        print(f"  Previous Score   : {data['previous_score']}")
        print(f"  Assignment Score : {data['assignment_score']}")
        print(f"  Midterm Score    : {data['midterm_score']}")
        sep("-")

        # ── Call the ML model ──────────────────────────────────────
        result = predict_student(data)

        # ── Print OUTPUT to terminal ───────────────────────────────
        print("  PREDICTION RESULTS:")
        print(f"  Predicted Performance : {result['performance']}")
        print(f"  Predicted Final Score : {result['final_score']} / 100")
        print()
        print(f"  Classifier Model      : {result['clf_name']}")
        print(f"  Classifier Accuracy   : {result['accuracy']}%")
        print(f"  Precision             : {result['precision']}%")
        print(f"  Recall                : {result['recall']}%")
        print(f"  F1-score              : {result['f1']}%")
        print()
        print(f"  Regressor Model       : {result['reg_name']}")
        print(f"  MAE                   : {result['mae']} marks")
        print(f"  RMSE                  : {result['rmse']} marks")
        print(f"  R²                    : {result['r2']}")
        sep()

        # Return everything to the browser
        return jsonify({
            "success":     True,
            "student_id":  data["student_id"],
            "performance": result["performance"],
            "final_score": result["final_score"],
            "clf_name":    result["clf_name"],
            "accuracy":    result["accuracy"],
            "precision":   result["precision"],
            "recall":      result["recall"],
            "f1":          result["f1"],
            "reg_name":    result["reg_name"],
            "mae":         result["mae"],
            "rmse":        result["rmse"],
            "r2":          result["r2"],
        })

    except RuntimeError as e:
        print(f"\n  MODEL ERROR: {e}\n")
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}\n")
        return jsonify({"error": "Prediction failed. Check the terminal for details."}), 500


# ══════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({
        "status":       "running",
        "models_ready": MODELS_READY,
        "error":        LOAD_ERROR,
    })


# ══════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sep()
    print("  STUDENT PERFORMANCE PREDICTION APP")
    sep()
    if MODELS_READY:
        print("  ✔ Models loaded successfully")
    else:
        print("  ✘ Models NOT loaded!")
        print(f"    Reason: {LOAD_ERROR}")
        print("    Run: python src/train.py  first")
    sep()
    print("  Starting Flask server ...")
    print("  Open your browser at:  http://127.0.0.1:5000")
    sep()
    app.run(debug=True, use_reloader=False)
