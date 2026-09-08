# Student Performance Prediction

## Project Overview
This project uses **real Machine Learning** to predict a student's performance grade (**Good / Average / Poor**) and their predicted **Final Score (0–100)** based on student information entered through a web form.

It uses **two trained ML models**:
- **Model 1 — Classifier**: Predicts performance category (Good / Average / Poor)
- **Model 2 — Regressor**: Predicts the final exam score (0–100)

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Library | scikit-learn |
| Data | pandas, numpy |
| Model Storage | joblib |
| Web Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |

---

## Project Structure

```
student_performance_prediction_model/
│
├── dataset/
│   └── performance.csv         ← generated dataset (1000 students)
│
├── models/
│   ├── best_classifier.pkl     ← saved classification model
│   ├── best_regressor.pkl      ← saved regression model
│   └── model_metadata.json    ← accuracy & metric values
│
├── src/
│   ├── train.py                ← ML training script
│   └── predict.py              ← prediction helper for Flask
│
├── templates/
│   └── index.html              ← web UI
│
├── static/
│   ├── css/style.css           ← styling
│   └── js/script.js            ← frontend logic
│
├── generate_dataset.py         ← creates performance.csv
├── app.py                      ← Flask web server
├── requirements.txt
└── README.md
```

---

## Input Features

| Field | Type | Range |
|---|---|---|
| Name | Text | Any |
| Student ID | Text | Any |
| Gender | Dropdown | Male / Female |
| Attendance (%) | Number | 50 – 100 |
| Study Hours/Day | Number | 1 – 12 |
| Previous Score | Number | 0 – 100 |
| Assignment Score | Number | 0 – 100 |
| Midterm Score | Number | 0 – 100 |

> Note: **Name** and **Student ID** are for display only. The ML model uses the 6 numeric/categorical fields.

---

## ML Workflow

```
Dataset → Data Inspection → Feature/Target Split → Train/Test Split (80/20)
       → Preprocessing Pipeline (OneHotEncoder + StandardScaler)
       → Train 4 Classifiers  → Compare → Select Best Classifier
       → Train 4 Regressors   → Compare → Select Best Regressor
       → Save Models → Flask loads models → User Input → Prediction → Display
```

---

## Step-by-Step Setup (Windows)

### 1. Open a terminal in the project folder

Right-click inside the folder → "Open in Terminal" (or use VS Code terminal)

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

```bash
venv\Scripts\activate
```

You should see `(venv)` at the start of the line.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Generate the dataset

```bash
python generate_dataset.py
```

This creates `dataset/performance.csv` with 1000 student records.

### 6. Train the ML models

```bash
python src/train.py
```

This will:
- Load the dataset
- Train 4 classification models + 4 regression models
- Print a comparison table
- Save the best models to the `models/` folder

### 7. Run the web application

```bash
python app.py
```

### 8. Open in browser

Go to: **http://127.0.0.1:5000**

---

## Terminal Output (What You'll See)

When you click **Predict Result**, the terminal will show:

```
============================================================
  PREDICTION REQUEST RECEIVED
------------------------------------------------------------
  Student Name     : Priya Sharma
  Student ID       : SP0042
  Gender           : Female
  Attendance (%)   : 85
  Study Hours/Day  : 6
  Previous Score   : 72
  Assignment Score : 78
  Midterm Score    : 68
------------------------------------------------------------
  PREDICTION RESULTS:
  Predicted Performance : Good
  Predicted Final Score : 74.5 / 100

  Classifier Model      : Random Forest
  Classifier Accuracy   : 86.40%
  Precision             : 85.21%
  Recall                : 86.40%
  F1-score              : 85.72%

  Regressor Model       : Random Forest Regressor
  MAE                   : 3.21 marks
  RMSE                  : 4.85 marks
  R²                    : 0.8412
============================================================
```

---

## How the ML Works (Beginner Explanation)

### Training
1. We load 1000 student records from the CSV.
2. We split them: 800 for training, 200 for testing.
3. We train multiple ML models on the 800 training records.
4. We test each model on the 200 unseen test records.
5. We pick the best-performing model and save it.

### Prediction
1. You enter a student's details in the form.
2. Flask sends the data to the ML pipeline.
3. The pipeline preprocesses the data exactly the same way as during training.
4. The model predicts a performance category and final score.
5. The result appears on screen.

### Why the model works
The model finds patterns in the training data. For example, it learns that students with 90% attendance and 8 study hours tend to score well. It uses these learned patterns to predict for new students.

---

## Evaluation Metrics Explained

### Classification (Performance: Good/Average/Poor)
- **Accuracy**: % of students correctly classified
- **Precision**: Of all predicted "Good" students, how many actually were Good
- **Recall**: Of all actual "Good" students, how many did the model find
- **F1-score**: Balance of Precision and Recall

### Regression (Final Score)
- **MAE**: Average error in marks (e.g., 3.2 means predictions are ±3.2 marks off)
- **RMSE**: Like MAE but penalises larger errors more
- **R²**: Closer to 1.0 = better. 0.85 means 85% of score variation is explained by the model

---

## Notes

- The model uses **scikit-learn Pipelines** to prevent data leakage.
- All preprocessing (encoding, scaling) is fitted only on training data.
- The test set accuracy is truly "unseen" data performance.
- No prediction values are hardcoded — all results come from `model.predict()`.
