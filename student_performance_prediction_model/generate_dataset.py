"""
generate_dataset.py
====================
Generates a realistic student performance dataset (1000 rows).

COLUMNS:
  student_id       - Unique ID like SP0001 (display only, not used in ML)
  gender           - Male / Female
  attendance       - Attendance percentage (50-100%)
  study_hours      - Hours studied per day (1-12)
  previous_score   - Score from previous exam (20-100)
  assignment_score - Assignment marks (20-100)
  midterm_score    - Midterm exam marks (20-100)
  final_score      - Final exam score  → REGRESSION TARGET
  performance      - Good / Average / Poor → CLASSIFICATION TARGET

NOTE: 'name' column has been removed as requested.
"""

import numpy as np
import pandas as pd
import os

# Fix random seed so the same dataset is generated every time
np.random.seed(42)

N = 1000  # number of student records

print("=" * 60)
print("  GENERATING STUDENT PERFORMANCE DATASET")
print("=" * 60)

# ── Hidden 'academic ability' factor ──────────────────────────────
# In real life, students have different inherent abilities.
# This latent factor influences all their scores realistically.
ability = np.clip(np.random.normal(65, 15, N), 25, 95)

# ── Gender ────────────────────────────────────────────────────────
gender = np.random.choice(["Male", "Female"], N, p=[0.50, 0.50])

# ── Student IDs ───────────────────────────────────────────────────
student_id = [f"SP{str(i).zfill(4)}" for i in range(1, N + 1)]

# ── Attendance (%) ────────────────────────────────────────────────
# Range: 50–100. Higher ability → more motivated → better attendance
attendance = np.clip(ability * 0.4 + np.random.normal(40, 8, N), 50, 100).round(1)

# ── Study Hours/Day ───────────────────────────────────────────────
# Range: 1–12. Higher ability → more disciplined study habits
study_hours = np.clip(ability * 0.06 + np.random.normal(2.5, 1.2, N), 1, 12).round(1)

# ── Previous Score ────────────────────────────────────────────────
# Range: 20–100. Strongly correlated with ability (best predictor)
previous_score = np.clip(ability + np.random.normal(0, 8, N), 20, 100).round(1)

# ── Assignment Score ──────────────────────────────────────────────
# Range: 20–100. Influenced by ability and study hours
assignment_score = np.clip(
    ability * 0.55 + study_hours * 1.5 + np.random.normal(0, 7, N),
    20, 100
).round(1)

# ── Midterm Score ─────────────────────────────────────────────────
# Range: 20–100. Harder exam; influenced by ability, attendance, study
midterm_score = np.clip(
    ability * 0.6 + (attendance - 50) * 0.15 + study_hours * 1.2
    + np.random.normal(0, 9, N),
    20, 100
).round(1)

# ── Final Score  (REGRESSION TARGET) ─────────────────────────────
# Weighted combination of all factors + small noise
final_score = (
    0.15 * attendance
    + 0.10 * (study_hours / 12 * 100)   # normalised to 0-100
    + 0.20 * previous_score
    + 0.25 * assignment_score
    + 0.30 * midterm_score
    + np.random.normal(0, 3, N)
)
final_score = np.clip(final_score, 0, 100).round(1)

# ── Performance  (CLASSIFICATION TARGET) ─────────────────────────
# Label derived from final_score so ML can learn what drives Good/Poor
performance = np.where(final_score >= 70, "Good",
              np.where(final_score >= 50, "Average", "Poor"))

# ── Build DataFrame (NO 'name' column) ───────────────────────────
df = pd.DataFrame({
    "student_id":       student_id,
    "gender":           gender,
    "attendance":       attendance,
    "study_hours":      study_hours,
    "previous_score":   previous_score,
    "assignment_score": assignment_score,
    "midterm_score":    midterm_score,
    "final_score":      final_score,
    "performance":      performance,
})

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/performance.csv", index=False)

print(f"  Saved  : dataset/performance.csv")
print(f"  Rows   : {N}")
print(f"  Columns: {list(df.columns)}")
print()
print("  Performance distribution:")
for label, cnt in df["performance"].value_counts().items():
    print(f"    {label:<8}: {cnt} students ({cnt/N*100:.1f}%)")
print()
print("  Score statistics:")
print(f"    Final Score — mean: {df['final_score'].mean():.1f}  "
      f"min: {df['final_score'].min():.1f}  "
      f"max: {df['final_score'].max():.1f}")
print("=" * 60)
print("  Dataset generation complete!")
print("=" * 60)
