/**
 * script.js  —  Frontend Logic
 * ==============================
 * What this file does:
 *   1. Intercepts the form submit (prevents full page reload)
 *   2. Reads all input values from the form
 *   3. Sends them to Flask at POST /predict as JSON
 *   4. Receives the prediction result
 *   5. Updates the result card in the DOM
 *   6. Shows colour-coded performance banner (green/orange/red)
 */

"use strict";

// ── Element references ─────────────────────────────────────────────
const form        = document.getElementById("prediction-form");
const submitBtn   = document.getElementById("submit-btn");
const btnText     = document.getElementById("btn-text");
const btnSpinner  = document.getElementById("btn-spinner");
const formError   = document.getElementById("form-error");
const resultCard  = document.getElementById("result-card");
const mainCard    = document.getElementById("main-card");

// Result elements
const resultStudentName  = document.getElementById("result-student-name");
const resultStudentId    = document.getElementById("result-student-id");
const perfBanner         = document.getElementById("performance-banner");
const perfValue          = document.getElementById("perf-value");
const scoreValue         = document.getElementById("score-value");
const clfModelName       = document.getElementById("clf-model-name");
const clfAccuracy        = document.getElementById("clf-accuracy");
const clfPrecision       = document.getElementById("clf-precision");
const clfRecall          = document.getElementById("clf-recall");
const clfF1              = document.getElementById("clf-f1");
const regModelName       = document.getElementById("reg-model-name");
const regR2              = document.getElementById("reg-r2");
const regMae             = document.getElementById("reg-mae");
const regRmse            = document.getElementById("reg-rmse");


// ── Helpers ────────────────────────────────────────────────────────

/** Show an error message below the form */
function showError(msg) {
  formError.textContent = msg;
  formError.classList.remove("hidden");
}

/** Clear error message */
function clearError() {
  formError.textContent = "";
  formError.classList.add("hidden");
  // Also reset any red-border inputs
  document.querySelectorAll(".input-error")
          .forEach(el => el.classList.remove("input-error"));
}

/** Set loading state on the button */
function setLoading(loading) {
  submitBtn.disabled = loading;
  if (loading) {
    btnText.classList.add("hidden");
    btnSpinner.classList.remove("hidden");
  } else {
    btnText.classList.remove("hidden");
    btnSpinner.classList.add("hidden");
  }
}

/** Apply colour class based on performance */
function applyPerformanceStyle(performance) {
  // Remove previous colour classes
  perfBanner.classList.remove("bg-good", "bg-average", "bg-poor");
  perfValue.classList.remove("perf-good", "perf-average", "perf-poor");

  const p = performance.toLowerCase();
  if (p === "good") {
    perfBanner.classList.add("bg-good");
    perfValue.classList.add("perf-good");
  } else if (p === "average") {
    perfBanner.classList.add("bg-average");
    perfValue.classList.add("perf-average");
  } else {
    perfBanner.classList.add("bg-poor");
    perfValue.classList.add("perf-poor");
  }
}

/** Reset everything back to the form view */
function resetForm() {
  resultCard.classList.add("hidden");
  mainCard.classList.remove("hidden");
  clearError();
  // Smooth scroll to top of form
  mainCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Expose to HTML onclick
window.resetForm = resetForm;


// ── Form Submit ────────────────────────────────────────────────────
form.addEventListener("submit", async function (e) {
  e.preventDefault();   // Stop normal HTML form submission
  clearError();

  // ── Collect form values ──────────────────────────────────────────
  const data = {
    student_id:       document.getElementById("student_id").value.trim(),
    gender:           document.getElementById("gender").value,
    attendance:       document.getElementById("attendance").value,
    study_hours:      document.getElementById("study_hours").value,
    previous_score:   document.getElementById("previous_score").value,
    assignment_score: document.getElementById("assignment_score").value,
    midterm_score:    document.getElementById("midterm_score").value,
  };

  // ── Client-side validation ───────────────────────────────────────
  const textFields = ["student_id", "gender"];
  const numFields  = [
    { key: "attendance",      lo: 50,  hi: 100, label: "Attendance (%)" },
    { key: "study_hours",     lo: 1,   hi: 12,  label: "Study Hours/Day" },
    { key: "previous_score",  lo: 0,   hi: 100, label: "Previous Score" },
    { key: "assignment_score",lo: 0,   hi: 100, label: "Assignment Score" },
    { key: "midterm_score",   lo: 0,   hi: 100, label: "Midterm Score" },
  ];

  for (const f of textFields) {
    if (!data[f]) {
      document.getElementById(f).classList.add("input-error");
      showError(`Please fill in: ${f.replace("_", " ")}`);
      return;
    }
  }

  for (const { key, lo, hi, label } of numFields) {
    const val = parseFloat(data[key]);
    const el  = document.getElementById(key);
    if (data[key] === "" || isNaN(val)) {
      el.classList.add("input-error");
      showError(`"${label}" is required.`);
      return;
    }
    if (val < lo || val > hi) {
      el.classList.add("input-error");
      showError(`"${label}" must be between ${lo} and ${hi}.`);
      return;
    }
  }

  // ── Send to Flask ────────────────────────────────────────────────
  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok || result.error) {
      showError(result.error || "Server error. Please try again.");
      setLoading(false);
      return;
    }

    // ── Populate result card ──────────────────────────────────────

    // Helper: safely set text only if element exists in the DOM
    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    }

    // Student info
    resultStudentName.textContent = `🎓 Student Result`;
    resultStudentId.textContent   = `ID: ${result.student_id}`;

    // Performance
    perfValue.textContent = result.performance.toUpperCase();
    applyPerformanceStyle(result.performance);

    // Final score
    scoreValue.textContent = `${result.final_score}`;

    // Classification metrics (safe — elements may have been removed from HTML)
    setText("clf-model-name", result.clf_name);
    setText("clf-accuracy",   `${result.accuracy}%`);
    setText("clf-precision",  `${result.precision}%`);
    setText("clf-recall",     `${result.recall}%`);
    setText("clf-f1",         `${result.f1}%`);

    // Regression metrics
    setText("reg-model-name", result.reg_name);
    setText("reg-r2",         result.r2);
    setText("reg-mae",        `${result.mae} marks`);
    setText("reg-rmse",       `${result.rmse} marks`);

    // ── Show result, hide form ────────────────────────────────────
    mainCard.classList.add("hidden");
    resultCard.classList.remove("hidden");

    // Smooth scroll to result
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showError("Connection error. Is the Flask server running?");
  } finally {
    setLoading(false);
  }
});
