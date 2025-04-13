# classifier_calibration

## Overview
In this project, we train an XGBoost classifier on an imbalanced binary classification problem and perform calibration to improve the predicted probabilities. The calibration is done using Platt scaling via the CalibratedClassifierCV from scikit-learn. We evaluate the model's performance before and after calibration, using various evaluation metrics such as ROC AUC, Precision-Recall AUC (AUC-PR), and accuracy.

# 📏 Model Calibration: Sigmoid and Isotonic Regression

## 🌡️ What is Model Calibration?

Model calibration ensures that a model’s **predicted probabilities reflect true likelihoods**.

- A well-calibrated model that predicts "80% chance of positive" should be correct **80% of the time** when it says that.
- Important for decision-making, risk scoring, and thresholding.

---

## ⚖️ Goal of Calibration

Given a model's uncalibrated probabilities $\hat{p}$, learn a function $f(\hat{p})$ such that:

$$
f(\hat{p}) \approx \text{true likelihood}
$$

This function is learned on a **validation set**, and applied post-training to adjust output confidence.

---

## 🔁 1. Sigmoid Calibration (Platt Scaling)

### ✅ Idea:
Fits a **logistic regression** on the model's output scores to remap them into calibrated probabilities.

### ✅ Formula:

$$
f(s) = \frac{1}{1 + \exp(As + B)}
$$

Where:
- $s$ = raw model score (logit or probability)
- $A$, $B$ = learned parameters from logistic regression on validation set

### ✅ Pros:
- Simple and fast
- Effective when miscalibration has a sigmoid-like shape

### ❌ Cons:
- Too restrictive for more complex miscalibration patterns

---

## 📈 2. Isotonic Regression

### ✅ Idea:
Fits a **monotonic, non-parametric** function to match predicted probabilities to observed outcomes.

- Assumes: if prediction A > prediction B, then probability(A) ≥ probability(B)
- Doesn't assume any specific curve shape

### ✅ How it works:
- Sort predictions and true outcomes
- Fit a **step-wise, monotonic increasing function** that best matches the true label frequencies

### ✅ Pros:
- Very flexible (no fixed shape)
- Can model arbitrary calibration curves

### ❌ Cons:
- Needs more data to avoid overfitting
- Can be unstable with small datasets

---

## 🎨 Visual Comparison

| Raw Prediction | True Frequency |
|----------------|----------------|
| 0.1            | 0.2            |
| 0.5            | 0.4            |
| 0.8            | 0.7            |

- **Sigmoid**: fits a smooth S-curve through the data
- **Isotonic**: fits a piecewise constant (step) function that follows the empirical frequencies

---

## 📊 Evaluation Metrics

- **Reliability Diagram**: plot of predicted vs actual probability
- **Expected Calibration Error (ECE)**: average gap between confidence and accuracy
- **Brier Score**: mean squared error between predicted probs and true outcomes

---

## ✍️ Summary Table

| Method             | Type                    | Flexibility | Risk         | Best Use Case                   |
|--------------------|--------------------------|-------------|--------------|----------------------------------|
| **Platt Scaling**  | Sigmoid / logistic       | Low         | Underfitting | Small validation set             |
| **Isotonic**       | Piecewise monotonic      | High        | Overfitting  | Large validation set             |

---

## 🛠️ Example (scikit-learn)

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (sigmoid)
calibrated_model = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv='prefit')

# Isotonic regression
calibrated_model_iso = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv='prefit')

calibrated_model.fit(X_val, y_val)
```

---
## 📖 Further Reading

- 🔗 [Calibration – scikit-learn docs](https://scikit-learn.org/stable/modules/calibration.html)
- 🔗 [Isotonic Regression Overview – Wikipedia](https://en.wikipedia.org/wiki/Isotonic_regression)
