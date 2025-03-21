﻿# classifier_calibration

## Overview
In this project, we train an XGBoost classifier on an imbalanced binary classification problem and perform calibration to improve the predicted probabilities. The calibration is done using Platt scaling via the CalibratedClassifierCV from scikit-learn. We evaluate the model's performance before and after calibration, using various evaluation metrics such as ROC AUC, Precision-Recall AUC (AUC-PR), and accuracy.
