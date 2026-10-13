# Gradient Boosting: From Zero to Confident

A comprehensive, zero-to-hero interactive masterclass on **Gradient Boosted Decision Trees (GBDT)** for tabular data regression and classification, featuring:
- **Theory in plain English**: Bagging vs. Boosting, residuals as negative gradients, learning rate (shrinkage), and step-by-step hand calculations on a 6-row dataset.
- **From Scratch Implementation**: Pure Python / NumPy + shallow Decision Trees for both regression (MSE loss) and binary classification (Log-Loss).
- **The "Big Three" Compared**: Head-to-head benchmarking of **XGBoost**, **LightGBM**, and **CatBoost** on identical tabular datasets.
- **Hyperparameter Mastery**: Systematic Optuna tuning for tree architecture, sampling, and regularization.
- **Early Stopping & Validation Dynamics**: Tracking train vs. validation divergence and stopping before overfitting.
- **Real-World Failure Modes**: Diagnosing silent train-test generalization gaps, plus production-grade CUDA / GPU detection with seamless CPU fallback.
- **Explainability**: Global and local feature attributions via TreeSHAP and summary beeswarm plots.
- **Active Recall**: 8 "Prove It" questions, 3 coding challenges, and 1 real-world extension proposal.

---

## Repository Structure

```text
gradient-boosting-regression-zero-to-one/
│
├── gradient_boosting_masterclass.ipynb  # The core interactive masterclass notebook
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Ignored caches, checkpoints, models, and data
└── README.md                           # Project documentation, benchmarks, and run guide
```

---

## Quickstart

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Arnav-Shrivastava/gradient-boosting-regression-zero-to-one.git
   cd gradient-boosting-regression-zero-to-one
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the masterclass notebook**:
   ```bash
   jupyter notebook gradient_boosting_masterclass.ipynb
   ```
