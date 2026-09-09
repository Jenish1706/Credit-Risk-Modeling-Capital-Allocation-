> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# boosting: EXP01 and EXP02

Feature set: **primary** (46 features, 39 dropped by the availability guard and constancy checks).
Out-of-time test window: 20,078 loans, observed default rate 0.1657.

## EXP01 - does non-linear modelling improve default prediction?

|          |     n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------|------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| logistic | 20078 |         0.16565 |           0.14935 |   0.69728 |  0.29468 | 0.39457 | 0.305   | 0.12923 |    0.4166  | 0.01747 | 0.04411 |             0.98311 |                 0.10472 |                   0.90158 |
| lightgbm | 20078 |         0.16565 |           0.14689 |   0.69601 |  0.29876 | 0.39201 | 0.30391 | 0.12953 |    0.41759 | 0.02234 | 0.05356 |             1.13228 |                 0.36604 |                   0.8867  |

**Verdict.** LightGBM moves out-of-time ROC-AUC by -0.0013 against logistic regression (0.6973 -> 0.6960). That is a small margin for the extra complexity, and the logistic scorecard remains defensible.

## EXP02 - does calibration improve the reliability of PD estimates?

| method   |     n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------|------:|----------------:|------------------:|----------:|---------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| identity | 20078 |         0.16565 |           0.14689 |   0.69601 |  0.29876 | 0.30391 | 0.12953 |    0.41759 | 0.02234 | 0.05356 |             1.13228 |                 0.36604 |                   0.8867  |
| platt    | 20078 |         0.16565 |           0.1574  |   0.69601 |  0.29876 | 0.30391 | 0.12898 |    0.41602 | 0.01225 | 0.04303 |             1.02087 |                 0.09717 |                   0.9502  |
| isotonic | 20078 |         0.16565 |           0.1569  |   0.69526 |  0.28818 | 0.29902 | 0.12907 |    0.41874 | 0.01198 | 0.0477  |             0.94198 |                -0.01965 |                   0.94716 |

**Verdict.** platt reduced Brier by 0.43% (0.12953 -> 0.12898) and moved ROC-AUC by +0.00000; ECE 0.0223 -> 0.0123, calibration slope 1.132 -> 1.021

Note the ROC-AUC column: it barely moves. That is the whole point. Calibration is a monotone transform of the score, so every ranking metric is blind to it, while the probabilities the decision layer multiplies by an exposure become materially more accurate. A model that ranks well but is mis-calibrated will systematically mis-state expected loss.
