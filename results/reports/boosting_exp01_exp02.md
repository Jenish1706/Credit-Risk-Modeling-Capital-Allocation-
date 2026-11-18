> Computed on **real** data.

# boosting: EXP01 and EXP02

Feature set: **primary** (40 features, 42 dropped by the availability guard and constancy checks).
Out-of-time test book: 283,026 loans, observed default rate 0.1489. **Headline window: 2015-07..12 (H2, untouched holdout)** (162,236 loans, default rate 0.1471). Model and calibrator choices were made on validation and 2015-01..06 only.

## EXP01 - does non-linear modelling improve default prediction?

2015-07..12 (headline):

|          |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   predicted_observed_ratio |   observed_expected_ratio |
|:---------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|---------------------------:|--------------------------:|
| lightgbm | 162236 |         0.14709 |           0.1276  |   0.67316 |  0.2503  | 0.34633 | 0.24982 | 0.11993 |    0.39505 | 0.01982 | 0.03342 |             1.00312 |                 0.17836 |                    0.8675  |                   1.15274 |
| logistic | 162236 |         0.14709 |           0.12627 |   0.66059 |  0.23773 | 0.32118 | 0.23074 | 0.12095 |    0.39903 | 0.02101 | 0.03736 |             0.98975 |                 0.16552 |                    0.85847 |                   1.16486 |

Full 2015 (continuity):

|          |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   predicted_observed_ratio |   observed_expected_ratio |
|:---------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|---------------------------:|--------------------------:|
| lightgbm | 283026 |         0.14886 |           0.12973 |   0.6705  |  0.25019 | 0.34101 | 0.24634 | 0.12124 |    0.39864 | 0.01931 | 0.02822 |             0.99098 |                 0.15153 |                    0.87147 |                   1.14749 |
| logistic | 283026 |         0.14886 |           0.12857 |   0.65806 |  0.23663 | 0.31613 | 0.22797 | 0.1223  |    0.40263 | 0.02044 | 0.03359 |             0.97228 |                 0.12721 |                    0.86368 |                   1.15784 |

**Verdict (2015-07..12).** LightGBM moves out-of-time ROC-AUC by +0.0126 against logistic regression (0.6606 -> 0.6732). That is a material margin and justifies the non-linear model.

## EXP02 - does calibration improve the reliability of PD estimates?

Production calibrator: **intercept_only**, chosen on 2015-01..06 Brier.

2015-07..12 (headline):

| method         |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   predicted_observed_ratio |
|:---------------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|---------------------------:|
| identity       | 162236 |         0.14709 |           0.1276  |   0.67316 |  0.2503  | 0.24982 | 0.11993 |    0.39505 | 0.01982 | 0.03342 |             1.00312 |                 0.17836 |                    0.8675  |
| intercept_only | 162236 |         0.14709 |           0.136   |   0.67316 |  0.2503  | 0.24982 | 0.11963 |    0.39388 | 0.01168 | 0.02271 |             1.0042  |                 0.10305 |                    0.92462 |
| platt          | 162236 |         0.14709 |           0.13605 |   0.67316 |  0.2503  | 0.24982 | 0.11963 |    0.39388 | 0.01177 | 0.02291 |             1.01725 |                 0.12494 |                    0.92494 |
| isotonic       | 162236 |         0.14709 |           0.13605 |   0.67289 |  0.24566 | 0.24886 | 0.11967 |    0.39402 | 0.01111 | 0.03269 |             0.99056 |                 0.07911 |                    0.92497 |

**Verdict (2015-07..12).** intercept_only reduced Brier by 0.25% (0.11993 -> 0.11963) and moved ROC-AUC by +0.00000; ECE 0.0198 -> 0.0117, calibration slope 1.003 -> 1.004

Full 2015 (continuity):

| method         |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   predicted_observed_ratio |
|:---------------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|---------------------------:|
| identity       | 283026 |         0.14886 |           0.12973 |   0.6705  |  0.25019 | 0.24634 | 0.12124 |    0.39864 | 0.01931 | 0.02822 |             0.99098 |                 0.15153 |                    0.87147 |
| intercept_only | 283026 |         0.14886 |           0.13825 |   0.6705  |  0.25019 | 0.24634 | 0.12095 |    0.39749 | 0.011   | 0.01803 |             0.99197 |                 0.077   |                    0.92869 |
| platt          | 283026 |         0.14886 |           0.13827 |   0.6705  |  0.25019 | 0.24634 | 0.12095 |    0.39749 | 0.01111 | 0.0181  |             1.00487 |                 0.09863 |                    0.92885 |
| isotonic       | 283026 |         0.14886 |           0.13829 |   0.67023 |  0.24563 | 0.24496 | 0.12098 |    0.39759 | 0.01057 | 0.03183 |             0.98252 |                 0.06048 |                    0.929   |

Note the ROC-AUC column: it barely moves. That is the whole point. Calibration is a monotone transform of the score, so every ranking metric is blind to it, while the probabilities the decision layer multiplies by an exposure become materially more accurate. A model that ranks well but is mis-calibrated will systematically mis-state expected loss.
