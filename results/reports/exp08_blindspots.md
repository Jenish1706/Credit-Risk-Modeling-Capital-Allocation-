> Computed on **real** data.

# EXP08: tabular anomaly as a blind-spot signal

An autoencoder (32 numeric features, latent dim 16) was fitted on the training window and never saw the target. Reconstruction error scores the 283,026 out-of-time test borrowers.

## Marginal view (descriptive only)

| quantile     |     n |   mean_signal |   default_rate |   mean_pd |   calibration_gap |   roc_auc |   brier |    ece |   predicted_observed_ratio |   observed_expected_ratio |
|:-------------|------:|--------------:|---------------:|----------:|------------------:|----------:|--------:|-------:|---------------------------:|--------------------------:|
| Q1_normal    | 56606 |        0.0117 |         0.1376 |    0.135  |            0.0027 |    0.6472 |  0.1148 | 0.0047 |                     0.9807 |                    1.0197 |
| Q2           | 56605 |        0.024  |         0.1433 |    0.1365 |            0.0068 |    0.6681 |  0.1175 | 0.0084 |                     0.9528 |                    1.0496 |
| Q3           | 56605 |        0.0383 |         0.1445 |    0.1359 |            0.0087 |    0.6731 |  0.1181 | 0.0114 |                     0.94   |                    1.0639 |
| Q4           | 56605 |        0.0629 |         0.1498 |    0.1376 |            0.0122 |    0.6864 |  0.1204 | 0.0133 |                     0.9185 |                    1.0887 |
| Q5_anomalous | 56605 |        0.2401 |         0.1691 |    0.1463 |            0.0228 |    0.6706 |  0.1339 | 0.0228 |                     0.8653 |                    1.1556 |

Read this table with care. If the anomaly score correlates with the PD level, calibration will appear to degrade across quintiles simply because a boosted model is less well calibrated at high PD. That is not a blind spot; it is a known property.

## Conditional view, full 2015 (continuity)

Within each PD decile the population is split at the median anomaly score, and the calibration gap (observed minus predicted default rate) is compared. A positive `gap_difference` means the model under-states risk more for anomalous borrowers *at the same predicted PD*.

|   pd_stratum |   mean_pd |   n_high |   n_low |   predicted_high |   observed_high |   gap_high |   predicted_low |   observed_low |   gap_low |   gap_difference |
|-------------:|----------:|---------:|--------:|-----------------:|----------------:|-----------:|----------------:|---------------:|----------:|-----------------:|
|            0 |    0.0385 |    14151 |   14152 |           0.0362 |          0.0371 |     0.0009 |          0.0407 |         0.036  |   -0.0046 |           0.0055 |
|            1 |    0.0638 |    14151 |   14152 |           0.0637 |          0.0711 |     0.0074 |          0.0639 |         0.0631 |   -0.0008 |           0.0082 |
|            2 |    0.0818 |    14151 |   14151 |           0.0818 |          0.0944 |     0.0126 |          0.0818 |         0.0866 |    0.0047 |           0.0079 |
|            3 |    0.0988 |    14151 |   14152 |           0.0988 |          0.1182 |     0.0194 |          0.0988 |         0.1025 |    0.0036 |           0.0158 |
|            4 |    0.1162 |    14151 |   14151 |           0.1162 |          0.1394 |     0.0232 |          0.1162 |         0.1214 |    0.0052 |           0.018  |
|            5 |    0.1353 |    14151 |   14152 |           0.1352 |          0.1562 |     0.0209 |          0.1353 |         0.1487 |    0.0134 |           0.0076 |
|            6 |    0.1567 |    14151 |   14151 |           0.1568 |          0.1838 |     0.027  |          0.1566 |         0.1633 |    0.0067 |           0.0202 |
|            7 |    0.1829 |    14151 |   14152 |           0.1829 |          0.2061 |     0.0231 |          0.1828 |         0.1933 |    0.0105 |           0.0127 |
|            8 |    0.2183 |    14151 |   14151 |           0.2185 |          0.2324 |     0.014  |          0.2181 |         0.2248 |    0.0067 |           0.0072 |
|            9 |    0.2903 |    14151 |   14152 |           0.2961 |          0.3155 |     0.0194 |          0.2845 |         0.2835 |   -0.001  |           0.0204 |

## Selection window vs untouched holdout

The decision below is taken on 2015-01..06 only. 2015-07..12 had no part in it.

| window | strata with larger gap | mean gap difference | sign test (two-sided) | odds ratio / SD (95% CI) |
|---|---|---|---|---|
| 2015-01..06 (H1, selection) | 9/10 | +0.0108 | 0.0215 | 1.030 [1.015, 1.045] |
| 2015-07..12 (H2, untouched holdout) | 10/10 | +0.0129 | 0.0020 | 1.033 [1.021, 1.046] |
| 2015-01..12 (full test book, continuity) | 10/10 | +0.0123 | 0.0020 | 1.032 [1.022, 1.041] |

## Verdict

On 2015-01..06 (selection window): Anomaly score: correlation with PD itself is +0.023. Within PD strata the high-signal half shows a larger gap in 9/10 strata (mean +0.0108, sign-test two-sided p=0.0215, one-sided p=0.0107), so the effect **survives** the PD control -- the model under-states risk for high-signal borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal multiplies default odds by 1.030 (95% CI [1.015, 1.045]).

**Decision: KEEP.** The signal carries information about model reliability that the PD does not. Note that the effect on the outcome itself is modest -- the value is in flagging *miscalibration*, not in improving discrimination.

**Scope limit.** Only numeric features are encoded; the 8 categorical columns are not. A borrower unusual only in loan purpose or state would not be flagged.
