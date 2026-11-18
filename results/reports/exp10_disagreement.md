> Computed on **real** data.

# EXP10: model disagreement as a blind-spot signal

Two LightGBM models were fitted under an **identical** configuration and calibrator, differing only in whether the 18 graph cohort features were available. Disagreement is `|logit(p_tabular) - logit(p_graph)|` on the 283,026 out-of-time test borrowers.

Measured on the **logodds** scale, disagreement correlates with the PD level at Spearman -0.150. On the raw probability scale the same quantity correlates at +0.258 -- which is why the log-odds scale is used: an absolute probability difference is mechanically larger in the middle of the PD range and so partly restates the PD rather than measuring disagreement.

## Marginal view (descriptive only)

| quantile    |     n |   mean_signal |   default_rate |   mean_pd |   calibration_gap |   roc_auc |   brier |    ece |   predicted_observed_ratio |   observed_expected_ratio |
|:------------|------:|--------------:|---------------:|----------:|------------------:|----------:|--------:|-------:|---------------------------:|--------------------------:|
| Q1_agree    | 57639 |        0.0201 |         0.156  |    0.1462 |            0.0097 |    0.6683 |  0.1255 | 0.0104 |                     0.9376 |                    1.0666 |
| Q2          | 58522 |        0.0639 |         0.1561 |    0.1482 |            0.0079 |    0.6787 |  0.125  | 0.0091 |                     0.9492 |                    1.0535 |
| Q3          | 53794 |        0.1274 |         0.1546 |    0.1454 |            0.0092 |    0.6568 |  0.1255 | 0.0125 |                     0.9405 |                    1.0632 |
| Q4          | 56815 |        0.2232 |         0.143  |    0.1345 |            0.0084 |    0.6641 |  0.1174 | 0.0089 |                     0.941  |                    1.0627 |
| Q5_disagree | 56256 |        0.4258 |         0.1345 |    0.1166 |            0.0179 |    0.6771 |  0.1113 | 0.018  |                     0.8672 |                    1.1532 |

## Conditional view, full 2015 (continuity)

|   pd_stratum |   mean_pd |   n_high |   n_low |   predicted_high |   observed_high |   gap_high |   predicted_low |   observed_low |   gap_low |   gap_difference |
|-------------:|----------:|---------:|--------:|-----------------:|----------------:|-----------:|----------------:|---------------:|----------:|-----------------:|
|            0 |    0.0385 |    13676 |   14627 |           0.0394 |          0.0396 |     0.0003 |          0.0376 |         0.0337 |   -0.0039 |           0.0042 |
|            1 |    0.0638 |    11902 |   16401 |           0.0632 |          0.0655 |     0.0022 |          0.0642 |         0.0683 |    0.0041 |          -0.0019 |
|            2 |    0.0818 |    13546 |   14756 |           0.0814 |          0.0916 |     0.0102 |          0.0822 |         0.0895 |    0.0073 |           0.0029 |
|            3 |    0.0988 |    13986 |   14317 |           0.0987 |          0.1113 |     0.0126 |          0.0989 |         0.1095 |    0.0105 |           0.002  |
|            4 |    0.1162 |    14147 |   14155 |           0.1164 |          0.1337 |     0.0174 |          0.116  |         0.1271 |    0.0111 |           0.0063 |
|            5 |    0.1353 |    13634 |   14669 |           0.1348 |          0.1537 |     0.0189 |          0.1357 |         0.1512 |    0.0155 |           0.0034 |
|            6 |    0.1567 |    13618 |   14684 |           0.1566 |          0.177  |     0.0204 |          0.1569 |         0.1704 |    0.0135 |           0.0069 |
|            7 |    0.1829 |    14092 |   14211 |           0.1832 |          0.205  |     0.0218 |          0.1826 |         0.1944 |    0.0118 |           0.01   |
|            8 |    0.2183 |    13896 |   14406 |           0.2181 |          0.2296 |     0.0115 |          0.2185 |         0.2277 |    0.0092 |           0.0023 |
|            9 |    0.2903 |    14001 |   14302 |           0.2902 |          0.3027 |     0.0125 |          0.2903 |         0.2963 |    0.006  |           0.0065 |

## Selection window vs untouched holdout

The decision below is taken on 2015-01..06 only. 2015-07..12 had no part in it.

| window | strata with larger gap | mean gap difference | sign test (two-sided) | odds ratio / SD (95% CI) |
|---|---|---|---|---|
| 2015-01..06 (H1, selection) | 8/10 | +0.0032 | 0.1094 | 1.036 [1.016, 1.056] |
| 2015-07..12 (H2, untouched holdout) | 7/10 | +0.0038 | 0.3438 | 1.050 [1.033, 1.068] |
| 2015-01..12 (full test book, continuity) | 9/10 | +0.0043 | 0.0215 | 1.045 [1.032, 1.058] |

## Verdict

On 2015-01..06 (selection window): Disagreement (logodds scale): correlation with PD itself is -0.152. Within PD strata the high-signal half shows a larger gap in 8/10 strata (mean +0.0032, sign-test two-sided p=0.1094, one-sided p=0.0547), so the effect **does not survive** the PD control -- the model under-states risk for high-signal borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal multiplies default odds by 1.036 (95% CI [1.016, 1.056]).

**Decision: REJECT.**

> **Correction.** The first version of this experiment compared the production PD against an alternative model that differed in hyper-parameters *and* calibrator *and* features, using an absolute probability difference. The disagreement it measured was therefore partly a measure of those choices and partly a restatement of the PD level. See `docs/AUDIT_REPORT.md`.
