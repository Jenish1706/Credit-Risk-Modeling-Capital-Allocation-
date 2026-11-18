> Computed on **real** data.

# EXP09: relational uncertainty as a blind-spot signal

A VGAE (2 GraphSAGE layers, latent dim 16) was trained to reconstruct the cohort graph, stopping early on held-out link reconstruction after 19 epochs (best validation loss 0.3555). Per-node posterior sigma scores the 283,026 out-of-time test borrowers.

## Marginal view (descriptive only)

| quantile     |     n |   mean_signal |   default_rate |   mean_pd |   calibration_gap |   roc_auc |   brier |    ece |   predicted_observed_ratio |   observed_expected_ratio |
|:-------------|------:|--------------:|---------------:|----------:|------------------:|----------:|--------:|-------:|---------------------------:|--------------------------:|
| Q1_certain   | 56606 |        0.0241 |         0.1494 |    0.1366 |            0.0128 |    0.6687 |  0.1215 | 0.0134 |                     0.9144 |                    1.0936 |
| Q2           | 56605 |        0.038  |         0.1492 |    0.1391 |            0.0101 |    0.6681 |  0.1213 | 0.0103 |                     0.9324 |                    1.0725 |
| Q3           | 56605 |        0.048  |         0.1478 |    0.1389 |            0.0089 |    0.6673 |  0.1204 | 0.0102 |                     0.9398 |                    1.0641 |
| Q4           | 56605 |        0.0596 |         0.1493 |    0.1389 |            0.0104 |    0.6719 |  0.1211 | 0.012  |                     0.9304 |                    1.0749 |
| Q5_uncertain | 56605 |        0.085  |         0.1487 |    0.1378 |            0.0109 |    0.6765 |  0.1205 | 0.012  |                     0.9266 |                    1.0792 |

## Conditional view, full 2015 (continuity)

|   pd_stratum |   mean_pd |   n_high |   n_low |   predicted_high |   observed_high |   gap_high |   predicted_low |   observed_low |   gap_low |   gap_difference |
|-------------:|----------:|---------:|--------:|-----------------:|----------------:|-----------:|----------------:|---------------:|----------:|-----------------:|
|            0 |    0.0385 |    14151 |   14152 |           0.0382 |          0.0345 |    -0.0037 |          0.0387 |         0.0387 |   -0      |          -0.0037 |
|            1 |    0.0638 |    14151 |   14152 |           0.0638 |          0.0655 |     0.0017 |          0.0638 |         0.0687 |    0.0049 |          -0.0031 |
|            2 |    0.0818 |    14151 |   14151 |           0.0818 |          0.0906 |     0.0088 |          0.0819 |         0.0904 |    0.0085 |           0.0003 |
|            3 |    0.0988 |    14151 |   14152 |           0.0988 |          0.1119 |     0.0132 |          0.0988 |         0.1087 |    0.0099 |           0.0033 |
|            4 |    0.1162 |    14151 |   14151 |           0.1162 |          0.1328 |     0.0166 |          0.1162 |         0.128  |    0.0119 |           0.0047 |
|            5 |    0.1353 |    14151 |   14152 |           0.1353 |          0.1522 |     0.017  |          0.1353 |         0.1526 |    0.0173 |          -0.0004 |
|            6 |    0.1567 |    14151 |   14151 |           0.1567 |          0.1705 |     0.0139 |          0.1568 |         0.1766 |    0.0198 |          -0.006  |
|            7 |    0.1829 |    14151 |   14152 |           0.1829 |          0.2041 |     0.0212 |          0.1829 |         0.1952 |    0.0124 |           0.0088 |
|            8 |    0.2183 |    14151 |   14151 |           0.2184 |          0.2292 |     0.0108 |          0.2182 |         0.228  |    0.0099 |           0.0009 |
|            9 |    0.2903 |    14151 |   14152 |           0.2911 |          0.2984 |     0.0074 |          0.2895 |         0.3005 |    0.0111 |          -0.0037 |

## Selection window vs untouched holdout

The decision below is taken on 2015-01..06 only. 2015-07..12 had no part in it.

| window | strata with larger gap | mean gap difference | sign test (two-sided) | odds ratio / SD (95% CI) |
|---|---|---|---|---|
| 2015-01..06 (H1, selection) | 4/10 | +0.0003 | 0.7539 | 1.003 [0.987, 1.019] |
| 2015-07..12 (H2, untouched holdout) | 3/10 | -0.0002 | 0.3438 | 1.002 [0.988, 1.017] |
| 2015-01..12 (full test book, continuity) | 5/10 | +0.0001 | 1.0000 | 1.002 [0.992, 1.013] |

## Verdict

On 2015-01..06 (selection window): Graph uncertainty (VGAE sigma): correlation with PD itself is -0.003. Within PD strata the high-signal half shows a larger gap in 4/10 strata (mean +0.0003, sign-test two-sided p=0.7539, one-sided p=0.8281), so the effect **does not survive** the PD control -- the model under-states risk for high-signal borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal multiplies default odds by 1.003 (95% CI [0.987, 1.019]).

**Decision: REJECT.** Relational uncertainty does **not** predict where the PD model fails. This is a clean negative: the model was trained to convergence with early stopping, so the null is a property of the signal rather than of the training budget. Reporting it is the point -- a complex model that adds nothing should be dropped, and the tabular anomaly signal in EXP08 is doing different work.

Sigma correlates with node degree at Spearman +0.006. It is not simply a proxy for neighbourhood size.

> **Correction.** The first version of this experiment trained for 10 epochs with no validation split and no early stopping, which made 'no signal' and 'undertrained' indistinguishable. See `docs/AUDIT_REPORT.md`.
