# Deployment Realism Audit (Finding P1-5)

## Motivation
The standard out-of-time split trains on 2010-01..2014-06 and calibrates on 2014-07..12 using terminal 
labels, most of which were not yet knowable on 2015-01-01 (the 36-month outcomes arrive in 2015-2018). 
The realistic arm keeps only loans whose outcome was resolved by 2014-12.

## Results Table (all figures from the table below; regenerate with this script)

| model                                      |   train_loans |   valid_loans |   test_auc |   test_ece |   observed_expected_ratio |      brier |   h2_test_auc |   h2_test_ece |   h2_observed_expected_ratio |
|:-------------------------------------------|--------------:|--------------:|-----------:|-----------:|--------------------------:|-----------:|--------------:|--------------:|-----------------------------:|
| Standard (Omniscient Labels <= 2018)       |        239104 |         90615 |  0.668968  |  0.0104557 |                  1.07554  | 0.12108    |     0.671249  |     0.0108915 |                     1.07997  |
| Deployment-Realistic (Outcomes <= 2014-12) |         56842 |         20713 |  0.647815  |  0.0374455 |                  1.28598  | 0.12396    |     0.650254  |     0.0366574 |                     1.29382  |
| Delta (Realistic - Standard)               |       -182262 |        -69902 | -0.0211532 |  0.0269899 |                  0.210435 | 0.00287952 |    -0.0209952 |     0.0257659 |                     0.213849 |

## Key Findings
- **Discrimination:** 2015 AUC 0.6690 -> 0.6478 (-0.0212); 2015-07..12: 0.6712 -> 0.6503.
- **Calibration:** 2015 ECE 0.0105 -> 0.0374; 2015-07..12: 0.0109 -> 0.0367.
- **Level:** observed/expected 1.076 -> 1.286 (2015-07..12: 1.080 -> 1.294).

## What this comparison can and cannot say
The realistic arm is not a random sample of the loans issued in the same months. Selection on resolution:

| window                                       |   loans_issued |   loans_resolved_by_as_of |   share_resolved |   default_rate_all_loans |   default_rate_resolved_loans |
|:---------------------------------------------|---------------:|--------------------------:|-----------------:|-------------------------:|------------------------------:|
| train window (issued <= 2013-06)             |         107851 |                     56842 |           0.5270 |                   0.1264 |                        0.1629 |
| calibration window (issued 2013-07..2014-12) |         221868 |                     20713 |           0.0934 |                   0.1328 |                        0.1040 |

In the training window resolved loans default at 16.3% against 12.6% for all loans issued then; in the calibration window the figures are 10.4% and 13.3%. So the drop in performance mixes staleness (older vintages, fewer recent examples) with selection on resolution. It measures the cost of the naive approach 'train on what has resolved', not staleness alone. There is no arm here that trains on fully matured vintages only, so the two effects are not separated.