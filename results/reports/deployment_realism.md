# Deployment Realism Audit (Finding P1-5)

## Motivation
In academic credit scoring benchmarks, researchers split data chronologically by origination date 
(e.g., train on 2010-01..2014-06, valid on 2014-07..2014-12, test on 2015) and assume all 36-month loan outcomes 
are fully known. However, at deployment time on 2015-01-01, 68.4% of training loans and 97.9% of validation loans 
had not yet matured. Their final outcomes occurred during 2015-2018—the exact period in which the test loans ran.

## Results Table

| model                                      |   train_loans |   valid_loans |   test_auc |   test_ece |   observed_expected_ratio |      brier |   h2_test_auc |   h2_test_ece |   h2_observed_expected_ratio |
|:-------------------------------------------|--------------:|--------------:|-----------:|-----------:|--------------------------:|-----------:|--------------:|--------------:|-----------------------------:|
| Standard (Omniscient Labels <= 2018)       |        239104 |         90615 |  0.668968  |  0.0104557 |                  1.07554  | 0.12108    |     0.671249  |     0.0108915 |                     1.07997  |
| Deployment-Realistic (Outcomes <= 2014-12) |         56842 |         20713 |  0.647815  |  0.0374455 |                  1.28598  | 0.12396    |     0.650254  |     0.0366574 |                     1.29382  |
| Delta (Realistic - Standard)               |       -182262 |        -69902 | -0.0211532 |  0.0269899 |                  0.210435 | 0.00287952 |    -0.0209952 |     0.0257659 |                     0.213849 |

## Key Findings
- **Discrimination Drop:** Out-of-time test AUC drops from 0.6690 to 0.6478 (-0.0212).
- **Calibration Deterioration:** ECE increases nearly four-fold from 0.0105 to 0.0374 (+0.0270).
- **Level Bias:** Observed/Expected ratio rises from 1.0755 to 1.2860, meaning the realistic model underpredicts 2015 defaults by 28.6% (compared to 7.5% under omniscient labeling).

## Conclusion for Interview Defense
The standard out-of-time evaluation significantly flatters true operational performance because the calibrator 
is allowed to peek at outcomes from the test era. When constrained to truly observable historical data, 
the model suffers severe calibration drift, proving why continuous vintage monitoring and adaptive calibration 
are essential in production credit risk systems.