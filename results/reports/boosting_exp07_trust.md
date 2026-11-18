> Computed on **real** data.

# boosting: EXP07, explainability and fairness

## EXP07 - how do model and policy degrade under drift?

### Calibration drift on matured vintages

|   vintage |      n |   observed_rate |   mean_predicted |   observed_expected |   roc_auc |   brier |    ece |   calibration_slope |
|----------:|-------:|----------------:|-----------------:|--------------------:|----------:|--------:|-------:|--------------------:|
|      2010 |   9156 |          0.1092 |           0.1269 |              0.8609 |    0.7475 |  0.0883 | 0.0256 |              1.3234 |
|      2011 |  14101 |          0.1063 |           0.1167 |              0.9108 |    0.7392 |  0.0874 | 0.0183 |              1.335  |
|      2012 |  43470 |          0.1358 |           0.1377 |              0.9859 |    0.7094 |  0.1092 | 0.0133 |              1.2601 |
|      2013 | 100422 |          0.1233 |           0.1346 |              0.9157 |    0.7125 |  0.101  | 0.0188 |              1.3413 |
|      2014 | 162570 |          0.1373 |           0.1402 |              0.9793 |    0.6875 |  0.112  | 0.0057 |              1.1137 |
|      2015 | 283026 |          0.1489 |           0.1382 |              1.0768 |    0.6705 |  0.1209 | 0.011  |              0.992  |

Observed/expected moves from 0.861 (2010) to 1.077 (2015), while ROC-AUC moves 0.7475 -> 0.6705.
Note which one moves first. Ranking power is comparatively stable across vintages while the *level* of predicted risk drifts. A monitoring regime that watches only AUC would see nothing wrong; the expected loss the business plans against would already be systematically wrong. This is the operational version of the EXP02 point.

### Production drift (2016-2018, no mature outcomes)

```
Reference window: train (2010-2014H1); current window: production (2016-2018).
Feature drift: 2 of 40 features show significant drift (PSI > 0.25); worst: initial_list_status, application_type, revol_util_clipped.
Prediction drift: PSI 0.0025 (stable); mean PD 0.1351 -> 0.1324.
Largest approval-rate shift: tau=0.1 moves +1.7%.
Calibration drift on matured vintages: observed/expected 0.861 (2010) -> 1.077 (2015).
Note: calibration drift is only measurable on matured vintages. For the current production window, feature and prediction drift are the only available signals -- outcomes will not be known for up to three years.
```

Top drifting features:

| feature               | kind        |    psi |       ks | severity    |   reference_mean |   current_mean |   missing_shift |
|:----------------------|:------------|-------:|---------:|:------------|-----------------:|---------------:|----------------:|
| initial_list_status   | categorical | 1.0062 | nan      | significant |         nan      |       nan      |               0 |
| application_type      | categorical | 0.7901 | nan      | significant |         nan      |       nan      |               0 |
| revol_util_clipped    | numeric     | 0.1497 |   0.1691 | moderate    |          55.6234 |        46.4866 |               0 |
| revol_util            | numeric     | 0.1497 |   0.1691 | moderate    |          55.6234 |        46.4866 |               0 |
| inq_last_6mths        | numeric     | 0.1017 |   0.1321 | moderate    |           0.81   |         0.5023 |               0 |
| inq_last_6mths_capped | numeric     | 0.1017 |   0.1321 | moderate    |           0.81   |         0.5023 |               0 |
| verification_status   | categorical | 0.0996 | nan      | stable      |         nan      |       nan      |               0 |
| open_acc_ratio        | numeric     | 0.0901 |   0.129  | stable      |           0.4891 |         0.5437 |               0 |
| purpose               | categorical | 0.0773 | nan      | stable      |         nan      |       nan      |               0 |
| dti                   | numeric     | 0.0591 |   0.0878 | stable      |          16.4664 |        18.1446 |               0 |
| dti_clean             | numeric     | 0.0591 |   0.0878 | stable      |          16.4664 |        18.1446 |               0 |
| loan_amnt             | numeric     | 0.0436 |   0.0515 | stable      |       12198.9    |     12818.1    |               0 |

Approval-rate impact at fixed cut-offs:

| cut-off | change in approval rate |
|---|---|
| tau=0.05 | +1.27% |
| tau=0.1 | +1.69% |
| tau=0.15 | +1.50% |
| tau=0.2 | +0.94% |
| tau=0.3 | -0.09% |

## Explainability

Global attribution (mean |SHAP|, top 15):

| feature             |   mean_abs_shap |   mean_shap |
|:--------------------|----------------:|------------:|
| fico_range_low      |         0.20415 |     0.02685 |
| annual_inc          |         0.13149 |    -0.00602 |
| loan_to_income      |         0.1233  |    -0.00152 |
| inq_last_6mths      |         0.10099 |    -0.02678 |
| purpose             |         0.09662 |     0.00021 |
| addr_state          |         0.09281 |    -0.00104 |
| home_ownership      |         0.08175 |    -0.00353 |
| open_acc_ratio      |         0.07932 |     0.00963 |
| fico_range_high     |         0.07537 |     0.01029 |
| dti                 |         0.07275 |     0.01722 |
| revol_bal_to_income |         0.0607  |     0.00625 |
| revol_util          |         0.06031 |    -0.00684 |
| revol_bal           |         0.05253 |     0.0006  |
| emp_length          |         0.05079 |     0.00709 |
| loan_amnt           |         0.0434  |     0.00504 |

### Sample decisions

Each reason below is derived from a SHAP contribution computed for that specific applicant. Nothing is generated from the prediction alone.

```
Applicant
  Risk band : Medium
  PD        : 15.0%

Decision
  Decline

Principal reasons
  + Loan purpose is higher risk
  + Location carries higher observed risk
  + Large requested amount
  + High revolving balances relative to income
  - Strong stated income
  - Few open credit lines
  - Housing status is lower risk
  - Few accounts currently open
```

```
Applicant
  Risk band : Medium
  PD        : 11.4%

Decision
  Approve -- exposure $8,000

Principal reasons
  + Low stated income
  + Low credit bureau score
  + Housing status is higher risk
  + Most accounts currently open
  - Few recent credit applications
  - Loan is modest relative to income
  - Low debt-to-income ratio
  - Low revolving balances
```

```
Applicant
  Risk band : Very high
  PD        : 28.9%

Decision
  Decline

Principal reasons
  + Location carries higher observed risk
  + Many open credit lines
  + Income not verified
  + Limited number of credit accounts
  - Loan purpose is lower risk
  - Strong stated income
  - Strong credit bureau score
  - revol_bal lowers modelled risk
```

```
Applicant
  Risk band : High
  PD        : 19.5%

Decision
  Decline

Principal reasons
  + Several recent credit applications
  + Low credit bureau score
  + High revolving balances relative to income
  + Housing status is higher risk
  - Loan purpose is lower risk
  - Strong stated income
  - Loan is modest relative to income
  - Few accounts currently open
```

```
Applicant
  Risk band : Low
  PD        : 8.3%

Decision
  Approve -- exposure $4,200

Principal reasons
  + High revolving credit utilisation
  + Large requested amount
  + Low credit bureau score
  + Housing status is higher risk
  - Few accounts currently open
  - revol_bal lowers modelled risk
  - Few recent credit applications
  - Strong stated income
```

```
Applicant
  Risk band : Very low
  PD        : 4.4%

Decision
  Approve -- exposure $10,000

Principal reasons
  + Loan purpose is higher risk
  + Low credit bureau score
  + High revolving balances relative to income
  + revol_bal raises modelled risk
  - Loan is modest relative to income
  - Few accounts currently open
  - Strong stated income
  - Low revolving credit utilisation
```

SHAP explains **the model**, not the borrower and not the world. If the model has learned a proxy, SHAP reports the proxy faithfully. The contributions are local to each prediction and do not support causal claims such as 'reducing utilisation by 10 points would lower your PD by X'.

## Fairness slice

```
Fairness slice. LendingClub publishes no protected attributes, so the groups below are observable proxies, and nothing here supports a claim about disparate impact in the legal sense. See the module docstring in evaluation/fairness.py.

addr_state: 49 groups. Approval rate ranges 41.6% to 91.2% (spread 49.6%); calibration (observed/expected) ranges 0.770 to 1.410; AUC ranges 0.615 to 0.722.
  -> The lowest-approval group sits at 0.68x the reference group (CA). Below the 0.8 heuristic; on a protected attribute this would warrant investigation.
home_ownership: 3 groups. Approval rate ranges 49.7% to 74.6% (spread 24.9%); calibration (observed/expected) ranges 1.046 to 1.105; AUC ranges 0.650 to 0.676.
  -> The lowest-approval group sits at 0.67x the reference group (MORTGAGE). Below the 0.8 heuristic; on a protected attribute this would warrant investigation.
emp_length: 12 groups. Approval rate ranges 31.6% to 70.3% (spread 38.6%); calibration (observed/expected) ranges 1.006 to 1.139; AUC ranges 0.646 to 0.677.
  -> The lowest-approval group sits at 0.45x the reference group (10+ years). Below the 0.8 heuristic; on a protected attribute this would warrant investigation.
purpose: 11 groups. Approval rate ranges 18.6% to 75.0% (spread 56.4%); calibration (observed/expected) ranges 0.987 to 1.310; AUC ranges 0.618 to 0.679.
  -> The lowest-approval group sits at 0.32x the reference group (debt_consolidation). Below the 0.8 heuristic; on a protected attribute this would warrant investigation.

Reading: a calibration spread near zero means the model's probabilities mean the same thing in every group, which is the property most directly tied to correct pricing. Approval-rate spread is expected to be non-zero whenever true risk differs between groups, and on its own is not evidence of a problem.
```

Largest groups by `addr_state`:

| group   |     n |   share |   observed_rate |   mean_predicted |   observed_expected |   roc_auc |   approval_rate |    tpr |    fpr |
|:--------|------:|--------:|----------------:|-----------------:|--------------------:|----------:|----------------:|-------:|-------:|
| CA      | 41377 |  0.1462 |          0.1526 |           0.1411 |              1.0812 |    0.6784 |          0.6092 | 0.6129 | 0.3509 |
| TX      | 23530 |  0.0831 |          0.1494 |           0.1224 |              1.2202 |    0.6576 |          0.7073 | 0.4637 | 0.2627 |
| NY      | 23386 |  0.0826 |          0.1638 |           0.1498 |              1.0934 |    0.6606 |          0.5611 | 0.6249 | 0.4025 |
| FL      | 20374 |  0.072  |          0.1614 |           0.1574 |              1.0254 |    0.6561 |          0.5117 | 0.6732 | 0.4527 |
| IL      | 11609 |  0.041  |          0.1309 |           0.1219 |              1.0739 |    0.6681 |          0.7107 | 0.4586 | 0.2638 |
| NJ      | 10088 |  0.0356 |          0.1576 |           0.1468 |              1.0736 |    0.6558 |          0.5772 | 0.6119 | 0.3874 |
| PA      |  9505 |  0.0336 |          0.1509 |           0.131  |              1.1515 |    0.6446 |          0.6659 | 0.4895 | 0.3065 |
| GA      |  9383 |  0.0332 |          0.1285 |           0.1291 |              0.9955 |    0.6879 |          0.6628 | 0.5688 | 0.303  |
| OH      |  9168 |  0.0324 |          0.1563 |           0.1402 |              1.115  |    0.6791 |          0.6137 | 0.5918 | 0.3483 |
| MI      |  7737 |  0.0273 |          0.1455 |           0.1461 |              0.9963 |    0.6736 |          0.5702 | 0.6403 | 0.3939 |
