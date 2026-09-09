> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# boosting: EXP07, explainability and fairness

## EXP07 - how do model and policy degrade under drift?

### Calibration drift on matured vintages

|   vintage |     n |   observed_rate |   mean_predicted |   observed_expected |   roc_auc |   brier |    ece |   calibration_slope |
|----------:|------:|----------------:|-----------------:|--------------------:|----------:|--------:|-------:|--------------------:|
|      2010 |  1375 |          0.1338 |           0.1574 |              0.8503 |    0.7576 |  0.1044 | 0.0427 |              1.455  |
|      2011 |  2363 |          0.1439 |           0.1561 |              0.9216 |    0.7233 |  0.113  | 0.0188 |              1.2172 |
|      2012 |  4139 |          0.1331 |           0.1568 |              0.8488 |    0.7474 |  0.1048 | 0.0349 |              1.4006 |
|      2013 |  7021 |          0.1505 |           0.1572 |              0.9577 |    0.7305 |  0.1162 | 0.021  |              1.2583 |
|      2014 | 12499 |          0.1578 |           0.1578 |              1      |    0.7035 |  0.1229 | 0.0103 |              1.077  |
|      2015 | 20078 |          0.1657 |           0.1574 |              1.0524 |    0.696  |  0.129  | 0.0123 |              1.0209 |

Observed/expected moves from 0.850 (2010) to 1.052 (2015), while ROC-AUC moves 0.7576 -> 0.6960.
Note which one moves first. Ranking power is comparatively stable across vintages while the *level* of predicted risk drifts. A monitoring regime that watches only AUC would see nothing wrong; the expected loss the business plans against would already be systematically wrong. This is the operational version of the EXP02 point.

### Production drift (2016-2018, no mature outcomes)

```
Reference window: train (2010-2014H1); current window: production (2016-2018).
Feature drift: 1 of 43 features show significant drift (PSI > 0.25); worst: application_type, initial_list_status, addr_state.
Prediction drift: PSI 0.0011 (stable); mean PD 0.1573 -> 0.1577.
Largest approval-rate shift: tau=0.2 moves -0.5%.
Calibration drift on matured vintages: observed/expected 0.850 (2010) -> 1.052 (2015).
Note: calibration drift is only measurable on matured vintages. For the current production window, feature and prediction drift are the only available signals -- outcomes will not be known for up to three years.
```

Top drifting features:

| feature               | kind        |    psi |       ks | severity    |   reference_mean |   current_mean |   missing_shift |
|:----------------------|:------------|-------:|---------:|:------------|-----------------:|---------------:|----------------:|
| application_type      | categorical | 0.5471 | nan      | significant |         nan      |       nan      |               0 |
| initial_list_status   | categorical | 0.191  | nan      | moderate    |         nan      |       nan      |               0 |
| addr_state            | categorical | 0.0029 | nan      | stable      |         nan      |       nan      |               0 |
| credit_history_months | numeric     | 0.001  |   0.0058 | stable      |         191.368  |       191.471  |               0 |
| open_acc_ratio        | numeric     | 0.0009 |   0.0111 | stable      |           0.4773 |         0.4796 |               0 |
| revol_util_clipped    | numeric     | 0.0008 |   0.0127 | stable      |          52.1957 |        51.9862 |               0 |
| revol_util            | numeric     | 0.0008 |   0.0127 | stable      |          52.1957 |        51.9862 |               0 |
| emp_length            | categorical | 0.0007 | nan      | stable      |         nan      |       nan      |               0 |
| purpose               | categorical | 0.0006 | nan      | stable      |         nan      |       nan      |               0 |
| dti_clean             | numeric     | 0.0006 |   0.0067 | stable      |          17.597  |        17.5072 |               0 |
| dti                   | numeric     | 0.0006 |   0.0067 | stable      |          17.597  |        17.5072 |               0 |
| revol_bal_to_income   | numeric     | 0.0005 |   0.0077 | stable      |           0.3894 |         0.3878 |               0 |

Approval-rate impact at fixed cut-offs:

| cut-off | change in approval rate |
|---|---|
| tau=0.05 | +0.18% |
| tau=0.1 | +0.36% |
| tau=0.15 | -0.20% |
| tau=0.2 | -0.49% |
| tau=0.3 | +0.01% |

## Explainability

Global attribution (mean |SHAP|, top 15):

| feature               |   mean_abs_shap |   mean_shap |
|:----------------------|----------------:|------------:|
| fico_range_low        |         0.32356 |     0.00756 |
| fico_range_high       |         0.09671 |     0.00287 |
| dti                   |         0.096   |    -0.00781 |
| addr_state            |         0.073   |     0.00142 |
| fico_mid              |         0.03493 |     0.00048 |
| revol_util            |         0.03197 |    -0.00152 |
| loan_to_income        |         0.03051 |    -0.00209 |
| revol_bal_to_income   |         0.02296 |    -0.00029 |
| dti_clean             |         0.02048 |    -0.00212 |
| emp_length            |         0.01734 |     0.00023 |
| revol_util_clipped    |         0.01585 |    -0.00066 |
| loan_amnt             |         0.01523 |    -0.00135 |
| credit_history_months |         0.01302 |    -0.00015 |
| initial_list_status   |         0.01224 |     0.00542 |
| annual_inc            |         0.00735 |    -0.00057 |

### Sample decisions

Each reason below is derived from a SHAP contribution computed for that specific applicant. Nothing is generated from the prediction alone.

```
Applicant
  Risk band : Medium
  PD        : 15.6%

Decision
  Decline

Principal reasons
  + Low credit bureau score
  + High debt-to-income ratio
  + Loan is large relative to income
  + Location carries higher observed risk
  - Low revolving credit utilisation
  - Low revolving balances
  - Listing channel carries lower observed risk
  - Income verified
```

```
Applicant
  Risk band : Medium
  PD        : 11.1%

Decision
  Approve -- exposure $15,125

Principal reasons
  + High debt-to-income ratio
  + Low credit bureau score
  + Loan is large relative to income
  + emp_length raises modelled risk
  - Location carries lower observed risk
  - Long, established credit history
  - Low revolving credit utilisation
  - Low revolving balances
```

```
Applicant
  Risk band : Very high
  PD        : 34.8%

Decision
  Decline

Principal reasons
  + Low credit bureau score
  + Location carries higher observed risk
  + Listing channel carries higher observed risk
  + Many open credit lines
  - Loan is modest relative to income
  - Low debt-to-income ratio
  - Low revolving credit utilisation
  - Low revolving balances
```

```
Applicant
  Risk band : High
  PD        : 22.0%

Decision
  Decline

Principal reasons
  + Low credit bureau score
  + Listing channel carries higher observed risk
  + Loan is large relative to income
  + Large requested amount
  - Location carries lower observed risk
  - Low debt-to-income ratio
  - Low revolving credit utilisation
  - Low revolving balances
```

```
Applicant
  Risk band : Low
  PD        : 7.8%

Decision
  Approve -- exposure $18,075

Principal reasons
  + Loan is large relative to income
  + Large requested amount
  + Short credit history
  + Loan purpose is higher risk
  - Low debt-to-income ratio
  - Strong credit bureau score
  - Low revolving balances
  - Location carries lower observed risk
```

```
Applicant
  Risk band : Low
  PD        : 5.4%

Decision
  Approve -- exposure $7,175

Principal reasons
  + Loan is large relative to income
  + emp_length raises modelled risk
  + Limited number of credit accounts
  + Listing channel carries higher observed risk
  - Strong credit bureau score
  - Low debt-to-income ratio
  - Low revolving credit utilisation
  - Low revolving balances
```

SHAP explains **the model**, not the borrower and not the world. If the model has learned a proxy, SHAP reports the proxy faithfully. The contributions are local to each prediction and do not support causal claims such as 'reducing utilisation by 10 points would lower your PD by X'.

## Fairness slice

```
Fairness slice. LendingClub publishes no protected attributes, so the groups below are observable proxies, and nothing here supports a claim about disparate impact in the legal sense. See the module docstring in evaluation/fairness.py.

addr_state: 51 groups. Approval rate ranges 42.2% to 71.8% (spread 29.6%); calibration (observed/expected) ranges 0.675 to 1.402; AUC ranges 0.607 to 0.783.
  -> The lowest-approval group sits at 0.61x the reference group (ND). Below the 0.8 heuristic; on a protected attribute this would warrant investigation.
home_ownership: 3 groups. Approval rate ranges 56.0% to 57.9% (spread 1.9%); calibration (observed/expected) ranges 0.937 to 1.050; AUC ranges 0.692 to 0.698.
emp_length: 12 groups. Approval rate ranges 53.3% to 62.4% (spread 9.2%); calibration (observed/expected) ranges 0.845 to 1.026; AUC ranges 0.672 to 0.727.
purpose: 8 groups. Approval rate ranges 54.7% to 62.0% (spread 7.3%); calibration (observed/expected) ranges 0.763 to 1.084; AUC ranges 0.675 to 0.776.

Reading: a calibration spread near zero means the model's probabilities mean the same thing in every group, which is the property most directly tied to correct pricing. Approval-rate spread is expected to be non-zero whenever true risk differs between groups, and on its own is not evidence of a problem.
```

Largest groups by `addr_state`:

| group   |   n |   share |   observed_rate |   mean_predicted |   observed_expected |   roc_auc |   approval_rate |    tpr |    fpr |
|:--------|----:|--------:|----------------:|-----------------:|--------------------:|----------:|----------------:|-------:|-------:|
| ND      | 430 |  0.0214 |          0.1488 |           0.1422 |              0.9556 |    0.7612 |          0.6884 | 0.6719 | 0.2486 |
| CA      | 428 |  0.0213 |          0.1519 |           0.1457 |              0.9593 |    0.6558 |          0.6355 | 0.5692 | 0.3278 |
| AR      | 426 |  0.0212 |          0.1643 |           0.1594 |              0.9703 |    0.6726 |          0.5798 | 0.5857 | 0.3876 |
| MS      | 421 |  0.021  |          0.1591 |           0.1509 |              0.9484 |    0.752  |          0.5986 | 0.7463 | 0.3362 |
| TX      | 416 |  0.0207 |          0.1659 |           0.1276 |              0.7695 |    0.7267 |          0.7019 | 0.5652 | 0.245  |
| OH      | 416 |  0.0207 |          0.1442 |           0.1489 |              1.0325 |    0.7827 |          0.6394 | 0.7    | 0.3034 |
| NE      | 415 |  0.0207 |          0.1566 |           0.1667 |              1.064  |    0.734  |          0.5157 | 0.7692 | 0.4314 |
| FL      | 415 |  0.0207 |          0.1614 |           0.1627 |              1.0076 |    0.6506 |          0.588  | 0.5522 | 0.3851 |
| NV      | 413 |  0.0206 |          0.155  |           0.1701 |              1.098  |    0.6662 |          0.4794 | 0.7344 | 0.4814 |
| MD      | 412 |  0.0205 |          0.1869 |           0.1606 |              0.8595 |    0.6858 |          0.517  | 0.7013 | 0.4328 |
