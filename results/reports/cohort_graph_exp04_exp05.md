> Computed on **real** data.

# cohort_graph: EXP04 and EXP05

Out-of-time test window: 283,026 loans.

|                         |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   predicted_observed_ratio |   observed_expected_ratio |
|:------------------------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|---------------------------:|--------------------------:|
| Control_macro_default   | 283026 |         0.14886 |           0.13854 |   0.66909 |  0.24267 | 0.33817 | 0.24361 | 0.12108 |    0.39814 | 0.01032 | 0.02649 |             0.96461 |                 0.02799 |                    0.93069 |                   1.07447 |
| A_tabular               | 283026 |         0.14886 |           0.13841 |   0.66897 |  0.24403 | 0.33794 | 0.24387 | 0.12108 |    0.39796 | 0.01046 | 0.02656 |             0.97142 |                 0.0407  |                    0.92976 |                   1.07554 |
| B_tabular_plus_graph    | 283026 |         0.14886 |           0.14867 |   0.66793 |  0.24223 | 0.33587 | 0.24109 | 0.12105 |    0.39764 | 0.00324 | 0.01127 |             0.97617 |                -0.03603 |                    0.9987  |                   1.0013  |
| Control_shuffled_cohort | 283026 |         0.14886 |           0.1414  |   0.66772 |  0.24246 | 0.33545 | 0.24223 | 0.12112 |    0.39806 | 0.00767 | 0.02092 |             0.9788  |                 0.02757 |                    0.94985 |                   1.0528  |
| C_graphsage             | 283026 |         0.14886 |           0.13698 |   0.66158 |  0.23501 | 0.32316 | 0.23418 | 0.1217  |    0.40047 | 0.01188 | 0.02457 |             0.95621 |                 0.02607 |                    0.92016 |                   1.08676 |

## EXP04 - do graph-derived cohort features add signal beyond tabular features?

Arm A (tabular) ROC-AUC 0.6690; arm B (tabular + cohort aggregates) 0.6679; difference **-0.0010**.

The 18 cohort features account for 10.7% of total LightGBM gain in arm B.
**Verdict.** Relational information adds little or nothing beyond the borrower's own file on this dataset. That is a legitimate result, and the more likely one for a shared-attribute cohort graph: geography and employment are already partly encoded in the tabular features (`addr_state`, `emp_length`), so the cohort aggregates are largely redundant with them.

### Non-Graph Controls & Paired Bootstrap Verification (P0-4 Audit)

Does the calibration improvement of Arm B over Arm A survive non-graph controls?
Two placebo arms rule out trivial non-graph explanations:
1. **Control (a) - Macro default rate:** A single time-respecting recent portfolio-wide default rate column.
2. **Control (b) - Shuffled cohort keys:** Cohort aggregates recomputed with `zip3` and `emp_norm` randomly permuted within issue month (preserving time and marginals but breaking cohort links).

- **Arm B vs Macro Default:** Delta ECE -0.00707 (95% CI: [-0.00786, -0.00462], p=0.0000); Delta |O/E - 1|: -0.07317 (95% CI: [-0.07408, -0.05792], p=0.0000).
- **Arm B vs Shuffled Keys:** Delta ECE -0.00442 (95% CI: [-0.00526, -0.00220], p=0.0000); Delta |O/E - 1|: -0.05149 (95% CI: [-0.05225, -0.03675], p=0.0000).

**Control Verdict:** The calibration gain survives both controls with non-overlapping bootstrap intervals (p < 0.0001). The improvement is genuinely relational, attributable to local cohort shocks rather than aggregate macro drift.

## EXP05 - does GraphSAGE add value beyond the aggregates?

Arm C (GraphSAGE) ROC-AUC 0.6616, against arm B at 0.6679 (**-0.0064**) and arm A at 0.6690 (**-0.0074**).
Trained on `cuda`, best epoch 3, 125 node features.

**Verdict.** The GNN does **not** beat the hand-built cohort aggregates. This is the expected outcome and worth stating plainly: mean-aggregation over a one-hop cohort is close to what the aggregate features already compute in closed form, so the GNN is re-learning a known function from far less signal, with more parameters. On a genuine transaction network -- where multi-hop structure carries information no aggregate captures -- the comparison could well go the other way. This dataset has no such network.

Practical note: arm B is also the more deployable of the two. Its contribution is legible in a SHAP plot and can be given to an applicant as a reason; the GNN's neighbourhood aggregation cannot, which matters under adverse-action requirements.
