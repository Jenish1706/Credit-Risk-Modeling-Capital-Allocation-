> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# cohort_graph: EXP04 and EXP05

Out-of-time test window: 20,078 loans.

|                      |     n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------------------|------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| A_tabular            | 20078 |         0.16565 |           0.15744 |   0.69345 |  0.28809 | 0.3869  | 0.2933  | 0.12927 |    0.41782 | 0.00931 | 0.03083 |             0.97332 |                 0.0242  |                   0.95041 |
| B_tabular_plus_graph | 20078 |         0.16565 |           0.16077 |   0.68953 |  0.28398 | 0.37906 | 0.29237 | 0.12969 |    0.41834 | 0.0108  | 0.03747 |             0.99719 |                 0.03395 |                   0.97053 |
| C_graphsage          | 20078 |         0.16565 |           0.15741 |   0.68903 |  0.28488 | 0.37807 | 0.29041 | 0.12951 |    0.41928 | 0.01281 | 0.0377  |             0.9692  |                 0.01765 |                   0.95022 |

## EXP04 - do graph-derived cohort features add signal beyond tabular features?

Arm A (tabular) ROC-AUC 0.6935; arm B (tabular + cohort aggregates) 0.6895; difference **-0.0039**.

The 18 cohort features account for 13.8% of total LightGBM gain in arm B.
**Verdict.** Relational information adds little or nothing beyond the borrower's own file on this dataset. That is a legitimate result, and the more likely one for a shared-attribute cohort graph: geography and employment are already partly encoded in the tabular features (`addr_state`, `emp_length`), so the cohort aggregates are largely redundant with them.

## EXP05 - does GraphSAGE add value beyond the aggregates?

Arm C (GraphSAGE) ROC-AUC 0.6890, against arm B at 0.6895 (**-0.0005**) and arm A at 0.6935 (**-0.0044**).
Trained on `cuda`, best epoch 6, 132 node features.

**Verdict.** The GNN does **not** beat the hand-built cohort aggregates. This is the expected outcome and worth stating plainly: mean-aggregation over a one-hop cohort is close to what the aggregate features already compute in closed form, so the GNN is re-learning a known function from far less signal, with more parameters. On a genuine transaction network -- where multi-hop structure carries information no aggregate captures -- the comparison could well go the other way. This dataset has no such network.

Practical note: arm B is also the more deployable of the two. Its contribution is legible in a SHAP plot and can be given to an applicant as a reason; the GNN's neighbourhood aggregation cannot, which matters under adverse-action requirements.
