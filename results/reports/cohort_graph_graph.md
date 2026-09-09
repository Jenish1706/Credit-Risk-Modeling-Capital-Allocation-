> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# cohort_graph: graph construction

Nodes: **108,085**. Edges: **2,035,795** across 2 relations.

## Relations

| relation   | key_column   |   cohorts |   edges |   mean_degree |   median_degree |   isolated_share |   at_cap_share |
|:-----------|:-------------|----------:|--------:|--------------:|----------------:|-----------------:|---------------:|
| zip3       | zip3         |       900 | 1022193 |        9.4573 |              10 |           0.0089 |         0.8982 |
| emp        | emp_norm     |        40 | 1013602 |        9.3778 |              10 |           0.0605 |         0.936  |

## The two information sets

A neighbour's *application* is visible as soon as it is filed. A neighbour's *outcome*
is not visible until the loan resolves, which for a 36-month loan is up to three years
later. Those are different information sets, and the gap between them is large:

| relation   |   mean_prior_applications |   mean_resolved_outcomes |   share_of_predecessors_with_known_outcome |
|:-----------|--------------------------:|-------------------------:|-------------------------------------------:|
| zip3       |                   42.6281 |                  16.7397 |                                     0.3927 |
| emp        |                  847.924  |                 332.86   |                                     0.3926 |

Feature aggregates use the first column; label aggregates use the second. Using the
first for a label aggregate is the standard way graph credit models leak, and it would
inflate AUC substantially, because the cohort default rate would then contain outcomes
from the same period the model is being scored on. See docs/GRAPH_DESIGN.md.

## Leakage checks

- every edge satisfies `issue_period[neighbour] < issue_period[node]`: **PASSED**
- no self-loops: **PASSED**
- no loan resolves on or before its own issue month: **PASSED**
