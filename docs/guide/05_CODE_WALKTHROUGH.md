# 05 · Code walkthrough

Grouped by pipeline stage. For each stage: the scripts, the library modules behind them, what goes in
and out, the key functions with their signatures, and the lines a reader is most likely to trip on.
Paths are relative to the repository root. The library is `src/credit_risk/`; the scripts in
`scripts/` are thin drivers that read a config, call the library and write artifacts.

**Conventions shared by every script** ([`utils/cli.py`](../../src/credit_risk/utils/cli.py),
[`utils/config.py`](../../src/credit_risk/utils/config.py)):

- `base_parser(description, default_config)` adds `--config`, `--seed`, `--set key=value`
  overrides and `--max-threads`; `resolve(args)` returns `(config, paths, run_name, seed)`.
- Configs are YAML with `_base_` inheritance; `Config.get_path("a.b.c", default)` reads a dotted key.
- Outputs go through `save_table`, `save_json`, `write_report` and `write_manifest`; `detect_source`
  stamps every output `real` or `synthetic`.
- Scripts on disk use CRLF line endings; git normalises them.

---

## Stage 1 · Data preparation

| | |
|---|---|
| Scripts | [`scripts/download_data.py`](../../scripts/download_data.py), [`scripts/prepare_data.py`](../../scripts/prepare_data.py) |
| Config | [`configs/data/lending_club.yaml`](../../configs/data/lending_club.yaml) |
| In | raw `accepted_2007_to_2018Q4.csv.gz` (or `--synthetic`) |
| Out | `data/processed/loans.parquet`, `results/reports/data_profile.md`, `results/tables/data_profile.json` |

Key functions:

- `download_lending_club(out_dir, files=..., dataset=..., force=False) -> dict[str, str]` and
  `sha256sum(path)` ([`data/download.py`](../../src/credit_risk/data/download.py)).
- `prepare_loans(raw_path, windows=None, term_months=36, chunksize=250000, usecols=None) -> (DataFrame, PrepareReport)`
  ([`data/prepare.py`](../../src/credit_risk/data/prepare.py)): reads in chunks, filters to
  36-month loans, parses months, maps `loan_status` to `default`, assigns splits, computes
  `outcome_observed_period`.
- `assign_split(issue_period, windows)`, `SplitWindows`, `assert_temporal_ordering(df)`
  ([`data/splits.py`](../../src/credit_risk/data/splits.py)).
- `feature_columns(...)`, `assert_no_leakage(columns)`, `FORBIDDEN_AS_FEATURES`
  ([`data/schema.py`](../../src/credit_risk/data/schema.py)).
- `generate(spec)` and `write_synthetic_raw(...)` build the schema-faithful synthetic extract
  ([`data/synthetic.py`](../../src/credit_risk/data/synthetic.py)), which every test uses.

Non-obvious lines:

- **`outcome_observed_period`** (`_outcome_observed_period` in `prepare.py`) is the month an
  observer could know a loan's outcome: the last payment month for a repaid loan, and that plus a
  charge-off lag for a defaulted one. A loan with no terminal status gets **NA**, not "not defaulted"; that
  is what stops unresolved loans being counted as good neighbours. It is derived from
  post-origination fields, so it is used only for graph construction and the deployment-realism
  audit, never as a feature.
- `default_observed` versus `default`: `default` is the modelling target for labelled windows;
  `default_observed` keeps the outcome for every row so the graph can use history rows' labels.
- The H1/H2 cut is implemented by `test_window_masks` and `assert_selection_window` in `splits.py`.

Tests: `tests/test_pipeline.py` (synthetic schema, parsing, split mapping, report counts),
`tests/test_leakage.py` (deny-list, split overlap, windows).

---

## Stage 2 · Features

| | |
|---|---|
| Library | [`features/build.py`](../../src/credit_risk/features/build.py), [`features/financial.py`](../../src/credit_risk/features/financial.py) |
| Used by | every model script |

- `FeatureBuilder(spec).fit(train, valid) -> self`, `.transform(df) -> DataFrame`, `.save(path)`,
  `FeatureBuilder.load(path)`. `FeatureSpec(feature_set="primary" | "with_lc_grade", ...)`.
- `add_derived_features(df, reference_apr=..., term_months=36)` adds ratios such as loan-to-income.
- `to_numeric_matrix(...)` converts a built frame into a standardised numeric matrix for the
  neural models.

Non-obvious lines:

- Everything stateful (vocabularies, medians, clip bounds, the missing-block guard) is fitted on
  `train` only. The **availability guard** drops columns whose missing rate shifts by more than
  `availability_shift_tolerance` (0.05) between train and validation, and uses validation rather
  than test so the test book never influences the feature set.
- The `primary` set removes `installment`, `est_payment_to_income` and `dti_post_loan` because the
  interest rate can be recovered from them (see the class docstring).
- `assert_no_leakage` runs on every build and raises; it is not a warning.

Tests: `tests/test_leakage.py` (`test_built_matrix_contains_no_forbidden_column`,
`test_feature_builder_statistics_come_from_train_only`, `test_availability_guard_*`).

---

## Stage 3 · Graph and cohort features

| | |
|---|---|
| Scripts | [`scripts/build_graph.py`](../../scripts/build_graph.py) |
| Config | [`configs/graph/cohort_graph.yaml`](../../configs/graph/cohort_graph.yaml) |
| Out | `data/processed/graph.npz`, `data/processed/cohort_features.parquet` |

- `build_graph(df, relations=DEFAULT_RELATIONS, period_column="issue_period") -> TimeRespectingGraph`
  and `assert_no_future_edges(graph, df)`
  ([`graph/construction.py`](../../src/credit_risk/graph/construction.py)).
- `CohortFeatureBuilder(config).fit(train)` then `.transform(df, ...)`
  ([`graph/features.py`](../../src/credit_risk/graph/features.py)).
- `compute_macro_default_rate(df, ...)` and `shuffle_keys_within_period(df, keys, ...)` are the two
  non-graph controls.

Non-obvious lines:

- Feature aggregates use `issue[j] < issue[i]`; label aggregates use
  `resolved[j] <= issue[i]`. `n_prior` and `n_resolved` are windows on different time axes, so
  `n_resolved <= n_prior` does **not** hold in general (see the module docstring in `features.py`).
- The only fitted quantity is the portfolio prior default rate, learned on the training window.
- Both aggregates use prefix sums and `searchsorted` instead of a pairwise join.
- `scripts/build_cohort_controls.py` imports both control helpers from `graph/features.py` (an earlier
  duplicate copy in the script was removed).

Tests: `tests/test_graph_leakage.py` (including a planted-violation test and a test that
an ungated cohort rate leaks measurably).

---

## Stage 4 · Risk models and calibration (EXP01, EXP02)

| | |
|---|---|
| Script | [`scripts/train_baseline.py`](../../scripts/train_baseline.py) (`--skip-logistic`, `--skip-search`, `--no-figures`), then [`scripts/analyze_exp01_ci.py`](../../scripts/analyze_exp01_ci.py) (paired-bootstrap interval on the EXP01 AUC difference) |
| Config | [`configs/models/boosting.yaml`](../../configs/models/boosting.yaml), [`configs/models/logistic.yaml`](../../configs/models/logistic.yaml) |
| Out | `artifacts/boosting_{features,logistic,boosting}.pkl`, `artifacts/boosting_calibrator_*.pkl`, `artifacts/boosting_pd_test.npy`, `results/tables/boosting_exp01_*`, `boosting_exp02_*`, `boosting_calibration_choice.json` |

- `LogisticBaseline(numeric, categorical, config).fit(X, y)`
  ([`models/baseline.py`](../../src/credit_risk/models/baseline.py)).
- `BoostingModel(categorical_features, config).fit(X_train, y_train, X_valid, y_valid, seed=42, search=None)`
  and `SearchSpace.sample(rng)`
  ([`models/boosting.py`](../../src/credit_risk/models/boosting.py)): early stopping on validation
  log-loss, optional small random search (15 trials) on validation.
- `ProbabilityCalibrator(method)` with `.fit(y, p)` and `.transform(p)`;
  `compare_calibrations(y_valid, p_valid, y_test, p_test, methods)`
  ([`models/calibration.py`](../../src/credit_risk/models/calibration.py)).
- Metrics: `compute_metrics`, `expected_calibration_error`, `calibration_slope_intercept`,
  `ks_statistic`, `compare_models`
  ([`evaluation/metrics.py`](../../src/credit_risk/evaluation/metrics.py)).

Non-obvious lines:

- **The production calibrator is chosen on 2015-01…06 Brier** (`calibration_h1` in
  `train_baseline.py`) and recorded in `boosting_calibration_choice.json`. Any script that re-scores
  a split must apply that same calibrator through `production_calibrator_path(paths, stem)` in
  `utils/cli.py`, or it will disagree with `boosting_pd_test.npy`.
- Calibrators are fitted on the **validation** window, never on training data.
- `boosting_pd_test.npy` holds PDs for the *whole* 2015 test book; H1/H2 are taken by mask.
- Tables without a suffix are full 2015; the `_h2` tables are the headline.

Tests: `tests/test_models.py`, `tests/test_leakage.py`.

---

## Stage 5 · Graph models and EXP04 / EXP05

| | |
|---|---|
| Scripts | [`scripts/train_gnn.py`](../../scripts/train_gnn.py) (`--skip-gnn`), [`scripts/build_cohort_controls.py`](../../scripts/build_cohort_controls.py) |
| Out | `artifacts/cohort_graph_graphsage.pt`, `results/tables/cohort_graph_exp04_exp05*.csv`, `cohort_controls_exp04*.csv/json` |

- `GraphSAGETrainer(config).fit(features, neighbour_index, neighbour_mask, train_nodes, train_labels, valid_nodes, ...)`,
  `.predict(nodes)`, `.embeddings(nodes)`
  ([`models/gnn.py`](../../src/credit_risk/models/gnn.py)).
- `paired_bootstrap_comparison(y_true, p_b, p_ctrl, n_resamples=2000, seed=42)` in
  `build_cohort_controls.py`.

Non-obvious lines:

- `build_cohort_controls.py` fits each arm for seeds 42 to 46; the **seed varies only the LightGBM
  fit**. The data, the cohort features and the single key shuffle are fixed.
- Per-seed outputs: `cohort_controls_exp04_seeds.csv`, `_seed_deltas.csv`, `_seed_summary.csv`; seed 42
  also feeds the single-fit files.
- `balance_classes: false` in the GNN config keeps the logits calibratable.

Tests: `tests/test_graph_leakage.py`, `tests/test_pipeline.py` (torch-dependent tests skip without
torch).

---

## Stage 6 · Blind-spot signals (EXP08, EXP09, EXP10) and swaps (EXP11)

| | |
|---|---|
| Scripts | [`scripts/analyze_blind_spots.py`](../../scripts/analyze_blind_spots.py), [`scripts/train_vgae.py`](../../scripts/train_vgae.py), [`scripts/analyze_disagreement.py`](../../scripts/analyze_disagreement.py), [`scripts/analyze_decision_swaps.py`](../../scripts/analyze_decision_swaps.py) |
| Configs | `configs/experiments/exp08_blindspots.yaml`, `exp09_vgae.yaml`, `exp10_disagreement.yaml`, `exp11_swaps.yaml` |
| Out | `artifacts/{anomaly,vgae_sigma,disagreement}_test.npy`, `results/tables/exp08_*`, `exp09_*`, `exp10_*`, `exp11_swaps*` |

- `conditional_signal_table(y_true, pd_estimate, signal, n_strata=10, min_per_half=200) -> ConditionalResult`,
  `ConditionalResult.keep`, `conditional_by_window(y, p, signal, issue_period, ...) -> WindowedConditional`,
  `marginal_signal_table(...)`
  ([`decision/blindspot.py`](../../src/credit_risk/decision/blindspot.py)).
- `compare_capital_constrained(df, pd_estimate, flagged, cashflow_model, economics, budget_fraction=0.8, ...)`,
  `greedy_allocate(density, exposure, budget, min_density=0.0)`,
  `exposure_capped_policy(...)`, `evaluate_placebo_controls(...)`.
- Models: `TabularAutoencoder(...).fit(X_train, X_valid)` and `.score_anomalies(X)`
  ([`models/autoencoder.py`](../../src/credit_risk/models/autoencoder.py)); `VGAETrainer(config).fit(...)`
  and `.embed(...)` returning mean and sigma ([`models/vgae.py`](../../src/credit_risk/models/vgae.py)).

Non-obvious lines:

- Each signal script calls `conditional_by_window`, which asserts that the selection frame ends at
  2015-06 before reading it; `decision = "KEEP"` is taken from H1 only and `holdout_confirms` from H2.
- `analyze_decision_swaps.py` reads the EXP08 and EXP10 JSON `decision` fields; only KEEP signals
  are flagged, and the flag cut-point is the 90th percentile **on H1**.
- `--flag-quantile`, `--budget-fraction`, `--exposure-cap` are pre-specified defaults, not tuned.
- The conclusion text about placebos is chosen from the placebo result
  (`_placebo_conclusion`); it is not fixed prose.
- The VGAE step takes about 90 minutes; it dominates a full re-run.

Tests: `tests/test_blindspot.py` (noise fails, PD-restating signal fails, genuine signal
passes, greedy allocation respects the budget, observed-cashflow profit).

---

## Stage 7 · Decision layer, simulator and RL (EXP03, EXP06)

| | |
|---|---|
| Script | [`scripts/train_policy.py`](../../scripts/train_policy.py) (`--pd-run`, `--skip-rl`, `--multi-exposure`) |
| Configs | `configs/simulation/base.yaml`, `configs/experiments/exp06_rl.yaml` |
| Out | `results/tables/exp06_rl_*` (headline) and `exp06_rl_sens_*` (sensitivity), `results/reports/exp06_rl_exp03_exp06.md` |

Economics ([`decision/expected_loss.py`](../../src/credit_risk/decision/expected_loss.py)):
`fit_cashflow_model(train, config, target_column="default") -> CashflowModel`,
`expected_profit(pd_estimate, exposure, apr, cashflow_model, config)`,
`breakeven_pd(apr, exposure, cashflow_model, config)`, `expected_loss(pd, exposure, lgd)`,
`realised_profit(df)`, `realised_return(df)`.

Policies ([`decision/policies.py`](../../src/credit_risk/decision/policies.py)): `ActionSpace`
(`.at_observed_exposure()`, `.without_review()`), `FixedThresholdPolicy`, `RiskBandPolicy`,
`ExpectedProfitPolicy(capital_shadow_price=λ)`, `ApproveAllPolicy`.

Simulator ([`simulation/environment.py`](../../src/credit_risk/simulation/environment.py)):
`build_applicant_pool(df, pd_estimate, cashflow_model, ...)`,
`LendingEnvironment(pool, cashflow_model, economics, action_space, config)` with
`.reset(seed)`, `.step(action_index)`, `.run(policy, seed, keep_log=False, learn=True)`.
Evaluation ([`simulation/evaluation.py`](../../src/credit_risk/simulation/evaluation.py)):
`evaluate_policy`, `compare_policies`, `paired_difference`, `difference_table`, `summary_table`.

RL ([`rl/formulation.py`](../../src/credit_risk/rl/formulation.py),
[`rl/policy.py`](../../src/credit_risk/rl/policy.py)): `StateEncoder`, `analytical_shadow_price`,
`LinUCBPolicy`, `ThompsonSamplingPolicy`, `BudgetAwareQPolicy`, `train_policy(env, policy, n_episodes, seed_offset)`.

Non-obvious lines:

- **`run(..., learn=False)`** is how evaluation stays read-only: `evaluate_policy` always passes it,
  and `train_policy` (the training loop) uses the default `learn=True`. Tests assert parameters are
  byte-identical after evaluation and that results do not depend on episode order.
- **Observed-outcome path** in `LendingEnvironment.step`: when the approved exposure equals the
  observed funded exposure the simulator uses the loan's real default flag and real return per dollar
  (`use_observed_outcomes_at_observed_exposure`). With the headline accept/reject space that is
  every approval. Other exposures use a coupled draw `u` and the fitted cashflow model.
- `build_action_space(max_exposure, include_review, multi_exposure)` in `train_policy.py` builds the
  headline space (reject and 1.0x only) or, with `--multi-exposure`, the 0.5x/1.0x/1.5x sensitivity
  space; the run name then gets a `_sens` suffix.
- The shadow price λ is fitted on the **validation** pool (`train_pool`). The policy named
  `expected_profit(lambda_eval_pool=...)` fits λ on the evaluation pool and is a diagnostic, not a ceiling.
- `RiskBandPolicy` collapses its `approve_*` bands to the single 1.0x approval when the action space
  has one exposure.
- The simulator draws 1,200 applicants per episode and 30 episodes (`evaluation.seeds`).

Tests: `tests/test_decision.py`, `tests/test_simulation.py`.

---

## Stage 8 · Audits of the decision layer

| | |
|---|---|
| Scripts | [`scripts/analyze_profit_calibration.py`](../../scripts/analyze_profit_calibration.py), [`scripts/analyze_apr_in_pd.py`](../../scripts/analyze_apr_in_pd.py), [`scripts/analyze_deployment_realism.py`](../../scripts/analyze_deployment_realism.py) |
| Library | [`decision/profit_audit.py`](../../src/credit_risk/decision/profit_audit.py) |
| Out | `results/tables/profit_calibration_*.csv`, `apr_in_pd_*.csv`, `deployment_realism_p1_5.csv`, `deployment_realism_selection.csv` |

- `gap_decomposition(pd_hat, exposure, apr, realised_return, defaulted, cashflow) -> dict` splits
  realised minus predicted profit into `pd_effect`, `yield_effect` and `severity_effect`, which sum to
  the gap exactly (identity in the docstring).
- `profit_decile_table(...)`, `rule_realised_profit(accept, exposure, apr, realised_return, economics)`,
  `paired_bootstrap_ci(differences, n_bootstrap=2000, seed=0, alpha=0.05)`.
- `evaluate_deployment_realism(df, seed=42)` retrains with only labels observable by 2014-12.

Non-obvious lines:

- `analyze_profit_calibration.py` imports `_load_pd` and `_score_split` from `scripts.train_policy`,
  so it inserts the repository root on `sys.path` (scripts are not a package install).
- Pools are `valid`, `test_h1`, `test_h2`, `test_full`; the "oracle" cashflow refit is fitted on the
  pool it is scored on and is labelled a diagnostic.
- `analyze_apr_in_pd.py` selects its calibrator on H1 and reports H2.

Tests: `tests/test_profit_audit.py` (decomposition sums exactly; each error source is isolated),
`tests/test_pipeline.py::test_deployment_realism_evaluates_on_synthetic_frame`.

---

## Stage 9 · Drift, explainability, fairness (EXP07)

| | |
|---|---|
| Script | [`scripts/evaluate.py`](../../scripts/evaluate.py) |
| Library | [`monitoring/drift.py`](../../src/credit_risk/monitoring/drift.py), [`explainability/`](../../src/credit_risk/explainability/), [`evaluation/fairness.py`](../../src/credit_risk/evaluation/fairness.py) |
| Out | `results/tables/boosting_exp07_*`, `boosting_fairness_*`, `boosting_shap_importance.csv` |

- `population_stability_index(reference, current, n_bins=10)`, `feature_drift(...)`,
  `prediction_drift(...)`, `calibration_drift(df, score_column, ..., freq="Q")`.
- `explain_boosting(model, X, max_rows=5000, seed=0)`, `explain_row(...)` for reason codes;
  `group_analysis(...)` and `fairness_report(...)` for slices.

Non-obvious lines: `evaluate.py` also writes `boosting_exp07_calibration_drift_2015h2.csv`, the same
statistic on the holdout half only; it re-scores with the calibrator named in
`boosting_calibration_choice.json`.

---

## Stage 10 · Rendering and repository validation

- [`scripts/render_readme.py`](../../scripts/render_readme.py): `render_results_table()`,
  `render_rl_paragraph()`, `render_profit_paragraph()`, `render_apr_ablation_paragraph()`,
  `render_data_table()`, `render_data_profile()`, `render_guide_page(text)`, `update_readme()`,
  `update_guide()`. It fills README regions and named BEGIN/END marker blocks from `results/tables/`.
- [`scripts/validate_repository.py`](../../scripts/validate_repository.py): a standalone check, separate from
  pytest, that the results on disk trace to the current code, the pipeline chain is intact and the
  leakage controls hold on the real processed data (it needs `data/processed/loans.parquet`).
- [`scripts/run_home_credit.py`](../../scripts/run_home_credit.py) with
  [`data/home_credit.py`](../../src/credit_risk/data/home_credit.py): a side experiment on a second
  dataset, outside the main pipeline. The repository carries no outputs from it: the earlier outputs came
  from the `--synthetic` smoke test (not real credit data) and were removed. Real results need the Kaggle
  extract (see [`docs/reference/DATASET.md`](../reference/DATASET.md)).

Tests: `tests/test_readme_tables.py` re-renders every block and fails on any difference.

---

## Configs

| File | Purpose |
|---|---|
| `configs/data/lending_club.yaml` | dataset, files, split windows |
| `configs/models/boosting.yaml`, `logistic.yaml` | model and calibration settings (the calibration methods list includes `intercept_only`) |
| `configs/graph/cohort_graph.yaml` | relations, cohort-feature smoothing (`smoothing_alpha: 25`), GraphSAGE settings |
| `configs/simulation/base.yaml` | economics, simulator, evaluation seeds, RL hyper-parameters |
| `configs/experiments/exp01..exp11_*.yaml` | per-experiment overrides (EXP06's budget sweep: 100%, 60%, 35%, 20% of demand) |
