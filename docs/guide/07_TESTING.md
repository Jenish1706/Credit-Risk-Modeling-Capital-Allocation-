# 07 · Testing

```powershell
python -m pytest -q
```

About 25 seconds, no real data and no network needed. Everything is built from the
synthetic generator at a size (12,000 loans) that keeps the whole suite fast; the tests check
**invariants**, not model quality, so a small sample is enough.

## Philosophy

1. **A test should fail before the fix and pass after.** Every audit fix in this repository has a
   regression test written first (for example the byte-identical-parameters test for
   evaluation-time learning).
2. **Leakage rules raise; they do not warn.** `assert_no_leakage` and `assert_no_future_edges` are exercised by planting a violation and checking it
   is caught, and the selection functions are exercised by poisoning the holdout
   (`tests/test_selection_windows.py`).
3. **Rendered documentation is tested.** The README and the guide blocks are regenerated from the
   artifacts and compared; a hand-edited number cannot survive.

## What is covered, by module

| Test file | Covers |
|---|---|
| [`tests/test_selection_windows.py`](../../tests/test_selection_windows.py) | the calibrator choice and flag cut-points read only 2015-01…06 rows: replacing every holdout label and prediction changes nothing |
| [`tests/test_leakage.py`](../../tests/test_leakage.py) | the deny-list and its enforcement, primary vs `with_lc_grade` feature sets, train-only statistics, the availability guard, split non-overlap and temporal ordering, the H1/H2 window masks and `assert_selection_window`, the windowed conditional test deciding on H1 |
| [`tests/test_graph_leakage.py`](../../tests/test_graph_leakage.py) | no future edges (strictly earlier, not merely not later), no self-loops, padding masked, lookback enforced, a planted violation caught, the label information set being strictly smaller than the feature set, shrinkage and evidence weights, the macro and shuffled controls, and that an ungated cohort rate leaks measurably |
| [`tests/test_models.py`](../../tests/test_models.py) | AUC blind to calibration, slope detects over-confidence, the intercept-only baseline isolating a base-rate shift, O/E direction, KS against a manual computation, calibrators fixing a distorted model and preserving ranking, feature-builder behaviour (indicators, unseen levels, disk round trip), logistic and boosting baselines |
| [`tests/test_decision.py`](../../tests/test_decision.py) | weighted-average life, realised profit arithmetic and no double-counted recoveries, empirical LGD, the textbook expected-loss identity, expected profit monotone in PD and price, breakeven PD zeroing expected profit, action-space exposure cap |
| [`tests/test_simulation.py`](../../tests/test_simulation.py) | the outcome coupling, common random numbers, rejecting everything earns zero, budget never exceeded, episode accounting, reproducibility, paired-difference tables, the shadow price (zero when unbinding, the marginal applicant, validation not test), LinUCB/Thompson/Q-learning mechanics, evaluation leaving learner parameters byte-identical, order independence, the headline action space containing only the 1.0x multiplier, accept/reject episodes earning only realised outcomes |
| [`tests/test_blindspot.py`](../../tests/test_blindspot.py) | profit from observed cashflow, greedy allocation, equal-budget comparison, noise and PD-restating signals failing the conditional test, a genuine signal passing, confidence intervals and two-sided p, placebo controls |
| [`tests/test_profit_audit.py`](../../tests/test_profit_audit.py) | the predicted-vs-realised decomposition sums exactly and isolates each error source, decile table shape, cost symmetry |
| [`tests/test_pipeline.py`](../../tests/test_pipeline.py) | synthetic generator faithfulness, month parsing, employment normalisation, split mapping, target mapping, report counts, config loading and inheritance, PSI, prediction and calibration drift, reason-code phrasing, the deployment-realism audit, GraphSAGE (torch) |
| [`tests/test_readme_tables.py`](../../tests/test_readme_tables.py) | README tables and prose equal their rendering, one calibration-ratio convention, EXP04 over at least three seeds, EXP06 lambda rows, approval rates and collapsed learners present, superseded banners, the APR ablation, guide blocks |
| [`tests/test_docs.py`](../../tests/test_docs.py) | every relative link in README and docs resolves, every repository path named in the code walkthrough exists, every function named in the code pages is defined (`scripts/check_docs.py`) |

Fixtures live in [`tests/conftest.py`](../../tests/conftest.py): a session-scoped synthetic raw file,
the prepared frame, `split_frames` (train, valid, test) and an `enriched` frame with derived
features.

## What is deliberately not covered

- **End-to-end `main()` of each script on real data.** Scripts are drivers; their logic lives in
  tested library functions. Running them is the runbook's job ([06_RUNBOOK.md](06_RUNBOOK.md)), and
  `scripts/validate_repository.py` checks the artifacts on disk against the processed data.
- **Model quality on real data.** The suite does not assert an AUC or an ECE; those are results,
  reported with uncertainty.
- **Training quality of the neural models.** The autoencoder and VGAE have no tests; GraphSAGE has
  smoke tests (it trains, predicts, does not distort the base rate, ignores masked neighbours).
- **Plots, fairness slices, SHAP explanations of the boosting model** beyond the reason-code
  phrasing.
- **The Home Credit side experiment** and the download code (no network in tests).

## How skips work on a fresh clone

- Tests that need torch import it with `pytest.importorskip("torch")` and are skipped with a reason
  when it is absent; they are not counted as failures.
- Tests do not need `data/processed/` or `artifacts/` (both are untracked). The README tests read
  `results/tables/`, which **is** tracked. The data-table renderer reads `data/processed/loans.parquet`
  when it exists and otherwise falls back to the recorded counts, so the suite passes on a fresh clone.

## Adding a test

1. Reproduce the defect or gap first, then write the test so it **fails**: run
   `python -m pytest tests/test_x.py -k name -q`.
2. Use the synthetic fixtures; do not read real data.
3. Test an invariant (for example "the sum of the three effects equals the gap") rather than a
   number copied from a result.
4. For a leakage rule, plant a violation and assert it is caught, as in
   `test_assert_no_future_edges_catches_a_planted_violation`.
5. If the change alters a result, regenerate the artifacts, run `python scripts/render_readme.py`, and
   let `tests/test_readme_tables.py` confirm the rendering.
6. Run the targeted test while iterating and the full suite before committing.
