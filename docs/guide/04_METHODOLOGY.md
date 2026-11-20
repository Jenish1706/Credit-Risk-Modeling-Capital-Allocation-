# 04 · Methodology

How every number in this repository is produced, why the protocol is shaped the way it is, and what
result would falsify each claim. Concepts are explained in [01_CONCEPTS.md](01_CONCEPTS.md); the
results themselves are in [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md).

## 1. Windows

| Window | Issue months | Used for |
|---|---|---|
| history | 2007-06 to 2009-12 | graph neighbours only; never modelled |
| train | 2010-01 to 2014-06 | fit encoders, models, the cashflow curves, the cohort prior |
| valid | 2014-07 to 2014-12 | early stopping, hyper-parameter search, calibrator fitting, the shadow price, bandit training |
| test H1 | 2015-01 to 2015-06 | **every selection** (production calibrator, KEEP/REJECT verdicts, flag cut-points) |
| test H2 | 2015-07 to 2015-12 | **headline numbers**; nothing is chosen on it |
| test (full 2015) | H1 plus H2 | a labelled continuity column next to the headline |
| monitor | 2016-01 to 2018-12 | unlabelled drift monitoring |

Counts and default rates are rendered in [03_DATA.md](03_DATA.md). The rule is enforced in the functions that make each selection, which take the issue period and
read only the 2015-01…06 rows: `select_calibrator_on_window`
([`models/calibration.py`](../../src/credit_risk/models/calibration.py)), `flag_cutpoints` and
`conditional_by_window` ([`decision/blindspot.py`](../../src/credit_risk/decision/blindspot.py)), built on
`test_window_masks` and `assert_selection_window` in
[`data/splits.py`](../../src/credit_risk/data/splits.py). `tests/test_selection_windows.py` poisons the
holdout and checks that none of the three selections moves: the calibrator choice and the flag cut-points
(every holdout label and prediction replaced), and the KEEP/REJECT decision (the holdout made to point
the other way, with the reported holdout column shown to change). Each selection function also asserts
that the rows it reads end at 2015-06, but that assertion is a tripwire, not the proof; the poisoning is.

**Why two halves of 2015.** An earlier version chose the production calibrator, the KEEP/REJECT
verdicts and the flag cut-points on the same book the headlines were reported on, so those
numbers were optimistic by construction. Splitting the test year in time removes that reuse, at the
cost of smaller samples (about 120,000 loans to select on, about 162,000 to report on).

**What is held fixed rather than tuned.** The 15% PD threshold, the risk-band edges, learner
hyper-parameters, the flag quantile (90th percentile), the budget fraction (80% of profitable
demand) and the half-exposure cap are specified in configs or script defaults *before* evaluation
and are not tuned. The shadow price is fitted on the validation pool.

## 2. Baselines

| Question | Baseline | Challenger |
|---|---|---|
| Does non-linear modelling help? | regularised logistic regression | LightGBM |
| Does calibration help? | uncalibrated LightGBM, **and** an intercept-only shift | Platt, isotonic |
| Do cohort features help? | tabular LightGBM | tabular plus cohort features; GraphSAGE |
| Does the calibration gain come from the graph? | tabular plus macro default rate; tabular plus **shuffled** cohort keys | tabular plus real cohort features |
| Does the expected-profit rule help? | fixed 15% PD threshold; approve-all; risk-band | expected-profit rule (and its shadow-priced version) |
| Does RL help? | the myopic expected-profit rule, the 15% threshold, the validation-priced shadow-price policy | LinUCB, Thompson sampling, tabular Q-learning |
| Do blind-spot signals help? | PD ranking alone | PD ranking that refuses (or caps) flagged borrowers |

## 3. Metrics

| Metric | Definition | Used for |
|---|---|---|
| ROC-AUC, Gini, KS | ranking quality (see concepts §3) | EXP01, EXP04, EXP05 |
| Brier score | mean squared error of the probability | choosing the production calibrator (on H1) |
| ECE | 20 quantile bins, weighted mean absolute gap | EXP02, EXP04 |
| Calibration slope and intercept | logistic regression of outcome on predicted log-odds | EXP02 |
| **O/E** | observed default rate over mean prediction | EXP02, EXP04, EXP07, deployment realism |
| PSI | population stability index on reference quantile bins | EXP07 |
| Realised profit | `total_pymnt - collection_recovery_fee - funded_amnt` (cash received already includes recoveries) | economics, EXP03, EXP11 |
| Return on capital | profit over capital deployed | EXP11 |
| Episode reward | profit net of funding and operating cost over one simulated funding period | EXP03, EXP06 |

Every table names its window. Where two windows are shown, H2 is the headline and full 2015 is the
second column.

## 4. Statistical tests and why each fits

| Comparison | Test | Why this one |
|---|---|---|
| Two policies on the same simulated episodes | paired bootstrap of per-episode differences (10,000 resamples, 30 episodes) | the episodes share applicants and outcome draws, so pairing removes shared noise |
| Two rules on the real book | paired bootstrap over loans of the per-loan profit difference (10,000 resamples, `evaluation.n_bootstrap`) | loans are the unit; both rules see the same loans |
| ECE or O/E gap between two models | paired bootstrap on the same test loans (2,000 resamples) | ECE has no closed-form interval; pairing matters because both arms score identical loans |
| Model-fit variance | five LightGBM seeds per arm; mean and standard deviation across seeds, bootstrap within each seed | a single fit cannot separate a real gain from fit-to-fit noise |
| Signal survives PD control | exact two-sided binomial sign test across PD deciles | needs no distributional assumption about decile gaps; conservative with ten strata |
| Signal strength | odds ratio per standard deviation, Wald 95% interval from the Fisher information | incremental information about the outcome beyond `logit(PD)` |
| Does the signal beat budget refilling? | 200-draw placebos: random exclusion and APR-matched exclusion | the draws give an interval for what "no signal" produces |

## 5. Controls

| Claim under test | Control | What it rules out |
|---|---|---|
| Calibration improves | intercept-only baseline | the gain being a base-rate shift between train and test |
| Cohort features improve calibration | macro default-rate feature; cohort keys shuffled within issue month | the gain being just "a recent default rate" or "any cohort-shaped features" |
| A blind-spot signal is real | conditioning on PD decile | the signal merely restating the PD level |
| Acting on signals helps | random and APR-matched placebo exclusion | budget refilling producing the default-rate drop |
| An RL policy beats rules | frozen evaluation (`learn=False`), accept/reject at observed exposure, comparison against the strongest simple rule | leakage of test outcomes into the learner, and counterfactual-outcome artefacts |
| Predicted profit is reliable | decomposition of predicted versus realised profit into PD, yield and severity effects; ablation with a PD that sees LendingClub's pricing | blaming the wrong input |

## 6. What each experiment measures, and what would falsify it

| ID | Measures | Script | A result that would falsify the claim |
|---|---|---|---|
| EXP01 | LightGBM vs logistic regression, AUC on H2 | `train_baseline.py` | an AUC difference within noise |
| EXP02 | calibration methods on H2 (ECE, slope, O/E) | `train_baseline.py` | the intercept-only baseline matching the gain, which would mean the improvement is only a base-rate shift |
| EXP03 | expected-profit rule vs 15% threshold, simulated and on the real book | `train_policy.py`, `analyze_profit_calibration.py` | a real-book interval excluding zero in the other direction |
| EXP04 | cohort features vs tabular, with controls, over five seeds | `train_gnn.py`, `build_cohort_controls.py` | any seed where B fails to beat a control with an interval excluding zero |
| EXP05 | GraphSAGE vs aggregates vs tabular | `train_gnn.py` | GraphSAGE beating both on H2 AUC |
| EXP06 | learned policies vs static rules across capital budgets | `train_policy.py` | a learner whose paired interval against the threshold excludes zero in its favour |
| EXP07 | feature, prediction and calibration drift | `evaluate.py` | O/E staying near 1 across vintages while AUC falls |
| EXP08/09/10 | does the signal flag miscalibration beyond PD | `analyze_blind_spots.py`, `train_vgae.py`, `analyze_disagreement.py` | a signal failing the H1 sign test (REJECT), or passing H1 and failing on H2 (not confirmed) |
| EXP11 | acting on kept signals at equal budget | `analyze_decision_swaps.py` | the signal portfolio falling inside both placebo intervals (no attributable effect) |
| Deployment realism | cost of using labels not yet known at the deployment date | `analyze_deployment_realism.py` | no difference between omniscient and deployment-realistic training |
| Profit audit | why the expected-profit rule underperforms | `analyze_profit_calibration.py` | predicted and realised profit agreeing decile by decile |
| APR ablation | PD with LendingClub's pricing | `analyze_apr_in_pd.py` | flat O/E across APR bands and the rule then matching the threshold |

## 7. The simulator, stated honestly

`LendingEnvironment` replays applicants from a pool in random order, one funding period (1,200
applicants) per episode, with the same applicants and outcome draws for every policy (common random
numbers). The headline action space is **accept or reject at the exposure that was actually funded**,
so every approved loan earns the return it really earned and the simulator contains no counterfactual
outcome. A second, labelled sensitivity run adds 0.5x and 1.5x exposure actions; their outcomes at
non-observed exposures use a model coupling and are not measurements. Learners train on the
validation pool for 400 (Q-learning) or 120 (bandit) episodes and are frozen during evaluation.
Costs: 3% annual cost of funds over a 1.55-year weighted-average life (the amortisation schedule gives about 1.6 years), $150 fixed plus 1%
variable operating cost. The full assumption catalogue is in [ASSUMPTIONS.md](../reference/ASSUMPTIONS.md) and
the formulation in [RL_FORMULATION.md](../reference/RL_FORMULATION.md).

## 8. Reproducibility contract

- Every script takes `--config` and a seed (default 42); outputs record the data source (real or
  synthetic), and synthetic results are refused as findings.
- Every number in the README and in embedded guide blocks is rendered from `results/tables/` by
  `scripts/tools/render_readme.py`; a test fails if a hand edit drifts from the artifacts.
- Selection windows are enforced by selection functions that read only the selection rows, and tested by
  poisoning the holdout.
