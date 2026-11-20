# Graph-Based Credit Risk and Lending Decision System

A credit-risk and lending-decision pipeline on real LendingClub loans. It was built to answer one
question and a second that followed from it:

> Can borrower-level credit-risk modelling be improved by relational financial information, and can
> calibrated risk predictions then be used by a decision policy to optimise lending outcomes rather
> than merely predict default?
>
> Can representation-learning signals identify where a conventional credit-risk model is
> unreliable, and can that information improve lending decisions?

Several answers are null or negative and are reported as findings. A number of earlier headline
conclusions were reversed by audit (see [`docs/guide/09_RESULTS_AND_LIMITATIONS.md`](docs/guide/09_RESULTS_AND_LIMITATIONS.md)).

## Approach

- **Risk.** Logistic baseline and LightGBM on decision-time features, with calibration treated as a
  separate step and compared against an intercept-only baseline.
- **Graph.** A time-respecting cohort graph (shared 3-digit ZIP, shared employer text), cohort
  aggregates with empirical-Bayes shrinkage, and a hand-written GraphSAGE. Neighbour *labels* are
  used only once the neighbour's outcome had resolved.
- **Decision.** Economics estimated from observed cashflows, a ladder of policies from a fixed PD
  threshold to an expected-profit rule, and a simulator that replays real applicants at the funded
  exposure. LinUCB, Thompson sampling and Q-learning are compared against the simple rules.
- **Trust.** Leakage tests that fail when a violation is planted, blind-spot signals tested conditional on the PD level,
  placebo controls, drift monitoring and a predicted-versus-realised profit audit.

## Data

**LendingClub 2007–2018Q4** (`wordsforthewise/lending-club`, Kaggle, data licence **CC0**), restricted
to **36-month loans**. The labelled population is loans issued **2010-01 … 2015-12**, so every outcome has
matured; the 2007–2009 history and 2016–2018 monitoring windows are unlabelled. Splits are out-of-time,
never random.

<!-- BEGIN:data-table -->
| Split | Window | Loans | Default rate |
|---|---|---|---|
| history (graph only) | 2007-06 … 2009-12 | 8,277 | unlabelled by design |
| **train** | 2010-01 … 2014-06 | 239,104 | 12.67% |
| **valid** | 2014-07 … 2014-12 | 90,615 | 14.13% |
| **test** (out-of-time) | 2015-01 … 2015-12 | 283,026 | 14.89% |
| ↳ test H1 (all selections) | 2015-01 … 2015-06 | 120,790 | 15.12% |
| ↳ test H2 (untouched holdout, headline) | 2015-07 … 2015-12 | 162,236 | 14.71% |
| monitor (production traffic) | 2016-01 … 2018-12 | 988,585 | unlabelled by design |
<!-- END:data-table -->

## How the project is built

<!-- BEGIN:how-built -->
Each step answers a question the previous result made necessary. The experiment ids refer to the table below, which keeps them as references.

1. **Predict (EXP01).** Can a model rank borrowers out of time? Yes, modestly: gradient boosting beats a logistic baseline on a window no selection touched.
2. **Trust the probability (EXP02, EXP07, deployment realism).** A decision multiplies the probability by dollars, so ranking is not enough. Calibration gains are mostly a base-rate correction, the PD's level drifts across vintages while out-of-sample ranking is flat over the months observed, and a model trained only on the loans already resolved at the deployment date does clearly worse than the usual benchmark, though that comparison mixes staleness with selection on resolution.
3. **Add relational information (EXP04, EXP05).** Do cohorts help? Not ranking. They fix the *level* of the PD in every training seed tested, though shuffled cohort keys reproduce part of that gain; GraphSAGE is the weakest arm.
4. **Decide (EXP03, EXP06).** Does a profit rule built on the PD beat a plain PD threshold, and does learning or capital pricing help? The profit rule loses to the threshold on the real book; no learned policy significantly beats the threshold in any capital condition; the validation-fitted shadow-price policy is worse than the myopic rule.
5. **Diagnose the failure.** Where does predicted profit go wrong? A dollar decomposition of predicted versus realised profit, a PD refit with LendingClub's pricing, and a PD refit with the cohort features (correct level, remaining APR gradient). The PD's default-count error is the largest single source, it varies systematically with APR, and correcting the PD's level alone does not fix the decision.
6. **Find where the PD is unreliable (EXP08, EXP09, EXP10).** Blind-spot signals, judged conditional on the PD level and decided on a selection window; only one passes.
7. **Act on it (EXP11).** Excluding borrowers flagged by the kept signal, judged against random and APR-matched placebos.

**Root cause, measured.** The decision layers inherit the PD's errors. The production PD under-predicts defaults where APR is high (observed/expected 0.67 in the lowest APR quintile to 1.38 in the highest), so a rule that rewards APR selects understated risk: the PD's default-count error is 55% of the gap between predicted and realised profit (EXP03; the rest is drift in the cashflow curves), refitting the PD for level alone does not fix the decision, and the risk reduction in EXP11 is matched by excluding loans of the same APR mix.
<!-- END:how-built -->

## Results

Every **selection** (the production calibrator, KEEP/REJECT verdicts, flag cut-points) uses
**2015-01…06 (H1)** only. Headline numbers are measured on **2015-07…12 (H2)**, which no selection
touched; the full-2015 figure is in brackets for continuity. `OBSERVED` is measured from data,
`SIMULATED` is produced inside the lending simulator. The tables are rendered from
`results/tables/` by `scripts/render_readme.py`; the prose around them states no result numbers, only
fixed protocol parameters (a test enforces this).

<!-- BEGIN:results-table -->
| # | Question | Result | Verdict |
|---|---|---|---|
| **EXP01** | Does non-linear modelling beat logistic regression? | AUC **0.6606 → 0.6732** (+0.0126, paired-bootstrap 95% CI [+0.0110, +0.0142], 2,000 draws) on 2015-07..12 (full 2015: 0.6581 → 0.6705) `OBSERVED` | Material: the whole interval is above the +0.01 AUC threshold. Boosting justified |
| **EXP02** | Does calibration improve PD reliability? | AUC **unchanged to 5 d.p.**; ECE **0.0198 → 0.0117** (intercept_only, chosen on 2015-01..06) on 2015-07..12 (full 2015: 0.0193 → 0.0110); slope 1.003 → 1.004 `OBSERVED` | Ranking is unchanged and ECE falls, but this is mostly a base-rate shift: a one-parameter intercept shift recovers 94% of the best calibrator's ECE gain |
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | Simulated diff vs 15% threshold: **-28,262**/episode (95% CI [-54,837, -2,616]) `SIMULATED` on 2015-07..12. Real book at observed exposure: **-36.9**/pool applicant (95% CI [-47.1, -26.9]) `OBSERVED` on 2015-07..12 | **Expected-profit rule loses to a PD threshold on the real book.** Simulation CI excludes zero. Mechanism is measured below, not assumed |
| **EXP04** | Do graph cohort features add signal? | AUC **-0.0012** (mean over 5 LightGBM seeds). On 2015-07..12: ECE **0.0110 ± 0.0001 → 0.0048 ± 0.0005**, O/E **1.081 → 0.998** (EXP04/05 single-fit run: ECE 0.0109 → 0.0043, O/E 1.080 → 0.989) `OBSERVED` | No discrimination gain; **calibration gain survives both controls in every one of 5 seeds** (ΔECE B−tabular -0.0062 ± 0.0005, B−macro -0.0066 ± 0.0002, B−shuffled -0.0038 ± 0.0006; mean ± sd over seeds, per-seed paired bootstrap, 2015-07..12) |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6637** vs 0.6704 (aggregates) vs 0.6712 (tabular) on 2015-07..12; the GNN is one training run `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | Frozen-at-test learners, accept/reject at observed exposure (2015-07..12). Unconstrained: best learner thompson +17,398 vs myopic expected-profit (95% CI [-2,574, +37,136]). 60% budget: thompson -160 vs the 15% threshold (95% CI [-15,333, +14,180]) `SIMULATED` | Control: no learner significantly beats the myopic rule with unconstrained capital (not significant is not equal: the best interval reaches +37,136); under a binding budget some learners do significantly beat the myopic rule: thompson, linucb@alpha=1 (budget=0.60); **no learner significantly beats the fixed 15% threshold in any condition**; significantly worse than the threshold: budget_aware_q(greedy) (unconstrained, budget=0.60); linucb@alpha=1 (budget=0.35, budget=0.20); thompson (budget=0.20); learners that stopped lending (approval below 1%): linucb@alpha=1 (budget=0.35); thompson (budget=0.20); the validation-fitted shadow-price policy is significantly worse than the myopic rule at budget=0.35 |
| **EXP07** | How does the model degrade under vintage drift? | O/E **0.861 (2010, a training vintage) → 1.082 (2015-07..12)** (full 2015: 1.077). AUC by quarter: in-sample training vintages 0.700–0.761, out-of-sample (validation and test) 0.665–0.675 `OBSERVED` | Calibration drifts across vintages (O/E rises from below 1 to above 1). Out-of-sample ranking is flat over the 18 months observed, so the fall in AUC from the training vintages to the out-of-sample quarters is mostly in-sample optimism. Every out-of-sample quarter is a late one, so this cannot rule out drift in ranking between earlier vintages |
| **Deployment realism** | What if the model only knows the labels available at the deployment date? | Trained only on loans resolved by 2014-12, on 2015-07..12: AUC **0.6712 → 0.6503**, ECE **0.0109 → 0.0367**, O/E **1.080 → 1.294** (full 2015: AUC 0.6690 → 0.6478, O/E 1.076 → 1.286). The resolved loans are a selected sample: default rate 16.3% against 12.6% for all loans issued in the training window, 10.4% against 13.3% in the calibration window `OBSERVED` | **A naive 'train on what has resolved' model is clearly worse on day one:** ranking and calibration both fall. The comparison mixes staleness with selection on resolution, so it does not isolate staleness, and no arm here trains on matured vintages only |
| **EXP08** | Does autoencoder anomaly flag untrustworthy PDs? | H1 decision **KEEP**. On 2015-07..12: survives PD control in **10/10** PD strata, two-sided sign test p=0.0020, odds ratio 1.033/SD [1.021, 1.046] (full 2015: 10/10, OR 1.032) `OBSERVED` | **KEEP**, confirmed on holdout |
| **EXP09** | Does VGAE relational uncertainty flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **3/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.002/SD [0.988, 1.017] (full 2015: 5/10, OR 1.002) `OBSERVED` | **REJECT** on H1 (4/10 strata, two-sided sign-test p=0.7539; the rule needs p<0.05 with the gap in the right direction). The odds-ratio interval is reported but is not part of the rule |
| **EXP10** | Does model disagreement flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **7/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.050/SD [1.033, 1.068] (full 2015: 9/10, OR 1.045) `OBSERVED` | **REJECT** on H1 (8/10 strata, two-sided sign-test p=0.1094; the rule needs p<0.05 with the gap in the right direction). The odds-ratio interval is reported but is not part of the rule |
| **EXP11** | Does acting on those signals improve the book? | Default rate **16.62% → 15.88%** on 2015-07..12. Placebo exclusion (200 draws, 95% intervals): random 16.08%–16.26%, APR-matched 15.87%–16.05%. Return on capital 6.82% → 6.87% (random 6.73%–6.87%, APR-matched 6.77%–6.91%) `OBSERVED` | **Beats random exclusion but not the APR-matched placebo.** The signal book is 0.09 points below the APR-matched placebo mean, 12% of its 0.74-point reduction, and +0.008 points from that interval's lower edge, so most of the reduction is the lower-APR refill; the margin is borderline; return on capital sits inside both placebo intervals, so there is no evidence of a profit gain |
<!-- END:results-table -->

What the table says, in words:

- The graph adds nothing to discrimination and GraphSAGE is the weakest arm. The cohort features lower
  calibration error, the gain holds in every training seed tested (the count is in the table), and the shuffled-key control
  shows part of it is not specific to real cohort membership.
- An expected-profit rule built on these PDs loses to a plain 15% PD threshold on the real book. The
  audit attributes the gap mainly to the PD understating risk where APR is high, and partly to drift in
  the cashflow curves; giving the PD LendingClub's own pricing narrows but does not close it.
- Refitting the PD with the cohort features fixes its level but not the decision: the rule still loses,
  because an APR gradient remains in the PD's errors. A model trained only on the loans already resolved
  at the deployment date does clearly worse than the usual out-of-time benchmark, but that comparison mixes
  staleness with selection on resolution.
- With learning frozen at evaluation and every outcome real, no learned policy beats the simple rule.
- One of three blind-spot signals passes the pre-specified rule. Acting on it beats a random placebo, but
  the margin over an APR-matched placebo is borderline and most of the risk reduction is the lower-APR
  refill.

The full tables with uncertainty (every policy in every capital condition, the profit audit, the
APR ablation) are in
[`docs/guide/09_RESULTS_AND_LIMITATIONS.md`](docs/guide/09_RESULTS_AND_LIMITATIONS.md).

## Quickstart

Windows PowerShell, conda environment (Smart App Control blocks unsigned PyPI wheels, so conda is
required; details in the runbook):

```powershell
conda create -n credit-risk python=3.13 numpy pandas scipy scikit-learn pyarrow lightgbm shap matplotlib pyyaml pytest numba
conda activate credit-risk
pip install -e . --no-deps
python -m pytest -q
python scripts/render_readme.py
```

Without a Kaggle token the pipeline can run on a synthetic extract (stamped `synthetic` and never
reported as a finding). The full pipeline from raw data, with run times, is in
[`docs/guide/06_RUNBOOK.md`](docs/guide/06_RUNBOOK.md); it takes about three hours, half of it the
variational graph autoencoder.

## Documentation

Start at [`docs/guide/00_START_HERE.md`](docs/guide/00_START_HERE.md).

| Page | Contents |
|---|---|
| [`01_CONCEPTS`](docs/guide/01_CONCEPTS.md) | every concept from scratch, with worked examples |
| [`02_ARCHITECTURE`](docs/guide/02_ARCHITECTURE.md) | components, data flow, life of one record |
| [`03_DATA`](docs/guide/03_DATA.md) | source, tables, quality problems, exclusions |
| [`04_METHODOLOGY`](docs/guide/04_METHODOLOGY.md) | windows, baselines, metrics, tests, controls, falsifiers |
| [`05_CODE_WALKTHROUGH`](docs/guide/05_CODE_WALKTHROUGH.md) | scripts and modules by pipeline stage |
| [`06_RUNBOOK`](docs/guide/06_RUNBOOK.md) | environment, commands, run times, troubleshooting |
| [`07_TESTING`](docs/guide/07_TESTING.md) | what is and is not tested |
| [`08_DESIGN_DECISIONS`](docs/guide/08_DESIGN_DECISIONS.md) | the decisions most likely to be questioned |
| [`09_RESULTS_AND_LIMITATIONS`](docs/guide/09_RESULTS_AND_LIMITATIONS.md) | results with uncertainty, null results, threats to validity |
| [`GLOSSARY`](docs/guide/GLOSSARY.md) | every term and acronym |

Appendices: [`docs/reference/DATA_DICTIONARY.md`](docs/reference/DATA_DICTIONARY.md),
[`docs/reference/LEAKAGE.md`](docs/reference/LEAKAGE.md). Earlier design documents
([`ASSUMPTIONS`](docs/reference/ASSUMPTIONS.md),
[`GRAPH_DESIGN`](docs/reference/GRAPH_DESIGN.md), [`RL_FORMULATION`](docs/reference/RL_FORMULATION.md),
[`DATASET`](docs/reference/DATASET.md)) remain for detail.

## Limitations

- **Not deployable.** The EXP06 policy results come from a simulator, and the real-book audits are
  retrospective reallocations measured on observed cashflows. Both use an *already-approved*
  population from one institution and one credit cycle in a long expansion; nothing is tested
  through a downturn.
- **No reject inference.** The model estimates `P(default | approved)`. The rejected-applications file
  is not analysed here.
- **No causal claims.** SHAP explains the model, not the borrower or the world.
- **The holdout is the second half of the same year** as the selection window, not a later cycle.
- **Training labels were not all knowable at the deployment date;** a delayed-feedback learning mode
  is not built.
- **No real-world impact.** No money was lent.

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Data: LendingClub loan data via Kaggle, CC0; the raw data is not
redistributed in this repository.
