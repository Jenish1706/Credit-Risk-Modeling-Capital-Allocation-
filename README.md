# Graph-Based Credit Risk and Lending Decision System

A credit-risk and lending-decision pipeline on real LendingClub loans. It was built to answer one
question and a second that followed from it:

> Can borrower-level credit-risk modelling be improved by relational financial information, and can
> calibrated risk predictions then be used by a decision policy to optimise lending outcomes rather
> than merely predict default?
>
> Can representation-learning signals identify where a conventional credit-risk model is
> unreliable, and can that information improve lending decisions?

Several answers are null or negative and are reported as findings. Two earlier headline conclusions
were reversed by audit (see [`docs/WHAT_BROKE.md`](docs/WHAT_BROKE.md)).

## Approach

- **Risk.** Logistic baseline and LightGBM on decision-time features, with calibration treated as a
  separate step and compared against an intercept-only baseline.
- **Graph.** A time-respecting cohort graph (shared 3-digit ZIP, shared employer text), cohort
  aggregates with empirical-Bayes shrinkage, and a hand-written GraphSAGE. Neighbour *labels* are
  used only once the neighbour's outcome had resolved.
- **Decision.** Economics estimated from observed cashflows, a ladder of policies from a fixed PD
  threshold to an expected-profit rule, and a simulator that replays real applicants at the funded
  exposure. LinUCB, Thompson sampling and Q-learning are compared against the simple rules.
- **Trust.** Leakage tests that fail the build, blind-spot signals tested conditional on the PD level,
  placebo controls, drift monitoring and a predicted-versus-realised profit audit.

## Data

**LendingClub 2007–2018Q4** (`wordsforthewise/lending-club`, Kaggle, data licence **CC0**), restricted
to **36-month loans issued 2010-01 … 2015-12** so every outcome has matured. Splits are out-of-time,
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

## Results

Every **selection** (the production calibrator, KEEP/REJECT verdicts, flag cut-points) uses
**2015-01…06 (H1)** only. Headline numbers are measured on **2015-07…12 (H2)**, which no selection
touched; the full-2015 figure is in brackets for continuity. `OBSERVED` is measured from data,
`SIMULATED` is produced inside the lending simulator. Every number below is rendered from
`results/tables/` by `scripts/render_readme.py`.

<!-- BEGIN:results-table -->
| # | Question | Result | Verdict |
|---|---|---|---|
| **EXP01** | Does non-linear modelling beat logistic regression? | AUC **0.6606 → 0.6732** (+0.0126) on 2015-07..12 (full 2015: 0.6581 → 0.6705) `OBSERVED` | Material. Boosting justified |
| **EXP02** | Does calibration improve PD reliability? | AUC **unchanged to 5 d.p.**; ECE **0.0198 → 0.0117** (intercept_only, chosen on 2015-01..06) on 2015-07..12 (full 2015: 0.0193 → 0.0110); slope 1.003 → 1.004 `OBSERVED` | The point of the project in one row: ranking metrics are blind to calibration |
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | Simulated diff vs 15% threshold: **-28,262**/episode (95% CI [-54,837, -2,616]) `SIMULATED` on 2015-07..12. Real book at observed exposure: **-36.9**/loan (95% CI [-47.1, -26.9]) `OBSERVED` on 2015-07..12 | **Expected-profit rule loses to a PD threshold on the real book.** Simulation CI excludes zero. Mechanism is measured below, not assumed |
| **EXP04** | Do graph cohort features add signal? | AUC **-0.0009**. Mean over 5 LightGBM seeds on 2015-07..12: ECE **0.0110 ± 0.0001 → 0.0048 ± 0.0005**, O/E **1.081 → 0.998** (EXP04/05 single-fit run: ECE 0.0109 → 0.0043, O/E 1.080 → 0.989) `OBSERVED` | No discrimination gain; **calibration gain survives both controls in every one of 5 seeds** (ΔECE B−tabular -0.0062 ± 0.0005, B−macro -0.0066 ± 0.0002, B−shuffled -0.0038 ± 0.0006; mean ± sd over seeds, per-seed paired bootstrap, 2015-07..12) |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6637** vs 0.6704 (aggregates) vs 0.6712 (tabular) on 2015-07..12 `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | Frozen-at-test learners, accept/reject at observed exposure (2015-07..12). Unconstrained: best learner thompson +17,398 vs myopic expected-profit (95% CI [-2,574, +37,136]). 60% budget: thompson -160 vs the 15% threshold (95% CI [-15,333, +14,180]) `SIMULATED` | Control holds: no learner significantly beats the myopic rule with unconstrained capital; **no learner significantly beats the fixed 15% threshold in any condition**; significantly worse than the threshold: budget_aware_q(greedy) (unconstrained, budget=0.60); linucb@alpha=1 (budget=0.35, budget=0.20); thompson (budget=0.20) |
| **EXP07** | How does the model degrade under vintage drift? | O/E **0.861 (2010) → 1.082 (2015-07..12)** (full 2015: 1.077) while AUC moves 0.7475 → 0.6732 `OBSERVED` | Calibration drifts before ranking does |
| **EXP08** | Does autoencoder anomaly flag untrustworthy PDs? | H1 decision **KEEP**. On 2015-07..12: survives PD control in **10/10** PD strata, two-sided sign test p=0.0020, odds ratio 1.033/SD [1.021, 1.046] (full 2015: 10/10, OR 1.032) `OBSERVED` | **KEEP**, confirmed on holdout |
| **EXP09** | Does VGAE relational uncertainty flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **3/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.002/SD [0.988, 1.017] (full 2015: 5/10, OR 1.002) `OBSERVED` | **REJECT** |
| **EXP10** | Does model disagreement flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **7/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.050/SD [1.033, 1.068] (full 2015: 9/10, OR 1.045) `OBSERVED` | **REJECT** |
| **EXP11** | Does acting on those signals improve the book? | Default rate **16.62% → 15.88%** on 2015-07..12, placebo controls show **15.87%–16.26%** `OBSERVED` | **Beats random exclusion but not the APR-matched placebo:** the risk reduction is the APR-mix shift, not blind-spot selectivity |
<!-- END:results-table -->

What the table says, in words:

- The graph adds nothing to discrimination and GraphSAGE is the weakest arm. The cohort features lower
  calibration error, the gain holds in every one of five training seeds, and the shuffled-key control
  shows part of it is not specific to real cohort membership.
- An expected-profit rule built on these PDs loses to a plain 15% PD threshold on the real book. The
  audit attributes the gap mainly to the PD understating risk where APR is high, and partly to drift in
  the cashflow curves; giving the PD LendingClub's own pricing narrows but does not close it.
- With learning frozen at evaluation and every outcome real, no learned policy beats the simple rule.
- One of three blind-spot signals passes the pre-specified rule, and acting on it does not beat an
  APR-matched placebo.

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

Appendices: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md),
[`docs/LEAKAGE.md`](docs/LEAKAGE.md). Earlier design documents
([`ASSUMPTIONS`](docs/ASSUMPTIONS.md),
[`GRAPH_DESIGN`](docs/GRAPH_DESIGN.md), [`RL_FORMULATION`](docs/RL_FORMULATION.md),
[`DATASET`](docs/DATASET.md)) and the audit record ([`AUDIT_REPORT`](docs/AUDIT_REPORT.md),
[`WHAT_BROKE`](docs/WHAT_BROKE.md)) remain for detail.

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
