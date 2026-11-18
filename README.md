# Graph-Based Credit Risk & Reinforcement Learning Lending Decision System

An end-to-end ML credit decision system on real LendingClub data, built to answer one question:

> Can borrower-level credit-risk modelling be improved by incorporating relational financial
> information, and can calibrated risk predictions then be used by a decision policy to optimise
> lending outcomes rather than merely predict default?

A second question was added later, and the two are reported side by side:

> Can representation-learning signals identify situations where a conventional credit-risk model is
> unreliable, and can that information improve lending decisions?

The governing philosophy is a **small theoretical surface area, fully explainable**. It is not a
survey of techniques. Several experiments return null or negative results, and those are reported as
findings rather than buried — including one headline conclusion that reversed under audit
(see [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md)).

---

## Data

**LendingClub 2007–2018Q4** (`wordsforthewise/lending-club`, Kaggle, licence **CC0**), restricted to
**36-month loans issued 2010-01 … 2015-12** so every outcome is fully matured — no censoring, no
term-mix confound. Splits are strictly **out-of-time**, never random:

| Split | Window | Loans | Default rate |
|---|---|---|---|
| history (graph only) | 2007-06 … 2009-12 | 8,277 | unlabelled by design |
| **train** | 2010-01 … 2014-06 | 239,104 | 12.67% |
| **valid** | 2014-07 … 2014-12 | 90,615 | 14.13% |
| **test** (out-of-time) | 2015-01 … 2015-12 | 283,026 | 14.89% |
| ↳ test H1 (all selections) | 2015-01 … 2015-06 | 120,790 | 15.12% |
| ↳ test H2 (untouched holdout, headline) | 2015-07 … 2015-12 | 162,236 | 14.71% |
| monitor (production traffic) | 2016-01 … 2018-12 | 988,585 | unlabelled by design |

The rising default rate across windows is real vintage deterioration, and it is the reason the
out-of-time split matters.

---

## Results

The 2015 test book is split in time. **Every selection** (the production calibrator, KEEP/REJECT
verdicts, flag cut-points) uses **2015-01…06 (H1)** only; the headline numbers below are measured on
**2015-07…12 (H2)**, which no selection has seen. The full-2015 figure is given in brackets for
continuity. Scripts assert that their selection frame ends at 2015-06. `OBSERVED` = measured from
data; `SIMULATED` = produced inside the lending simulator under the assumptions in
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

| # | Question | Result | Verdict |
|---|---|---|---|
| **EXP01** | Does non-linear modelling beat logistic regression? | AUC **0.6606 → 0.6732** (+0.0126) on 2015-07..12 (full 2015: 0.6581 → 0.6705) `OBSERVED` | Material. Boosting justified |
| **EXP02** | Does calibration improve PD reliability? | AUC **unchanged to 5 d.p.**; ECE **0.0198 → 0.0117** (intercept_only, chosen on 2015-01..06) on 2015-07..12 (full 2015: 0.0193 → 0.0110); slope 1.003 → 1.004 `OBSERVED` | The point of the project in one row: ranking metrics are blind to calibration |
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | Simulated diff vs 15% threshold: **-29,042**/episode (95% CI [-62,824, +4,620]) `SIMULATED` on 2015-07..12. Real book at observed exposure: **-36.9**/loan (95% CI [-47.1, -26.9]) `OBSERVED` on 2015-07..12 | **Expected-profit rule loses to a PD threshold on the real book.** Simulation CI straddles zero (null). Mechanism is measured below, not assumed |
| **EXP04** | Do graph cohort features add signal? | AUC **-0.0009**. ECE **0.0109 → 0.0043**, P/O **0.926 → 1.011** on 2015-07..12 (full 2015 ECE: 0.0105 → 0.0032) `OBSERVED` | No discrimination gain; **calibration gain confirmed by controls** (ΔECE -0.0066 vs macro [-0.0083, -0.0037], -0.0029 vs shuffled [-0.0047, -0.0002], 2015-07..12) |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6637** vs 0.6704 (aggregates) vs 0.6712 (tabular) on 2015-07..12 `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | Frozen-at-test learners vs myopic expected-profit (2015-07..12): unconstrained linucb@alpha=1 **+39,697** (95% CI [+13,802, +65,501]); at 60% budget best learner linucb@alpha=1 **-1,379** (95% CI [-30,139, +27,086]) `SIMULATED` | **Control breached:** learners significantly beat the myopic rule with unconstrained capital, where it should be near-optimal (see the real-book audit below); **no learner significantly beats expected-profit in any capital-constrained condition**; significantly worse at budget=0.60 |
| **EXP07** | How does the model degrade under vintage drift? | O/E **0.861 (2010) → 1.082 (2015-07..12)** (full 2015: 1.077) while AUC moves 0.7475 → 0.6732 `OBSERVED` | Calibration drifts before ranking does |
| **EXP08** | Does autoencoder anomaly flag untrustworthy PDs? | H1 decision **KEEP**. On 2015-07..12: survives PD control in **10/10** PD strata, two-sided sign test p=0.0020, odds ratio 1.033/SD [1.021, 1.046] (full 2015: 10/10, OR 1.032) `OBSERVED` | **KEEP**, confirmed on holdout |
| **EXP09** | Does VGAE relational uncertainty flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **3/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.002/SD [0.988, 1.017] (full 2015: 5/10, OR 1.002) `OBSERVED` | **REJECT** |
| **EXP10** | Does model disagreement flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **7/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.050/SD [1.033, 1.068] (full 2015: 9/10, OR 1.045) `OBSERVED` | **REJECT** |
| **EXP11** | Does acting on those signals improve the book? | Default rate **16.62% → 15.88%** on 2015-07..12, placebo controls show **15.87%–16.26%** `OBSERVED` | **Beats random exclusion but not the APR-matched placebo:** the risk reduction is the APR-mix shift, not blind-spot selectivity |

### Four results worth reading twice

**Calibration is where the graph might pay.** EXP04/EXP05 show the graph adds nothing to
*discrimination*: GraphSAGE is the worst of the three arms. The cohort features lower expected
calibration error (table above), and that gain is larger than the gain from two non-graph controls:
a time-respecting portfolio macro default rate, and cohort keys shuffled within issue month (paired
bootstrap intervals in the EXP04 verdict). This rests on one fitted LightGBM model per arm, so it
says nothing yet about variation across training seeds.

**Only one blind-spot signal passes the pre-specified rule.** Each signal is tested *conditional on
the PD level*; KEEP requires a two-sided sign test below 5% across PD strata on 2015-01…06, and the
holdout is then used to confirm. The autoencoder anomaly score passes and is confirmed; the VGAE and
the disagreement signal do not pass on H1 (the table gives their holdout figures). The odds ratios are
small, a few percent per standard deviation: they flag *miscalibration*, not bad borrowers.

**Acting on the surviving signal buys risk reduction, not profit, and the reduction does not exceed
what an APR-matched placebo produces.** EXP11's earlier claim of a $9.8M effect was an artefact of an
assumed-LGD profit formula and a swap-set comparison. On observed cashflows the profit difference is a
fraction of a percentage point of return on capital. 200-draw placebo controls (randomly excluding the
same share of loans, or an APR-decile matched share, and refilling to budget) are in the EXP11 row of
the table and in `results/reports/exp11_swaps.md`.

**RL does not earn its place here, and that is the finding.** Learning policies (LinUCB, Thompson sampling, Q-learning) are trained on the validation pool and their parameters are **frozen** during every test-pool episode; an earlier version kept updating them on realised test outcomes after every decision, which no deployed policy can do (loan outcomes take up to 36 months to arrive).

The experiment carries a control. With unconstrained capital the myopic expected-profit rule should be near-optimal, yet unconstrained Thompson sampling earns **63,814** vs the myopic rule's **32,707** (paired difference +31,107, 95% CI [+5,461, +56,958]). The control is breached. This experiment cannot say why, but the real-book audit below shows the myopic rule's inputs (PD and cashflow curves) mis-predict profit, and the learners are fitted on validation-window outcomes instead.

Paired difference vs the myopic expected-profit policy, by capital condition (mean over 30 episodes, 95% bootstrap CI):

| Condition | Learner | Difference | 95% CI | Significant |
|---|---|---|---|---|
| unconstrained | linucb@alpha=1 | +39,697 | [+13,802, +65,501] | yes |
| unconstrained | thompson | +31,107 | [+5,461, +56,958] | yes |
| unconstrained | budget_aware_q(greedy) | -26,476 | [-50,945, -2,355] | yes |
| budget=0.60 | linucb@alpha=1 | -1,379 | [-30,139, +27,086] | no |
| budget=0.60 | budget_aware_q(greedy) | -19,172 | [-44,115, +5,601] | no |
| budget=0.60 | thompson | -40,215 | [-74,306, -6,107] | yes |
| budget=0.35 | linucb@alpha=1 | -3,537 | [-20,689, +14,115] | no |
| budget=0.35 | thompson | -6,453 | [-28,219, +15,903] | no |
| budget=0.35 | budget_aware_q(greedy) | -22,308 | [-50,454, +5,496] | no |
| budget=0.20 | thompson | +3,482 | [-10,128, +17,280] | no |
| budget=0.20 | linucb@alpha=1 | -3,938 | [-17,246, +9,299] | no |
| budget=0.20 | budget_aware_q(greedy) | -6,876 | [-19,427, +5,400] | no |

**Why the expected-profit rule underperforms.** Measured on the real book at the exposure that was actually funded, so no simulator is involved:

- On the 2015-07..12 holdout, accepting every loan with predicted profit above zero earns **-36.9** per loan vs the 15% PD threshold (95% CI [-47.1, -26.9], -5.99M in total). On the validation pool the two rules are indistinguishable (-1.5 per loan, CI [-14.8, +11.8]).
- Predicted profit on the holdout exceeds realised profit by **31.0M** (19% of the prediction). By source: default count (PD too low) 55%, good-loan yield curve 22%, loss-severity curve 23%. Refitting the two cashflow curves on the holdout itself (an oracle diagnostic, not deployable) removes the yield and severity errors (+0.4M, -0.9M) and leaves the PD effect (-18.5M).
- Cashflow drift is therefore a real but secondary cause. Refitting the curves on the validation window narrows the shortfall to -28.2 per loan (CI [-38.2, -18.6]), still significantly negative.
- The PD model excludes `int_rate` by design, while the rule's revenue term rises with APR. Observed/expected defaults on the holdout run from **0.67** in the lowest APR quintile to **1.38** in the highest, so risk is understated exactly where predicted profit is largest. Predicted profit is over-dispersed: the top predicted-profit decile is predicted at +1,426 per loan and realises +132.
- Simulator caveat: 0% of the expected-profit policy's simulated approvals are at the observed exposure (mean multiplier 1.47x) vs 100% for the threshold, so the simulated policy comparison mixes realised and modelled returns.

2015-07..12 holdout, deciles of predicted profit (per loan, dollars):

| Decile | Mean PD | Observed default | Predicted profit | Realised profit |
|---|---|---|---|---|
| 1 | 0.221 | 0.172 | -706 | -386 |
| 2 | 0.168 | 0.142 | -312 | -180 |
| 3 | 0.141 | 0.126 | -186 | -124 |
| 4 | 0.123 | 0.116 | -109 | -101 |
| 5 | 0.120 | 0.129 | -42 | -110 |
| 6 | 0.120 | 0.136 | +36 | -67 |
| 7 | 0.119 | 0.145 | +136 | -34 |
| 8 | 0.117 | 0.154 | +282 | +9 |
| 9 | 0.116 | 0.169 | +541 | +21 |
| 10 | 0.114 | 0.182 | +1,426 | +132 |

---

## Architecture

- **Layer A — Risk.** Logistic baseline vs LightGBM, then Platt/isotonic calibration. A hard
  deny-list of 49 post-origination columns, enforced on every feature build.
- **Layer B — Graph.** A time-respecting cohort graph (`zip3`, normalised `emp_title`) and a
  GraphSAGE written directly in PyTorch — no `torch-geometric`, so message passing is explainable
  line by line. Neighbour *features* and neighbour *labels* are gated separately and strictly.
- **Layer C — Decision.** Economics estimated from observed cashflows (empirical loss rate 0.518;
  accounting strictly counts post-charge-off recoveries once), a policy ladder from fixed threshold to
  budget-aware Q-learning, and a simulator whose conditional coupling reproduces real outcomes
  exactly at the funded exposure.
- **Layer D — Trust.** Leakage tests that fail the build, SHAP reason codes, PSI/calibration drift on
  unlabelled production traffic, and a fairness slice over proxy groups.
- **Blind-spot layer.** Autoencoder anomaly, VGAE relational uncertainty and model disagreement,
  each assessed *conditional on the PD level* rather than marginally.

---

## Quickstart

**Environment notes** (these cost real time to rediscover):

- **Smart App Control** on Windows blocks unsigned `.pyd` files, so PyPI wheels for
  scipy/scikit-learn/LightGBM fail to import. Use a **conda** environment; Anaconda's packages are
  signed.
- **Windows long paths** are disabled by default, so the conda `pytorch` package cannot extract.
  Install torch from **pip**.
- The resulting MKL/torch OpenMP clash is handled by `KMP_DUPLICATE_LIB_OK` in
  `src/credit_risk/__init__.py`. Do not remove it.

```bash
conda create -n credit-risk python=3.13 numpy pandas scipy scikit-learn pyarrow lightgbm shap matplotlib pyyaml pytest numba
conda activate credit-risk
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -e . --no-deps
```

Data (needs a Kaggle token at `~/.kaggle/kaggle.json`):

```bash
python scripts/download_data.py --config configs/data/lending_club.yaml
```

No token? Every stage runs on a schema-faithful synthetic generator, and every artefact built from it
is stamped `data_source: synthetic` and refused as a finding:

```bash
python scripts/download_data.py --synthetic --n-loans 200000
```

Full pipeline. `--max-threads` keeps a long run from monopolising the machine:

```bash
python scripts/prepare_data.py          --config configs/data/lending_club.yaml --max-threads 10
python scripts/build_graph.py           --config configs/graph/cohort_graph.yaml
python scripts/train_baseline.py        --config configs/models/boosting.yaml
python scripts/train_gnn.py             --config configs/graph/cohort_graph.yaml
python scripts/analyze_blind_spots.py   --config configs/experiments/exp08_blindspots.yaml
python scripts/train_vgae.py            --config configs/experiments/exp09_vgae.yaml
python scripts/analyze_disagreement.py  --config configs/experiments/exp10_disagreement.yaml
python scripts/analyze_decision_swaps.py --config configs/experiments/exp11_swaps.yaml
python scripts/train_policy.py          --config configs/experiments/exp06_rl.yaml
python scripts/evaluate.py              --config configs/models/boosting.yaml
```

Tests:

```bash
python -m pytest -q
```

---

## What this project does not claim

- **Not deployable.** Every policy result comes from a simulator, on an *already-approved*
  population, from one institution and one credit cycle in a long expansion. Nothing here has been
  tested through a downturn.
- **No causal claims.** SHAP explains the model, not the borrower and not the world.
- **No reject inference.** The model estimates `P(default | approved)`, not `P(default | applied)`.
  The gap is quantified against the rejected-applications file but not corrected — doing it properly
  needs propensity data LendingClub never published.
- **No real-world impact claim.** No money was lent.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) | The contract the code implements. **Read first** |
| [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) | Bugs, mathematical errors and the result that reversed |
| [`docs/DATASET.md`](docs/DATASET.md) | Dataset evaluation, licensing, and why Home Credit was rejected |
| [`docs/LEAKAGE.md`](docs/LEAKAGE.md) | Every leakage rule and the test that enforces it |
| [`docs/GRAPH_DESIGN.md`](docs/GRAPH_DESIGN.md) | Time-respecting graph, the two information sets |
| [`docs/RL_FORMULATION.md`](docs/RL_FORMULATION.md) | Why capital constraints make lending sequential — and when they do not |
| [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) | Observed vs estimated vs assumed, itemised |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Column-by-column decision-time rulings |
