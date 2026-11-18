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
| monitor (production traffic) | 2016-01 … 2018-12 | 988,585 | unlabelled by design |

The rising default rate across windows is real vintage deterioration, and it is the reason the
out-of-time split matters.

---

## Results

All figures are on the **out-of-time 2015 test book (283,026 loans)** and were produced by the
current code. `OBSERVED` = measured from data; `SIMULATED` = produced inside the lending simulator
under the assumptions in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

| # | Question | Result | Verdict |
|---|---|---|---|
| **EXP01** | Does non-linear modelling beat logistic regression? | AUC **0.6581 → 0.6705** (+0.0124) `OBSERVED` | Material. Boosting justified |
| **EXP02** | Does calibration improve PD reliability? | AUC **unchanged to 5 d.p.**; ECE **0.0193 → 0.0111**; calibration slope 0.991 → 1.005 `OBSERVED` | The point of the project in one row: ranking metrics are blind to calibration |
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | Simulated diff vs 15% threshold: **-26,405**/episode (95% CI [-59,902, +7,655]) `SIMULATED`. Real book at observed exposure: **-33.2**/loan (95% CI [-40.9, -25.5]) `OBSERVED` | **Expected-profit rule loses to a PD threshold on the real book.** Simulation CI straddles zero (null). Mechanism is measured below, not assumed |
| **EXP04** | Do graph cohort features add signal? | AUC **-0.0010**. ECE **0.0105 → 0.0032**, P/O **0.930 → 0.999** `OBSERVED` | No discrimination gain; **calibration gain confirmed by controls** (ΔECE −0.0071 vs macro, −0.0044 vs shuffled) |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6616** vs 0.6679 (aggregates) vs 0.6690 (tabular) `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | Frozen-at-test learners vs myopic expected-profit: unconstrained thompson **+55,753** (95% CI [+29,266, +80,634]); at 60% budget best learner linucb@alpha=1 **+6,916** (95% CI [-17,235, +30,160]) `SIMULATED` | **Control breached:** learners significantly beat the myopic rule with unconstrained capital, where it should be near-optimal (see the real-book audit below); **no learner significantly beats expected-profit in any capital-constrained condition**; significantly worse at budget=0.20 |
| **EXP07** | How does the model degrade under vintage drift? | O/E **0.860 (2010) → 1.077 (2015)** while AUC moves 0.7475 → 0.6705 `OBSERVED` | Calibration drifts before ranking does |
| **EXP08** | Does autoencoder anomaly flag untrustworthy PDs? | Survives PD control in **10/10** deciles, sign-test **p=0.0010**, odds ratio 1.032/SD `OBSERVED` | **KEEP** |
| **EXP09** | Does VGAE relational uncertainty flag them? | Fails PD control (**5/10** deciles, sign-test **p=0.6230**), odds ratio 1.002/SD `OBSERVED` | **REJECT** |
| **EXP10** | Does model disagreement flag them? | Survives in **9/10** deciles, **p=0.0107**, odds ratio 1.043/SD `OBSERVED` | **KEEP** |
| **EXP11** | Does acting on those signals improve the book? | Default rate **16.88% → 15.98%**, but placebo controls show **15.94%–16.16%** `OBSERVED` | **Null result.** Risk reduction is driven by refilling budget with safer loans, not blind-spot selectivity |

### Four results worth reading twice

**Calibration is where the graph pays.** EXP04/EXP05 show the graph adds nothing to *discrimination*
— GraphSAGE is the worst of the three arms. But with matched model configurations the cohort features
cut expected calibration error by two-thirds and move observed/expected from 0.930 to 0.999. Tested
against two non-graph controls (a time-respecting portfolio macro default rate, and cohort keys shuffled
within issue month), the calibration improvement decisively survives with paired bootstrap intervals
(ΔECE −0.0071 [−0.0079, −0.0046] vs macro, and −0.0044 [−0.0053, −0.0022] vs shuffled keys; both p < 0.0001).
The gain is genuinely relational, capturing local geographic and occupational cohort shocks rather than aggregate macro drift.

**The blind-spot signals are real but small.** Both survive a conditional test against the PD level
— the test the first version of these experiments did not run — but the odds ratios are 1.03–1.04
per standard deviation. They flag *miscalibration*, not bad borrowers.

**Acting on them buys risk reduction, not profit — and the reduction is a budget-refill effect.**
EXP11's earlier claim of a $9.8M effect was an artefact of an assumed-LGD profit formula and a swap-set
comparison. On observed cashflows, the profit difference is a fraction of a percentage point. Furthermore,
200-draw placebo controls (randomly excluding 19% or APR-decile matched 19% and refilling to budget)
match the 15.98% default rate (random CI [15.94%, 16.09%], APR-matched CI [15.98%, 16.16%]). The risk
drop is an accounting consequence of deploying unspent budget into lower-risk loans, not representation-learning
selectivity.

**RL does not earn its place here, and that is the finding.** Learning policies (LinUCB, Thompson sampling, Q-learning) are trained on the validation pool and their parameters are **frozen** during every test-pool episode; an earlier version kept updating them on realised test outcomes after every decision, which no deployed policy can do (loan outcomes take up to 36 months to arrive).

The experiment carries a control. With unconstrained capital the myopic expected-profit rule should be near-optimal, yet unconstrained Thompson sampling earns **92,696** vs the myopic rule's **36,943** (paired difference +55,753, 95% CI [+29,266, +80,634]). The control is breached. This experiment cannot say why, but the real-book audit below shows the myopic rule's inputs (PD and cashflow curves) mis-predict profit, and the learners are fitted on validation-window outcomes instead.

Paired difference vs the myopic expected-profit policy, by capital condition (mean over 30 episodes, 95% bootstrap CI):

| Condition | Learner | Difference | 95% CI | Significant |
|---|---|---|---|---|
| unconstrained | thompson | +55,753 | [+29,266, +80,634] | yes |
| unconstrained | linucb@alpha=1 | +54,951 | [+29,273, +79,091] | yes |
| unconstrained | budget_aware_q(greedy) | -6,499 | [-27,055, +14,423] | no |
| budget=0.60 | linucb@alpha=1 | +6,916 | [-17,235, +30,160] | no |
| budget=0.60 | thompson | -10,209 | [-36,599, +15,626] | no |
| budget=0.60 | budget_aware_q(greedy) | -24,354 | [-53,771, +4,544] | no |
| budget=0.35 | budget_aware_q(greedy) | -8,541 | [-30,982, +13,788] | no |
| budget=0.35 | linucb@alpha=1 | -16,716 | [-41,512, +8,368] | no |
| budget=0.35 | thompson | -21,443 | [-47,513, +5,321] | no |
| budget=0.20 | linucb@alpha=1 | -13,343 | [-26,403, -490] | yes |
| budget=0.20 | budget_aware_q(greedy) | -16,708 | [-32,578, -1,264] | yes |
| budget=0.20 | thompson | -19,706 | [-33,928, -5,896] | yes |

**Why the expected-profit rule underperforms.** Measured on the real book at the exposure that was actually funded, so no simulator is involved:

- On the 2015 test book, accepting every loan with predicted profit above zero earns **-33.2** per loan vs the 15% PD threshold (95% CI [-40.9, -25.5], -9.39M in total). On the validation pool the two rules are indistinguishable (-0.4 per loan, CI [-13.7, +13.0]).
- Predicted profit on the test book exceeds realised profit by **50.3M** (18% of the prediction). By source: default count (PD too low) 57%, good-loan yield curve 23%, loss-severity curve 20%. Refitting the two cashflow curves on the test book itself (an oracle diagnostic, not deployable) removes the yield and severity errors (+1.4M, -1.4M) and leaves the PD effect (-31.0M).
- Cashflow drift is therefore a real but secondary cause. Refitting the curves on the validation window narrows the shortfall to -26.0 per loan (CI [-33.4, -18.6]), still significantly negative.
- The PD model excludes `int_rate` by design, while the rule's revenue term rises with APR. Observed/expected defaults on the test book run from **0.67** in the lowest APR quintile to **1.36** in the highest, so risk is understated exactly where predicted profit is largest. Predicted profit is over-dispersed: the top predicted-profit decile is predicted at +1,460 per loan and realises +192.
- Simulator caveat: 0% of the expected-profit policy's simulated approvals are at the observed exposure (mean multiplier 1.48x) vs 100% for the threshold, so the simulated policy comparison mixes realised and modelled returns.

Test book, deciles of predicted profit (per loan, dollars):

| Decile | Mean PD | Observed default | Predicted profit | Realised profit |
|---|---|---|---|---|
| 1 | 0.221 | 0.173 | -698 | -349 |
| 2 | 0.169 | 0.142 | -308 | -172 |
| 3 | 0.143 | 0.125 | -183 | -104 |
| 4 | 0.126 | 0.121 | -104 | -104 |
| 5 | 0.123 | 0.130 | -35 | -88 |
| 6 | 0.122 | 0.138 | +44 | -45 |
| 7 | 0.121 | 0.146 | +147 | -3 |
| 8 | 0.120 | 0.156 | +296 | +27 |
| 9 | 0.120 | 0.172 | +558 | +49 |
| 10 | 0.118 | 0.186 | +1,460 | +192 |

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

Tests (161, all passing):

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
