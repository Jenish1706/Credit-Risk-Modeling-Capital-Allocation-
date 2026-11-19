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
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | Simulated diff vs 15% threshold: **-28,262**/episode (95% CI [-54,837, -2,616]) `SIMULATED` on 2015-07..12. Real book at observed exposure: **-36.9**/loan (95% CI [-47.1, -26.9]) `OBSERVED` on 2015-07..12 | **Expected-profit rule loses to a PD threshold on the real book.** Simulation CI excludes zero. Mechanism is measured below, not assumed |
| **EXP04** | Do graph cohort features add signal? | AUC **-0.0009**. Mean over 5 LightGBM seeds on 2015-07..12: ECE **0.0110 ± 0.0001 → 0.0048 ± 0.0005**, O/E **1.081 → 0.998** (EXP04/05 single-fit run: ECE 0.0109 → 0.0043, O/E 1.080 → 0.989) `OBSERVED` | No discrimination gain; **calibration gain survives both controls in every one of 5 seeds** (ΔECE B−tabular -0.0062 ± 0.0005, B−macro -0.0066 ± 0.0002, B−shuffled -0.0038 ± 0.0006; mean ± sd over seeds, per-seed paired bootstrap, 2015-07..12) |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6637** vs 0.6704 (aggregates) vs 0.6712 (tabular) on 2015-07..12 `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | Frozen-at-test learners, accept/reject at observed exposure (2015-07..12). Unconstrained: best learner thompson +17,398 vs myopic expected-profit (95% CI [-2,574, +37,136]). 60% budget: thompson -160 vs the 15% threshold (95% CI [-15,333, +14,180]) `SIMULATED` | Control holds: no learner significantly beats the myopic rule with unconstrained capital; **no learner significantly beats the fixed 15% threshold in any condition**; significantly worse than the threshold: budget_aware_q(greedy) (unconstrained, budget=0.60); linucb@alpha=1 (budget=0.35, budget=0.20); thompson (budget=0.20) |
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
bootstrap intervals in the EXP04 verdict). Every arm is fitted for five LightGBM seeds; the result
is reported as mean ± sd over seeds, and the claim is kept only if it holds in each seed. The
shuffled-key control also lowers ECE somewhat, so part of the gain is not specific to true cohort
membership.

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

**RL does not earn its place here, and that is the finding.** Learning policies (LinUCB, Thompson sampling, Q-learning) choose between accept and reject at the exposure that was actually funded, so every simulated outcome is a real one. They are trained on the validation pool and their parameters are **frozen** during every test-pool episode; an earlier version kept updating them on realised test outcomes after every decision, which no deployed policy can do (loan outcomes take up to 36 months to arrive).

The experiment carries a control. With unconstrained capital the myopic expected-profit rule should be near-optimal. The best learner (thompson) earns 50,883 vs the myopic rule's 33,486 (paired difference +17,398, 95% CI [-2,574, +37,136]): **the control holds**. The plain 15% PD threshold beats the myopic rule by +28,262 (95% CI [+2,616, +54,837]), consistent with the real-book audit below.

Policy-level results by capital condition (mean over 30 episodes, 95% bootstrap CI on the paired difference), against the myopic expected-profit policy and the fixed 15% PD threshold:

| Condition | Policy | Mean profit | Approval rate | Capital deployed | vs expected-profit [95% CI] | vs 15% threshold [95% CI] |
|---|---|---|---|---|---|---|
| unconstrained | fixed_threshold@0.150 | +61,748 | 63.4% | 10.22M | +28,262* [+2,616, +54,837] | baseline |
| unconstrained | thompson | +50,883 | 39.0% | 9.41M | +17,398 [-2,574, +37,136] | -10,865 [-30,148, +7,562] |
| unconstrained | linucb@alpha=1 | +44,347 | 47.0% | 10.56M | +10,861 [-7,087, +29,571] | -17,401 [-37,640, +1,251] |
| unconstrained | expected_profit | +33,486 | 49.4% | 8.86M | baseline | -28,262* [-54,837, -2,616] |
| unconstrained | budget_aware_q(greedy) | +268 | 41.7% | 7.36M | -33,218* [-47,280, -18,552] | -61,480* [-91,513, -32,778] |
| unconstrained | risk_band | -46,114 | 95.3% | 14.75M | -79,600* [-98,773, -60,798] | -107,862* [-133,989, -82,728] |
| unconstrained | approve_all | -95,984 | 100.0% | 15.41M | -129,470* [-150,815, -109,765] | -157,733* [-190,223, -126,730] |
| budget=0.60 | fixed_threshold@0.150 | +60,208 | 57.7% | 9.32M (100% of budget) | +26,396* [+654, +52,743] | baseline |
| budget=0.60 | thompson | +60,048 | 39.8% | 9.30M (100% of budget) | +26,236* [+7,044, +44,592] | -160 [-15,333, +14,180] |
| budget=0.60 | linucb@alpha=1 | +58,557 | 39.6% | 9.30M (100% of budget) | +24,745* [+5,999, +42,610] | -1,651 [-16,958, +12,362] |
| budget=0.60 | expected_profit(lambda=0.0053) | +38,649 | 43.4% | 7.85M (84% of budget) | +4,836 [-3,352, +13,424] | -21,559 [-48,018, +4,330] |
| budget=0.60 | expected_profit | +33,812 | 49.3% | 8.84M (95% of budget) | baseline | -26,396* [-52,743, -654] |
| budget=0.60 | budget_aware_q(greedy) | +15,182 | 32.2% | 5.87M (63% of budget) | -18,630* [-36,370, -1,326] | -45,026* [-76,395, -14,339] |
| budget=0.60 | risk_band | +4,568 | 60.2% | 9.32M (100% of budget) | -29,244 [-58,062, +1,296] | -55,640* [-83,601, -28,556] |
| budget=0.60 | approve_all | -31,463 | 60.5% | 9.32M (100% of budget) | -65,276* [-93,044, -36,445] | -91,671* [-120,701, -64,641] |
| budget=0.35 | expected_profit | +51,909 | 30.4% | 5.44M (100% of budget) | baseline | +6,008 [-14,256, +25,782] |
| budget=0.35 | thompson | +51,731 | 24.1% | 5.43M (100% of budget) | -178 [-16,318, +16,402] | +5,830 [-9,719, +20,364] |
| budget=0.35 | fixed_threshold@0.150 | +45,901 | 33.8% | 5.44M (100% of budget) | -6,008 [-25,782, +14,256] | baseline |
| budget=0.35 | expected_profit(lambda_eval_pool=0.0168) | +34,191 | 29.4% | 5.42M (100% of budget) | -17,718 [-42,626, +7,416] | -11,710 [-46,201, +21,784] |
| budget=0.35 | budget_aware_q(greedy) | +20,162 | 29.0% | 5.32M (98% of budget) | -31,747* [-60,165, -5,221] | -25,739 [-60,443, +7,692] |
| budget=0.35 | expected_profit(lambda=0.0291) | +2,041 | 22.0% | 4.17M (77% of budget) | -49,868* [-73,894, -26,243] | -43,860* [-74,604, -15,568] |
| budget=0.35 | linucb@alpha=1 | -202 | 0.1% | 0.01M (0% of budget) | -52,111* [-76,843, -26,224] | -46,103* [-67,057, -26,201] |
| budget=0.35 | risk_band | -4,306 | 34.9% | 5.44M (100% of budget) | -56,215* [-72,648, -38,539] | -50,207* [-66,652, -33,746] |
| budget=0.35 | approve_all | -29,379 | 35.1% | 5.44M (100% of budget) | -81,288* [-99,393, -61,909] | -75,279* [-95,310, -55,511] |
| budget=0.20 | fixed_threshold@0.150 | +23,140 | 19.1% | 3.11M (100% of budget) | +994 [-14,272, +15,672] | baseline |
| budget=0.20 | budget_aware_q(greedy) | +22,990 | 17.1% | 3.11M (100% of budget) | +844 [-19,385, +18,911] | -150 [-28,657, +26,114] |
| budget=0.20 | expected_profit | +22,146 | 17.2% | 3.11M (100% of budget) | baseline | -994 [-15,672, +14,272] |
| budget=0.20 | linucb@alpha=1 | +8,690 | 14.5% | 3.10M (100% of budget) | -13,456* [-26,812, -291] | -14,450* [-25,601, -3,329] |
| budget=0.20 | expected_profit(lambda_eval_pool=0.0394) | +5,515 | 15.4% | 2.99M (96% of budget) | -16,631 [-41,679, +9,120] | -17,625 [-46,585, +11,205] |
| budget=0.20 | expected_profit(lambda=0.0492) | +3,723 | 11.0% | 2.18M (70% of budget) | -18,423 [-43,120, +5,576] | -19,418 [-49,386, +8,968] |
| budget=0.20 | thompson | -38 | 0.0% | 0.00M (0% of budget) | -22,184* [-38,366, -5,401] | -23,179* [-38,016, -9,083] |
| budget=0.20 | risk_band | -8,350 | 20.0% | 3.11M (100% of budget) | -30,496* [-43,689, -17,626] | -31,490* [-46,203, -17,053] |
| budget=0.20 | approve_all | -15,482 | 20.0% | 3.11M (100% of budget) | -37,628* [-51,460, -23,857] | -38,622* [-54,976, -22,969] |

\* = 95% CI excludes zero.

The knapsack shadow-price policy (lambda fitted on the validation pool) is the analytical solution in the formulation, and here it does not help: budget=0.60: +4,836 vs expected-profit [-3,352, +13,424], -21,559 vs the 15% threshold [-48,018, +4,330], deploying 84% of the budget; budget=0.35: -49,868 vs expected-profit [-73,894, -26,243], -43,860 vs the 15% threshold [-74,604, -15,568], deploying 77% of the budget; budget=0.20: -18,423 vs expected-profit [-43,120, +5,576], -19,418 vs the 15% threshold [-49,386, +8,968], deploying 70% of the budget.

`expected_profit(lambda_eval_pool=...)` fits lambda on the evaluation pool itself. It is a diagnostic, not an upper bound: it scores below the myopic rule at budget=0.35, budget=0.20 and below the 15% threshold at budget=0.35, budget=0.20. Its lambda is applied to predicted profit that is over-dispersed relative to realised profit (profit audit below), so the knapsack optimum is for the wrong objective.

Collapsed learners: linucb@alpha=1 at budget=0.35 (approval rate 0.10%, 0.010M deployed); thompson at budget=0.20 (approval rate 0.04%, 0.000M deployed). These stopped lending (approval rate below 1%); "significantly worse" understates it.

**Sensitivity: exposure-changing actions.** Adding 0.5x and 1.5x exposure actions makes off-diagonal outcomes model counterfactuals rather than measurements, so these are not headline numbers. Expected-profit vs the 15% threshold in that space: -29,042 per episode (95% CI [-62,824, +4,620]). Learners vs the myopic expected-profit policy:

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
- Simulator caveat: in the exposure-changing sensitivity space 0% of the expected-profit policy's approvals are at the observed exposure (mean multiplier 1.47x) vs 100% for the threshold, so that comparison mixes realised and modelled returns. The headline EXP03/EXP06 tables use accept/reject at the observed exposure, where every outcome is real.

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

**Ablation: a PD that sees LendingClub's own pricing.** The production PD excludes `int_rate`, `grade` and `sub_grade` by design, because they are LendingClub's own risk score. This ablation refits the PD with them (calibrator chosen on 2015-01..06; every number on 2015-07..12). It is not a replacement: a PD that consumes the incumbent's pricing does not underwrite from primary evidence.

- ROC-AUC 0.6732 → 0.6934; ECE 0.0117 → 0.0178; O/E 1.082 → 1.134.
- Observed/expected defaults from the lowest to the highest APR quintile: production PD 0.67 → 1.38; PD with LendingClub's pricing 0.97 → 1.18.
- With that PD, accepting every loan with predicted profit above zero earns -23.6 per loan vs the 15% threshold under the same PD (95% CI [-33.9, -13.4]), so the rule still loses to the 15% PD threshold under that PD. Against the production-PD threshold the figure is -25.8 [-35.8, -16.2] (production-PD expected-profit rule: -36.9).

---

## Architecture

- **Layer A — Risk.** Logistic baseline vs LightGBM, then calibration (Platt, isotonic and an intercept-only baseline are compared; the production calibrator is chosen on 2015-H1). A hard
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

- **Not deployable.** The EXP06 policy results come from a simulator and the real-book audits (EXP03
  real-book column, EXP11) are retrospective reallocations measured on observed cashflows. Both use an
  *already-approved* population from one institution and one credit cycle in a long expansion. Nothing
  here has been tested through a downturn.
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
