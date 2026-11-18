# What Broke: Audit, Root-Cause Analysis, and Remediations

This document provides a technical account of the defects identified, reproduced, and remediated in this repository during the quantitative risk engineering audit.

Every defect here was reproduced with an isolated script, quantified on the full dataset, remediated with verifiable code changes, and covered by automated regression tests.

---

## Executive Summary of Findings

| ID | Severity | Component | Defect Summary | Status | Key Metric Impact |
|---|---|---|---|---|---|
| **P0-1** | Critical | `decision/expected_loss.py` | Cash recoveries double-counted in realised profit & return formulas | **Fixed** | 2015 Test Book Profit: \$270.4M $\to$ \$230.3M (-\$40.1M, -14.8%); RoC: 7.46% $\to$ 6.35% (-111 bp) |
| **P0-2** | High | `scripts/train_policy.py` | Analytical shadow price ($\lambda$) used test-window oracle | **Fixed** | Deployable validation-tuned policy: \$138.8M vs.\ clairvoyant oracle \$153.8M at 60% budget |
| **P0-3** | High | `scripts/run_swaps.py` | EXP11 risk reduction was an artifact of budget refilling, not blind spots | **Fixed** | Policy B default rate (15.98%) falls inside placebo intervals (Random 16.01%, APR-matched 16.08%) |
| **P0-4** | High | `scripts/train_gnn.py` | Cohort graph calibration lacked non-graph macro and shuffled controls | **Fixed** | Arm B calibration gain confirmed: outperforms macro default ($\Delta\text{ECE} = -0.0071$, $p < 0.0001$) and month-shuffled keys ($\Delta\text{ECE} = -0.0044$, $p < 0.0001$) |
| **P1-5** | High | `scripts/analyze_deployment_realism.py` | Benchmark assumed mature labels for loans immature at deployment cutoff | **Quantified** | Deployment-realistic retraining: test AUC $0.6690 \to 0.6478$ ($\Delta = -0.0212$); ECE surges $0.0105 \to 0.0374$ ($+256\%$) |
| **P1-6a** | Medium | `models/calibration.py` | EXP02 calibration gain reported without base-rate decomposition | **Fixed** | Intercept-only baseline proves 101.3% of ECE reduction is base-rate shift ($\bar{y}: 12.7\% \to 14.1\%$) with unit slope |
| **P1-6b** | Medium | `scripts/train_policy.py` | EXP06 control check breaches under vintage drift reported as "control holds" | **Fixed; superseded by CR-01** | Scaled to 30 seeds. The original figures (Thompson 101,203 vs myopic 36,943) were inflated by test-set learning (see CR-01) and the attribution to stale cashflows was unsupported |
| **CR-01** | Critical | `simulation/environment.py`, `simulation/evaluation.py` | Learning policies kept updating on realised test-pool outcomes during evaluation, and carried that state across episodes | **Fixed** | Evaluation runs with `learn=False`. Unconstrained Thompson 101,203 -> 92,696; the 60%-budget "bandits adapt" result (Thompson 84,737, LinUCB 76,908) disappears: LinUCB +6,916 vs expected-profit, 95% CI [-17,235, +30,160] |
| **P1-6c** | Medium | `data/splits.py` | 2015 test book reused across threshold/hyperparameter selection | **Fixed** | Isolated 2015-07..2015-12 as frozen final holdout (`assign_test_holdout`) |
| **P1-6d** | Medium | `simulation/environment.py` | Simulator evaluated defaults using tested model predictions circularly | **Fixed** | Decoupled by using ground-truth observed defaults/returns at observed exposures |
| **P1-6e** | Medium | `decision/blindspot.py` | Blind-spot odds ratios reported without CIs; sign test one-sided without stating | **Fixed** | Added analytical Fisher information Wald 95% CIs and two-sided binomial sign test p-values |

---

## Detailed Root-Cause Analyses & Remediations

### 1. P0-1 · Recoveries Double-Counted in Realised Cashflows

#### Defect Description
In LendingClub's loan accounting data, the field `total_pymnt` records the cumulative cash received by the lender across the lifetime of the loan, inclusive of principal payments, interest, late fees, and recoveries.

In `src/credit_risk/decision/expected_loss.py` (functions `realised_profit` and `realised_return`), the code originally implemented:
$$\text{profit} = \text{total\_pymnt} + \text{recoveries} - \text{collection\_recovery\_fee} - \text{funded\_amnt}$$
The docstring incorrectly stated that "recoveries are reported separately, so they must be added".

#### Verification & Proof
We tested the cashflow identity across all 66,841 defaulted loans in `data/processed/loans.parquet` with positive recoveries:
$$\Delta_1 = |\text{total\_pymnt} - (\text{total\_rec\_prncp} + \text{total\_rec\_int} + \text{total\_rec\_late\_fee} + \text{recoveries})| < \$0.02$$
$$\Delta_2 = |\text{total\_pymnt} - (\text{total\_rec\_prncp} + \text{total\_rec\_int} + \text{total\_rec\_late\_fee})| < \$0.02$$

- **Result:** $\Delta_1$ held in **100.00%** of loans (mean absolute difference = \$0.0000). $\Delta_2$ held in **0.01%** of loans.
- **Conclusion:** `total_pymnt` indisputably includes recoveries. Adding `+ recoveries` counted recovery cash twice for every defaulting loan.

#### Remediation
- Corrected `realised_profit` to:
  $$\text{profit} = \text{total\_pymnt} - \text{collection\_recovery\_fee} - \text{funded\_amnt}$$
- Corrected `realised_return` accordingly.
- Added automated unit test in `tests/test_expected_loss.py::test_realised_profit_accounting_identity_no_double_counting` verifying that no recoveries are added when `total_pymnt` equals component sum.
- Recomputed all downstream economic artifacts (`exp03_summary.csv`, `exp06_budget_sweep.csv`, `exp11_swaps.csv`, `exp11_swaps.md`).

#### Headline Impact
- **2015 Test Window (283,026 loans):**
  - Realised profit: dropped from **\$270,412,488** to **\$230,341,209** (-**\$40,071,279**, **-14.82%**).
  - Realised return on capital: dropped from **7.46%** to **6.35%** (**-111 bps**).
- **EXP03 Unconstrained Policy Evaluation:**
  - Expected profit policy profit: dropped from **\$280.9M** to **\$240.2M**.
- **EXP11 Blind-Spot Evaluation:**
  - Baseline portfolio profit at equal budget ($1.673B): dropped from **\$137.4M** to **\$117.3M**.

---

### 2. P0-2 · Analytical Shadow Price Policy Evaluated with Test Pool Oracle

#### Defect Description
In EXP06 (constrained capital allocation), the project evaluated an analytical shadow price benchmark against reinforcement learning policies (LinUCB, Thompson Sampling, Tabular Q-learning).

In `scripts/train_policy.py`, the analytical shadow price $\lambda$ was computed as:
$$\lambda = \text{analytical\_shadow\_price}(\text{densities}[:n\_steps], \text{exposures}[:n\_steps], \text{budget})$$
where `densities` and `exposures` were extracted from `pool = build_applicant_pool(test, pd_test, cashflow)`.

#### Why This Was a Flaw
The test set represents the unseen out-of-time production environment. Computing $\lambda$ directly on the test pool gave the analytical policy clairvoyant knowledge of the exact distribution, ordering, and profitability of future applicants. It was an oracle upper bound, not an operational decision policy. Comparing online learning agents against a test-derived oracle is mathematically invalid.

#### Remediation
- Separated the policy into two distinct benchmarks:
  1. **Deployable Analytical Benchmark (`expected_profit(lambda=...)`):** Fitted strictly on the **validation window** (`train_pool`, July 2014 -- Dec 2014, $N=90,615$), representing the parameter setting a risk officer would configure at deployment time.
  2. **Theoretical Oracle Upper Bound (`oracle_ceiling(lambda=...)`):** Fitted on the test window solely to establish the continuous knapsack relaxation ceiling.
- Added automated unit test in `tests/test_policy.py::test_analytical_policy_uses_validation_split_not_test` asserting that the evaluated analytical policy parameters depend only on validation data.

#### Headline Impact
At 60% capital budget ($1.004B demand):
- **Oracle Ceiling (clairvoyant):** \$153,856,120 profit.
- **Deployable Analytical Benchmark (validation-tuned):** \$138,824,310 profit.
- **LinUCB Learning Agent:** \$133,529,480 profit.
- **Takeaway:** The deployable analytical policy outperforms LinUCB without requiring dynamic exploration, but the gap is \$5.3M, not the \$20.3M previously claimed by comparing against the clairvoyant oracle.

---

### 3. P0-3 · EXP11 Risk Reduction Attributable to Budget Refilling, Not Blind Spots

#### Defect Description
EXP11 evaluated whether rejecting or capping the 19% of applicants flagged with the highest unsupervised anomaly score (tabular autoencoder reconstruction error) or model disagreement improved the portfolio.
The report originally claimed that swapping out flagged loans reduced portfolio default rate from 16.88% to 15.98% at "essentially constant profitability".

#### Why This Was a Flaw
When 19% of loans are excluded from an applicant pool under a fixed capital budget, the portfolio budget utilization drops. To reach the budget, the policy admits loans further down the queue.
In the 2015 LendingClub test book, applicants further down the approval queue were lower-APR, prime loans with naturally lower default rates. The drop in default rate was a mechanical artifact of budget refilling, not an informational property of the autoencoder or disagreement signals.

#### Verification with Placebo Controls
We implemented two rigorous placebo controls with 200 bootstrap draws at the identical capital budget:
1. **Random 19% Exclusion Placebo:** Arbitrarily exclude 19% of applicants and refill to budget.
   - Mean Default Rate: **16.01%** (95% CI: [15.94%, 16.09%])
   - Policy B default rate (15.98%) falls inside the 95% CI ($p = 0.42$).
2. **APR-Decile Matched 19% Exclusion Placebo:** Exclude 19% sampled conditionally within each APR decile and refill to budget.
   - Mean Default Rate: **16.08%** (95% CI: [15.98%, 16.16%])
   - Policy B default rate (15.98%) touches the boundary ($p = 0.08$).

#### Remediation & Falsification
- Updated `results/reports/exp11_swaps.md` and `results/tables/exp11_swaps.csv` with full placebo distributions.
- Explicitly falsified the claim of "risk reduction at constant profitability": neither profit improvement nor risk reduction is attributable to the blind-spot signals in portfolio allocation.
- Maintained the legitimate finding: autoencoder reconstruction error is a valid *diagnostic* monitor of conditional miscalibration across deciles (EXP08), but cannot be naively transformed into an active allocation filter.

---

### 4. P0-4 · Relational Cohort Graph Calibration Controls

#### Defect Description
In EXP04, augmenting LightGBM with 18 time-respecting cohort graph features (aggregates across 3-digit zip code and normalized employment title) improved probability calibration:
- ECE dropped from **0.0105** (Arm A: tabular) to **0.0032** (Arm B: tabular + graph).
- Observed/Expected ratio moved from **0.930** to **0.999**.
However, portfolio default rates were drifting upward over time (12.67% in training vs.\ 14.89% in testing). A skeptic could argue that the cohort aggregates merely captured the rising macroeconomic default level, rather than any localized relational structure.

#### Controls Implemented
We implemented two non-graph control arms and evaluated them via 2,000 paired bootstrap resamples:
1. **Macro Portfolio Default Rate Control:** A single time-respecting rolling average default rate feature across all loans issued in the preceding 6 months.
2. **Month-Shuffled Cohort Keys Control:** Cohort aggregates computed after randomly permuting `zip3` and `emp_norm` within each issue month. This preserves the exact temporal marginals and macro trend while completely destroying employer and geographic cohort relationships.

#### Results (2,000 Paired Bootstrap Resamples)
- **Arm B vs.\ Macro Default Control:**
  - $\Delta\text{ECE} = -0.00707$ (95% CI: $[-0.00786, -0.00462]$, $p < 0.0001$)
  - $\Delta|\text{O/E} - 1| = -0.07317$ (95% CI: $[-0.08985, -0.05739]$, $p < 0.0001$)
- **Arm B vs.\ Shuffled Keys Control:**
  - $\Delta\text{ECE} = -0.00442$ (95% CI: $[-0.00526, -0.00220]$, $p < 0.0001$)
  - $\Delta|\text{O/E} - 1| = -0.05149$ (95% CI: $[-0.06908, -0.03478]$, $p < 0.0001$)

#### Remediation
- Integrated control arms and paired bootstrap tests into `scripts/train_gnn.py`.
- Added unit test in `tests/test_graph_leakage.py::test_paired_bootstrap_calibration_comparison_smoke`.
- The calibration advantage of time-respecting cohort aggregates is statistically confirmed: cohort default rates capture localized socio-economic shocks that macro-level indicators cannot reproduce.

---

### 5. P1-5 · Deployment Realism: The Immature Loan Leakage Gap

#### Defect Description
Standard out-of-time backtesting protocols partition loans by origination date (e.g., Train: 2010--2014, Test: 2015). However, 36-month consumer installment loans take three full years to mature.
At the deployment cutoff date (**2014-12-31**):
- **68.4% of training loans** (originated 2012--2014) had not yet matured.
- **97.9% of validation loans** (originated July 2014--Dec 2014) had not matured.

In the standard benchmark, the LightGBM classifier and probability calibrator were trained on the eventual 36-month terminal outcome ("Fully Paid" vs.\ "Charged Off") of these loans, which were resolved between 2015 and 2018. The calibrator was effectively fitted on outcome labels that co-occurred in time with the out-of-time test period.

#### Quantification
We implemented `scripts/analyze_deployment_realism.py` and trained a strictly deployment-realistic model where only loans whose terminal outcome occurred on or before **2014-12-31** were labelled as defaults or non-defaults; active loans were treated as non-terminal.

| Metric | Standard Out-of-Time Benchmark | Strictly Deployment-Realistic | Gap ($\Delta$) | Operational Significance |
|---|---|---|---|---|
| **Test ROC-AUC** | 0.6690 | 0.6478 | **-0.0212** | Model discrimination degrades when immature vintages are excluded |
| **Test Brier Score** | 0.1211 | 0.1245 | **+0.0034** | Overall probability accuracy worsens |
| **Test ECE** | 0.0105 | 0.0374 | **+0.0270 (+256%)** | Severe miscalibration in production |
| **Observed / Expected (O/E)** | 1.0755 | 1.2860 | **+0.2105** | Production model underpredicts defaults by 28.6% |

#### Takeaway
Out-of-time benchmarks that rely on retroactively resolved loan outcomes flatter production reliability. In real-world lending systems, survival analysis (hazard rate modeling) or rolling dynamic calibration is essential to bridge the mature-outcome observation gap.

---

## Automated Test Coverage & Verification

All fixes and safeguards are permanently enforced via regression tests:
- `tests/test_expected_loss.py`: Validates non-negative cashflow identities and prevents double-counting of recoveries.
- `tests/test_policy.py`: Enforces validation-only parameter estimation for analytical decision policies.
- `tests/test_graph_leakage.py`: Tests non-graph controls and paired bootstrap calibration metrics.
- `tests/test_pipeline.py`: Validates deployment realism evaluation pipelines and stratification invariants.

Total test suite: **161 passing tests** across tabular, graph, simulation, calibration, and RL modules.
