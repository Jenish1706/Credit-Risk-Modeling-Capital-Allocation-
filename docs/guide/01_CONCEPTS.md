# 01 · Concepts

Every idea the project relies on, in the order you need it. Each entry gives the intuition, the
formula, a tiny worked example with made-up numbers (the numbers in the examples are for arithmetic
only; none is a project result), and a pointer to the code. Real results are in
[09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md). Terms are collected in
[GLOSSARY.md](GLOSSARY.md).

Contents: [1 PD, LGD, EAD](#1-pd-lgd-ead-and-expected-loss) ·
[2 Out-of-time validation](#2-out-of-time-validation-and-label-maturity-look-ahead) ·
[3 AUC, Gini, KS](#3-auc-gini-and-ks) · [4 Calibration](#4-calibration) ·
[5 Cohort features](#5-time-respecting-cohort-features-and-empirical-bayes-shrinkage) ·
[6 GraphSAGE](#6-graphsage) · [7 Conditional signal test](#7-the-conditional-pd-stratified-signal-test) ·
[8 Placebos and the bootstrap](#8-placebo-controls-and-the-paired-bootstrap) ·
[9 Expected profit and breakeven PD](#9-the-expected-profit-decision-rule-and-breakeven-pd) ·
[10 Knapsack shadow price](#10-the-knapsack-shadow-price) ·
[11 LinUCB and Thompson](#11-linucb-and-thompson-sampling) ·
[12 Delayed feedback](#12-why-delayed-feedback-matters-in-lending) · [13 PSI and drift](#13-psi-and-drift)

---

## 1. PD, LGD, EAD, and expected loss

**Intuition.** A lender asks three questions about a loan: how likely is it to go bad (**PD**,
probability of default), how much of the money is lost if it does (**LGD**, loss given default),
and how much is outstanding when it does (**EAD**, exposure at default).

**Formula.** `EL = PD × LGD × EAD`.

**Worked example.** PD 10%, LGD 50%, EAD $10,000 → EL = 0.10 × 0.5 × 10,000 = **$500**.

**Two traps.**

1. *Expected loss is not a decision rule.* It ignores revenue, so minimising it means lending to
   nobody. The decision rule is expected **profit** (§9).
2. *This project's "LGD" is not Basel LGD.* Here it is the **net lifetime loss rate on the funded
   principal** of a defaulted loan: `1 − (principal repaid + net recoveries) / funded amount`
   (`observed_lgd_mean` on `CashflowModel`). Basel LGD is defined over exposure at default, and
   because loans amortise before they default, Basel LGD is higher. The decision layer uses the
   funded-principal definition because that is the capital committed at origination.

**Code.** `expected_loss` and `fit_cashflow_model` in
[`decision/expected_loss.py`](../../src/credit_risk/decision/expected_loss.py).

---

## 2. Out-of-time validation and label-maturity look-ahead

**Intuition.** Credit performance moves with the economic cycle. If you split loans at random, the
model sees loans from the same month as the ones it is scored on and learns that month's conditions,
which it could never have known in advance. The honest question is "trained on the past, how does it
do on a later period?".

**Mechanics.** Splits are contiguous windows of issue month (train, valid, test, plus a history
window used only for graph neighbours and an unlabelled monitoring window). There is no random split
anywhere. The test year is further halved in time: **H1 (2015-01…06) is used for every selection,
H2 (2015-07…12) is the untouched holdout** where headline numbers are measured. See
[04_METHODOLOGY.md](04_METHODOLOGY.md).

**Label-maturity look-ahead** is subtler. A 36-month loan issued in January 2013 does not resolve
until January 2016. A model trained "as of" the end of 2014 could not have known that loan's
outcome, yet an out-of-time split that labels it from the final data quietly gives the model that
future knowledge. The same applies to the **calibrator** (fitted on the validation window with
mature labels).

**Worked example.** Deploy on 2015-01-01. A loan issued 2013-01 and charged off 2016-01 has the
label "default" in the final table, but on the deployment date it looked like a current loan. A
deployment-realistic training set contains only outcomes observable on or before the cutoff.

**Code.** [`data/splits.py`](../../src/credit_risk/data/splits.py) (`SplitWindows`, `assign_split`,
`assert_temporal_ordering`, `test_window_masks`, `assert_selection_window`);
`outcome_observed_period` in [`data/prepare.py`](../../src/credit_risk/data/prepare.py);
[`scripts/analyze_deployment_realism.py`](../../scripts/analyze_deployment_realism.py) quantifies
the cost of the look-ahead.

---

## 3. AUC, Gini, and KS

All three measure **ranking**: does the model put defaulters above non-defaulters?

- **AUC** = probability that a randomly chosen defaulter has a higher score than a randomly chosen
  non-defaulter (0.5 is chance, 1.0 is perfect).
- **Gini** = `2 × AUC − 1`.
- **KS** = the largest vertical gap between the cumulative score distributions of defaulters and
  non-defaulters.

**Worked example.** Scores 0.1 (good), 0.3 (good), 0.2 (bad), 0.4 (bad). Of the four
(bad, good) pairs, three rank the bad loan higher → **AUC = 3/4 = 0.75**, **Gini = 0.5**. Sweeping
the threshold, the cumulative-share gap peaks at 0.5 after score 0.1 and again after 0.3 →
**KS = 0.5**.

**Why they are not enough.** A monotone transform of the scores (calibration) leaves all three
unchanged, so they are blind to whether "12%" really means 12% (§4).

**Code.** `ks_statistic` and `compute_metrics` in
[`evaluation/metrics.py`](../../src/credit_risk/evaluation/metrics.py).

---

## 4. Calibration

**Intuition.** Among loans a model scores at 12%, about 12% should default. A model can rank
perfectly and still be wrong about the level, and the decision layer multiplies the probability by
an exposure, so the level matters.

**Metrics.**

| Metric | Definition | Reading |
|---|---|---|
| **ECE** | Sort loans into 20 quantile bins; average `|mean predicted − observed rate|` weighted by bin size | 0 is perfect |
| **Calibration slope / intercept** | Logistic regression of the outcome on the predicted log-odds | slope 1, intercept 0 is perfect; slope < 1 means over-confident |
| **O/E** | `observed default rate / mean predicted` | > 1: model **under**-predicts risk; < 1: over-predicts |

The project uses **O/E** (observed over expected) everywhere. `predicted_observed_ratio` (the
reciprocal) still exists in some tables; public text never quotes it.

**Recalibration methods** (all fitted on the validation window, never on training data):

- **Platt**: logistic regression on the model's log-odds (two parameters).
- **Isotonic**: a monotone step function (flexible, can overfit).
- **Intercept-only**: shift the log-odds by one constant so the mean prediction matches the base
  rate. This is the baseline that tells you whether a calibration "gain" is only the base rate
  having moved between training and the later window.

**Worked example.** Two equal bins: predicted 10% / observed 12% and predicted 30% / observed 26%.
Gaps 0.02 and 0.04 → ECE = 0.03. If the mean prediction is 20% and the observed rate is 25%,
O/E = 1.25.

**Code.** `expected_calibration_error`, `calibration_slope_intercept` in
[`evaluation/metrics.py`](../../src/credit_risk/evaluation/metrics.py);
`ProbabilityCalibrator`, `compare_calibrations` in
[`models/calibration.py`](../../src/credit_risk/models/calibration.py). The production calibrator is
chosen on 2015-01…06 Brier score (`scripts/train_baseline.py`).

---

## 5. Time-respecting cohort features and empirical-Bayes shrinkage

**Intuition.** Borrowers in the same 3-digit ZIP or with the same normalised employer title may
share risk (a local factory closing). A **cohort feature** summarises what happened to *earlier*
members of the borrower's cohort.

**Time-respecting means two different rules**, because there are two kinds of information:

| Quantity | Allowed neighbour `j` of loan `i` | Why |
|---|---|---|
| Feature aggregates (mean FICO, counts) | `issue[j] < issue[i]` | the application existed and was visible |
| **Label** aggregates (cohort default rate) | `resolved[j] <= issue[i]` | the outcome had actually happened |

Using the first rule for a label aggregate is the classic graph leak: a loan issued a year ago has
not resolved yet, so its outcome cannot be in the feature.

**Shrinkage.** A cohort with three past loans cannot have a reliable default rate. The smoothed rate
pulls it toward the portfolio prior with a pseudo-count `α` (25 here):

`smoothed = (defaults + α × prior) / (resolved + α)`

**Worked example.** Prior 15%, α = 25. A cohort with 4 resolved loans and 2 defaults:
`(2 + 25×0.15) / (4 + 25) = 5.75 / 29 = 0.198`. A cohort with 400 resolved loans and 80 defaults:
`(80 + 3.75) / 425 = 0.197`: the evidence now dominates. `evidence_weight = resolved / (resolved + α)`
is stored so the model can discount thin cohorts.

**Code.** `CohortFeatureBuilder` in
[`graph/features.py`](../../src/credit_risk/graph/features.py); the construction and its
`assert_no_future_edges` guard in
[`graph/construction.py`](../../src/credit_risk/graph/construction.py).

---

## 6. GraphSAGE

**Intuition.** Instead of hand-built cohort aggregates, let a neural network learn to summarise a
node's neighbours. GraphSAGE does it with a sampled **mean aggregator**.

**Formula** (one layer, for node `v`; `R` relations, each with its sampled neighbour set `N_r(v)`):

`h_v' = act( W · [ h_v ‖ mean_{u ∈ N_1(v)} h_u ‖ … ‖ mean_{u ∈ N_R(v)} h_u ] )`

where `‖` is concatenation, `act` is ReLU on hidden layers and L2-normalisation on the last (the
embedding), and an MLP head turns the embedding into a default logit. Two layers (neighbours of
neighbours), sampling 10 then 5 neighbours (`fanout`), keep it tractable; three or more over-smooth
on a cohort graph. The two relations are shared ZIP3 and shared normalised employer, and each
contributes its own mean. A node with no eligible earlier neighbour gets a zero mean for that
relation.

**Why it is written by hand.** The implementation avoids a graph-learning library so that every line
of message passing can be explained ([`models/gnn.py`](../../src/credit_risk/models/gnn.py),
`GraphSAGETrainer`). Neighbour sampling uses only the time-respecting neighbours from §5.

---

## 7. The conditional (PD-stratified) signal test

**Question.** A "blind-spot signal" (autoencoder reconstruction error, graph-embedding uncertainty,
disagreement between two models) is supposed to flag borrowers whose PD is unreliable. Looking at
the marginal table (calibration by signal quintile) is misleading: if the signal merely correlates
with PD itself, calibration degrades across quintiles simply because boosted models are less well
calibrated at high PD.

**Test.** Hold PD fixed.

1. Cut the book into PD deciles.
2. Within each decile split at the median of the signal.
3. For each half, compute the calibration gap `observed − predicted default rate`; take the
   difference (high-signal half minus low-signal half). Positive means the model under-states risk
   more for high-signal borrowers at the same PD.
4. Count deciles with a positive difference and apply an exact two-sided **sign test**; also fit a
   logistic regression of the outcome on `logit(PD)` plus the standardised signal and report the
   **odds ratio per standard deviation** with a Wald 95% interval.

**KEEP rule** (one definition, `ConditionalResult.keep`): two-sided sign-test p < 0.05 **and** mean
gap difference > 0. The decision is taken on 2015-01…06 only and confirmed on 2015-07…12.

**Worked example.** A decile where the high-signal half has predicted 20% / observed 25% (gap +0.05)
and the low-signal half has predicted 20% / observed 21% (gap +0.01): gap difference +0.04. If 9 of
10 deciles are positive, the two-sided binomial p-value is 2 × (10+1)/1024 ≈ 0.021.

**Code.** `conditional_signal_table`, `conditional_by_window`, `WindowedConditional` in
[`decision/blindspot.py`](../../src/credit_risk/decision/blindspot.py).

---

## 8. Placebo controls and the paired bootstrap

**Placebo.** When an intervention appears to help, ask what a meaningless version of it would do.
Excluding flagged borrowers and refilling the budget with others lowers the default rate for a
mechanical reason: the replacements tend to be lower-APR, lower-risk loans. Excluding the **same
share of loans at random**, or at random **within matching APR deciles**, shows how much of the
effect needs no signal at all. If the signal's result sits inside the placebo intervals (200 draws
each), the signal is not what produced it.

**Paired bootstrap.** To put an interval on a difference between two policies or models evaluated
on the same data, resample the *units* (loans or episodes) with replacement, recompute the
difference on every resample (2,000+), and take the 2.5th and 97.5th percentiles. "Paired" means
both arms are evaluated on the identical resample, which removes the noise they share.

**Worked example.** Two policies' per-episode profits on five episodes: A = (10, 12, 9, 11, 13),
B = (8, 11, 9, 10, 10). Differences A − B = (2, 1, 0, 1, 3), mean 1.4. Resampling those five
differences many times gives a distribution of means; if the 95% interval excludes zero, the
improvement is distinguishable from noise.

**Code.** `evaluate_placebo_controls` in
[`decision/blindspot.py`](../../src/credit_risk/decision/blindspot.py);
`paired_bootstrap_calibration_comparison` in
[`evaluation/metrics.py`](../../src/credit_risk/evaluation/metrics.py);
`paired_difference` and `difference_table` in
[`simulation/evaluation.py`](../../src/credit_risk/simulation/evaluation.py);
`paired_bootstrap_ci` in [`decision/profit_audit.py`](../../src/credit_risk/decision/profit_audit.py).

---

## 9. The expected-profit decision rule and breakeven PD

**Rule.** Lend if the expected profit is positive:

`E[profit] = (1 − PD) · y(APR) · E − PD · l(APR) · E − funding(E) − opex(E)`

`y` is the realised return per dollar on loans that repaid, `l` the loss rate per dollar on loans
that defaulted, both fitted as straight lines in APR on training-window loans (`CashflowModel`).

**Breakeven PD** is the PD at which the expression is zero:

`PD* = (y·E − costs) / ((y + l)·E)`

A fixed "approve if PD < 15%" rule ignores that `PD*` rises with the interest rate: a high-APR loan
can carry more risk and still pay.

**Worked example.** Exposure $10,000, `y` = 15%, `l` = 40%, costs $600 (funding plus opex).
`PD* = (1,500 − 600) / ((0.15 + 0.40) × 10,000) = 900 / 5,500 = 0.164`. A borrower with PD 12% has
profit `0.88×1,500 − 0.12×4,000 − 600 = 1,320 − 480 − 600 = +240`; at PD 20% it is
`0.80×1,500 − 0.20×4,000 − 600 = 1,200 − 800 − 600 = −200`.

**Caution.** The rule is only as good as `PD`, `y` and `l`. The project measures how far its
predicted profit is from realised profit (see
[09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md)).

**Code.** `expected_profit`, `breakeven_pd`, `realised_profit` in
[`decision/expected_loss.py`](../../src/credit_risk/decision/expected_loss.py);
`ExpectedProfitPolicy` in [`decision/policies.py`](../../src/credit_risk/decision/policies.py).

---

## 10. The knapsack shadow price

**Intuition.** With unlimited capital you approve every positive-profit loan. With a fixed budget
you should approve the loans with the best profit **per dollar of capital** until the money runs
out. The profit density of the last loan that fits is the **shadow price** `λ` of capital: the
value of one more dollar. The decision becomes "approve if profit per dollar > λ".

**Worked example.** Budget $100. Loans (exposure, profit): A ($40, $8), B ($40, $4), C ($40, $6).
Densities 0.20, 0.10, 0.15. Sorted: A, C, B. A and C fit ($80); B would overflow ($120) so `λ = 0.10`.
Approve A and C.

**Where it can fail.** The optimum is optimal *for the predicted profit it is given*. If predicted
profit is mis-ordered (§9 caution), the optimiser faithfully optimises the wrong thing; the
README's EXP06 table reports exactly that.

**Code.** `analytical_shadow_price` in [`rl/formulation.py`](../../src/credit_risk/rl/formulation.py);
`ExpectedProfitPolicy(capital_shadow_price=λ)`.

---

## 11. LinUCB and Thompson sampling

Both are **contextual bandits**: at each applicant, pick an action (accept or reject) given
features `x`, see a reward, update. They fit one ridge regression per action,
`reward ≈ xᵀθ_a`, with `A_a = λI + Σ x xᵀ` and `b_a = Σ r·x`, `θ_a = A_a⁻¹ b_a`.

- **LinUCB** picks `argmax_a  xᵀθ_a + α · sqrt(xᵀ A_a⁻¹ x)`: the estimate plus an optimism bonus
  that is large where the data is thin.
- **Thompson sampling** draws `θ̃_a ~ N(θ_a, v² A_a⁻¹)` and picks `argmax_a xᵀθ̃_a`: it explores in
  proportion to its uncertainty.

**Worked example.** After few observations of "accept" in some region, the bonus
`sqrt(xᵀ A⁻¹ x)` is large, so LinUCB tries "accept" there even if the mean estimate is slightly
negative; as data accumulates the bonus shrinks and the mean decides.

**In this project.** Learners are trained on the validation pool and **frozen** during every
evaluation episode (no `update`), so scoring a policy cannot teach it the test outcomes. A tabular
budget-aware Q-learner is also included. All are compared with static rules on identical episodes.

**Code.** `LinUCBPolicy`, `ThompsonSamplingPolicy`, `BudgetAwareQPolicy`, `train_policy` in
[`rl/policy.py`](../../src/credit_risk/rl/policy.py); `LendingEnvironment.run(learn=False)` in
[`simulation/environment.py`](../../src/credit_risk/simulation/environment.py).

---

## 12. Why delayed feedback matters in lending

A bandit assumes it sees the reward right after it acts. In lending the reward for approving a
36-month loan (repaid or defaulted) arrives up to three years later, and meanwhile hundreds of
thousands of similar decisions have been made. Two consequences:

1. **Evaluation must not leak it.** An early version fed realised outcomes back to the policy after
   every simulated decision, including on the test pool, and the policy learned from the very
   outcomes it was scored on. The fix is `learn=False` during evaluation.
2. **Training on "outcomes" is itself optimistic.** Most validation-window loans were still open at
   the deployment cutoff, so training on their final outcomes assumes information a deployed system
   would not have. A delayed-feedback training mode (release each reward at the loan's
   `outcome_observed_period`) is not built; it is listed as future work.

---

## 13. PSI and drift

**PSI** (population stability index) compares a feature or score distribution between a reference
period and a current one:

`PSI = Σ_bins (current_share − reference_share) · ln(current_share / reference_share)`

Bin edges come from the **reference** quantiles. Conventional reading: below 0.10 stable, 0.10 to
0.25 moderate shift, above 0.25 significant shift (`classify_psi`).

**Worked example.** Two bins; reference 50% / 50%, current 70% / 30%:
`(0.7−0.5)·ln(0.7/0.5) + (0.3−0.5)·ln(0.3/0.5) = 0.2×0.336 + (−0.2)×(−0.511) = 0.067 + 0.102 = 0.169`
(moderate).

**Why labelled and unlabelled drift differ.** In production, features and predictions arrive
immediately but outcomes arrive years later, so PSI on features and scores is the early warning,
while calibration drift (O/E by issue vintage) can only be measured on vintages whose loans have
matured. The monitoring window (2016-2018) is deliberately unlabelled for this reason.

**Code.** `population_stability_index`, `feature_drift`, `prediction_drift`, `calibration_drift` in
[`monitoring/drift.py`](../../src/credit_risk/monitoring/drift.py); run by
[`scripts/evaluate.py`](../../scripts/evaluate.py).
