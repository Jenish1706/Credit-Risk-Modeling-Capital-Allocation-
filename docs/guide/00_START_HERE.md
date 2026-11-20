# 00 · Start here

## What this project is

A credit-risk and lending-decision pipeline built on real LendingClub loans (2007 to 2018). It
predicts default, checks whether those probabilities can be trusted, tests whether relational
(cohort) information helps, and then asks whether a lender should act on the predictions with a
simple rule, an expected-profit rule or a learned policy. It is a research and engineering
exercise, not a deployable system: no money was lent.

## The questions it answers

1. Can borrower-level default modelling be improved by relational financial information, and can
   calibrated risk predictions then be used by a decision policy to optimise lending outcomes
   rather than merely predict default?
2. Can representation-learning signals (an autoencoder, a variational graph autoencoder, model
   disagreement) identify the borrowers for whom the PD model is unreliable, and does acting on
   that information improve the book?

Several answers are negative or null, and they are reported as findings. Two earlier headline
conclusions were reversed by audit (test-set learning in the reinforcement-learning evaluation, and
selections made on the reporting data); the corrections are in [WHAT_BROKE.md](../WHAT_BROKE.md).

## How the project is built

<!-- BEGIN:how-built -->
Each step answers a question the previous result made necessary. The experiment ids refer to the table below, which keeps them as references.

1. **Predict (EXP01).** Can a model rank borrowers out of time? Yes, modestly: gradient boosting beats a logistic baseline on a window no selection touched.
2. **Trust the probability (EXP02, EXP07, deployment realism).** A decision multiplies the probability by dollars, so ranking is not enough. Calibration gains are mostly a base-rate correction, the PD's level drifts across vintages while out-of-sample ranking does not, and training only on labels that were knowable at the deployment date is worse than the usual benchmark suggests.
3. **Add relational information (EXP04, EXP05).** Do cohorts help? Not ranking. They fix the *level* of the PD in every training seed tested, though shuffled cohort keys reproduce part of that gain; GraphSAGE is the weakest arm.
4. **Decide (EXP03, EXP06).** Does a profit rule built on the PD beat a plain PD threshold, and does learning or capital pricing help? No: the profit rule loses to the threshold on the real book, and neither a learned policy nor the shadow-price policy recovers it.
5. **Diagnose the failure.** Where does predicted profit go wrong? A dollar decomposition of predicted versus realised profit, a PD refit with LendingClub's pricing, and a PD refit with the cohort features (correct level, remaining APR gradient). The PD's default-count error is the largest single source, it varies systematically with APR, and correcting the PD's level alone does not fix the decision.
6. **Find where the PD is unreliable (EXP08, EXP09, EXP10).** Blind-spot signals, judged conditional on the PD level and decided on a selection window; only one passes.
7. **Act on it (EXP11).** Excluding borrowers flagged by the kept signal, judged against random and APR-matched placebos.

**Root cause, measured.** The decision layers inherit the PD's errors. The production PD under-predicts defaults where APR is high (observed/expected 0.67 in the lowest APR quintile to 1.38 in the highest), so a rule that rewards APR selects understated risk: the PD's default-count error is 55% of the gap between predicted and realised profit (EXP03; the rest is drift in the cashflow curves), refitting the PD for level alone does not fix the decision, and the risk reduction in EXP11 is matched by excluding loans of the same APR mix.
<!-- END:how-built -->

## Headline results

Every row is rendered from `results/tables/` by `scripts/render_readme.py`, so this table cannot
drift from the artifacts. Headlines are measured on **2015-07…12**, a window no selection touched;
selections used 2015-01…06 only. `OBSERVED` means measured from data, `SIMULATED` means produced in
the lending simulator. How to read each row, with its uncertainty, is in
[09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md).

<!-- BEGIN:results-table -->
| # | Question | Result | Verdict |
|---|---|---|---|
| **EXP01** | Does non-linear modelling beat logistic regression? | AUC **0.6606 → 0.6732** (+0.0126) on 2015-07..12 (full 2015: 0.6581 → 0.6705) `OBSERVED` | Material. Boosting justified |
| **EXP02** | Does calibration improve PD reliability? | AUC **unchanged to 5 d.p.**; ECE **0.0198 → 0.0117** (intercept_only, chosen on 2015-01..06) on 2015-07..12 (full 2015: 0.0193 → 0.0110); slope 1.003 → 1.004 `OBSERVED` | Ranking is unchanged and ECE falls, but this is mostly a base-rate shift: a one-parameter intercept shift recovers 94% of the best calibrator's ECE gain |
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | Simulated diff vs 15% threshold: **-28,262**/episode (95% CI [-54,837, -2,616]) `SIMULATED` on 2015-07..12. Real book at observed exposure: **-36.9**/loan (95% CI [-47.1, -26.9]) `OBSERVED` on 2015-07..12 | **Expected-profit rule loses to a PD threshold on the real book.** Simulation CI excludes zero. Mechanism is measured below, not assumed |
| **EXP04** | Do graph cohort features add signal? | AUC **-0.0012** (mean over 5 LightGBM seeds). On 2015-07..12: ECE **0.0110 ± 0.0001 → 0.0048 ± 0.0005**, O/E **1.081 → 0.998** (EXP04/05 single-fit run: ECE 0.0109 → 0.0043, O/E 1.080 → 0.989) `OBSERVED` | No discrimination gain; **calibration gain survives both controls in every one of 5 seeds** (ΔECE B−tabular -0.0062 ± 0.0005, B−macro -0.0066 ± 0.0002, B−shuffled -0.0038 ± 0.0006; mean ± sd over seeds, per-seed paired bootstrap, 2015-07..12) |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6637** vs 0.6704 (aggregates) vs 0.6712 (tabular) on 2015-07..12 `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | Frozen-at-test learners, accept/reject at observed exposure (2015-07..12). Unconstrained: best learner thompson +17,398 vs myopic expected-profit (95% CI [-2,574, +37,136]). 60% budget: thompson -160 vs the 15% threshold (95% CI [-15,333, +14,180]) `SIMULATED` | Control holds: no learner significantly beats the myopic rule with unconstrained capital; **no learner significantly beats the fixed 15% threshold in any condition**; significantly worse than the threshold: budget_aware_q(greedy) (unconstrained, budget=0.60); linucb@alpha=1 (budget=0.35, budget=0.20); thompson (budget=0.20); learners that stopped lending (approval below 1%): linucb@alpha=1 (budget=0.35); thompson (budget=0.20); the validation-fitted shadow-price policy is significantly worse than the myopic rule at budget=0.35 |
| **EXP07** | How does the model degrade under vintage drift? | O/E **0.861 (2010, a training vintage) → 1.082 (2015-07..12)** (full 2015: 1.077). AUC by quarter: in-sample training vintages 0.700–0.761, out-of-sample (validation and test) 0.665–0.675 `OBSERVED` | Calibration drifts across vintages (O/E rises from below 1 to above 1). Out-of-sample ranking is flat, so most of the fall in AUC across vintages is in-sample optimism, not drift |
| **Deployment realism** | What if the model only knows the labels available at the deployment date? | Trained only on outcomes knowable by 2014-12, on 2015-07..12: AUC **0.6712 → 0.6503**, ECE **0.0109 → 0.0367**, O/E **1.080 → 1.294** (full 2015: AUC 0.6690 → 0.6478, O/E 1.076 → 1.286) `OBSERVED` | **The standard out-of-time benchmark overstates day-one performance:** most training labels were not yet knowable at the deployment date, and using only those that were makes ranking and calibration worse |
| **EXP08** | Does autoencoder anomaly flag untrustworthy PDs? | H1 decision **KEEP**. On 2015-07..12: survives PD control in **10/10** PD strata, two-sided sign test p=0.0020, odds ratio 1.033/SD [1.021, 1.046] (full 2015: 10/10, OR 1.032) `OBSERVED` | **KEEP**, confirmed on holdout |
| **EXP09** | Does VGAE relational uncertainty flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **3/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.002/SD [0.988, 1.017] (full 2015: 5/10, OR 1.002) `OBSERVED` | **REJECT** |
| **EXP10** | Does model disagreement flag them? | H1 decision **REJECT**. On 2015-07..12: survives PD control in **7/10** PD strata, two-sided sign test p=0.3438, odds ratio 1.050/SD [1.033, 1.068] (full 2015: 9/10, OR 1.045) `OBSERVED` | **REJECT** |
| **EXP11** | Does acting on those signals improve the book? | Default rate **16.62% → 15.88%** on 2015-07..12. Placebo exclusion (200 draws, 95% intervals): random 16.08%–16.26%, APR-matched 15.87%–16.05% `OBSERVED` | **Beats random exclusion but not the APR-matched placebo.** The signal book is 0.09 points below the APR-matched placebo mean, 12% of its 0.74-point reduction, and +0.008 points from that interval's lower edge, so most of the reduction is the lower-APR refill; the margin is borderline |
<!-- END:results-table -->

Reading the table as a whole:

- **Ranking is not the point.** Non-linear modelling helps a little, but the graph adds nothing to
  discrimination and GraphSAGE is the weakest arm.
- **Calibration is where the effort pays**, and the main calibration claim is tested against
  controls and across training seeds. Much of any calibration gain is a base-rate shift, and training
  only on labels that were knowable at the deployment date is worse than the usual benchmark suggests.
- **The decision layer is the hard part.** An expected-profit rule built on these PDs loses to a
  plain PD threshold on the real book, and the audit measures why. Fixing the PD's level (the
  cohort-feature PD) does not fix the decision; the APR gradient in the PD's errors does the damage.
- **RL does not earn its place.** With learning frozen at evaluation and every outcome real, no
  learner beats the simple rule.
- **Blind-spot signals are small**, only one passes the pre-specified rule, and acting on it only
  matches a placebo that excludes loans of the same APR mix.

## Reading order

| If you want to... | Read |
|---|---|
| understand the ideas | [01_CONCEPTS.md](01_CONCEPTS.md), then [GLOSSARY.md](GLOSSARY.md) |
| see how the pieces fit | [02_ARCHITECTURE.md](02_ARCHITECTURE.md) |
| know exactly what data was used | [03_DATA.md](03_DATA.md) |
| check how results are produced and tested | [04_METHODOLOGY.md](04_METHODOLOGY.md) |
| find the code | [05_CODE_WALKTHROUGH.md](05_CODE_WALKTHROUGH.md) |
| run it | [06_RUNBOOK.md](06_RUNBOOK.md) |
| see what is tested | [07_TESTING.md](07_TESTING.md) |
| understand why it is built this way | [08_DESIGN_DECISIONS.md](08_DESIGN_DECISIONS.md) |
| see results with uncertainty and limits | [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) |

Appendices: [DATA_DICTIONARY.md](../DATA_DICTIONARY.md) (column ruling) and
[LEAKAGE.md](../LEAKAGE.md) (every leakage rule and its test).

## Ten-minute quickstart

This path needs no data download; it runs the tests and re-renders every table from the committed
artifacts.

1. Create the environment (Windows PowerShell; conda is required because Smart App Control blocks
   unsigned wheels, see [06_RUNBOOK.md](06_RUNBOOK.md)):

   ```powershell
   conda create -n credit-risk python=3.13 numpy pandas scipy scikit-learn pyarrow lightgbm shap matplotlib pyyaml pytest numba
   conda activate credit-risk
   pip install -e . --no-deps
   ```

2. Run the tests (under a minute; tests that need torch skip with a reason when it is absent):

   ```powershell
   python -m pytest -q
   ```

3. Re-render the README and guide tables from `results/tables/` (changes nothing if nothing drifted):

   ```powershell
   python scripts/render_readme.py
   ```

4. Look at one artifact behind a headline row, for example the paired differences for the
   learned policies:

   ```powershell
   python -c "import pandas as pd; print(pd.read_csv('results/tables/exp06_rl_exp06_diffthr_budget035.csv', index_col=0).round(0))"
   ```

Reproducing the numbers from raw data (several hours, mostly the variational graph autoencoder)
is described step by step in [06_RUNBOOK.md](06_RUNBOOK.md).
