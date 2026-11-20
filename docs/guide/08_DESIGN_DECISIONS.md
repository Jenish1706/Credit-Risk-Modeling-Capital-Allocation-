# 08 · Design decisions

Eleven decisions an informed reader is most likely to question, in ADR form: context, decision,
alternatives considered, consequences. Where a decision was revised after an audit, the entry says
so. Results referred to here are in [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md).

---

## ADR-01 · Out-of-time splits, and a second split inside the test year

**Context.** Credit performance moves with the cycle, so a random split inflates every metric. The
first version also chose the production calibrator, the KEEP/REJECT verdicts and the flag cut-points
on the same 2015 book it then reported, which makes those numbers optimistic.

**Decision.** Contiguous windows by issue month (train 2010-01 to 2014-06, valid 2014-07 to
2014-12, test 2015). The test year is split in time: **2015-01…06 for every selection, 2015-07…12 for
headline numbers**, with full 2015 as a labelled continuity column. Scripts assert the selection window.

**Alternatives.** Random or stratified k-fold (leaks the cycle). One test year used for everything
(selection reuse). Rolling-origin evaluation (more honest still, but several times the compute and
more complexity than the question needed).

**Consequences.** Smaller samples for both halves; the sign tests on ten PD strata in H1 have
little power, so the KEEP rule is conservative. Every headline number is a window further from the
training data than the full-year figure.

---

## ADR-02 · The primary PD excludes LendingClub's own grade and APR

**Context.** `grade`, `sub_grade` and `int_rate` are available at decision time but are the
incumbent underwriter's risk score and the price derived from it. A PD built on them partly
reproduces an existing scorecard. `installment` is an invertible function of `loan_amnt`,
`int_rate` and `term`, so it leaks the rate too.

**Decision.** The `primary` feature set excludes all four; the economics layer still uses APR as
revenue. An ablation refits the PD with the `with_lc_grade` set and repeats the profit audit, labelled
as using LendingClub's pricing.

**Alternatives.** Include them (higher AUC, but the question becomes "can we beat the incumbent
score?"). Include them everywhere (hides the PD's blind spot along APR).

**Consequences.** The PD is blind to APR, and the profit audit shows that this blindness is part of
why the expected-profit rule underperforms. The ablation shows the miscalibration along APR shrinks
with the grade features but the rule still does not match the threshold; the cost of the choice is
real and measured.

---

## ADR-03 · 36-month loans only

**Context.** Mixed terms have different outcome horizons, and 60-month loans issued in 2015 have not
resolved by the data cutoff.

**Decision.** Keep 36-month loans issued 2010 to 2015, so every labelled loan has had at least 36
months to resolve; no censoring, no term-mix confound across vintages.

**Alternatives.** Include 60-month loans with survival modelling (a different, larger project). Use
only an early-default window (a different target).

**Consequences.** Results speak only for 36-month unsecured loans. Non-terminal statuses inside the
window (a tiny share of rows) are data anomalies and are dropped and counted.

---

## ADR-04 · LightGBM plus explicit calibration as the production model

**Context.** The decision layer multiplies the probability by an exposure, so the probability level
matters as much as the ranking.

**Decision.** Logistic regression as the baseline, one boosting framework (LightGBM) as the
challenger, and calibration treated as its own step with four methods compared (identity,
intercept-only, Platt, isotonic) and the production calibrator **chosen on 2015-H1 Brier**.

**Alternatives.** Several boosting libraries (more surface area, no new insight). Neural tabular
models (not justified at this signal strength). Choosing the calibrator on the full test book (the
original behaviour, replaced).

**Consequences.** The calibrator choice is close to a tie (differences in Brier of the order of
1e-6), so the selected method matters little, which is itself informative. The intercept-only
baseline shows how much of a "calibration gain" is just a base-rate shift.

---

## ADR-05 · A time-respecting cohort graph, with separate rules for features and labels

**Context.** `member_id` is scrubbed, so there is no borrower identity and no counterparty network.
The graph can only link loans that share an attribute (3-digit ZIP, employer text).

**Decision.** Edges point strictly backwards in time. Neighbour **features** need only
`issue[j] < issue[i]`; neighbour **labels** need `resolved[j] <= issue[i]`. Cohort default rates are
shrunk toward the portfolio prior (pseudo-count 25). A guard (`assert_no_future_edges`) and tests
check the rules, including that an ungated rate leaks measurably.

**Alternatives.** A transductive graph over all loans (leaks the future). Using issue date for label
aggregates (the classic graph leak). A learned graph (no ground truth to learn from).

**Consequences.** It is a cohort graph, not a transaction network: it tests whether correlated-risk
structure helps, not whether a real counterparty graph would. The employer relation is weaker than it
looks because the column changed meaning around late 2013.

---

## ADR-06 · GraphSAGE written by hand

**Context.** The project's governing principle is a small theoretical surface that can be explained
line by line.

**Decision.** A two-layer mean-aggregator GraphSAGE in plain PyTorch with a masked mean per relation
and neighbour sampling over time-respecting neighbours; no graph-learning library.

**Alternatives.** PyTorch Geometric or DGL (faster to write, harder to defend and to guarantee that
sampling respects time).

**Consequences.** Slower and less general than a library, but every operation, including the masking
that keeps padded neighbours out of the mean, is readable and tested. The result is that the GNN does
not beat the aggregates, so the extra complexity buys nothing here.

---

## ADR-07 · Expected profit from a two-parameter cashflow model

**Context.** A decision needs revenue, not only risk. Realised cashflows are available per loan.

**Decision.** Fit good-loan yield and defaulted-loan loss rate as straight lines in APR on matured
training loans; subtract funding cost (3% over a 1.55-year weighted-average life, slightly below the schedule's 1.6) and
operating cost ($150 plus 1%). "LGD" is the net lifetime loss on funded principal, not Basel LGD.

**Alternatives.** A full amortisation and prepayment model (more parameters, more places to be
wrong). A fixed assumed LGD (an earlier version used 0.5 and reversed the sign of a result).

**Consequences.** Transparent and easy to audit, and the audit found it is **not** the dominant error:
cashflow-curve drift explains less than half of the gap between predicted and realised profit; the
PD level and its miscalibration along APR explain the larger share.

---

## ADR-08 · The simulator's headline action space is accept or reject at the observed exposure

**Context.** Only one exposure per borrower was ever observed. An early action space allowed 0.5x,
1.0x and 1.5x exposure; the expected-profit rule always chose 1.5x, so its outcomes were model
counterfactuals while a fixed threshold's were real, and part of the headline RL conclusion
came from that asymmetry.

**Decision.** Headline comparisons use accept or reject at the funded amount, so every approval
earns the return it really earned. The three-exposure space is a labelled sensitivity run
(`--multi-exposure`).

**Alternatives.** Keep the multi-exposure space as headline (mixes measurements and models).
Remove it entirely (loses the product-design question).

**Consequences.** The headline cannot say anything about lending more or less than requested. The
unconstrained RL "control breach" disappeared under this change, which is how the asymmetry was
identified as its source.

---

## ADR-09 · Learners are trained on the validation pool and frozen at evaluation

**Context.** In lending, a decision's reward arrives up to three years later. An early version kept
updating learners on realised test outcomes after every simulated decision and carried that state
across episodes.

**Decision.** `LendingEnvironment.run(..., learn=False)` for every evaluation episode; learners
train only on the validation pool. Tests assert parameters are byte-identical after evaluation and
that results do not depend on episode order.

**Alternatives.** Online learning during evaluation (what a bandit would do in production but with
delayed rewards). A delayed-feedback mode releasing rewards at `outcome_observed_period` (not built).

**Consequences.** Training still uses validation outcomes that were mostly unobservable at the
deployment cutoff, which is optimistic for the learners; building the delayed-feedback mode is the
open follow-up. A more pessimistic training regime would, if anything, weaken the learners further.

---

## ADR-10 · Blind-spot signals are judged conditionally on PD, with the decision made on H1

**Context.** A marginal table (calibration by signal quintile) is confounded by the PD level.

**Decision.** Stratify by PD decile, compare the calibration gap between signal halves, and apply
a two-sided sign test across strata plus an odds ratio per standard deviation. One KEEP rule
(`ConditionalResult.keep`) is decided on 2015-H1 and confirmed on 2015-H2.

**Alternatives.** A one-sided test (weaker, and inconsistent with the stated verdict). Deciding on
the full year (selection reuse).

**Consequences.** One of three signals passes; the disagreement signal fails the H1 rule even though
its holdout odds ratio is clearly above 1, which is a limitation of a sign test across ten strata
rather than evidence the effect is absent.

---

## ADR-11 · Evidence standard: controls, paired bootstrap, seeds, and rendered results

**Context.** Observational results are associations, and an interesting number invites
over-interpretation.

**Decision.** Attribution claims need a control arm (intercept-only, macro rate, shuffled keys,
placebo exclusion). Comparisons carry paired-bootstrap intervals (at least 2,000 resamples). A claim
about a model-fit effect must hold in every one of five seeds. Every number in the README and the
guide is rendered from `results/tables/` by a script and checked by tests.

**Alternatives.** Point estimates with commentary. Hand-maintained tables (they drifted from the
artifacts in earlier versions).

**Consequences.** Slower iteration, and some results that read worse (a calibration claim reported as
partly reproduced by meaningless cohort keys; a decision rule that loses). The cost is deliberate.
