# Interview Guide: ML-Driven Credit Decision System

This guide is designed as a Q&A companion to the codebase. It covers the rationale behind the architectural and theoretical decisions made in this project. The guiding philosophy of this project is *small theoretical surface area, fully explainable*. 

## 1. Dataset & Modelling Scope

**Q: Why LendingClub and not Home Credit or Kaggle competitions?**
**A:** Credit risk models in production must handle time. Home Credit does not contain absolute calendar dates (all dates are relative to the application). This makes it impossible to do out-of-time splits, vintage analysis, or drift monitoring. Furthermore, LendingClub contains `total_rec_prncp` (recovery amounts) and `int_rate` (pricing), allowing us to build an actual lending simulator with empirical cashflows. It also has borrower-level groupings (`zip_code`, `emp_title`) to build a graph.

**Q: Why limit the modelling window to 2010-2015 36-month loans?**
**A:** We need fully matured loans to have unbiased targets. A 36-month loan issued in December 2015 matures in December 2018 (the end of the LendingClub dataset). By enforcing this, every outcome in our dataset is fully realised, avoiding censoring issues and eliminating the confound of term-mix (36-month vs 60-month).

**Q: Why an out-of-time test split instead of random cross-validation?**
**A:** Random splitting in a time-series context is a massive leak. It allows a model to "see the future" of the same macroeconomic cycle it is being tested on. Our test split (2015-01 to 2015-12) strictly follows the training and validation windows. 

## 2. Feature Engineering & Leakage

**Q: What is the most dangerous leak in the LendingClub dataset?**
**A:** `last_fico_range_high` and `last_fico_range_low`. These are FICO scores pulled *after* origination. A borrower who is defaulting will have a collapsed FICO. Including this gives an AUC of ~0.95 but the model is completely useless for a decision made *at origination*. 

**Q: Why is `installment` excluded from the primary feature set?**
**A:** The `installment` is mechanically derived from `loan_amnt`, `term`, and `int_rate`. Given the amount and term, the model can back out the assigned interest rate to 0.05pp precision. Since `int_rate` is LendingClub's *decision*, this allows the model to just reverse-engineer the incumbent scorecard rather than learn to underwrite from primary evidence.

## 3. Layer B - Graph Structure

**Q: How do you build a graph without counterparty transaction data?**
**A:** We use a shared-attribute cohort graph. The primary relation is `zip3` (shared local labour/housing market shocks). The secondary relation is `emp_norm` (shared employer). 

**Q: How do you prevent the graph from leaking future information?**
**A:** By enforcing strictly time-respecting edges. For borrower $i$ applying today, they can only have an edge to neighbour $j$ if $j$ applied *before* $i$ (`issue_d[j] < issue_d[i]`). Furthermore, when using neighbour *labels* (did $j$ default?), we only use it if $j$'s outcome was resolved *before* $i$'s application date. This mimics exactly what a lender knows in production.

**Q: Why a custom GraphSAGE implementation in plain PyTorch instead of PyTorch Geometric (PyG)?**
**A:** Two reasons. First, `torch-geometric` sampling wheels can be unreliable on Windows. Second, a handwritten sampler (~200 LOC) demonstrates a fundamental understanding of the algorithm and can be explained line-by-line in an interview. A library call cannot.

## 4. Layer A - Risk Model & Layer C - Reinforcement Learning

**Q: Why is model calibration so important here?**
**A:** A standard classifier outputs a score that ranks risk (e.g., AUC). But Layer C (the decision engine) needs to calculate expected profit: $E[Profit] = PD \times Loss + (1 - PD) \times Gain$. This requires $PD$ to be a true probability. Without Platt/Isotonic calibration, the downstream economic calculations will be drastically wrong.

**Q: You mention Reinforcement Learning (RL), but isn't credit decisioning just supervised learning?**
**A:** If capital is unconstrained (infinite budget), the optimal policy is purely myopic: fund every applicant whose expected profit > 0. RL adds zero value here. 

However, when capital is constrained (a finite budget for the year), funding a marginal applicant today consumes capital that a better applicant tomorrow cannot use. The decision becomes sequential. The RL value function learns the opportunity cost (the shadow price of capital) over the finite horizon, rejecting marginal loans early in the period to save budget for better loans later.

**Q: How do you measure the success of the RL policy?**
**A:** By comparing a `BudgetAwareQPolicy` against a `ExpectedProfitPolicy` (myopic) under a constrained budget (e.g., 35% of total request volume). The claim is: RL $\approx$ Myopic when capital is free, but RL > Myopic when capital is scarce.
