# 09 · Results and limitations

Every result below is rendered from `results/tables/` by `scripts/tools/render_readme.py`; nothing here is
typed by hand. Headlines are on **2015-07…12** (a window no selection touched), with full 2015 in
brackets where both are shown. Selections used 2015-01…06 only. The protocol and what would falsify
each claim are in [04_METHODOLOGY.md](04_METHODOLOGY.md).

## Headline results with uncertainty

Rows carry a paired-bootstrap interval (EXP01, EXP03, EXP06), a seed spread (EXP04), a sign test with an
odds-ratio interval (EXP08 to EXP10) or placebo intervals (EXP11). EXP02, EXP05, EXP07 and the deployment-realism
row are single-fit point estimates with no interval. The artifact behind each row is listed in the runbook
([06_RUNBOOK.md](06_RUNBOOK.md), section 6).

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

## Decision layer and reinforcement learning in detail

The tables below hold every policy in every capital condition (with approval rate and capital
deployed), the exposure-changing sensitivity run, the real-book profit audit and the APR ablation.
Intervals are 95% paired-bootstrap intervals; a starred difference has an interval that excludes zero.

<!-- BEGIN:rl-section -->
**RL does not earn its place here, and that is the finding.** Learning policies (LinUCB, Thompson sampling, Q-learning) choose between accept and reject at the exposure that was actually funded, so every simulated outcome is a real one. They are trained on the validation pool and their parameters are **frozen** during every test-pool episode; an earlier version kept updating them on realised test outcomes after every decision, which no deployed policy can do (loan outcomes take up to 36 months to arrive).

The experiment carries a control. With unconstrained capital the myopic expected-profit rule should be near-optimal. The best learner (thompson) earns 50,883 vs the myopic rule's 33,486 (paired difference +17,398, 95% CI [-2,574, +37,136]): **the control is not breached** (the difference is not significant, though the interval is wide). The plain 15% PD threshold beats the myopic rule by +28,262 (95% CI [+2,616, +54,837]), consistent with the real-book audit below.

Under budget=0.60, learners do significantly beat the myopic rule (thompson +26,236 [+7,044, +44,592]; linucb@alpha=1 +24,745 [+5,999, +42,610]) (the myopic rule is itself -26,396 against the threshold there); against the 15% threshold the best of them, thompson, is -160 [-15,333, +14,180].

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

- On the 2015-07..12 holdout, accepting every loan with predicted profit above zero earns **-36.9** per pool applicant vs the 15% PD threshold (95% CI [-47.1, -26.9], -5.99M in total). On the validation pool the two rules are indistinguishable (-1.5 per pool applicant, CI [-14.8, +11.8]).
- Predicted profit on the holdout exceeds realised profit by **31.0M** (19% of the prediction). By source: default count (PD too low) 55%, good-loan yield curve 22%, loss-severity curve 23%. Refitting the two cashflow curves on the holdout itself (an oracle diagnostic, not deployable) removes the yield and severity errors (+0.4M, -0.9M) and leaves the PD effect (-18.5M).
- Cashflow drift is therefore a real but secondary cause. Refitting the curves on the validation window narrows the shortfall to -28.2 per pool applicant (CI [-38.2, -18.6]), still significantly negative.
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

**PD ablations: LendingClub's pricing and the cohort features.** The production PD excludes `int_rate`, `grade` and `sub_grade` by design (they are LendingClub's own risk score). Two refits are audited on the same real book (calibrator chosen on 2015-01..06, every number on 2015-07..12). Neither is a replacement for the production PD: the first consumes the incumbent's pricing, and the second is the EXP04 comparison arm.

| PD | ROC-AUC | ECE | O/E | O/E, lowest → highest APR quintile | Expected-profit rule vs 15% threshold, same PD (per pool applicant, 95% CI) | vs production-PD threshold |
|---|---|---|---|---|---|---|
| Production PD (primary features) | 0.6732 | 0.0117 | 1.082 | 0.67 → 1.38 | -36.9 [-47.1, -26.9] | -36.9 [-47.1, -26.9] |
| + LendingClub grade and APR (ablation) | 0.6934 | 0.0178 | 1.134 | 0.97 → 1.18 | -23.6 [-33.9, -13.4] | -25.8 [-35.8, -16.2] |
| + cohort features (EXP04 arm B) | 0.6704 | 0.0043 | 0.989 | 0.61 → 1.30 | -34.3 [-45.0, -23.8] | -27.7 [-38.1, -17.8] |

- With LendingClub's grade and APR in the PD, the observed/expected gradient across APR quintiles flattens to 0.97–1.18 and the expected-profit rule still loses to the 15% threshold under that PD (-23.6 per pool applicant, against -36.9 with the production PD).
- With the cohort features, overall observed/expected is 0.989 (level fixed), the gradient across APR quintiles is 0.61–1.30, and the expected-profit rule still loses to the 15% threshold under that PD (-34.3 per pool applicant).
- Fixing the level of the PD did not fix the decision: the rule still loses while an APR gradient in the PD's errors remains. The refit that flattens that gradient (LendingClub's pricing) narrows the shortfall more, though it does not close it, so pricing information along APR matters more here than calibration in the large.

**Is the cohort PD's gain a better PD or a stricter cut?** At a 15% cut-off the cohort-feature PD accepts a different number of loans than the production PD because its predictions sit at a different level (observed/expected 0.989 against 1.082). The control tightens the production PD's own cut-off until it accepts the same number of loans as the cohort PD at 15% (chosen from predicted PDs and a loan count only; no outcome is read). Realised profit on 2015-07..12, per pool applicant, paired bootstrap 95% CI:

| PD and rule | Loans accepted | PD cut-off | vs production PD at 15% | vs the volume-matched control |
|---|---|---|---|---|
| Production PD, 15% cut-off | 102,545 | 0.150 | reference | -6.8 [-12.9, -0.6] |
| + LendingClub grade and APR, 15% cut-off | 107,930 | 0.150 | -2.2 [-9.1, +4.7] | -9.0 [-16.4, -1.5] |
| + cohort features, 15% cut-off | 81,211 | 0.150 | +6.6 [+0.7, +12.7] | -0.2 [-5.1, +4.7] |
| Production PD, cut-off tightened to match the cohort PD's volume (control) | 81,211 | 0.123 | +6.8 [+0.6, +12.9] | reference |

The 15% cut-off on the cohort-feature PD accepts 81,211 loans against 102,545 and earns +6.6 [+0.7, +12.7] per pool applicant relative to the production PD's cut-off. The production PD tightened to the same volume earns +6.8 [+0.6, +12.9], and the cohort PD against that control earns -0.2 [-5.1, +4.7]. The gain is matched by a stricter cut on the production PD: at equal volume the two PDs are not distinguishable (the interval includes zero), and the cohort PD's ROC-AUC is 0.6704 against 0.6732. On this book the cohort features buy a better-levelled cut-off, with no detectable ranking advantage.
<!-- END:rl-section -->

## Null and negative results

These are findings, not failures to find something.

| Result | Where |
|---|---|
| The graph adds nothing to discrimination, and GraphSAGE is the weakest of the three arms | EXP04, EXP05 rows |
| The expected-profit rule loses to a plain 15% PD threshold on the real book | EXP03 row, profit audit |
| No learned policy beats the threshold in any capital condition, and some collapse to near-zero lending | EXP06 rows and the collapsed-learner note |
| The knapsack shadow-price policy does not help, and the "clairvoyant lambda" is not an upper bound | EXP06 notes |
| The VGAE relational-uncertainty signal fails the PD control | EXP09 row |
| The disagreement signal fails the pre-specified H1 rule | EXP10 row |
| Acting on the one kept signal does not beat an APR-matched placebo | EXP11 row |
| Giving the PD LendingClub's own pricing helps calibration along APR but does not rescue the rule | APR ablation |

## Corrections made during the audit

The first audit's conclusions that no longer stand (the earlier versions are in the git history) are:
test-set learning in the RL evaluation, comparisons that mixed real and counterfactual outcomes,
selections made on the reporting window (including the production calibrator), reciprocal
calibration ratios quoted under the wrong label, and unsupported causal explanations.

## Threats to validity

**Population and data**

- **One institution, one credit cycle.** US unsecured consumer loans in a long expansion; nothing is
  tested through a downturn.
- **Approved-only.** Everything estimates `P(default | approved)`. Reject inference is not done and
  the rejected-applications file is not analysed.
- **Marketplace economics.** The cost model is a simplification of an investor-funded platform.
- **Repeat borrowers are invisible** (`member_id` is scrubbed), so rows may not be independent and
  every interval is slightly too narrow.
- **A third-party mirror** of data LendingClub no longer publishes; checksums pin the copy.

**Evaluation design**

- **H1 and H2 are adjacent half-years of the same vintage year.** The holdout is untouched but not
  independent of 2015 conditions; it is not a later cycle.
- **Small selection window.** H1 has about 120,000 loans, and a sign test across ten PD strata has
  little power; the KEEP rule is conservative (the disagreement signal's H2 effect is clear but it
  fails H1).
- **The simulator replays one finite pool.** Episodes sample 1,200 applicants from the same
  holdout, so episodes overlap and the paired intervals understate population-level uncertainty.
- **Pre-specified, not tuned, but arbitrary:** the 15% threshold, the 80% budget fraction, the 90th
  percentile flag, the half-exposure cap, learner hyper-parameters.
- **Single runs.** The bandits and the Q-learner are trained once per capital condition, so the
  learners that stopped lending are single-run outcomes, not a measured failure rate. GraphSAGE, the
  autoencoder and the VGAE are single runs; only EXP04's LightGBM
  arms are repeated over seeds, and the seed varies the model fit only (the data, the cohort features
  and the single key shuffle are fixed).

**Method**

- **Training labels were not all knowable at the deployment date.** Most validation-window loans were
  still open at the cutoff; the deployment-realism audit measures what a resolved-only training set does to the PD model (a
  selected sample, so it does not isolate staleness), and
  the bandits' training on validation outcomes is optimistic for them. A delayed-feedback mode is not
  built.
- **The PD is blind to APR by design,** and the decision rule leans on APR; the audit and the
  ablation measure the consequence.
- **A part of the cohort-feature calibration gain is not specific to real cohort membership:** the
  shuffled-key control also lowers ECE.
- **The production calibrator is effectively a tie** between methods; its selection on H1 is
  principled but low-stakes.
- **Observed-exposure restriction.** The headline cannot speak to lending more or less than
  requested; that part rests on a model coupling and is a labelled sensitivity.

**Month granularity of the label gate**

Resolution and issue dates are both months, so the rule `resolved[j] <= issue[i]` admits a neighbour whose
outcome was booked in the same month as the applicant's issue. Whether that matters is measured, not assumed
(`scripts/analysis/analyze_label_gate_sensitivity.py`):

<!-- BEGIN:label-gate -->
**Same-month label rule.** Cohort label aggregates admit a neighbour when its resolution month is on or before the applicant's issue month, so a neighbour resolved in the same month is admitted. A strictly-before rule is more conservative. Rebuilding the cohort features under the strict rule and refitting (three LightGBM seeds, 2015-07..12, mean ± sd):

| Arm | ROC-AUC | ECE | O/E |
|---|---|---|---|
| Tabular only | 0.6715 ± 0.0005 | 0.0109 ± 0.0000 | 1.080 ± 0.000 |
| + cohort features, current rule (`resolved <= issue` month) | 0.6702 ± 0.0008 | 0.0048 ± 0.0006 | 0.993 ± 0.012 |
| + cohort features, strict rule (`resolved < issue` month) | 0.6707 ± 0.0005 | 0.0048 ± 0.0008 | 1.009 ± 0.016 |

Under the strict rule ECE moves by +0.0000 against the current rule (within seed noise), and the cohort features still lower ECE below the tabular arm. The headline cohort results use the current rule; this table states what the stricter rule would change.
<!-- END:label-gate -->

## What I would do next

1. Build the delayed-feedback bandit mode: release each reward at the loan's
   `outcome_observed_period`, and retrain the learners under that regime.
2. Replace the single 15% threshold comparison with a threshold chosen on H1 and evaluated on H2, so
   the strongest simple rule is tuned fairly.
3. Fit a PD that is calibrated within APR bands without consuming LendingClub's grade (for example a
   monotone constraint or an APR-conditional recalibration chosen on H1), then re-run the profit audit.
4. Repeat GraphSAGE, the autoencoder and the VGAE over seeds, and vary the key shuffle.
5. Move to rolling-origin evaluation so selection and reporting windows are further apart.
6. Use the rejected-applications file to quantify, if not correct, the selection bias.
7. Test on a downturn period or a second portfolio.
