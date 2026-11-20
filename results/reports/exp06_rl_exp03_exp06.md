> Computed on **real** data.

# exp06_rl: EXP03 and EXP06

Applicant pool: 162,236 loans from 2015-07..12 (H2, untouched holdout); 1,200 processed per episode; 30 seeds under common random numbers.
Learning policies are trained on 90,615 **validation-window** applicants and evaluated on the test window, so they never see the borrowers they are scored on. During evaluation episodes learning is switched off (`env.run(..., learn=False)`): parameters are frozen at what the validation window taught them, so no test-pool outcome is fed back. An earlier version trained them on the evaluation pool and then kept updating them on test outcomes after every decision, which static policies cannot do.
Estimated from training-window cashflows: good-loan yield -0.001+1.351*APR, defaulted-loan loss rate 0.393-0.179*APR, **empirical LGD 0.518** (mean) / 0.532 (median).

## EXP03 - does expected-profit optimisation beat a fixed PD threshold?

|                       |   n_episodes |   mean_reward |   std_reward |   min_reward |   p05_reward |   approval_rate |   default_rate_of_book |   capital_deployed |   return_on_capital |     revenue |   losses |            costs |   review_rate |
|:----------------------|-------------:|--------------:|-------------:|-------------:|-------------:|----------------:|-----------------------:|-------------------:|--------------------:|------------:|---------:|-----------------:|--------------:|
| fixed_threshold@0.150 |           30 |       61748.2 |      78133.7 |     -91264.4 |     -34684.5 |          0.6338 |                 0.0975 |        1.02241e+07 |              0.006  | 1.13942e+06 |   385930 | 691739           |             0 |
| expected_profit       |           30 |       33485.9 |      97338.6 |    -145037   |    -114455   |          0.4943 |                 0.1525 |        8.85956e+06 |              0.0038 | 1.15621e+06 |   533186 | 589540           |             0 |
| risk_band             |           30 |      -46113.7 |      95289   |    -187713   |    -176610   |          0.9532 |                 0.1369 |        1.47505e+07 |             -0.0031 | 1.7523e+06  |   793445 |      1.00497e+06 |             0 |
| approve_all           |           30 |      -95984.3 |     111273   |    -286015   |    -250078   |          1      |                 0.1468 |        1.54102e+07 |             -0.0063 | 1.84283e+06 |   888142 |      1.05067e+06 |             0 |

| treatment       | baseline              |   n_pairs |   mean_difference |    ci_low |   ci_high |   win_rate | significant   |   relative_improvement |
|:----------------|:----------------------|----------:|------------------:|----------:|----------:|-----------:|:--------------|-----------------------:|
| expected_profit | fixed_threshold@0.150 |        30 |          -28262.4 |  -54837.1 |   -2616.1 |        0.3 | True          |                   -0.5 |
| risk_band       | fixed_threshold@0.150 |        30 |         -107862   | -133989   |  -82728.1 |        0.1 | True          |                   -1.7 |
| approve_all     | fixed_threshold@0.150 |        30 |         -157732   | -190223   | -126730   |        0   | True          |                   -2.6 |

**Verdict.** No policy beat fixed_threshold@0.150. Even the best, expected_profit, was significantly worse at -28,262 per episode, 95% CI [-54,837, -2,616].

Why the expected-profit rule loses to a PD threshold is analysed on the real book, at the observed exposure, in `scripts/analyze_profit_calibration.py` (`results/tables/profit_calibration_*.csv`). Nothing is asserted about the mechanism here: a simulated difference with a CI that straddles zero is a null result.

## EXP06 - can a learned policy beat static policies?

| condition | validation shadow price lambda | evaluation-pool lambda (diagnostic, not a ceiling) | verdict |
|---|---|---|---|
| unconstrained | 0.0000 | 0.0000 | fixed_threshold@0.150 beat expected_profit by +28,262 per episode (+84.4%), 95% CI [+2,616, +54,837], winning on 67% of seeds. |
| budget=0.60 | 0.0053 | 0.0000 | fixed_threshold@0.150 beat expected_profit by +26,396 per episode (+78.1%), 95% CI [+654, +52,743], winning on 70% of seeds. |
| budget=0.35 | 0.0291 | 0.0168 | No policy beat expected_profit significantly. Best was thompson at -178 per episode, 95% CI [-16,318, +16,402] -- the interval includes zero, so this is a null result. |
| budget=0.20 | 0.0492 | 0.0394 | No policy beat expected_profit significantly. Best was fixed_threshold@0.150 at +994 per episode, 95% CI [-14,272, +15,672] -- the interval includes zero, so this is a null result. |

| policy                                   |   budget=0.20 |   budget=0.35 |   budget=0.60 |   unconstrained |
|:-----------------------------------------|--------------:|--------------:|--------------:|----------------:|
| approve_all                              |        -15482 |        -29379 |        -31463 |          -95984 |
| budget_aware_q(greedy)                   |         22990 |         20162 |         15182 |             268 |
| expected_profit                          |         22146 |         51909 |         33812 |           33486 |
| expected_profit(lambda=0.0053)           |           nan |           nan |         38649 |             nan |
| expected_profit(lambda=0.0291)           |           nan |          2041 |           nan |             nan |
| expected_profit(lambda=0.0492)           |          3723 |           nan |           nan |             nan |
| expected_profit(lambda_eval_pool=0.0168) |           nan |         34191 |           nan |             nan |
| expected_profit(lambda_eval_pool=0.0394) |          5515 |           nan |           nan |             nan |
| fixed_threshold@0.150                    |         23140 |         45901 |         60208 |           61748 |
| linucb@alpha=1                           |          8690 |          -202 |         58557 |           44347 |
| risk_band                                |         -8350 |         -4306 |          4568 |          -46114 |
| thompson                                 |           -38 |         51731 |         60048 |           50883 |

**Reading this table.** The `unconstrained` column is the control. There, the myopic expected-profit rule is optimal by construction and the learned policies should not beat it -- if they appear to, the simulator has a bug. As the budget tightens, the shadow price of capital rises and the ranking should change: pricing capital correctly (either analytically via `expected_profit(lambda=...)` or by learning it via `budget_aware_q`) starts to matter more than per-applicant accuracy.

**What this does not establish.** That any of these policies is safe to deploy. Every number is generated inside a simulator whose assumptions are listed in docs/ASSUMPTIONS.md, on a population of already-approved LendingClub borrowers from a single credit cycle. The defensible claim is a ranking under stated assumptions.
