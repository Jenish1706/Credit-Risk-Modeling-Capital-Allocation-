> Computed on **real** data.

# exp06_rl_sens: EXP03 and EXP06

Applicant pool: 162,236 loans from 2015-07..12 (H2, untouched holdout); 1,200 processed per episode; 30 seeds under common random numbers.
Learning policies are trained on 90,615 **validation-window** applicants and evaluated on the test window, so they never see the borrowers they are scored on. During evaluation episodes learning is switched off (`env.run(..., learn=False)`): parameters are frozen at what the validation window taught them, so no test-pool outcome is fed back. An earlier version trained them on the evaluation pool and then kept updating them on test outcomes after every decision, which static policies cannot do.
Estimated from training-window cashflows: good-loan yield -0.001+1.351*APR, defaulted-loan loss rate 0.393-0.179*APR, **empirical LGD 0.518** (mean) / 0.532 (median).

## EXP03 - does expected-profit optimisation beat a fixed PD threshold?

|                       |   n_episodes |   mean_reward |   std_reward |   min_reward |   p05_reward |   approval_rate |   default_rate_of_book |   capital_deployed |   return_on_capital |     revenue |   losses |            costs |   review_rate |
|:----------------------|-------------:|--------------:|-------------:|-------------:|-------------:|----------------:|-----------------------:|-------------------:|--------------------:|------------:|---------:|-----------------:|--------------:|
| fixed_threshold@0.150 |           30 |      61748.2  |      78133.7 |     -91264.4 |     -34684.5 |          0.6338 |                 0.0975 |        1.02241e+07 |              0.006  | 1.13942e+06 |   385930 | 691739           |             0 |
| expected_profit       |           30 |      32706.7  |     129717   |    -203657   |    -169385   |          0.5637 |                 0.1572 |        1.37599e+07 |              0.0024 | 1.75504e+06 |   843442 | 878895           |             0 |
| risk_band             |           30 |      -4959.68 |      64974.4 |    -114929   |     -91625   |          0.9532 |                 0.1297 |        1.12461e+07 |             -0.0005 | 1.26849e+06 |   466475 | 806977           |             0 |
| approve_all           |           30 |     -95984.3  |     111273   |    -286015   |    -250078   |          1      |                 0.1468 |        1.54102e+07 |             -0.0063 | 1.84283e+06 |   888142 |      1.05067e+06 |             0 |

| treatment       | baseline              |   n_pairs |   mean_difference |    ci_low |   ci_high |   win_rate | significant   |   relative_improvement |
|:----------------|:----------------------|----------:|------------------:|----------:|----------:|-----------:|:--------------|-----------------------:|
| expected_profit | fixed_threshold@0.150 |        30 |          -29041.6 |  -62824.4 |    4619.7 |        0.3 | False         |                   -0.5 |
| risk_band       | fixed_threshold@0.150 |        30 |          -66707.9 |  -84569.8 |  -49377.4 |        0.1 | True          |                   -1.1 |
| approve_all     | fixed_threshold@0.150 |        30 |         -157732   | -190223   | -126730   |        0   | True          |                   -2.6 |

**Verdict.** No policy beat fixed_threshold@0.150 significantly. Best was expected_profit at -29,042 per episode, 95% CI [-62,824, +4,620] -- the interval includes zero, so this is a null result.

Why the expected-profit rule loses to a PD threshold is analysed on the real book, at the observed exposure, in `scripts/analysis/analyze_profit_calibration.py` (`results/tables/profit_calibration_*.csv`). Nothing is asserted about the mechanism here: a simulated difference with a CI that straddles zero is a null result.

## EXP06 - can a learned policy beat static policies?

| condition | validation shadow price lambda | evaluation-pool lambda (diagnostic, not a ceiling) | verdict |
|---|---|---|---|
| unconstrained | 0.0000 | 0.0000 | linucb@alpha=1 beat expected_profit by +39,697 per episode (+121.4%), 95% CI [+13,802, +65,501], winning on 63% of seeds. |
| budget=0.60 | 0.0053 | 0.0000 | No policy beat expected_profit significantly. Best was expected_profit(lambda=0.0053) at +13,427 per episode, 95% CI [-4,512, +31,399] -- the interval includes zero, so this is a null result. |
| budget=0.35 | 0.0291 | 0.0168 | expected_profit(lambda_eval_pool=0.0168) beat expected_profit by +22,401 per episode (+62.8%), 95% CI [+3,860, +40,728], winning on 63% of seeds. |
| budget=0.20 | 0.0492 | 0.0394 | No policy beat expected_profit significantly. Best was expected_profit(lambda_eval_pool=0.0394) at +10,541 per episode, 95% CI [-17,016, +35,747] -- the interval includes zero, so this is a null result. |

### Control check

> CONTROL BREACHED: linucb@alpha=1, thompson significantly beat the myopic expected-profit policy with unconstrained capital, where it should be optimal. This does not support an RL claim. The cause is not established by this experiment (see profit_calibration_*.csv, the real-book predicted-vs-realised profit audit).

Learning-policy results in the unconstrained condition should therefore **not** be read as evidence for reinforcement learning. The myopic policy's PD and cashflow inputs are measurably mis-calibrated on the real book (see profit_calibration_*.csv), which this experiment does not isolate.

| policy                                   |   budget=0.20 |   budget=0.35 |   budget=0.60 |   unconstrained |
|:-----------------------------------------|--------------:|--------------:|--------------:|----------------:|
| approve_all                              |        -15482 |        -29379 |        -31463 |          -95984 |
| budget_aware_q(greedy)                   |          7500 |         13389 |         40487 |            6231 |
| expected_profit                          |         14376 |         35697 |         59659 |           32707 |
| expected_profit(lambda=0.0053)           |           nan |           nan |         73086 |             nan |
| expected_profit(lambda=0.0291)           |           nan |          4677 |           nan |             nan |
| expected_profit(lambda=0.0492)           |          9700 |           nan |           nan |             nan |
| expected_profit(lambda_eval_pool=0.0168) |           nan |         58098 |           nan |             nan |
| expected_profit(lambda_eval_pool=0.0394) |         24917 |           nan |           nan |             nan |
| fixed_threshold@0.150                    |         23140 |         45901 |         60208 |           61748 |
| linucb@alpha=1                           |         10438 |         32160 |         58280 |           72403 |
| risk_band                                |          1256 |         11990 |          6258 |           -4960 |
| thompson                                 |         17857 |         29243 |         19444 |           63814 |

**Reading this table.** The `unconstrained` column is the control. There, the myopic expected-profit rule is optimal by construction and the learned policies should not beat it -- if they appear to, the simulator has a bug. As the budget tightens, the shadow price of capital rises and the ranking should change: pricing capital correctly (either analytically via `expected_profit(lambda=...)` or by learning it via `budget_aware_q`) starts to matter more than per-applicant accuracy.

**What this does not establish.** That any of these policies is safe to deploy. Every number is generated inside a simulator whose assumptions are listed in docs/reference/ASSUMPTIONS.md, on a population of already-approved LendingClub borrowers from a single credit cycle. The defensible claim is a ranking under stated assumptions.
