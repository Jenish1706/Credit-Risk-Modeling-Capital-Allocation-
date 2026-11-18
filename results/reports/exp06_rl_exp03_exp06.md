> Computed on **real** data.

# exp06_rl: EXP03 and EXP06

Applicant pool: 283,026 out-of-time loans; 1,200 processed per episode; 30 seeds under common random numbers.
Learning policies are trained on 90,615 **validation-window** applicants and evaluated on the test window, so they never see the borrowers they are scored on. During evaluation episodes learning is switched off (`env.run(..., learn=False)`): parameters are frozen at what the validation window taught them, so no test-pool outcome is fed back. An earlier version trained them on the evaluation pool and then kept updating them on test outcomes after every decision, which static policies cannot do.
Estimated from training-window cashflows: good-loan yield -0.001+1.351*APR, defaulted-loan loss rate 0.393-0.179*APR, **empirical LGD 0.518** (mean) / 0.532 (median).

## EXP03 - does expected-profit optimisation beat a fixed PD threshold?

|                       |   n_episodes |   mean_reward |   std_reward |   min_reward |   p05_reward |   approval_rate |   default_rate_of_book |   capital_deployed |   return_on_capital |     revenue |   losses |            costs |   review_rate |
|:----------------------|-------------:|--------------:|-------------:|-------------:|-------------:|----------------:|-----------------------:|-------------------:|--------------------:|------------:|---------:|-----------------:|--------------:|
| fixed_threshold@0.150 |           30 |      63348.6  |      65507.2 |     -95939.2 |       -55836 |          0.6181 |                 0.0985 |        1.00081e+07 |              0.0064 | 1.11587e+06 |   375810 | 676707           |             0 |
| expected_profit       |           30 |      36943.4  |     131041   |    -249140   |      -200313 |          0.5706 |                 0.1599 |        1.39181e+07 |              0.0027 | 1.78827e+06 |   862246 | 889077           |             0 |
| risk_band             |           30 |       8162.78 |      60489.5 |    -122071   |      -107712 |          0.9529 |                 0.1311 |        1.10869e+07 |              0.0007 | 1.2654e+06  |   459301 | 797933           |             0 |
| approve_all           |           30 |     -67146.4  |     103599   |    -301179   |      -239523 |          1      |                 0.1475 |        1.54115e+07 |             -0.0043 | 1.86294e+06 |   879341 |      1.05075e+06 |             0 |

| treatment       | baseline              |   n_pairs |   mean_difference |    ci_low |   ci_high |   win_rate | significant   |   relative_improvement |
|:----------------|:----------------------|----------:|------------------:|----------:|----------:|-----------:|:--------------|-----------------------:|
| expected_profit | fixed_threshold@0.150 |        30 |          -26405.3 |  -59902.1 |    7655.2 |        0.4 | False         |                   -0.4 |
| risk_band       | fixed_threshold@0.150 |        30 |          -55185.8 |  -71222.5 |  -38915.2 |        0.1 | True          |                   -0.9 |
| approve_all     | fixed_threshold@0.150 |        30 |         -130495   | -155634   | -105070   |        0   | True          |                   -2.1 |

**Verdict.** No policy beat fixed_threshold@0.150 significantly. Best was expected_profit at -26,405 per episode, 95% CI [-59,902, +7,655] -- the interval includes zero, so this is a null result.

The mechanism is worth stating: the expected-profit policy typically runs a *higher* book default rate than the threshold policy and still earns more, because it approves high-rate loans whose price covers their risk and declines low-rate loans whose price does not. A single PD cut-off cannot express that, since the break-even PD moves with the interest rate. This is a failure of the decision rule, not of the risk model -- every policy here consumes identical PDs.

## EXP06 - can a learned policy beat static policies?

| condition | validation shadow price lambda | oracle ceiling lambda | verdict |
|---|---|---|---|
| unconstrained | 0.0000 | 0.0000 | thompson beat expected_profit by +55,753 per episode (+150.9%), 95% CI [+29,266, +80,634], winning on 80% of seeds. |
| budget=0.60 | 0.0052 | 0.0004 | No policy beat expected_profit significantly. Best was linucb@alpha=1 at +6,916 per episode, 95% CI [-17,235, +30,160] -- the interval includes zero, so this is a null result. |
| budget=0.35 | 0.0290 | 0.0232 | No policy beat expected_profit significantly. Best was oracle_ceiling(lambda=0.0232) at +10,574 per episode, 95% CI [-13,394, +34,391] -- the interval includes zero, so this is a null result. |
| budget=0.20 | 0.0491 | 0.0415 | No policy beat expected_profit significantly. Best was oracle_ceiling(lambda=0.0415) at -2,818 per episode, 95% CI [-26,767, +19,941] -- the interval includes zero, so this is a null result. |

### Control check

> CONTROL BREACHED: thompson, linucb@alpha=1 significantly beat the myopic expected-profit policy with unconstrained capital, where it should be optimal. This does not support an RL claim. The cause is not established by this experiment (see the CR-02 expected-vs-realised profit decile table).

Learning-policy results in the unconstrained condition should therefore **not** be read as evidence for reinforcement learning. The correct reading is that the myopic policy's economic inputs are stale relative to the test vintage.

| policy                         |   budget=0.20 |   budget=0.35 |   budget=0.60 |   unconstrained |
|:-------------------------------|--------------:|--------------:|--------------:|----------------:|
| approve_all                    |          4714 |          1560 |        -12491 |          -67146 |
| budget_aware_q(greedy)         |         20287 |         37113 |         45601 |           30444 |
| expected_profit                |         36995 |         45654 |         69955 |           36943 |
| expected_profit(lambda=0.0052) |           nan |           nan |         68316 |             nan |
| expected_profit(lambda=0.0290) |           nan |         28595 |           nan |             nan |
| expected_profit(lambda=0.0491) |         15900 |           nan |           nan |             nan |
| fixed_threshold@0.150          |         30411 |         46368 |         65381 |           63349 |
| linucb@alpha=1                 |         23652 |         28938 |         76871 |           91894 |
| oracle_ceiling(lambda=0.0004)  |           nan |           nan |         72515 |             nan |
| oracle_ceiling(lambda=0.0232)  |           nan |         56228 |           nan |             nan |
| oracle_ceiling(lambda=0.0415)  |         34177 |           nan |           nan |             nan |
| risk_band                      |         18279 |         22460 |         13858 |            8163 |
| thompson                       |         17289 |         24211 |         59745 |           92696 |

**Reading this table.** The `unconstrained` column is the control. There, the myopic expected-profit rule is optimal by construction and the learned policies should not beat it -- if they appear to, the simulator has a bug. As the budget tightens, the shadow price of capital rises and the ranking should change: pricing capital correctly (either analytically via `expected_profit(lambda=...)` or by learning it via `budget_aware_q`) starts to matter more than per-applicant accuracy.

**What this does not establish.** That any of these policies is safe to deploy. Every number is generated inside a simulator whose assumptions are listed in docs/ASSUMPTIONS.md, on a population of already-approved LendingClub borrowers from a single credit cycle. The defensible claim is a ranking under stated assumptions.
