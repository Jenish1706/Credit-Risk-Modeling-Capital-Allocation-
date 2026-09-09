> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# exp06_rl: EXP03 and EXP06

Applicant pool: 20,078 out-of-time loans; 1,000 processed per episode; 6 seeds under common random numbers.
Estimated from training-window cashflows: good-loan yield 0.177+0.317*APR, defaulted-loan loss rate 0.476-0.724*APR, **empirical LGD 0.584** (mean) / 0.628 (median).

## EXP03 - does expected-profit optimisation beat a fixed PD threshold?

|                       |   n_episodes |   mean_reward |   std_reward |   min_reward |   p05_reward |   approval_rate |   default_rate_of_book |   capital_deployed |   return_on_capital |     revenue |           losses |            costs |   review_rate |
|:----------------------|-------------:|--------------:|-------------:|-------------:|-------------:|----------------:|-----------------------:|-------------------:|--------------------:|------------:|-----------------:|-----------------:|--------------:|
| expected_profit       |            6 |        970018 |     169300   |       656946 |       728959 |          0.8282 |                 0.1609 |        1.66594e+07 |              0.0582 | 3.07632e+06 |      1.04083e+06 |      1.06548e+06 |             0 |
| approve_all           |            6 |        663291 |     134974   |       410745 |       470814 |          1      |                 0.1617 |        1.26697e+07 |              0.0523 | 2.34351e+06 | 814377           | 865840           |             0 |
| fixed_threshold@0.150 |            6 |        553474 |      74678.3 |       426566 |       455062 |          0.5765 |                 0.0916 |        6.51469e+06 |              0.0849 | 1.25673e+06 | 248705           | 454555           |             0 |
| risk_band             |            6 |        494040 |      74091.5 |       385093 |       404477 |          0.8695 |                 0.1301 |        7.18798e+06 |              0.0687 | 1.36738e+06 | 336798           | 536546           |             0 |

| treatment       | baseline              |   n_pairs |   mean_difference |   ci_low |   ci_high |   win_rate | significant   |   relative_improvement |
|:----------------|:----------------------|----------:|------------------:|---------:|----------:|-----------:|:--------------|-----------------------:|
| expected_profit | fixed_threshold@0.150 |         6 |          416543   | 333900   |    477950 |        1   | True          |                    0.8 |
| approve_all     | fixed_threshold@0.150 |         6 |          109817   |  46200.7 |    174948 |        0.8 | True          |                    0.2 |
| risk_band       | fixed_threshold@0.150 |         6 |          -59434.3 | -77521.3 |    -43181 |        0   | True          |                   -0.1 |

**Verdict.** expected_profit beat fixed_threshold@0.150 by +416,543 per episode (+75.3%), 95% CI [+333,900, +477,950], winning on 100% of seeds.

The mechanism is worth stating: the expected-profit policy typically runs a *higher* book default rate than the threshold policy and still earns more, because it approves high-rate loans whose price covers their risk and declines low-rate loans whose price does not. A single PD cut-off cannot express that, since the break-even PD moves with the interest rate. This is a failure of the decision rule, not of the risk model -- every policy here consumes identical PDs.

## EXP06 - can a learned policy beat static policies?

| condition | shadow price lambda | verdict |
|---|---|---|
| unconstrained | 0.0000 | budget_aware_q(greedy) beat expected_profit by -167,927 per episode (-17.3%), 95% CI [-221,554, -107,763], winning on 0% of seeds. |
| budget=0.35 | 0.0840 | expected_profit(lambda=0.0840) beat expected_profit by +159,623 per episode (+74.2%), 95% CI [+102,734, +217,512], winning on 100% of seeds. |

| policy                         |   budget=0.35 |   unconstrained |
|:-------------------------------|--------------:|----------------:|
| approve_all                    |        214434 |          663291 |
| budget_aware_q(greedy)         |        258897 |          802090 |
| expected_profit                |        214990 |          970018 |
| expected_profit(lambda=0.0840) |        374613 |             nan |
| fixed_threshold@0.150          |        346101 |          553474 |
| linucb@alpha=1                 |        242681 |          304611 |
| risk_band                      |        300729 |          494040 |
| thompson                       |        241674 |          672904 |

**Reading this table.** The `unconstrained` column is the control. There, the myopic expected-profit rule is optimal by construction and the learned policies should not beat it -- if they appear to, the simulator has a bug. As the budget tightens, the shadow price of capital rises and the ranking should change: pricing capital correctly (either analytically via `expected_profit(lambda=...)` or by learning it via `budget_aware_q`) starts to matter more than per-applicant accuracy.

**What this does not establish.** That any of these policies is safe to deploy. Every number is generated inside a simulator whose assumptions are listed in docs/ASSUMPTIONS.md, on a population of already-approved LendingClub borrowers from a single credit cycle. The defensible claim is a ranking under stated assumptions.
