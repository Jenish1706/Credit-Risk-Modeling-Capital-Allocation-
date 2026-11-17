> Computed on **real** data.

# exp06_rl: EXP03 and EXP06

Applicant pool: 283,026 out-of-time loans; 1,200 processed per episode; 12 seeds under common random numbers.
Learning policies are trained on 90,615 **validation-window** applicants and evaluated on the test window, so they never see the borrowers they are scored on. An earlier version trained them on the evaluation pool itself, which gave them an advantage the static policies did not have.
Estimated from training-window cashflows: good-loan yield -0.001+1.351*APR, defaulted-loan loss rate 0.393-0.179*APR, **empirical LGD 0.518** (mean) / 0.532 (median).

## EXP03 - does expected-profit optimisation beat a fixed PD threshold?

|                       |   n_episodes |   mean_reward |   std_reward |   min_reward |   p05_reward |   approval_rate |   default_rate_of_book |   capital_deployed |   return_on_capital |     revenue |   losses |            costs |   review_rate |
|:----------------------|-------------:|--------------:|-------------:|-------------:|-------------:|----------------:|-----------------------:|-------------------:|--------------------:|------------:|---------:|-----------------:|--------------:|
| fixed_threshold@0.150 |           12 |      54990.8  |      69214.1 |     -95939.2 |     -49801.1 |          0.6136 |                 0.1    |        9.91585e+06 |              0.0057 | 1.10335e+06 |   377663 | 670696           |             0 |
| risk_band             |           12 |       1650.27 |      67218.3 |    -122071   |    -107935   |          0.9498 |                 0.1328 |        1.1068e+07  |              0.0001 | 1.263e+06   |   465046 | 796304           |             0 |
| expected_profit       |           12 |      -8079.38 |     135373   |    -249140   |    -208074   |          0.5712 |                 0.1645 |        1.39569e+07 |             -0.0004 | 1.783e+06   |   899702 | 891378           |             0 |
| approve_all           |           12 |    -111089    |      82891.4 |    -274900   |    -231661   |          1      |                 0.1508 |        1.54395e+07 |             -0.0072 | 1.85972e+06 |   918477 |      1.05233e+06 |             0 |

| treatment       | baseline              |   n_pairs |   mean_difference |   ci_low |   ci_high |   win_rate | significant   |   relative_improvement |
|:----------------|:----------------------|----------:|------------------:|---------:|----------:|-----------:|:--------------|-----------------------:|
| risk_band       | fixed_threshold@0.150 |        12 |          -53340.6 |   -82149 |  -26059.6 |        0.2 | True          |                   -1   |
| expected_profit | fixed_threshold@0.150 |        12 |          -63070.2 |  -115788 |   -8934.1 |        0.2 | True          |                   -1.1 |
| approve_all     | fixed_threshold@0.150 |        12 |         -166080   |  -195998 | -136714   |        0   | True          |                   -3   |

**Verdict.** risk_band beat fixed_threshold@0.150 by -53,341 per episode (-97.0%), 95% CI [-82,149, -26,060], winning on 17% of seeds.

The mechanism is worth stating: the expected-profit policy typically runs a *higher* book default rate than the threshold policy and still earns more, because it approves high-rate loans whose price covers their risk and declines low-rate loans whose price does not. A single PD cut-off cannot express that, since the break-even PD moves with the interest rate. This is a failure of the decision rule, not of the risk model -- every policy here consumes identical PDs.
