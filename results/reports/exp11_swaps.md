> Computed on **real** data.

# EXP11: does acting on blind-spot signals improve the book?

Applicant pool: 283,026 out-of-time loans (2015 vintage). Capital budget $1,673,463,380, 80% of profitable demand.
Flagged as blind spots: 53,666 borrowers (19.0%) above the 90th percentile of the anomaly **or** disagreement signal.

## Portfolios at equal budget, measured on observed cashflows

| policy            |   n_loans |   capital_deployed |   observed_profit |   return_on_capital |   default_rate |   mean_apr |   budget_utilisation |
|:------------------|----------:|-------------------:|------------------:|--------------------:|---------------:|-----------:|---------------------:|
| A_pd_ranked       |    112078 |        1.67346e+09 |       1.17338e+08 |              0.0701 |         0.1688 |     0.1333 |               1      |
| B_blindspot_aware |    112792 |        1.63777e+09 |       1.15833e+08 |              0.0707 |         0.1598 |     0.1278 |               0.9787 |
| C_exposure_capped |    126581 |        1.67346e+09 |       1.18028e+08 |              0.0705 |         0.1638 |     0.13   |               1      |

## Verdict

B_blindspot_aware changed realised profit by **$-1,504,723** on a $1,673,462,150 book, deploying $-35,694,050 of capital (a +0.06% change in return on capital). That is **below 50bp of return on capital and should be read as no material difference in profitability**. The book default rate moved from 16.88% to 15.98% at essentially unchanged return, so the defensible claim is risk reduction at constant profitability -- not a profit improvement.

The swap involved 23,287 loans out (realised return +6.66%, default 16.87%) and 24,001 loans in (realised return +6.93%, default 12.63%).

Policy C, which caps flagged borrowers at 50% exposure rather than refusing them, changed return on capital by +0.04% against Policy A.

## Placebo Controls (Attribution Check)

To test whether the default reduction is attributable to the blind-spot signals or merely the mechanical consequence of excluding loans and refilling the portfolio, 200 draws of two placebo arms were evaluated at the exact same capital budget:

1. **Random 19% Excluded:**
   - Mean Default Rate: 16.01% (95% CI: [15.94%, 16.09%])
   - Mean RoC: 6.96% (95% CI: [6.90%, 7.02%])
   - Signal beats random placebo: **False**

2. **APR-Decile Matched 19% Excluded:**
   - Mean Default Rate: 16.08% (95% CI: [15.98%, 16.16%])
   - Mean RoC: 6.97% (95% CI: [6.91%, 7.04%])
   - Signal beats APR-matched placebo: **False**

**Conclusion:** Policy B's default rate (15.98%) falls inside the placebo intervals. The default reduction is **not attributable to the blind-spot signals**. Refilling the budget after any arbitrary 19% exclusion draws in lower-APR loans that naturally have lower default rates.

## What this does and does not show

- Profit here is **observed**: `total_pymnt - collection_recovery_fee - funded_amnt` for every matured loan (total_pymnt already includes recoveries). No loss-given-default is assumed.
- The comparison is between **whole portfolios at the same budget**, not between the two swap sets. Swap sets need not deploy equal capital, so differencing their profits is not a portfolio result.
- **Return on capital is the comparable quantity.** Profit alone rewards whichever book happens to deploy more capital.
- This is a *retrospective reallocation* on an already-approved population. It does not establish what would happen to borrowers LendingClub declined, and it is not evidence that the policy is safe to deploy.

> **Correction.** An earlier version of this experiment computed profit as `amount x apr x 1.55 - amount x 0.5 x default` and compared the two swap sets rather than the portfolios. That formula understated realised book profit by roughly a factor of 2.4, and it reported a **$9.8M loss** as a *'net increase in profit'* -- the conclusion was asserted regardless of the sign of the number printed above it. Measured on observed cashflows at equal budget, the effect is a fraction of a percentage point of return on capital in either direction, i.e. not a profit result at all. See `docs/AUDIT_REPORT.md`.
