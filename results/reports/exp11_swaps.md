> Computed on **real** data.

# EXP11: does acting on blind-spot signals improve the book?

**Headline window: 2015-07..12 (H2, untouched holdout)**, 162,236 loans. Signals used: anomaly (KEEP on 2015-01..06). Flag cut-points are the 90th percentile of each signal on 2015-01..06 (anomaly 0.1466). Capital budget $948,159,060, 80% of profitable demand. Flagged: 17,233 borrowers (10.6%).

## Portfolios at equal budget, measured on observed cashflows (2015-07..12)

| policy            |   n_loans |   capital_deployed |   observed_profit |   return_on_capital |   default_rate |   mean_apr |   budget_utilisation |
|:------------------|----------:|-------------------:|------------------:|--------------------:|---------------:|-----------:|---------------------:|
| A_pd_ranked       |     62820 |        9.48154e+08 |       6.46693e+07 |              0.0682 |         0.1662 |     0.1311 |                    1 |
| B_blindspot_aware |     64127 |        9.48146e+08 |       6.5142e+07  |              0.0687 |         0.1588 |     0.1268 |                    1 |
| C_exposure_capped |     67266 |        9.48157e+08 |       6.47703e+07 |              0.0683 |         0.1639 |     0.1294 |                    1 |

Full-2015 continuity column (283,026 loans, flagged 29,312):

| policy            |   n_loans |   capital_deployed |   observed_profit |   return_on_capital |   default_rate |   mean_apr |   budget_utilisation |
|:------------------|----------:|-------------------:|------------------:|--------------------:|---------------:|-----------:|---------------------:|
| A_pd_ranked       |    112292 |        1.67724e+09 |       1.17891e+08 |              0.0703 |         0.168  |     0.133  |                    1 |
| B_blindspot_aware |    114282 |        1.67723e+09 |       1.18937e+08 |              0.0709 |         0.1602 |     0.1286 |                    1 |
| C_exposure_capped |    119957 |        1.67724e+09 |       1.18221e+08 |              0.0705 |         0.1654 |     0.1312 |                    1 |

## Verdict

B_blindspot_aware changed realised profit by **$+472,752** on a $948,153,975 book, deploying $-8,400 of capital (a +0.05% change in return on capital). That is **below 50bp of return on capital and should be read as no material difference in profitability**. The book default rate moved from 16.62% to 15.88% at essentially unchanged return, so the defensible claim is risk reduction at constant profitability -- not a profit improvement.

The swap involved 7,384 loans out (realised return +6.37%, default 18.22%) and 8,691 loans in (realised return +6.77%, default 12.48%).

Policy C, which caps flagged borrowers at 50% exposure rather than refusing them, changed return on capital by +0.01% against Policy A.

## Placebo Controls (Attribution Check)

To test whether the default reduction is attributable to the blind-spot signals or merely the mechanical consequence of excluding loans and refilling the portfolio, 200 draws of two placebo arms were evaluated at the exact same capital budget (2015-07..12):

1. **Random exclusion (same share flagged):**
   - Mean Default Rate: 16.17% (95% CI: [16.08%, 16.26%])
   - Mean RoC: 6.81% (95% CI: [6.73%, 6.87%])
   - Signal beats random placebo: **True**

2. **APR-decile matched exclusion:**
   - Mean Default Rate: 15.97% (95% CI: [15.87%, 16.05%])
   - Mean RoC: 6.84% (95% CI: [6.77%, 6.91%])
   - Signal beats APR-matched placebo: **False**

**Conclusion:** Policy B's default rate (15.88%) lies outside the random-exclusion interval but inside the APR-matched interval: the reduction is what excluding the same share of loans with a matching APR profile produces, so it is **not attributable to the signal's selectivity** beyond the shift toward lower-APR loans.

## What this does and does not show

- Profit here is **observed**: `total_pymnt - collection_recovery_fee - funded_amnt` for every matured loan (total_pymnt already includes recoveries). No loss-given-default is assumed.
- The comparison is between **whole portfolios at the same budget**, not between the two swap sets. Swap sets need not deploy equal capital, so differencing their profits is not a portfolio result.
- **Return on capital is the comparable quantity.** Profit alone rewards whichever book happens to deploy more capital.
- This is a *retrospective reallocation* on an already-approved population. It does not establish what would happen to borrowers LendingClub declined, and it is not evidence that the policy is safe to deploy.

> **Correction.** An earlier version of this experiment computed profit as `amount x apr x 1.55 - amount x 0.5 x default` and compared the two swap sets rather than the portfolios. That formula understated realised book profit by roughly a factor of 2.4, and it reported a **$9.8M loss** as a *'net increase in profit'* -- the conclusion was asserted regardless of the sign of the number printed above it. The earlier version is in the git history.
