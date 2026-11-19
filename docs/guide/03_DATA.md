# 03 · Data

## Source and licence

**LendingClub accepted-loans file, 2007 to 2018Q4**, obtained from the Kaggle mirror
`wordsforthewise/lending-club` under licence **CC0**. LendingClub no longer publishes the original
files, so the mirror is a third party; `scripts/download_data.py` records a SHA-256 for each file in
`data/raw/checksums.json` so results stay reproducible against a fixed copy.

Raw data is not committed. Without a Kaggle token, `python scripts/download_data.py --synthetic`
writes a schema-faithful synthetic extract; every artifact built from it is stamped
`data_source: synthetic` and is not reported as a finding. The optional rejected-applications file
(`--with-rejected`) is downloadable but **no analysis in this repository uses it**.

**Why this dataset** (full evaluation of alternatives in [DATASET.md](../DATASET.md)): it has real
calendar time (needed for out-of-time validation and drift), real loan economics (cashflows from
which realised profit and loss are computed rather than assumed), and two usable relational keys
(3-digit ZIP, employer text). Home Credit was rejected because it has no absolute calendar date.

## The modelling population

Only **36-month loans issued 2010 to 2015** are labelled. Every such loan has had at least 36
months to resolve before the data cutoff, so there is no censoring and no term-mix confound
(60-month loans issued in 2015 would not have resolved).

<!-- BEGIN:data-profile -->
| Stage | Rows |
|---|---|
| Read from the raw accepted-loans file | 2,260,701 |
| After the 36-month term filter | 1,609,754 |
| After dropping loans issued outside every split window | 1,609,754 |
| Dropped: non-terminal or unmapped status inside a matured window | 147 |
| **Modelling table** | **1,609,607** |

Rows with a malformed id are dropped while reading: 33.

| Issue year | Default rate |
|---|---|
| 2010 | 10.92% |
| 2011 | 10.63% |
| 2012 | 13.58% |
| 2013 | 12.33% |
| 2014 | 13.73% |
| 2015 | 14.89% |
<!-- END:data-profile -->

<!-- BEGIN:data-table -->
| Split | Window | Loans | Default rate |
|---|---|---|---|
| history (graph only) | 2007-06 … 2009-12 | 8,277 | unlabelled by design |
| **train** | 2010-01 … 2014-06 | 239,104 | 12.67% |
| **valid** | 2014-07 … 2014-12 | 90,615 | 14.13% |
| **test** (out-of-time) | 2015-01 … 2015-12 | 283,026 | 14.89% |
| ↳ test H1 (all selections) | 2015-01 … 2015-06 | 120,790 | 15.12% |
| ↳ test H2 (untouched holdout, headline) | 2015-07 … 2015-12 | 162,236 | 14.71% |
| monitor (production traffic) | 2016-01 … 2018-12 | 988,585 | unlabelled by design |
<!-- END:data-table -->

The history window (2007-06 to 2009-12) is never modelled; it supplies past neighbours for the
graph. The monitoring window (2016 to 2018) has no mature outcomes by design. The test year is
split again in time for selection versus reporting (H1 and H2, see
[04_METHODOLOGY.md](04_METHODOLOGY.md)).

## Tables actually used

| Table | Written by | Rows | Contents |
|---|---|---|---|
| `data/processed/loans.parquet` | [`prepare_data.py`](../../scripts/prepare_data.py) | see above | one row per loan: raw application fields, cashflow fields (for realised outcomes only), `issue_period`, `outcome_observed_period`, `split`, target `default` |
| `data/processed/cohort_features.parquet` | [`build_graph.py`](../../scripts/build_graph.py) | one per loan | smoothed cohort default rates, counts, evidence weights, cohort means for the ZIP3 and employer relations |
| `data/processed/graph.npz` | `build_graph.py` | one per loan | time-respecting neighbour indices and masks |

Target. `default = 1` for `Charged Off`, `Default` and the "does not meet the credit policy"
charged-off status; `default = 0` for `Fully Paid` and its credit-policy variant.

Column groups in `loans.parquet`, as classified in
[`data/schema.py`](../../src/credit_risk/data/schema.py):

| Group | Role | Examples |
|---|---|---|
| Decision-time numeric and categorical features | model inputs | `loan_amnt`, `dti`, `annual_inc`, `fico_range_low`, `purpose`, `home_ownership`, `addr_state` |
| Vintage-sensitive numeric features | inputs, with an availability guard | the bureau block introduced around 2012 to 2015 |
| Pricing: `int_rate`, `grade`, `sub_grade` | **excluded from the primary PD**; used by the economics layer | LendingClub's own risk score and the price derived from it |
| Exposure | funding amounts | `funded_amnt`, `funded_amnt_inv` |
| Cashflow | realised outcomes only | `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee` |
| Post-origination | **forbidden as features** | `last_fico_range_*`, `last_pymnt_*`, `out_prncp`, `settlement_*`, `hardship_*` |
| Identifiers and dropped | not used | `id`, `member_id`, `url`, free-text `desc`, `title` |

The deny-list (`FORBIDDEN_AS_FEATURES`: post-origination, identifier, exposure, dropped columns and
the target) is enforced by `assert_no_leakage` on every feature build and by tests. The column-by-
column ruling is in [DATA_DICTIONARY.md](../DATA_DICTIONARY.md); the reasoning is in
[LEAKAGE.md](../LEAKAGE.md).

## Known quality problems and how each is handled

| Problem | Evidence | Handling |
|---|---|---|
| Rows with a malformed id | counted in the profile above | dropped while reading |
| Non-terminal status inside a matured window | counted in the profile above (a tiny share of rows) | dropped, and the count is reported; if it were material (over 0.5%) the "no censoring" assumption would be revisited |
| `member_id` is entirely null | scrubbed by LendingClub | repeat borrowers cannot be detected, so rows may not be independent and every interval is slightly too narrow; the graph is built from shared attributes instead |
| Informative missingness (`mths_since_last_delinq` and similar) | missing means "no such event" | explicit indicator plus sentinel, never mean-imputed (`INFORMATIVE_MISSING_NUMERIC`) |
| Bureau block missing in early vintages | the availability-shift table in `results/reports/data_profile.md` | an availability guard drops columns whose missingness shifts more than a tolerance between train and test, so a model cannot learn "is this populated?" as a vintage proxy |
| `emp_title` changed meaning (employer name before late 2013, job title after) | LendingClub's data dictionary | the employer relation is kept configurable, ablated separately and never presented as a clean occupation relation |
| Outliers and sentinels in numeric features (for example `dti`) | profile | numeric features are clipped to the training 0.1% and 99.9% quantiles (`FeatureSpec.clip_quantiles`) |
| Approved-only population | structural | the model estimates `P(default | approved)`, not `P(default | applied)`; reject inference is not implemented because the propensity data is not published |
| One institution, one credit cycle | structural | stated as a limitation; nothing is tested through a downturn |
| Vintage deterioration | the default-rate-by-year table above | real and the reason for out-of-time splits; also why calibration drifts |

## What is excluded, and why

| Excluded | Reason |
|---|---|
| 60-month loans | would not have matured, so outcomes are censored |
| Loans issued after 2015 from the labelled set | monitoring window; outcomes immature at the 2018 cutoff |
| Post-origination columns as features | outcome fields: they encode the target |
| `grade`, `sub_grade`, `int_rate` from the primary PD | they are the incumbent's own risk score; an ablation with them is reported separately and labelled |
| `installment` from the primary PD | recoverable from `loan_amnt`, `int_rate` and `term`, so it would smuggle the incumbent's score back in |
| Free-text `desc` and `title` | not used |
| The rejected-applications file | no analysis here uses it |
