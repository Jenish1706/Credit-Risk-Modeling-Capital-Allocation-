> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# Data profile

Rows read from raw file: **150,000**
After the 36-month term filter: **108,139**
After the issue-window filter: **108,139**
Final modelling table: **108,085** rows x 162 columns

## Rows dropped

- malformed / non-numeric id: 0
- non-terminal status inside a matured window: 54
- unmapped loan_status: 0

## Splits

| split | window | rows | default rate |
|---|---|---|---|
| history | 2007-06 .. 2009-12 | 1,447 | n/a (unlabelled by design) |
| train | 2010-01 .. 2014-06 | 20,440 | 0.1472 |
| valid | 2014-07 .. 2014-12 | 6,957 | 0.1575 |
| test | 2015-01 .. 2015-12 | 20,078 | 0.1657 |
| monitor | 2016-01 .. 2018-12 | 59,163 | n/a (unlabelled by design) |

## Default rate by vintage

| year | default rate |
|---|---|
| 2010 | 0.1338 |
| 2011 | 0.1439 |
| 2012 | 0.1331 |
| 2013 | 0.1505 |
| 2014 | 0.1578 |
| 2015 | 0.1657 |

## Highest missingness

| column | missing |
|---|---|
| `member_id` | 100.0% |
| `url` | 100.0% |
| `desc` | 100.0% |
| `title` | 100.0% |
| `next_pymnt_d` | 100.0% |
| `annual_inc_joint` | 100.0% |
| `dti_joint` | 100.0% |
| `verification_status_joint` | 100.0% |
| `revol_bal_joint` | 100.0% |
| `sec_app_fico_range_low` | 100.0% |
| `sec_app_fico_range_high` | 100.0% |
| `sec_app_earliest_cr_line` | 100.0% |
| `sec_app_inq_last_6mths` | 100.0% |
| `sec_app_mort_acc` | 100.0% |
| `sec_app_open_acc` | 100.0% |
| `sec_app_revol_util` | 100.0% |
| `sec_app_open_act_il` | 100.0% |
| `sec_app_num_rev_accts` | 100.0% |
| `sec_app_chargeoff_within_12_mths` | 100.0% |
| `sec_app_collections_12_mths_ex_med` | 100.0% |
| `sec_app_mths_since_last_major_derog` | 100.0% |
| `hardship_type` | 100.0% |
| `hardship_reason` | 100.0% |
| `hardship_status` | 100.0% |
| `hardship_loan_status` | 100.0% |

## Availability shift (train -> test)

Columns whose *missingness* moves between windows. These partly encode the
origination date, so a model can learn 'is this populated?' as a vintage proxy.
The feature builder's availability guard drops them; see docs/LEAKAGE.md section 5.

| column | shift in missing rate |
|---|---|
| `acc_open_past_24mths` | -30.5% |
| `avg_cur_bal` | -30.5% |
| `bc_open_to_buy` | -30.5% |
| `bc_util` | -30.5% |
| `mo_sin_old_il_acct` | -30.5% |
| `mo_sin_old_rev_tl_op` | -30.5% |
| `mo_sin_rcnt_rev_tl_op` | -30.5% |
| `mo_sin_rcnt_tl` | -30.5% |
| `mort_acc` | -30.5% |
| `num_accts_ever_120_pd` | -30.5% |
| `num_actv_bc_tl` | -30.5% |
| `num_actv_rev_tl` | -30.5% |
| `num_bc_sats` | -30.5% |
| `num_bc_tl` | -30.5% |
| `num_il_tl` | -30.5% |
| `num_op_rev_tl` | -30.5% |
| `num_rev_accts` | -30.5% |
| `num_rev_tl_bal_gt_0` | -30.5% |
| `num_sats` | -30.5% |
| `num_tl_120dpd_2m` | -30.5% |
| `num_tl_30dpd` | -30.5% |
| `num_tl_90g_dpd_24m` | -30.5% |
| `num_tl_op_past_12m` | -30.5% |
| `pct_tl_nvr_dlq` | -30.5% |
| `percent_bc_gt_75` | -30.5% |

## Notes

- dropped 54 labelled rows (0.11%) with a non-terminal or unmapped status inside a window that should be fully matured

Data source: **synthetic**.
