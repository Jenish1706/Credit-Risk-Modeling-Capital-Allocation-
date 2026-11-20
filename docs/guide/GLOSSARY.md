# Glossary

Acronyms and terms used anywhere in the repository, alphabetical. A pointer in brackets says where
the idea is explained or implemented.

| Term | Meaning |
|---|---|
| **A / B arms** | In EXP04: arm A is the tabular model, arm B is tabular plus cohort features. |
| **ADR** | Architecture decision record; the entries in [08_DESIGN_DECISIONS.md](08_DESIGN_DECISIONS.md). |
| **Anomaly score** | Reconstruction error of the tabular autoencoder; a candidate blind-spot signal (EXP08). |
| **APR** | Annual percentage rate; the interest rate charged on a loan. Called `int_rate` in the data (a percent) and used as a fraction in the economics. |
| **Approved-only population** | Every labelled loan passed LendingClub's underwriting, so models estimate `P(default | approved)`. |
| **Approve-all** | The policy that funds every applicant; a lower-bound baseline. |
| **Arm** | One variant in a comparison (a feature set, a model, a policy). |
| **Artifact** | A file a script writes: a model or score array under `artifacts/`, a table or report under `results/`. |
| **AUC / ROC-AUC** | Probability a random defaulter is ranked above a random non-defaulter ([concepts §3](01_CONCEPTS.md)). |
| **Availability guard** | Feature-builder rule dropping columns whose missing rate shifts between windows, so "is this populated?" cannot act as a vintage proxy. |
| **Bandit (contextual)** | A learner that picks an action given features, sees a reward and updates ([concepts §11](01_CONCEPTS.md)). |
| **Basel LGD** | Loss given default defined over exposure at default; not what this project calls LGD. |
| **Blind-spot signal** | A score meant to flag borrowers whose PD is unreliable: anomaly (EXP08), VGAE sigma (EXP09), disagreement (EXP10). |
| **Bootstrap, paired** | Resample units with replacement and recompute a difference between two arms on the same resample ([concepts §8](01_CONCEPTS.md)). |
| **Breakeven PD** | The PD at which expected profit is zero; the economically correct approval cut-off for a given price ([concepts §9](01_CONCEPTS.md)). |
| **Brier score** | Mean squared error of predicted probabilities; used to choose the production calibrator on H1. |
| **Budget fraction** | Capital budget as a share of the capital needed to fund every applicant (EXP06: 100%, 60%, 35%, 20%; EXP11: 80% of profitable demand). |
| **Calibration** | Agreement between predicted probabilities and observed frequencies ([concepts §4](01_CONCEPTS.md)). |
| **Calibration slope / intercept** | Coefficients from regressing the outcome on predicted log-odds; 1 and 0 are perfect. |
| **Capital constraint** | A limit on total exposure funded in a period; makes lending a knapsack problem. |
| **Cashflow model** | Two straight lines in APR: good-loan yield and defaulted-loan loss rate (`CashflowModel`). |
| **CC0** | Creative Commons zero, the dataset's licence. |
| **Charge-off** | A loan written off as uncollectable; the main default status. |
| **Cohort** | Loans sharing a key: 3-digit ZIP or normalised employer text. |
| **Cohort features** | Smoothed default rates, counts and means over earlier members of a cohort ([concepts §5](01_CONCEPTS.md)). |
| **Collapsed learner** | A learner whose approval rate fell below 1%; effectively stopped lending. |
| **Common random numbers** | Every policy sees the same applicants and outcome draws in an episode, making paired comparisons tight. |
| **Conditional coupling** | The simulator's per-applicant uniform draw `u`, sampled so the real outcome is reproduced at the observed exposure. |
| **Conditional test** | Testing a signal within PD strata ([concepts §7](01_CONCEPTS.md)). |
| **Continuity column** | A full-2015 figure shown next to the H2 headline so earlier results remain comparable. |
| **Control (arm)** | A deliberately meaningless or weaker variant used to test attribution (intercept-only, macro rate, shuffled keys, placebo). |
| **Counterfactual exposure** | A loan amount other than the funded one, whose outcome was never observed. |
| **Credit policy status** | LendingClub's "does not meet the credit policy" statuses, mapped to default or repaid by their suffix. |
| **Deny-list** | The forbidden-feature set (`FORBIDDEN_AS_FEATURES`) enforced by `assert_no_leakage`. |
| **Delayed feedback** | The reward for a lending decision arrives up to years later ([concepts §12](01_CONCEPTS.md)). |
| **Deployment-realism audit** | Retraining with only labels observable at the deployment date (`analyze_deployment_realism.py`). |
| **Default (target)** | 1 for charged-off or defaulted loans, 0 for fully paid. |
| **Dev window** | 2015-01…06 (H1), the only window used for selection. |
| **Disagreement** | Difference between the tabular and graph models' predictions on the log-odds scale; blind-spot signal for EXP10. |
| **DTI** | Debt-to-income ratio. |
| **EAD** | Exposure at default: the amount outstanding when a loan defaults. |
| **ECE / MCE** | Expected and maximum calibration error over 20 quantile bins. |
| **Empirical Bayes shrinkage** | Pulling a noisy rate toward a prior with a pseudo-count `α` ([concepts §5](01_CONCEPTS.md)). |
| **Episode** | One simulated funding period of 1,200 applicants. |
| **EL** | Expected loss `PD × LGD × EAD`. |
| **Evidence weight** | `resolved / (resolved + α)`: how much of a smoothed cohort rate is data rather than prior. |
| **Expected profit** | `(1−PD)·y·E − PD·l·E − funding − opex` ([concepts §9](01_CONCEPTS.md)). |
| **EXP01 to EXP11** | The eleven experiments in the README results table; mapping in [04_METHODOLOGY.md](04_METHODOLOGY.md). |
| **Exposure** | Dollar amount lent. |
| **Fanout** | Number of neighbours sampled per layer in GraphSAGE (10, then 5). |
| **FICO** | A US credit score; `fico_range_low/high` are decision-time features, `last_fico_range_*` are post-origination and forbidden. |
| **Fixed-threshold policy** | Approve if PD is below a constant (15%). |
| **Frozen learner** | A learning policy whose parameters are not updated during evaluation (`learn=False`). |
| **Funded amount** | The amount LendingClub actually funded; the observed exposure. |
| **Gini** | `2·AUC − 1`. |
| **GNN / GraphSAGE** | Graph neural network; GraphSAGE aggregates sampled neighbours with a mean ([concepts §6](01_CONCEPTS.md)). |
| **Grade / sub_grade** | LendingClub's own risk grade; excluded from the primary PD. |
| **H1 / H2** | 2015-01…06 (selection) and 2015-07…12 (untouched holdout). |
| **Headline** | The number reported on H2 in the README table. |
| **History window** | 2007-06 to 2009-12; supplies graph neighbours, never modelled. |
| **Holdout** | The window no selection has seen (H2). |
| **Informative missingness** | A missing value that itself carries meaning (for example "no delinquency on file"); encoded with an indicator. |
| **Intercept-only calibration** | Shifting log-odds by one constant to match the base rate; isolates a base-rate shift. |
| **Isotonic** | A monotone step-function calibrator. |
| **Issue period / `issue_d`** | Month a loan was funded; the decision timestamp. |
| **KEEP / REJECT** | The verdict on a blind-spot signal from the sign-test rule on H1. |
| **Knapsack** | Choosing items under a capacity constraint to maximise value; the constrained lending problem ([concepts §10](01_CONCEPTS.md)). |
| **KS** | Kolmogorov-Smirnov statistic: largest gap between cumulative score distributions of defaulters and non-defaulters. |
| **Label maturity** | Whether a loan's outcome was knowable at a given date ([concepts §2](01_CONCEPTS.md)). |
| **Lambda (λ)** | Shadow price of capital: profit per dollar of the marginal loan funded. |
| **LC** | LendingClub. |
| **Leakage** | Information in a feature or selection that would not have been available at decision time. |
| **LGD** | Loss given default; here the net lifetime loss on funded principal of a defaulted loan. |
| **LightGBM** | The gradient-boosted tree library used for the production PD. |
| **LinUCB** | Contextual bandit with an optimism bonus ([concepts §11](01_CONCEPTS.md)). |
| **Loan-to-income** | A derived affordability ratio. |
| **Loss severity** | The defaulted-loan loss rate; the "severity" term in the profit decomposition. |
| **Macro default rate** | A single time-respecting portfolio-wide recent default rate; the non-graph control in EXP04. |
| **Manifest** | JSON recording config, seed and data source for a run (`results/reports/*_manifest.json`). |
| **Marginal table** | Calibration by quintile of a signal without conditioning on PD; confounded by the PD level. |
| **Mirror** | The third-party Kaggle copy of the LendingClub files. |
| **Monitor window** | 2016-01 to 2018-12; features and predictions only, outcomes immature by design. |
| **Myopic rule** | The expected-profit policy without a shadow price; ignores the opportunity cost of capital. |
| **O/E** | Observed over expected: observed default rate divided by mean predicted; above 1 means the model under-predicts. |
| **Observed exposure** | The funded amount; the only exposure at which the outcome was observed. |
| **Opex** | Operating cost: fixed $150 per loan plus 1% of exposure. |
| **Out-of-time (OOT)** | Evaluating on a later period than training. |
| **`outcome_observed_period`** | Month a loan's outcome became knowable; used for label gating in the graph and the deployment-realism audit. |
| **Oracle / diagnostic** | A quantity fitted on the evaluation data (a cashflow refit, a lambda) to show what a clairvoyant would do; labelled and never used to choose anything. |
| **Pairing** | Comparing two arms on the identical units. |
| **PD** | Probability of default. |
| **Placebo control** | A meaningless version of an intervention (random or APR-matched exclusion) used to size the effect that needs no signal. |
| **Platt scaling** | Logistic regression on log-odds used as a calibrator. |
| **Policy** | A rule mapping an applicant's context to an action (accept or reject). |
| **Primary feature set** | Decision-time features excluding `grade`, `sub_grade`, `int_rate` and `installment`. |
| **Production calibrator** | The calibrator behind `boosting_pd_test.npy`, chosen on H1 Brier. |
| **Profit density** | Expected profit per dollar of exposure. |
| **PSI** | Population stability index ([concepts §13](01_CONCEPTS.md)). |
| **Q-learning** | Tabular reinforcement learning over (profit density, budget left, time left) states. |
| **Quantile bin** | A bin containing an equal share of the data; used for ECE and PSI. |
| **Realised profit** | `total_pymnt − collection_recovery_fee − funded_amnt` (cash received already includes recoveries). |
| **Reject inference** | Estimating how declined applicants would have performed; not implemented here. |
| **Reliability diagram** | Plot of observed against predicted rate by bin. |
| **Resolved** | A loan whose outcome has been booked (`outcome_observed_period` reached). |
| **Return on capital (RoC)** | Profit over capital deployed. |
| **Risk band** | A policy that maps PD ranges to actions; collapses to a PD cut-off at a single exposure. |
| **SAC (Smart App Control)** | Windows feature that blocks unsigned binaries, which breaks PyPI scientific wheels. |
| **Selection** | Any choice made using data: a calibrator, a threshold, a verdict, a flag cut-point. |
| **Sensitivity analysis** | A run that changes an assumption (here, adding 0.5x and 1.5x exposures) and is reported separately. |
| **Shadow price** | See lambda. |
| **SHAP** | Shapley-value attributions explaining a model's prediction. |
| **Shuffled keys** | The control in which cohort keys are permuted within issue month. |
| **Sign test** | Exact binomial test on how many strata show a positive difference. |
| **Split** | A named window: history, train, valid, test, monitor. |
| **Strata / stratum** | A PD decile in the conditional test. |
| **Synthetic data** | A schema-faithful generated extract used for tests and credential-free runs; never reported as a finding. |
| **Thompson sampling** | Bandit that samples parameters from its posterior and acts greedily on the sample. |
| **Time-respecting** | Using only information available at the decision date ([concepts §5](01_CONCEPTS.md)). |
| **Unconstrained condition** | EXP06 with unlimited capital, where the myopic rule should be near-optimal. |
| **Valid(ation) window** | 2014-07 to 2014-12; for early stopping, calibration, the shadow price and bandit training. |
| **VGAE** | Variational graph autoencoder; its per-node posterior sigma is a relational-uncertainty signal (EXP09). |
| **Vintage** | The origination period of a loan; default rates differ by vintage. |
| **Wald interval** | Confidence interval from a coefficient and its standard error (here from the Fisher information). |
| **WAL** | Weighted-average life of a 36-month amortising loan (configured as 1.55 years; the schedule gives about 1.6). |
| **Yield (good-loan)** | Realised return per dollar on loans that repaid. |
| **ZIP3** | The first three characters of the ZIP code; one cohort key. |
