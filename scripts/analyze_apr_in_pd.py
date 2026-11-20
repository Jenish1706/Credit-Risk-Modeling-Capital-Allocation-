#!/usr/bin/env python
"""CR-10 and CR-11: does the profit-rule failure go away with a better PD?

    python scripts/analyze_apr_in_pd.py --config configs/experiments/exp06_rl.yaml

The production PD excludes ``int_rate`` / ``grade`` / ``sub_grade`` by design (they are
LendingClub's own risk score). The expected-profit rule, however, uses APR as its revenue lever, and
the CR-02 audit found the PD under-states risk exactly where APR is high. This ablation refits the
PD with the ``with_lc_grade`` feature set and repeats the real-book audit, so the answer is measured
rather than argued.

It is an ablation, **not** a replacement: a PD that consumes LendingClub's pricing cannot claim to
underwrite from primary evidence. Selection (the calibrator) uses 2015-01..06 only; every reported
number is on 2015-07..12.

CR-11 adds a third PD: the tabular + cohort-feature arm of EXP04 (seed 42, same features and
configuration), with its calibrator chosen on 2015-01..06. EXP04 shows the cohort features fix the
*level* of the PD; this audit asks whether fixing the level fixes the *decision*.

A volume-matched control separates "better PD" from "stricter cut": the production PD with its cut-off
set so that it accepts as many loans as the cohort-feature PD does at 15%. The cut-off is chosen from
predicted PDs and a loan count only; no outcome is read to choose it.

Writes ``results/tables/apr_in_pd_{metrics,by_apr,rules}.csv`` with one row per PD source and rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.data.splits import (  # noqa: E402
    WINDOW_LABELS,
    test_window_masks,
)
from credit_risk.decision.expected_loss import (  # noqa: E402
    EconomicsConfig,
    expected_profit,
    fit_cashflow_model,
)
from credit_risk.decision.profit_audit import (  # noqa: E402
    paired_bootstrap_ci,
    rule_realised_profit,
)
from credit_risk.evaluation.metrics import compute_metrics  # noqa: E402
from credit_risk.features.build import FeatureBuilder, FeatureSpec  # noqa: E402
from credit_risk.models.boosting import BoostingConfig, BoostingModel  # noqa: E402
from credit_risk.models.calibration import select_calibrator_on_window  # noqa: E402
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    load_prepared,
    resolve,
    save_table,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402
from scripts.analyze_profit_calibration import THRESHOLD, _arrays, pd_by_apr_table  # noqa: E402

LOG = get_logger("scripts.apr_in_pd")

VOLUME_MATCHED = "volume_matched_to_cohort_15"


def fit_pd(spec, train, valid, test, seed, label):
    """Fit LightGBM on ``spec``'s features; calibrator chosen on 2015-01..06 Brier (library helper)."""
    builder = FeatureBuilder(spec).fit(train, valid)
    X_tr, X_va, X_te = (builder.transform(d) for d in (train, valid, test))
    y_tr = train["default"].astype(int).to_numpy()
    y_va = valid["default"].astype(int).to_numpy()
    y_te = test["default"].astype(int).to_numpy()
    model = BoostingModel(
        builder.categorical_features, BoostingConfig(n_estimators=2000, early_stopping_rounds=100)
    ).fit(X_tr, y_tr, X_va, y_va, seed=seed)
    raw_va, raw_te = model.predict_proba(X_va), model.predict_proba(X_te)

    best, briers, calibrated = select_calibrator_on_window(
        y_va, raw_va, y_te, raw_te, test["issue_period"]
    )
    for method, brier in briers.items():
        LOG.info("%s | %-14s | H1 Brier %.6f", label, method, brier)
    return best, calibrated[best]


def fit_grade_pd(train, valid, test, seed):
    """PD with LendingClub's grade and APR as features."""
    return fit_pd(FeatureSpec(feature_set="with_lc_grade"), train, valid, test, seed, "grade PD")


def fit_cohort_pd(train, valid, test, cohort_columns, seed):
    """PD with the cohort features of EXP04 arm B (the frames already carry the cohort columns)."""
    spec = FeatureSpec(feature_set="primary", extra_numeric=tuple(cohort_columns))
    return fit_pd(spec, train, valid, test, seed, "cohort PD")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp06_rl.yaml")
    args = parser.parse_args()
    config, paths, _, seed = resolve(args)

    df = load_prepared(paths)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    masks = test_window_masks(test["issue_period"])
    h2 = masks["holdout"]

    ecfg = config.get("economics", {})
    economics = EconomicsConfig(
        annual_cost_of_funds=float(ecfg.get("annual_cost_of_funds", 0.03)),
        opex_fixed=float(ecfg.get("opex_fixed", 150.0)),
        opex_variable_rate=float(ecfg.get("opex_variable_rate", 0.010)),
        manual_review_cost=float(ecfg.get("manual_review_cost", 75.0)),
        weighted_average_life_years=float(ecfg.get("weighted_average_life_years", 1.55)),
    )
    cashflow = fit_cashflow_model(train, economics)

    pd_primary = np.load(paths.artifacts / "boosting_pd_test.npy")
    method, pd_grade = fit_grade_pd(train, valid, test, seed)
    LOG.info("grade PD calibrator chosen on 2015-01..06: %s", method)

    # CR-11: the EXP04 arm B features. Cohort rows are aligned positionally with the table.
    cohort = pd.read_parquet(paths.data_processed / "cohort_features.parquet")
    split_values = df["split"].to_numpy()

    def with_cohort(frame, split_name):
        return pd.concat([frame, cohort.loc[split_values == split_name].set_index(frame.index)], axis=1)

    method_c, pd_cohort = fit_cohort_pd(
        with_cohort(train, "train"), with_cohort(valid, "valid"), with_cohort(test, "test"),
        list(cohort.columns), seed,
    )
    LOG.info("cohort PD calibrator chosen on 2015-01..06: %s", method_c)
    sources = {"primary_pd": pd_primary, "with_lc_grade_pd": pd_grade, "cohort_features_pd": pd_cohort}

    holdout = test[h2]
    y = holdout["default"].astype(int).to_numpy()

    # -- discrimination and calibration ------------------------------------------------
    metric_rows = []
    for name, p in sources.items():
        m = compute_metrics(y, p[h2])
        metric_rows.append(
            {"pd_source": name, "window": WINDOW_LABELS["holdout"], "roc_auc": m.roc_auc,
             "ece": m.ece, "brier": m.brier, "observed_expected_ratio": m.observed_expected_ratio}
        )
    save_table(pd.DataFrame(metric_rows), paths, "apr_in_pd_metrics", index=False)

    # -- calibration within APR bands ---------------------------------------------------
    by_apr = pd_by_apr_table({name: (holdout, p[h2]) for name, p in sources.items()})
    save_table(by_apr, paths, "apr_in_pd_by_apr", index=False)

    # -- real-book rules at the observed exposure ----------------------------------------
    n_boot = int(config.get_path("evaluation.n_bootstrap", 2000))
    per_loan, accepted, cutoff, arrays = {}, {}, {}, {}
    for name, p in sources.items():
        a = _arrays(holdout, p[h2])
        ok = np.isfinite(a["realised_return"]) & np.isfinite(a["exposure"]) & np.isfinite(a["apr"])
        a = {k: v[ok] for k, v in a.items()}
        arrays[name] = a
        ep = expected_profit(a["pd_hat"], a["exposure"], a["apr"], cashflow, economics)
        for rule, accept in (("fixed_threshold@0.15", a["pd_hat"] < THRESHOLD), ("expected_profit>0", ep > 0)):
            per_loan[(name, rule)] = rule_realised_profit(
                accept, a["exposure"], a["apr"], a["realised_return"], economics
            )
            accepted[(name, rule)] = int(accept.sum())
            cutoff[(name, rule)] = THRESHOLD if rule.startswith("fixed") else float("nan")

    # Volume-matched control: production PD, accept the n lowest-PD loans, n = what the cohort-feature PD
    # accepts at 15%. Chosen from PDs and a count only (no outcome is read).
    n_match = accepted[("cohort_features_pd", "fixed_threshold@0.15")]
    a_prod = arrays["primary_pd"]
    order = np.argsort(a_prod["pd_hat"], kind="stable")
    accept_vm = np.zeros(len(order), dtype=bool)
    accept_vm[order[:n_match]] = True
    key_vm = ("primary_pd", VOLUME_MATCHED)
    per_loan[key_vm] = rule_realised_profit(
        accept_vm, a_prod["exposure"], a_prod["apr"], a_prod["realised_return"], economics
    )
    accepted[key_vm] = int(accept_vm.sum())
    cutoff[key_vm] = float(a_prod["pd_hat"][order[n_match - 1]])
    LOG.info("volume-matched production-PD cut-off %.4f accepts %d loans", cutoff[key_vm], accepted[key_vm])

    reference = per_loan[("primary_pd", "fixed_threshold@0.15")]
    matched = per_loan[key_vm]
    rows = []
    for (name, rule), profit in per_loan.items():
        same_base = per_loan[(name, "fixed_threshold@0.15")]
        m_same, lo_s, hi_s = paired_bootstrap_ci(profit - same_base, n_bootstrap=n_boot, seed=seed)
        m_ref, lo_r, hi_r = paired_bootstrap_ci(profit - reference, n_bootstrap=n_boot, seed=seed)
        m_vm, lo_v, hi_v = paired_bootstrap_ci(profit - matched, n_bootstrap=n_boot, seed=seed)
        rows.append(
            {
                "pd_source": name, "rule": rule, "n_accepted": accepted[(name, rule)],
                "pd_cutoff": cutoff[(name, rule)],
                "realised_profit_total": float(profit.sum()),
                "diff_vs_same_pd_threshold_per_loan": m_same, "same_ci_low": lo_s, "same_ci_high": hi_s,
                "diff_vs_primary_pd_threshold_per_loan": m_ref, "ref_ci_low": lo_r, "ref_ci_high": hi_r,
                "diff_vs_volume_matched_per_loan": m_vm, "vm_ci_low": lo_v, "vm_ci_high": hi_v,
            }
        )
    save_table(pd.DataFrame(rows), paths, "apr_in_pd_rules", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
