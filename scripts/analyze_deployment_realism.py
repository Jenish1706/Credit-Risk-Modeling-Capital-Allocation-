#!/usr/bin/env python
"""P1-5: Quantify the deployment realism gap.

Evaluates the difference between:
1. Omniscient Out-of-Time Model: Trained on 2010-01..2014-06 and calibrated on 2014-07..12
   using mature terminal labels known only after 2014-12 (up to 2018).
2. Deployment-Realistic Model: Trained and calibrated strictly on outcomes observable
   on or before 2014-12-31, simulating an actual production deployment on 2015-01-01.

Reports test AUC, ECE, O/E, and Brier score on the 2015 test cohort.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from credit_risk.data.splits import test_window_masks
from credit_risk.evaluation.metrics import compute_metrics
from credit_risk.features.build import FeatureBuilder, FeatureSpec
from credit_risk.features.financial import add_derived_features
from credit_risk.models.boosting import BoostingConfig, BoostingModel
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.utils.runtime import get_logger

LOG = get_logger("scripts.deployment_realism")


def resolution_selection_table(df: pd.DataFrame, as_of: str = "2014-12", split_issue: str = "2013-06") -> pd.DataFrame:
    """G4-r2: is a training set restricted to loans resolved by ``as_of`` a fair sample of the loans issued?

    For the same issue windows the realistic arm uses, compare the default rate among loans whose outcome
    was already resolved at ``as_of`` with the default rate among *all* loans issued then (terminal labels).
    A gap means the realistic arm differs from the standard arm by *selection on resolution* as well as by
    staleness, so the two effects cannot be separated by this comparison alone.
    """
    dev = df[df["split"].isin(["train", "valid"])]
    windows = {
        f"train window (issued <= {split_issue})": dev["issue_period"] <= split_issue,
        f"calibration window (issued {pd.Period(split_issue, 'M') + 1}..2014-12)": dev["issue_period"] > split_issue,
    }
    rows = []
    for name, mask in windows.items():
        allw = dev[mask]
        resolved = allw[allw["outcome_observed_period"] <= as_of]
        rows.append(
            {
                "window": name,
                "loans_issued": len(allw),
                "loans_resolved_by_as_of": len(resolved),
                "share_resolved": len(resolved) / len(allw),
                "default_rate_all_loans": float(allw["default"].mean()),
                "default_rate_resolved_loans": float(resolved["default"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_deployment_realism(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train and compare standard vs deployment-realistic models."""
    spec = FeatureSpec(feature_set="primary")
    model_cfg = BoostingConfig(n_estimators=2000, early_stopping_rounds=100)

    # 1. Standard omniscient model
    train_mask = (df["split"] == "train").to_numpy()
    valid_mask = (df["split"] == "valid").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    tr_df, va_df, te_df = df[train_mask], df[valid_mask], df[test_mask]
    y_tr = tr_df["default"].astype("float32").to_numpy()
    y_va = va_df["default"].astype("float32").to_numpy()
    y_te = te_df["default"].astype("float32").to_numpy()

    b_std = FeatureBuilder(spec).fit(tr_df, va_df)
    X_tr = b_std.transform(tr_df)
    X_va = b_std.transform(va_df)
    X_te = b_std.transform(te_df)

    m_std = BoostingModel(b_std.categorical_features, model_cfg).fit(X_tr, y_tr, X_va, y_va, seed=seed)
    cal_std = ProbabilityCalibrator("isotonic").fit(y_va, m_std.predict_proba(X_va))
    pred_std = cal_std.transform(m_std.predict_proba(X_te))
    res_std = compute_metrics(y_te, pred_std)
    oe_std = float(y_te.mean() / pred_std.mean()) if pred_std.mean() > 0 else float("nan")

    # 2. Deployment-realistic model (outcomes <= 2014-12-31)
    obs_mask = (df["outcome_observed_period"] <= "2014-12") & df["split"].isin(["train", "valid"])
    obs_df = df[obs_mask].copy()

    obs_tr_mask = (obs_df["issue_period"] <= "2013-06").to_numpy()
    obs_va_mask = ~obs_tr_mask

    obs_tr_df = obs_df[obs_tr_mask]
    obs_va_df = obs_df[obs_va_mask]
    y_obs_tr = obs_tr_df["default"].astype("float32").to_numpy()
    y_obs_va = obs_va_df["default"].astype("float32").to_numpy()

    # Guard against single-class validation in small fixtures/subsamples
    if len(np.unique(y_obs_va)) < 2 and len(np.unique(obs_df["default"])) >= 2:
        from sklearn.model_selection import train_test_split

        obs_tr_idx, obs_va_idx = train_test_split(
            np.arange(len(obs_df)), test_size=0.25, random_state=seed, stratify=obs_df["default"]
        )
        obs_tr_df = obs_df.iloc[obs_tr_idx]
        obs_va_df = obs_df.iloc[obs_va_idx]
        y_obs_tr = obs_tr_df["default"].astype("float32").to_numpy()
        y_obs_va = obs_va_df["default"].astype("float32").to_numpy()


    b_obs = FeatureBuilder(spec).fit(obs_tr_df, obs_va_df)
    X_obs_tr = b_obs.transform(obs_tr_df)
    X_obs_va = b_obs.transform(obs_va_df)
    X_obs_te = b_obs.transform(te_df)

    m_obs = BoostingModel(b_obs.categorical_features, model_cfg).fit(
        X_obs_tr, y_obs_tr, X_obs_va, y_obs_va, seed=seed
    )
    cal_obs = ProbabilityCalibrator("isotonic").fit(y_obs_va, m_obs.predict_proba(X_obs_va))
    pred_obs = cal_obs.transform(m_obs.predict_proba(X_obs_te))
    res_obs = compute_metrics(y_te, pred_obs)
    oe_obs = float(y_te.mean() / pred_obs.mean()) if pred_obs.mean() > 0 else float("nan")

    # CR-03: nothing here is selected on the test book, so the headline is the untouched
    # 2015-07..12 holdout and the full-2015 columns above are the continuity figures.
    h2 = test_window_masks(te_df["issue_period"])["holdout"]
    h2_std = compute_metrics(y_te[h2], pred_std[h2])
    h2_obs = compute_metrics(y_te[h2], pred_obs[h2])
    oe_h2_std = float(y_te[h2].mean() / pred_std[h2].mean())
    oe_h2_obs = float(y_te[h2].mean() / pred_obs[h2].mean())

    table = pd.DataFrame(
        [
            {
                "model": "Standard (Omniscient Labels <= 2018)",
                "train_loans": len(tr_df),
                "valid_loans": len(va_df),
                "test_auc": res_std.roc_auc,
                "test_ece": res_std.ece,
                "observed_expected_ratio": oe_std,
                "brier": res_std.brier,
                "h2_test_auc": h2_std.roc_auc,
                "h2_test_ece": h2_std.ece,
                "h2_observed_expected_ratio": oe_h2_std,
            },
            {
                "model": "Deployment-Realistic (Outcomes <= 2014-12)",
                "train_loans": len(obs_tr_df),
                "valid_loans": len(obs_va_df),
                "test_auc": res_obs.roc_auc,
                "test_ece": res_obs.ece,
                "observed_expected_ratio": oe_obs,
                "brier": res_obs.brier,
                "h2_test_auc": h2_obs.roc_auc,
                "h2_test_ece": h2_obs.ece,
                "h2_observed_expected_ratio": oe_h2_obs,
            },
            {
                "model": "Delta (Realistic - Standard)",
                "train_loans": len(obs_tr_df) - len(tr_df),
                "valid_loans": len(obs_va_df) - len(va_df),
                "test_auc": res_obs.roc_auc - res_std.roc_auc,
                "test_ece": res_obs.ece - res_std.ece,
                "observed_expected_ratio": oe_obs - oe_std,
                "brier": res_obs.brier - res_std.brier,
                "h2_test_auc": h2_obs.roc_auc - h2_std.roc_auc,
                "h2_test_ece": h2_obs.ece - h2_std.ece,
                "h2_observed_expected_ratio": oe_h2_obs - oe_h2_std,
            },
        ]
    )

    details = {
        "standard_pred": pred_std,
        "realistic_pred": pred_obs,
        "y_test": y_te,
        "n_test": len(y_te),
    }
    return table, details


def main() -> int:
    paths_root = Path(".")
    data_path = paths_root / "data" / "processed" / "loans.parquet"
    LOG.info("Loading loans.parquet...")
    df = add_derived_features(pd.read_parquet(data_path).reset_index(drop=True))

    LOG.info("Running deployment realism audit...")
    table, _ = evaluate_deployment_realism(df)

    print("\n" + "=" * 80)
    print("P1-5 DEPLOYMENT REALISM AUDIT RESULTS")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80 + "\n")

    selection = resolution_selection_table(df)
    print(selection.to_string(index=False))
    out_sel = paths_root / "results" / "tables" / "deployment_realism_selection.csv"
    selection.to_csv(out_sel, index=False)
    LOG.info("Saved %s", out_sel)

    out_csv = paths_root / "results" / "tables" / "deployment_realism_p1_5.csv"
    table.to_csv(out_csv, index=False)
    LOG.info("Saved %s", out_csv)

    std, real, delta = table.iloc[0], table.iloc[1], table.iloc[2]
    tr_sel, va_sel = selection.iloc[0], selection.iloc[1]
    report_lines = [
        "# Deployment Realism Audit (Finding P1-5)",
        "",
        "## Motivation",
        "The standard out-of-time split trains on 2010-01..2014-06 and calibrates on 2014-07..12 using terminal ",
        "labels, most of which were not yet knowable on 2015-01-01 (the 36-month outcomes arrive in 2015-2018). ",
        "The realistic arm keeps only loans whose outcome was resolved by 2014-12.",
        "",
        "## Results Table (all figures from the table below; regenerate with this script)",
        "",
        table.to_markdown(index=False),
        "",
        "## Key Findings",
        f"- **Discrimination:** 2015 AUC {std['test_auc']:.4f} -> {real['test_auc']:.4f} ({delta['test_auc']:+.4f}); "
        f"2015-07..12: {std['h2_test_auc']:.4f} -> {real['h2_test_auc']:.4f}.",
        f"- **Calibration:** 2015 ECE {std['test_ece']:.4f} -> {real['test_ece']:.4f}; "
        f"2015-07..12: {std['h2_test_ece']:.4f} -> {real['h2_test_ece']:.4f}.",
        f"- **Level:** observed/expected {std['observed_expected_ratio']:.3f} -> {real['observed_expected_ratio']:.3f} "
        f"(2015-07..12: {std['h2_observed_expected_ratio']:.3f} -> {real['h2_observed_expected_ratio']:.3f}).",
        "",
        "## What this comparison can and cannot say",
        "The realistic arm is not a random sample of the loans issued in the same months. Selection on resolution:",
        "",
        selection.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"In the training window resolved loans default at {tr_sel['default_rate_resolved_loans']:.1%} against "
        f"{tr_sel['default_rate_all_loans']:.1%} for all loans issued then; in the calibration window the "
        f"figures are {va_sel['default_rate_resolved_loans']:.1%} and {va_sel['default_rate_all_loans']:.1%}. "
        "So the drop in performance mixes staleness (older vintages, fewer recent examples) with selection on "
        "resolution. It measures the cost of the naive approach 'train on what has resolved', not staleness alone. "
        "There is no arm here that trains on fully matured vintages only, so the two effects are not separated.",
    ]
    out_md = paths_root / "results" / "reports" / "deployment_realism.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    LOG.info("Saved %s", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
