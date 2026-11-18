#!/usr/bin/env python
"""P0-4: Evaluate non-graph controls for EXP04 cohort calibration claims.

Controls evaluated:
1. Arm A (tabular only)
2. Arm B (tabular + true cohort aggregates)
3. Control (a): Tabular + global time-respecting recent default rate
4. Control (b): Tabular + cohort aggregates with keys (zip3, emp_norm) shuffled within issue month

Uses paired bootstrap (>= 2,000 resamples) to test whether Arm B beats controls on ECE and O/E.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from credit_risk.data.splits import test_window_masks
from credit_risk.evaluation.metrics import (
    RiskMetrics,
    compare_models,
    compute_metrics,
    expected_calibration_error,
)
from credit_risk.features.build import FeatureBuilder, FeatureSpec
from credit_risk.features.financial import add_derived_features
from credit_risk.graph.features import (
    CohortFeatureBuilder,
    CohortFeatureConfig,
    _period_to_int,
    save_cohort_features,
)
from credit_risk.models.boosting import BoostingConfig, BoostingModel
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.utils.cli import load_prepared, resolve, save_json, save_table, write_report
from credit_risk.utils.runtime import get_logger

LOG = get_logger("scripts.cohort_controls")


def compute_macro_default_rate(df: pd.DataFrame, window_months: int = 36) -> np.ndarray:
    """Compute portfolio-wide time-respecting recent default rate for each loan."""
    months = _period_to_int(df["issue_period"])
    observed = _period_to_int(df["outcome_observed_period"], allow_na=True)
    labels = pd.to_numeric(df["default_observed"], errors="coerce").to_numpy(dtype=float)

    res_mask = ~np.isnan(observed)
    res_positions = np.where(res_mask)[0]
    res_order = np.argsort(observed[res_positions], kind="stable")
    pos_by_res = res_positions[res_order]
    o_sorted = observed[pos_by_res]
    lab_sorted = labels[pos_by_res]

    present = ~np.isnan(lab_sorted)
    csum_bad = np.concatenate([[0.0], np.cumsum(np.where(present, lab_sorted, 0.0))])
    ccnt_res = np.concatenate([[0.0], np.cumsum(present.astype(float))])

    hi = np.searchsorted(o_sorted, months, side="right")
    lo = np.searchsorted(o_sorted, months - window_months, side="left")
    macro_res = np.maximum(ccnt_res[hi] - ccnt_res[lo], 0)
    macro_bad = np.maximum(csum_bad[hi] - csum_bad[lo], 0)

    # Where no resolved loans are available, fallback to training prior ~0.1267
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(macro_res > 0, macro_bad / macro_res, 0.1267)
    return rate.astype(np.float32)


def shuffle_keys_within_period(
    df: pd.DataFrame,
    keys: tuple[str, ...] = ("zip3", "emp_norm"),
    period_col: str = "issue_period",
    seed: int = 42,
) -> pd.DataFrame:
    """Shuffle cohort keys within each issue period to destroy cohort identity while preserving marginal distributions."""
    rng = np.random.default_rng(seed)
    df_shuffled = df.copy()
    grouped = df.groupby(period_col).indices
    for period, indices in grouped.items():
        for k in keys:
            if k in df_shuffled.columns:
                vals = df_shuffled.loc[indices, k].to_numpy().copy()
                rng.shuffle(vals)
                df_shuffled.loc[indices, k] = vals
    return df_shuffled


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    p_b: np.ndarray,
    p_ctrl: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Compute paired bootstrap intervals for delta ECE and delta O/E (Arm B - Control)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    ece_b_orig, _ = expected_calibration_error(y_true, p_b)
    ece_c_orig, _ = expected_calibration_error(y_true, p_ctrl)
    delta_ece_orig = ece_b_orig - ece_c_orig

    oe_b_orig = float(y_true.mean() / p_b.mean()) if p_b.mean() > 0 else float("nan")
    oe_c_orig = float(y_true.mean() / p_ctrl.mean()) if p_ctrl.mean() > 0 else float("nan")
    # For O/E, distance from ideal 1.0 is what matters: |OE - 1|
    gap_b_orig = abs(oe_b_orig - 1.0)
    gap_c_orig = abs(oe_c_orig - 1.0)
    delta_oe_gap_orig = gap_b_orig - gap_c_orig

    delta_eces = np.empty(n_resamples, dtype=float)
    delta_oe_gaps = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_samp = y_true[idx]
        pb_samp = p_b[idx]
        pc_samp = p_ctrl[idx]

        eb, _ = expected_calibration_error(y_samp, pb_samp)
        ec, _ = expected_calibration_error(y_samp, pc_samp)
        delta_eces[i] = eb - ec

        y_m = y_samp.mean()
        oeb = y_m / pb_samp.mean() if pb_samp.mean() > 0 else 1.0
        oec = y_m / pc_samp.mean() if pc_samp.mean() > 0 else 1.0
        delta_oe_gaps[i] = abs(oeb - 1.0) - abs(oec - 1.0)

    return {
        "delta_ece": float(delta_ece_orig),
        "delta_ece_ci": (float(np.percentile(delta_eces, 2.5)), float(np.percentile(delta_eces, 97.5))),
        "p_val_ece": float((delta_eces >= 0).mean()),  # H0: delta_ece >= 0 (B is not better than Control)
        "delta_oe_gap": float(delta_oe_gap_orig),
        "delta_oe_gap_ci": (float(np.percentile(delta_oe_gaps, 2.5)), float(np.percentile(delta_oe_gaps, 97.5))),
        "p_val_oe_gap": float((delta_oe_gaps >= 0).mean()),
    }


def main() -> int:
    seed = 42
    paths_root = Path(".")
    data_dir = paths_root / "data" / "processed"

    t0 = time.time()
    LOG.info("Loading loans.parquet...")
    df = add_derived_features(pd.read_parquet(data_dir / "loans.parquet").reset_index(drop=True))

    cohort_path = data_dir / "cohort_features.parquet"
    if not cohort_path.exists():
        raise SystemExit("cohort_features.parquet missing. Run build_graph.py first.")
    cohort = pd.read_parquet(cohort_path)

    shuffled_path = data_dir / "cohort_features_shuffled.parquet"
    if not shuffled_path.exists():
        LOG.info("Building shuffled cohort features (Control b)...")
        df_shuffled = shuffle_keys_within_period(df, keys=("zip3", "emp_norm"), seed=seed)
        cohort_config = CohortFeatureConfig()
        builder = CohortFeatureBuilder(cohort_config).fit(df_shuffled[df_shuffled["split"] == "train"])
        cohort_shuffled = builder.transform(df_shuffled)
        save_cohort_features(cohort_shuffled, shuffled_path)
        LOG.info("Saved %s", shuffled_path)
    else:
        LOG.info("Loading existing %s...", shuffled_path)
        cohort_shuffled = pd.read_parquet(shuffled_path)

    LOG.info("Computing macro default rate (Control a)...")
    macro_default_rate = compute_macro_default_rate(df)
    df["macro_default_rate"] = macro_default_rate

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}
    train, valid, test = (df[masks[s]] for s in ("train", "valid", "test"))
    y = df["default"].astype("float32").to_numpy()
    y_train, y_valid, y_test = (y[masks[s]] for s in ("train", "valid", "test"))

    scores: dict[str, np.ndarray] = {}
    metrics: dict[str, RiskMetrics] = {}

    def fit_and_eval(name: str, extra_feature_df: pd.DataFrame | None, extra_cols: tuple[str, ...]) -> None:
        LOG.info("Fitting arm: %s...", name)
        if extra_feature_df is not None:
            combined_train = pd.concat([train, extra_feature_df[masks["train"]].set_index(train.index)], axis=1)
            combined_valid = pd.concat([valid, extra_feature_df[masks["valid"]].set_index(valid.index)], axis=1)
            combined_test = pd.concat([test, extra_feature_df[masks["test"]].set_index(test.index)], axis=1)
        else:
            combined_train, combined_valid, combined_test = train, valid, test

        spec = FeatureSpec(feature_set="primary", extra_numeric=extra_cols)
        builder = FeatureBuilder(spec).fit(combined_train, combined_valid)
        X_tr = builder.transform(combined_train)
        X_va = builder.transform(combined_valid)
        X_te = builder.transform(combined_test)

        model = BoostingConfig(n_estimators=2000, early_stopping_rounds=100)
        boosting = BoostingModel(builder.categorical_features, model).fit(
            X_tr, y_train, X_va, y_valid, seed=seed
        )
        calibrator = ProbabilityCalibrator("isotonic").fit(y_valid, boosting.predict_proba(X_va))
        pred = calibrator.transform(boosting.predict_proba(X_te))
        scores[name] = pred
        m = compute_metrics(y_test, pred)
        metrics[name] = m
        oe = float(y_test.mean() / pred.mean()) if pred.mean() > 0 else float("nan")
        LOG.info(
            "%-24s | AUC=%.4f | ECE=%.4f | O/E=%.4f (ratio=%.4f)",
            name,
            m.roc_auc,
            m.ece,
            oe,
            m.predicted_observed_ratio,
        )

    # 1. Arm A (tabular only)
    fit_and_eval("A_tabular", None, ())

    # 2. Arm B (tabular + cohort features)
    cohort_cols = tuple(c for c in cohort.columns if c in cohort.columns)
    fit_and_eval("B_tabular_plus_graph", cohort, cohort_cols)

    # 3. Control (a): Tabular + global recent default rate
    fit_and_eval("A_plus_macro_default", None, ("macro_default_rate",))

    # 4. Control (b): Tabular + shuffled cohort features
    fit_and_eval("B_shuffled_cohort", cohort_shuffled, cohort_cols)

    LOG.info("Running paired bootstrap comparisons (2,000 resamples)...")
    comp_macro = paired_bootstrap_comparison(y_test, scores["B_tabular_plus_graph"], scores["A_plus_macro_default"])
    comp_shuffled = paired_bootstrap_comparison(y_test, scores["B_tabular_plus_graph"], scores["B_shuffled_cohort"])

    # CR-03: the same comparisons on the untouched 2015-07..12 holdout (the headline window).
    holdout = test_window_masks(test["issue_period"])["holdout"]
    metrics_h2 = {k: compute_metrics(y_test[holdout], v[holdout]) for k, v in scores.items()}
    comp_macro_h2 = paired_bootstrap_comparison(
        y_test[holdout], scores["B_tabular_plus_graph"][holdout], scores["A_plus_macro_default"][holdout]
    )
    comp_shuffled_h2 = paired_bootstrap_comparison(
        y_test[holdout], scores["B_tabular_plus_graph"][holdout], scores["B_shuffled_cohort"][holdout]
    )
    compare_models(metrics_h2).to_csv(paths_root / "results" / "tables" / "cohort_controls_exp04_h2.csv")

    print("\n" + "=" * 80)
    print("EXP04 NON-GRAPH CONTROLS RESULTS")
    print("=" * 80)
    res_df = compare_models(metrics)
    print(res_df[["roc_auc", "ece", "observed_expected_ratio", "calibration_slope", "brier"]].to_string())

    print("\n--- Paired Bootstrap: Arm B vs Control (a) [Macro Default Rate] ---")
    print(f"Delta ECE (B - Macro):     {comp_macro['delta_ece']:+.5f} (95% CI: [{comp_macro['delta_ece_ci'][0]:.5f}, {comp_macro['delta_ece_ci'][1]:.5f}], p={comp_macro['p_val_ece']:.4f})")
    print(f"Delta |O/E - 1|:           {comp_macro['delta_oe_gap']:+.5f} (95% CI: [{comp_macro['delta_oe_gap_ci'][0]:.5f}, {comp_macro['delta_oe_gap_ci'][1]:.5f}], p={comp_macro['p_val_oe_gap']:.4f})")

    print("\n--- Paired Bootstrap: Arm B vs Control (b) [Shuffled Cohort Keys] ---")
    print(f"Delta ECE (B - Shuffled):  {comp_shuffled['delta_ece']:+.5f} (95% CI: [{comp_shuffled['delta_ece_ci'][0]:.5f}, {comp_shuffled['delta_ece_ci'][1]:.5f}], p={comp_shuffled['p_val_ece']:.4f})")
    print(f"Delta |O/E - 1|:           {comp_shuffled['delta_oe_gap']:+.5f} (95% CI: [{comp_shuffled['delta_oe_gap_ci'][0]:.5f}, {comp_shuffled['delta_oe_gap_ci'][1]:.5f}], p={comp_shuffled['p_val_oe_gap']:.4f})")
    print("=" * 80 + "\n")

    # Save results to tables and JSON
    table_path = paths_root / "results" / "tables" / "cohort_controls_exp04.csv"
    res_df.to_csv(table_path)
    LOG.info("Saved %s", table_path)

    json_path = paths_root / "results" / "tables" / "cohort_controls_exp04.json"
    import json
    with open(json_path, "w") as f:
        json.dump(
            {
                "arms": {k: v.to_dict() for k, v in metrics.items()},
                "bootstrap_vs_macro": comp_macro,
                "bootstrap_vs_shuffled": comp_shuffled,
                "holdout_h2": {
                    "window": "2015-07..12",
                    "arms": {k: v.to_dict() for k, v in metrics_h2.items()},
                    "bootstrap_vs_macro": comp_macro_h2,
                    "bootstrap_vs_shuffled": comp_shuffled_h2,
                },
            },
            f,
            indent=2,
        )
    LOG.info("Saved %s", json_path)
    LOG.info("Total elapsed: %.1f seconds", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
