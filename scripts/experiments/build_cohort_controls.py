#!/usr/bin/env python
"""P0-4: Evaluate non-graph controls for EXP04 cohort calibration claims.

Controls evaluated:
1. Arm A (tabular only)
2. Arm B (tabular + true cohort aggregates)
3. Control (a): Tabular + global time-respecting recent default rate
4. Control (b): Tabular + cohort aggregates with keys (zip3, emp_norm) shuffled within issue month

Uses paired bootstrap (>= 2,000 resamples) to test whether Arm B beats controls on ECE and O/E,
and repeats every arm over five LightGBM seeds (CR-06) so the claim does not rest on one fit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from credit_risk.data.splits import test_window_masks
from credit_risk.evaluation.metrics import (
    compare_models,
    compute_metrics,
    expected_calibration_error,
)
from credit_risk.features.build import FeatureBuilder, FeatureSpec
from credit_risk.features.financial import add_derived_features
from credit_risk.graph.features import (
    CohortFeatureBuilder,
    CohortFeatureConfig,
    compute_macro_default_rate,
    save_cohort_features,
    shuffle_keys_within_period,
)
from credit_risk.models.boosting import BoostingConfig, BoostingModel
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.utils.runtime import get_logger

LOG = get_logger("scripts.cohort_controls")


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

    # D6 / CR-06: the arms differ in their features, but each is one *fitted model*, and a single
    # LightGBM seed cannot tell a real calibration gain from fit-to-fit noise. Every arm is
    # therefore fitted for several model seeds; the data, the cohort features and the key shuffle
    # are held fixed, so the spread below is model-fit variance only.
    seeds = [42, 43, 44, 45, 46]
    cohort_cols = tuple(c for c in cohort.columns if c in cohort.columns)
    arm_specs = {
        "A_tabular": (None, ()),
        "B_tabular_plus_graph": (cohort, cohort_cols),
        "A_plus_macro_default": (None, ("macro_default_rate",)),
        "B_shuffled_cohort": (cohort_shuffled, cohort_cols),
    }
    win = test_window_masks(test["issue_period"])
    h2 = win["holdout"]

    prepared = {}
    for name, (extra_df, extra_cols) in arm_specs.items():
        LOG.info("Building features for arm %s...", name)
        if extra_df is not None:
            c_tr = pd.concat([train, extra_df[masks["train"]].set_index(train.index)], axis=1)
            c_va = pd.concat([valid, extra_df[masks["valid"]].set_index(valid.index)], axis=1)
            c_te = pd.concat([test, extra_df[masks["test"]].set_index(test.index)], axis=1)
        else:
            c_tr, c_va, c_te = train, valid, test
        builder = FeatureBuilder(FeatureSpec(feature_set="primary", extra_numeric=extra_cols)).fit(c_tr, c_va)
        prepared[name] = (builder, builder.transform(c_tr), builder.transform(c_va), builder.transform(c_te))

    scores: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    for fit_seed in seeds:
        for name, (builder, X_tr, X_va, X_te) in prepared.items():
            boosting = BoostingModel(
                builder.categorical_features, BoostingConfig(n_estimators=2000, early_stopping_rounds=100)
            ).fit(X_tr, y_train, X_va, y_valid, seed=fit_seed)
            calibrator = ProbabilityCalibrator("isotonic").fit(y_valid, boosting.predict_proba(X_va))
            pred = calibrator.transform(boosting.predict_proba(X_te))
            scores[(name, fit_seed)] = pred
            for window, mask in (("2015-07..12", h2), ("full 2015", win["full"])):
                m = compute_metrics(y_test[mask], pred[mask])
                rows.append(
                    {
                        "seed": fit_seed, "arm": name, "window": window, "roc_auc": m.roc_auc,
                        "ece": m.ece, "brier": m.brier, "calibration_slope": m.calibration_slope,
                        "observed_expected_ratio": m.observed_expected_ratio,
                    }
                )
            LOG.info("seed %d | %-24s | H2 ECE=%.4f", fit_seed, name,
                     [r for r in rows if r["seed"] == fit_seed and r["arm"] == name and r["window"] == "2015-07..12"][0]["ece"])
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(paths_root / "results" / "tables" / "cohort_controls_exp04_seeds.csv", index=False)

    # Primary seed (42): the existing single-fit outputs, kept for continuity.
    primary = seeds[0]
    p_scores = {name: scores[(name, primary)] for name in arm_specs}
    metrics = {k: compute_metrics(y_test, v) for k, v in p_scores.items()}
    metrics_h2 = {k: compute_metrics(y_test[h2], v[h2]) for k, v in p_scores.items()}

    LOG.info("Paired bootstrap, every seed, 2015-07..12 (2,000 resamples each)...")
    deltas = []
    boot = {}
    for fit_seed in seeds:
        b = scores[("B_tabular_plus_graph", fit_seed)][h2]
        for control, tag in (("A_plus_macro_default", "macro"), ("B_shuffled_cohort", "shuffled")):
            boot[(fit_seed, tag)] = paired_bootstrap_comparison(y_test[h2], b, scores[(control, fit_seed)][h2])
        a_arm = scores[("A_tabular", fit_seed)][h2]
        e_b, _ = expected_calibration_error(y_test[h2], b)
        e_a, _ = expected_calibration_error(y_test[h2], a_arm)
        deltas.append(
            {
                "seed": fit_seed,
                "delta_ece_vs_tabular": float(e_b - e_a),
                "delta_ece_vs_macro": boot[(fit_seed, "macro")]["delta_ece"],
                "delta_ece_vs_macro_ci_low": boot[(fit_seed, "macro")]["delta_ece_ci"][0],
                "delta_ece_vs_macro_ci_high": boot[(fit_seed, "macro")]["delta_ece_ci"][1],
                "delta_ece_vs_shuffled": boot[(fit_seed, "shuffled")]["delta_ece"],
                "delta_ece_vs_shuffled_ci_low": boot[(fit_seed, "shuffled")]["delta_ece_ci"][0],
                "delta_ece_vs_shuffled_ci_high": boot[(fit_seed, "shuffled")]["delta_ece_ci"][1],
                "delta_oe_gap_vs_macro": boot[(fit_seed, "macro")]["delta_oe_gap"],
                "delta_oe_gap_vs_shuffled": boot[(fit_seed, "shuffled")]["delta_oe_gap"],
            }
        )
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(paths_root / "results" / "tables" / "cohort_controls_exp04_seed_deltas.csv", index=False)

    h2_rows = per_seed[per_seed["window"] == "2015-07..12"]
    summary_table = h2_rows.groupby("arm")[["roc_auc", "ece", "observed_expected_ratio"]].agg(["mean", "std", "min", "max"])
    summary_table.columns = ["_".join(c) for c in summary_table.columns]
    summary_table.to_csv(paths_root / "results" / "tables" / "cohort_controls_exp04_seed_summary.csv")

    survives_all = bool(
        (delta_df["delta_ece_vs_macro_ci_high"] < 0).all() and (delta_df["delta_ece_vs_shuffled_ci_high"] < 0).all()
    )
    seed_summary = {
        "seeds": seeds,
        "window": "2015-07..12",
        "survives_controls_for_every_seed": survives_all,
        "delta_ece_vs_tabular": {"mean": float(delta_df["delta_ece_vs_tabular"].mean()), "std": float(delta_df["delta_ece_vs_tabular"].std(ddof=1)),
                                 "max": float(delta_df["delta_ece_vs_tabular"].max())},
        "delta_ece_vs_macro": {"mean": float(delta_df["delta_ece_vs_macro"].mean()), "std": float(delta_df["delta_ece_vs_macro"].std(ddof=1)),
                               "max": float(delta_df["delta_ece_vs_macro"].max())},
        "delta_ece_vs_shuffled": {"mean": float(delta_df["delta_ece_vs_shuffled"].mean()), "std": float(delta_df["delta_ece_vs_shuffled"].std(ddof=1)),
                                  "max": float(delta_df["delta_ece_vs_shuffled"].max())},
        "delta_oe_gap_vs_macro": {"mean": float(delta_df["delta_oe_gap_vs_macro"].mean()), "std": float(delta_df["delta_oe_gap_vs_macro"].std(ddof=1))},
        "delta_oe_gap_vs_shuffled": {"mean": float(delta_df["delta_oe_gap_vs_shuffled"].mean()), "std": float(delta_df["delta_oe_gap_vs_shuffled"].std(ddof=1))},
    }

    # Continuity outputs for the primary seed (existing file names).
    LOG.info("Primary-seed bootstrap on the full 2015 book...")
    comp_macro = paired_bootstrap_comparison(y_test, p_scores["B_tabular_plus_graph"], p_scores["A_plus_macro_default"])
    comp_shuffled = paired_bootstrap_comparison(y_test, p_scores["B_tabular_plus_graph"], p_scores["B_shuffled_cohort"])
    comp_macro_h2, comp_shuffled_h2 = boot[(primary, "macro")], boot[(primary, "shuffled")]
    res_df = compare_models(metrics)
    res_df.to_csv(paths_root / "results" / "tables" / "cohort_controls_exp04.csv")
    compare_models(metrics_h2).to_csv(paths_root / "results" / "tables" / "cohort_controls_exp04_h2.csv")

    print("\n" + "=" * 80)
    print("EXP04 NON-GRAPH CONTROLS: per-seed deltas on 2015-07..12 (B minus control; negative favours B)")
    print(delta_df.round(5).to_string(index=False))
    print(f"\nSurvives both controls for every seed: {survives_all}")
    print("=" * 80 + "\n")

    import json
    json_path = paths_root / "results" / "tables" / "cohort_controls_exp04.json"
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
                "seed_summary": seed_summary,
            },
            f,
            indent=2,
        )
    LOG.info("Saved %s", json_path)
    LOG.info("Total elapsed: %.1f seconds", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
