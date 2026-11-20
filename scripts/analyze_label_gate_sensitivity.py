#!/usr/bin/env python
"""G4: how much does the same-month rule for neighbour labels matter?

    python scripts/analyze_label_gate_sensitivity.py

Cohort label aggregates admit a neighbour ``j`` of loan ``i`` when ``resolved[j] <= issue[i]``. Both
are *months*, so a neighbour that resolved in the same month the applicant was issued is admitted
even though, within the month, its outcome may have been booked after the applicant's decision. A
strictly-before rule (``resolved[j] < issue[i]``) is more conservative.

This script rebuilds the cohort features under the strict rule (by moving every
``outcome_observed_period`` one month later, which is the same as requiring ``resolved < issue``),
refits the tabular and tabular-plus-cohort arms for three seeds, and compares calibration on
2015-07..12 with the current rule. Nothing is selected here; both rules are reported.

Writes ``results/tables/label_gate_sensitivity.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from credit_risk.data.splits import test_window_masks  # noqa: E402
from credit_risk.evaluation.metrics import compute_metrics  # noqa: E402
from credit_risk.features.build import FeatureBuilder, FeatureSpec  # noqa: E402
from credit_risk.features.financial import add_derived_features  # noqa: E402
from credit_risk.graph.features import CohortFeatureBuilder, CohortFeatureConfig  # noqa: E402
from credit_risk.models.boosting import BoostingConfig, BoostingModel  # noqa: E402
from credit_risk.models.calibration import ProbabilityCalibrator  # noqa: E402
from credit_risk.utils.cli import save_table  # noqa: E402
from credit_risk.utils.config import Paths, load_config  # noqa: E402
from credit_risk.utils.runtime import get_logger  # noqa: E402
from scripts.build_graph import _relations  # noqa: E402

LOG = get_logger("scripts.label_gate")
SEEDS = (42, 43, 44)


def main() -> int:
    paths = Paths()
    config = load_config("configs/graph/cohort_graph.yaml")
    df = add_derived_features(pd.read_parquet(paths.data_processed / "loans.parquet").reset_index(drop=True))
    relations = _relations(config)
    ccfg = config.get("cohort_features", {})
    cohort_config = CohortFeatureConfig(
        relations=relations,
        lookback_months=int(ccfg.get("lookback_months", 36)),
        smoothing_alpha=float(ccfg.get("smoothing_alpha", 25.0)),
        feature_columns=tuple(ccfg.get("feature_columns", ["fico_mid", "dti_clean", "loan_amnt", "annual_inc"])),
    )

    current = pd.read_parquet(paths.data_processed / "cohort_features.parquet")
    strict_df = df.copy()
    observed = strict_df["outcome_observed_period"]
    known = observed.notna()
    moved = observed.astype("object").copy()
    moved[known] = (pd.PeriodIndex(observed[known].astype(str), freq="M") + 1).astype(str)
    strict_df["outcome_observed_period"] = moved.astype("string")
    LOG.info("rebuilding cohort features with the strictly-before label rule...")
    builder = CohortFeatureBuilder(cohort_config).fit(strict_df[strict_df["split"] == "train"])
    strict = builder.transform(strict_df)

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}
    train, valid, test = (df[masks[s]] for s in ("train", "valid", "test"))
    y = df["default"].astype("float32").to_numpy()
    y_tr, y_va, y_te = (y[masks[s]] for s in ("train", "valid", "test"))
    h2 = test_window_masks(test["issue_period"])["holdout"]

    arms = {"A_tabular": (None, ()), "B_cohort_current_rule": (current, tuple(current.columns)),
            "B_cohort_strict_rule": (strict, tuple(strict.columns))}
    rows = []
    for name, (extra, cols) in arms.items():
        parts = [train, valid, test]
        if extra is not None:
            parts = [d.copy() for d in parts]
            for d, m in zip(parts, (masks["train"], masks["valid"], masks["test"]), strict=True):
                sub = extra.loc[m, list(cols)].set_index(d.index)
                for c in cols:
                    d[c] = sub[c]
        b = FeatureBuilder(FeatureSpec(feature_set="primary", extra_numeric=tuple(cols))).fit(parts[0], parts[1])
        X = [b.transform(d) for d in parts]
        for seed in SEEDS:
            model = BoostingModel(b.categorical_features, BoostingConfig(n_estimators=2000, early_stopping_rounds=100)).fit(
                X[0], y_tr, X[1], y_va, seed=seed
            )
            cal = ProbabilityCalibrator("isotonic").fit(y_va, model.predict_proba(X[1]))
            p = cal.transform(model.predict_proba(X[2]))
            m = compute_metrics(y_te[h2], p[h2])
            rows.append({"arm": name, "seed": seed, "roc_auc": m.roc_auc, "ece": m.ece,
                         "observed_expected_ratio": m.observed_expected_ratio, "window": "2015-07..12"})
            LOG.info("%-24s seed %d | AUC %.4f ECE %.4f O/E %.3f", name, seed, m.roc_auc, m.ece, m.observed_expected_ratio)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("arm")[["roc_auc", "ece", "observed_expected_ratio"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    save_table(summary.reset_index(), paths, "label_gate_sensitivity", index=False)
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
