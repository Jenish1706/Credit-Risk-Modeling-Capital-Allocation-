#!/usr/bin/env python
"""EXP01 interval: paired bootstrap of the LightGBM-minus-logistic AUC difference on 2015-07..12.

``train_baseline.py`` reports each model's AUC as a point estimate. This script loads the two saved
models, scores the untouched 2015-07..12 holdout and resamples loans (paired, 2,000 draws) to put an
interval on the difference. It selects nothing; it only attaches an interval to a reported number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from credit_risk.data.splits import test_window_masks  # noqa: E402
from credit_risk.features.build import FeatureBuilder  # noqa: E402
from credit_risk.models.baseline import LogisticBaseline  # noqa: E402
from credit_risk.models.boosting import BoostingModel  # noqa: E402
from credit_risk.utils.cli import base_parser, load_prepared, resolve, save_json  # noqa: E402
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.exp01_ci")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/models/boosting.yaml")
    parser.add_argument("--model-run", default="boosting")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    _, paths, _, seed = resolve(args)

    stem = args.model_run
    builder = FeatureBuilder.load(paths.artifacts / f"{stem}_features.pkl")
    boosting = BoostingModel.load(paths.artifacts / f"{stem}_boosting.pkl")
    logistic = LogisticBaseline.load(paths.artifacts / f"{stem}_logistic.pkl")

    df = load_prepared(paths)
    test = df[df["split"] == "test"]
    h2 = test_window_masks(test["issue_period"])["holdout"]
    X = builder.transform(test[h2])
    y = test.loc[h2, "default"].astype(int).to_numpy()
    p_lgbm, p_logit = boosting.predict_proba(X), logistic.predict_proba(X)

    auc_g, auc_l = float(roc_auc_score(y, p_lgbm)), float(roc_auc_score(y, p_logit))
    rng = np.random.default_rng(seed)
    n, diffs = len(y), np.empty(args.n_bootstrap)
    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if y[idx].min() == y[idx].max():  # cannot score a single-class resample
            diffs[b] = np.nan
            continue
        diffs[b] = roc_auc_score(y[idx], p_lgbm[idx]) - roc_auc_score(y[idx], p_logit[idx])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    out = {
        "window": "2015-07..12 (H2, untouched holdout)",
        "n_loans": int(n),
        "auc_logistic": auc_l,
        "auc_lightgbm": auc_g,
        "auc_difference": auc_g - auc_l,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(seed),
    }
    save_json(out, paths, "exp01_auc_difference")
    LOG.info("AUC lightgbm - logistic on H2: %+.4f [%+.4f, %+.4f]", out["auc_difference"], lo, hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
