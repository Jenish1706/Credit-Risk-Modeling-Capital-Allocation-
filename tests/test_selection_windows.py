"""G4: selections read the 2015-01..06 window only, proved by poisoning the 2015-07..12 holdout.

The scripts' `assert_selection_window` calls check the frame they build; these tests check the
*functions that make the selections*: changing anything in the holdout must not change a choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.decision.blindspot import flag_cutpoints
from credit_risk.models.calibration import select_calibrator_on_window


def _frames(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_valid, n_dev, n_hold = 6_000, 3_000, 3_000
    true_p = lambda n: np.clip(rng.beta(2, 12, n), 0.01, 0.9)  # noqa: E731
    raw_valid, raw_test = true_p(n_valid) * 0.8, true_p(n_dev + n_hold) * 0.8  # an under-predicting model
    y_valid = (rng.random(n_valid) < raw_valid / 0.8).astype(float)
    y_test = (rng.random(n_dev + n_hold) < raw_test / 0.8).astype(float)
    months = pd.Series(["2015-03"] * n_dev + ["2015-09"] * n_hold)
    return y_valid, raw_valid, y_test, raw_test, months, n_dev


def test_calibrator_choice_is_scored_on_the_selection_window_only():
    y_valid, p_valid, y_test, p_test, months, n_dev = _frames()
    best, briers, calibrated = select_calibrator_on_window(y_valid, p_valid, y_test, p_test, months)
    for method, brier in briers.items():
        manual = float(np.mean((calibrated[method][:n_dev] - y_test[:n_dev]) ** 2))
        assert np.isclose(brier, manual), f"{method} Brier was not computed on 2015-01..06 rows only"
    assert best == min(briers, key=briers.__getitem__)


def test_poisoning_the_holdout_cannot_change_the_calibrator_choice():
    y_valid, p_valid, y_test, p_test, months, n_dev = _frames()
    best, briers, _ = select_calibrator_on_window(y_valid, p_valid, y_test, p_test, months)

    y_poison, p_poison = y_test.copy(), p_test.copy()
    y_poison[n_dev:] = 1.0          # every holdout loan defaults
    p_poison[n_dev:] = 0.999        # and the model is wildly off on it
    best_p, briers_p, _ = select_calibrator_on_window(y_valid, p_valid, y_poison, p_poison, months)

    assert best_p == best
    assert briers_p == briers, "a holdout change moved the selection-window Brier scores"


def test_flag_cutpoints_come_from_the_selection_window_only():
    rng = np.random.default_rng(1)
    months = pd.Series(["2015-02"] * 4_000 + ["2015-11"] * 6_000)
    signals = {"anomaly": rng.normal(size=10_000), "disagreement": rng.normal(size=10_000)}
    cuts = flag_cutpoints(signals, ["anomaly"], months, 0.90)
    assert set(cuts) == {"anomaly"}
    assert np.isclose(cuts["anomaly"], np.quantile(signals["anomaly"][:4_000], 0.90))

    poisoned = {k: v.copy() for k, v in signals.items()}
    poisoned["anomaly"][4_000:] = 1e6
    assert flag_cutpoints(poisoned, ["anomaly"], months, 0.90) == cuts
