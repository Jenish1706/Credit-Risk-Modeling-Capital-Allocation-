"""G4: selections read the 2015-01..06 window only, proved by poisoning the 2015-07..12 holdout.

These test the *functions that make the three kinds of selection* (the calibrator, the flag
cut-points and the KEEP/REJECT decision): changing anything in the holdout must not change a choice.
Each function also asserts the exact rows it reads, but that assert is only a tripwire; the poisoning
below is the evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.decision.blindspot import conditional_by_window, flag_cutpoints
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


def _signal_book(seed: int = 3, n_dev: int = 30_000, n_hold: int = 30_000):
    """A book whose signal is informative in 2015-01..06 and pure noise in 2015-07..12."""
    rng = np.random.default_rng(seed)
    n = n_dev + n_hold
    p = np.clip(rng.beta(2, 10, n), 0.02, 0.8)
    s = rng.normal(size=n)
    boost = np.where(np.arange(n) < n_dev, 0.9, 0.0)  # the signal only matters in the dev window
    true = np.clip(p * np.exp(boost * s), 0, 0.95)
    y = (rng.random(n) < true).astype(float)
    months = pd.Series(["2015-03"] * n_dev + ["2015-09"] * n_hold)
    return y, p, s, months, n_dev


def test_keep_reject_decision_ignores_the_holdout():
    y, p, s, months, n_dev = _signal_book()
    clean = conditional_by_window(y, p, s, months)
    assert clean.decision == "KEEP", "fixture should be informative in the selection window"

    # Poison 2015-07..12: make the signal look harmful there (and break its link to the outcome).
    y_bad = y.copy()
    y_bad[n_dev:] = (s[n_dev:] < 0).astype(float)
    p_bad = p.copy()
    p_bad[n_dev:] = np.clip(p[n_dev:] * 0.1, 1e-3, 0.5)
    poisoned = conditional_by_window(y_bad, p_bad, s, months)

    assert poisoned.decision == clean.decision
    assert poisoned.selection.n_strata_positive == clean.selection.n_strata_positive
    assert poisoned.selection.sign_test_p_twosided == clean.selection.sign_test_p_twosided
    assert poisoned.selection.odds_ratio_per_sd == clean.selection.odds_ratio_per_sd
    # the poison must bite where it is allowed to: the reported holdout column changes
    assert poisoned.holdout.odds_ratio_per_sd != clean.holdout.odds_ratio_per_sd


def test_keep_reject_decision_reads_only_selection_rows():
    """The decision is the selection window's result, not the holdout's or the full book's."""
    y, p, s, months, n_dev = _signal_book()
    windowed = conditional_by_window(y, p, s, months)
    from credit_risk.decision.blindspot import conditional_signal_table

    direct = conditional_signal_table(y[:n_dev], p[:n_dev], s[:n_dev])
    assert windowed.selection.odds_ratio_per_sd == direct.odds_ratio_per_sd
    assert windowed.selection.n_strata_positive == direct.n_strata_positive
    assert windowed.decision == ("KEEP" if direct.keep else "REJECT")
