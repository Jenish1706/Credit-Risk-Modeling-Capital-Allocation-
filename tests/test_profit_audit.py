"""CR-02: predicted-vs-realised profit audit."""

from __future__ import annotations

import numpy as np

from credit_risk.decision.expected_loss import CashflowModel, EconomicsConfig
from credit_risk.decision.profit_audit import (
    gap_decomposition,
    paired_bootstrap_ci,
    profit_decile_table,
    realised_profit_after_costs,
    rule_realised_profit,
)

CASHFLOW = CashflowModel(good_intercept=0.10, good_slope=0.5, bad_intercept=0.70, bad_slope=-0.5)
ECON = EconomicsConfig()


def _book(n=5_000, seed=0, true_scale=1.0, yield_shift=0.0, loss_shift=0.0):
    rng = np.random.default_rng(seed)
    apr = rng.uniform(0.07, 0.28, n)
    exposure = rng.uniform(3_000, 30_000, n)
    pd_hat = np.clip(0.04 + 0.7 * apr * 0.5 + rng.normal(0, 0.02, n), 0.01, 0.8)
    defaulted = (rng.random(n) < np.clip(pd_hat * true_scale, 0, 1)).astype(float)
    ret = np.where(
        defaulted > 0.5,
        -(CASHFLOW.bad_loss_rate(apr) + loss_shift),
        CASHFLOW.good_yield(apr) + yield_shift,
    )
    return pd_hat, exposure, apr, ret, defaulted


def test_decomposition_sums_to_the_gap_exactly():
    pd_hat, e, apr, ret, d = _book(true_scale=1.4, yield_shift=-0.03, loss_shift=0.08)
    out = gap_decomposition(pd_hat, e, apr, ret, d, CASHFLOW)
    parts = out["pd_effect"] + out["yield_effect"] + out["severity_effect"]
    assert np.isclose(parts, out["gap_total"], rtol=1e-9)


def test_decomposition_isolates_each_error_source():
    base = gap_decomposition(*_book(), CASHFLOW)
    # Perfectly specified yield and loss curves: only default-count noise is left.
    assert abs(base["yield_effect"]) < 1e-6 and abs(base["severity_effect"]) < 1e-6

    worse_yield = gap_decomposition(*_book(yield_shift=-0.05), CASHFLOW)
    assert worse_yield["yield_effect"] < -1e3
    assert abs(worse_yield["severity_effect"]) < 1e-6

    worse_loss = gap_decomposition(*_book(loss_shift=0.10), CASHFLOW)
    assert worse_loss["severity_effect"] < -1e3
    assert abs(worse_loss["yield_effect"]) < 1e-6

    under_predicted = gap_decomposition(*_book(true_scale=2.0), CASHFLOW)
    assert under_predicted["pd_effect"] < -1e3


def test_decile_table_shapes_and_a_calibrated_book_has_small_gaps():
    pd_hat, e, apr, ret, d = _book(n=40_000)
    table = profit_decile_table(pd_hat, e, apr, ret, d, CASHFLOW, ECON)
    assert list(table["decile"]) == list(range(1, 11))
    assert table["n"].sum() == 40_000
    assert (table["mean_predicted_profit"].diff().dropna() >= 0).all()
    assert table["gap_realised_minus_predicted"].abs().max() < 0.05 * 30_000


def test_realised_profit_is_charged_the_same_costs_as_the_prediction():
    ret = np.array([0.10])
    out = realised_profit_after_costs(ret, np.array([10_000.0]), np.array([0.12]), ECON)
    expected = 1_000 - 10_000 * ECON.annual_cost_of_funds * ECON.weighted_average_life_years
    expected -= ECON.opex_fixed + ECON.opex_variable_rate * 10_000
    assert np.isclose(out[0], expected)


def test_rule_profit_is_zero_when_rejected():
    out = rule_realised_profit(
        np.array([True, False]), np.array([1e4, 1e4]), np.array([0.1, 0.1]),
        np.array([0.1, 0.1]), ECON,
    )
    assert out[1] == 0.0 and out[0] != 0.0


def test_paired_bootstrap_interval_contains_the_mean():
    diffs = np.random.default_rng(1).normal(5.0, 20.0, 2_000)
    mean, lo, hi = paired_bootstrap_ci(diffs, n_bootstrap=300)
    assert lo < mean < hi
