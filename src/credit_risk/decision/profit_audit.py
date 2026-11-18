"""Does the model's *predicted* expected profit match the profit that was actually realised?

CR-02. The expected-profit rule in :mod:`credit_risk.decision.policies` ranks loans by

    E[profit] = (1 - PD) * y_good(APR) * E  -  PD * l_bad(APR) * E  -  funding  -  opex

where ``y_good`` and ``l_bad`` come from a two-parameter fit on the training window. When that rule
loses to a fixed PD threshold, either the PD, the yield curve, or the loss curve is wrong for the
evaluated book. This module separates the three, in dollars, on loans whose real outcome is known.

Everything here is evaluated **at the observed exposure** (the amount actually funded), so every
realised number is a measurement and none depends on the simulator's counterfactual coupling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.decision.expected_loss import (
    CashflowModel,
    EconomicsConfig,
    expected_profit,
    funding_cost,
    operating_cost,
)


def realised_profit_after_costs(
    realised_return: np.ndarray,
    exposure: np.ndarray,
    apr: np.ndarray,
    economics: EconomicsConfig,
) -> np.ndarray:
    """Realised profit in dollars, charged the same funding and opex as ``expected_profit``.

    ``realised_return`` is profit per dollar lent *before* funding cost and opex (cash received
    minus funded amount, over funded amount), so this is directly comparable to the prediction.
    """
    exposure = np.asarray(exposure, dtype=float)
    return (
        np.asarray(realised_return, dtype=float) * exposure
        - funding_cost(exposure, economics, apr)
        - operating_cost(exposure, economics)
    )


def _predicted_before_costs(p, e, y, ell) -> np.ndarray:
    return (1.0 - p) * y * e - p * ell * e


def gap_decomposition(
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    apr: np.ndarray,
    realised_return: np.ndarray,
    defaulted: np.ndarray,
    cashflow: CashflowModel,
) -> dict[str, float]:
    """Split (realised - predicted) expected profit into three dollar effects that sum exactly.

    With ``p`` the predicted PD, ``d`` the 0/1 outcome, ``y``/``l`` the modelled good yield and bad
    loss rate and ``R`` the realised return per dollar, the funding and opex terms cancel and

        gap = sum E * [ (R - y)(1 - d)  +  (R + l) d  -  (y + l)(d - p) ]
                         yield error      severity error    default-count error

    * **pd_effect** ``-(y + l) E (d - p)``: more defaults than predicted cost ``y + l`` each.
    * **yield_effect** ``(R - y) E`` over loans that repaid: the good-loan yield curve is off.
    * **severity_effect** ``(R + l) E`` over loans that defaulted: the loss-rate curve is off.
    """
    p = np.asarray(pd_hat, dtype=float)
    e = np.asarray(exposure, dtype=float)
    d = np.asarray(defaulted, dtype=float)
    r = np.asarray(realised_return, dtype=float)
    y = np.asarray(cashflow.good_yield(apr), dtype=float)
    ell = np.asarray(cashflow.bad_loss_rate(apr), dtype=float)
    ok = np.isfinite(p) & np.isfinite(e) & np.isfinite(d) & np.isfinite(r)
    p, e, d, r, y, ell = (a[ok] for a in (p, e, d, r, y, ell))

    predicted = float(_predicted_before_costs(p, e, y, ell).sum())
    realised = float((r * e).sum())
    return {
        "n": int(ok.sum()),
        "predicted_total": predicted,
        "realised_total": realised,
        "gap_total": realised - predicted,
        "pd_effect": float((-(y + ell) * e * (d - p)).sum()),
        "yield_effect": float(((r - y) * e * (1.0 - d)).sum()),
        "severity_effect": float(((r + ell) * e * d).sum()),
    }


def profit_decile_table(
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    apr: np.ndarray,
    realised_return: np.ndarray,
    defaulted: np.ndarray,
    cashflow: CashflowModel,
    economics: EconomicsConfig,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs realised profit per loan, by decile of *predicted* expected profit.

    Decile 1 is the lowest predicted profit. A well-specified rule has realised profit rising with
    the decile and close to the prediction in each one.
    """
    pd_hat = np.asarray(pd_hat, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    apr = np.asarray(apr, dtype=float)
    ret = np.asarray(realised_return, dtype=float)
    d = np.asarray(defaulted, dtype=float)

    frame = pd.DataFrame(
        {
            "pd_hat": pd_hat, "default": d, "ret": ret,
            "predicted": expected_profit(pd_hat, exposure, apr, cashflow, economics),
            "realised": realised_profit_after_costs(ret, exposure, apr, economics),
            "y_hat": cashflow.good_yield(apr), "l_hat": cashflow.bad_loss_rate(apr),
        }
    ).dropna()
    frame["decile"] = pd.qcut(frame["predicted"].rank(method="first"), n_bins, labels=False) + 1

    rows = []
    for decile, g in frame.groupby("decile"):
        good, bad = g[g["default"] < 0.5], g[g["default"] >= 0.5]
        rows.append(
            {
                "decile": int(decile),
                "n": len(g),
                "mean_pd_hat": g["pd_hat"].mean(),
                "observed_default_rate": g["default"].mean(),
                "mean_predicted_profit": g["predicted"].mean(),
                "mean_realised_profit": g["realised"].mean(),
                "gap_realised_minus_predicted": g["realised"].mean() - g["predicted"].mean(),
                "mean_modelled_good_yield": good["y_hat"].mean(),
                "mean_realised_good_return": good["ret"].mean(),
                "mean_modelled_bad_loss": bad["l_hat"].mean(),
                "mean_realised_bad_loss": -bad["ret"].mean(),
            }
        )
    return pd.DataFrame(rows)


def rule_realised_profit(
    accept: np.ndarray,
    exposure: np.ndarray,
    apr: np.ndarray,
    realised_return: np.ndarray,
    economics: EconomicsConfig,
) -> np.ndarray:
    """Per-loan realised profit if ``accept`` loans are funded at the observed exposure, else 0."""
    profit = realised_profit_after_costs(realised_return, exposure, apr, economics)
    return np.where(np.asarray(accept, dtype=bool), np.nan_to_num(profit), 0.0)


def paired_bootstrap_ci(
    differences: np.ndarray, n_bootstrap: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile CI of per-loan paired differences (resampling loans)."""
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(differences)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        means[b] = differences[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(differences.mean()), float(lo), float(hi)
