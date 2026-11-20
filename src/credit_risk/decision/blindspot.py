"""Blind-spot signals and the portfolio comparison that uses them.

This module exists because the analysis scripts for EXP08-EXP11 had each grown their own copy of
the credit economics, and one of those copies was wrong in a way that inverted a headline result.
Everything economic now routes through :mod:`credit_risk.decision.expected_loss`.

The central research question these signals serve:

    *Can representation-learning signals identify situations where a conventional credit-risk
    model is unreliable, and can that information improve lending decisions?*

Two distinct claims are involved, and they need separate evidence:

1. **Detection.** Does the signal identify borrowers whose PD is untrustworthy? The honest test is
   *conditional*: within a narrow PD stratum, does a high signal still predict that the model
   under-states risk? A marginal quintile table cannot answer this, because both signals correlate
   with the PD level itself, and a boosted model is already less well calibrated at high PD.
2. **Decision value.** Does acting on the signal improve the book? This must be measured on the
   **whole portfolio at equal capital**, using **observed cashflows** -- not by comparing the two
   swap sets, and not with an assumed LGD.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.decision.expected_loss import (
    CashflowModel,
    EconomicsConfig,
    expected_profit,
    realised_profit,
)
from credit_risk.utils.runtime import get_logger

LOG = get_logger("decision.blindspot")


# ---------------------------------------------------------------------------
# 1. Detection: is the signal informative beyond the PD level?
# ---------------------------------------------------------------------------

def marginal_signal_table(
    y_true: np.ndarray,
    pd_estimate: np.ndarray,
    signal: np.ndarray,
    n_bins: int = 5,
    labels: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Model performance by signal quantile -- the *marginal* view.

    Useful for description, but **not** sufficient evidence that a signal finds blind spots: if the
    signal correlates with PD, this table will show calibration degrading simply because the model
    is less well calibrated at higher PD. Always read it beside
    :func:`conditional_signal_table`.
    """
    from credit_risk.evaluation.metrics import compute_metrics

    frame = pd.DataFrame({"y": y_true, "pd": pd_estimate, "signal": signal})
    frame["bin"] = pd.qcut(frame["signal"], q=n_bins, labels=labels, duplicates="drop")

    rows = []
    for name, group in frame.groupby("bin", observed=True):
        metrics = compute_metrics(group["y"].to_numpy(), group["pd"].to_numpy())
        rows.append(
            {
                "quantile": str(name),
                "n": len(group),
                "mean_signal": float(group["signal"].mean()),
                "default_rate": metrics.positive_rate,
                "mean_pd": metrics.mean_prediction,
                "calibration_gap": metrics.positive_rate - metrics.mean_prediction,
                "roc_auc": metrics.roc_auc,
                "brier": metrics.brier,
                "ece": metrics.ece,
                "predicted_observed_ratio": metrics.predicted_observed_ratio,
                "observed_expected_ratio": metrics.observed_expected_ratio,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ConditionalResult:
    """Does the signal predict miscalibration *within* PD strata?"""

    table: pd.DataFrame
    n_strata: int
    n_strata_positive: int
    mean_gap_difference: float
    sign_test_p: float
    odds_ratio_per_sd: float
    spearman_with_pd: float
    odds_ratio_ci_lower: float = 0.0
    odds_ratio_ci_upper: float = 0.0
    sign_test_p_twosided: float = 1.0
    sign_test_p_onesided: float = 1.0

    @property
    def keep(self) -> bool:
        """The single KEEP rule: two-sided sign test below 5% with the gap in the right direction."""
        return bool(self.sign_test_p_twosided < 0.05 and self.mean_gap_difference > 0)

    def verdict(self, name: str) -> str:
        direction = "under-states" if self.mean_gap_difference > 0 else "over-states"
        strength = (
            "survives" if self.sign_test_p_twosided < 0.05 and self.mean_gap_difference > 0
            else "does not survive"
        )
        return (
            f"{name}: correlation with PD itself is {self.spearman_with_pd:+.3f}. "
            f"Within PD strata the high-signal half shows a larger gap in "
            f"{self.n_strata_positive}/{self.n_strata} strata "
            f"(mean {self.mean_gap_difference:+.4f}, sign-test two-sided p={self.sign_test_p_twosided:.4f}, "
            f"one-sided p={self.sign_test_p_onesided:.4f}), so the "
            f"effect **{strength}** the PD control -- the model {direction} risk for high-signal "
            f"borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal "
            f"multiplies default odds by {self.odds_ratio_per_sd:.3f} "
            f"(95% CI [{self.odds_ratio_ci_lower:.3f}, {self.odds_ratio_ci_upper:.3f}])."
        )


def conditional_signal_table(
    y_true: np.ndarray,
    pd_estimate: np.ndarray,
    signal: np.ndarray,
    n_strata: int = 10,
    min_per_half: int = 200,
) -> ConditionalResult:
    """Test the signal *holding the PD level fixed*.

    Within each PD decile the population is split at the median of the signal, and the calibration
    gap (observed minus predicted default rate) is compared between halves. If the signal only
    restates the PD level, the two halves inside a stratum will show the same gap and the effect
    vanishes here even when the marginal table looks dramatic.

    Two summaries are returned: an exact binomial sign test across strata, and the coefficient on
    the standardised signal in a logistic regression of the outcome on ``logit(PD)`` plus the
    signal -- the incremental information the signal carries about the outcome itself.
    """
    from scipy.stats import binomtest
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(pd_estimate, dtype=float), 1e-6, 1 - 1e-6)
    s = np.asarray(signal, dtype=float)

    frame = pd.DataFrame({"y": y, "p": p, "s": s})
    frame["stratum"] = pd.qcut(frame["p"], n_strata, labels=False, duplicates="drop")

    rows = []
    for stratum, group in frame.groupby("stratum"):
        median = group["s"].median()
        high, low = group[group["s"] > median], group[group["s"] <= median]
        if len(high) < min_per_half or len(low) < min_per_half:
            continue
        rows.append(
            {
                "pd_stratum": int(stratum),
                "mean_pd": float(group["p"].mean()),
                "n_high": len(high),
                "n_low": len(low),
                "predicted_high": float(high["p"].mean()),
                "observed_high": float(high["y"].mean()),
                "gap_high": float(high["y"].mean() - high["p"].mean()),
                "predicted_low": float(low["p"].mean()),
                "observed_low": float(low["y"].mean()),
                "gap_low": float(low["y"].mean() - low["p"].mean()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no PD stratum had enough rows on both sides of the signal median")
    table["gap_difference"] = table["gap_high"] - table["gap_low"]

    positive = int((table["gap_difference"] > 0).sum())
    total = int(len(table))
    sign_p_onesided = float(binomtest(positive, total, 0.5, alternative="greater").pvalue)
    sign_p_twosided = float(binomtest(positive, total, 0.5, alternative="two-sided").pvalue)

    logit_p = np.log(p / (1 - p))
    z = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    X = np.column_stack([logit_p, z])
    model = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)

    # Wald 95% confidence interval for odds ratio from Fisher information matrix
    pred_p = model.predict_proba(X)[:, 1]
    w = pred_p * (1.0 - pred_p)
    H = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(H)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")

    beta1 = float(model.coef_[0][1])
    or_point = float(np.exp(beta1))
    if np.isfinite(se):
        ci_lower = float(np.exp(beta1 - 1.95996 * se))
        ci_upper = float(np.exp(beta1 + 1.95996 * se))
    else:
        ci_lower, ci_upper = or_point, or_point

    return ConditionalResult(
        table=table,
        n_strata=total,
        n_strata_positive=positive,
        mean_gap_difference=float(table["gap_difference"].mean()),
        sign_test_p=sign_p_onesided,
        odds_ratio_per_sd=or_point,
        spearman_with_pd=float(pd.Series(s).corr(pd.Series(p), method="spearman")),
        odds_ratio_ci_lower=ci_lower,
        odds_ratio_ci_upper=ci_upper,
        sign_test_p_twosided=sign_p_twosided,
        sign_test_p_onesided=sign_p_onesided,
    )


@dataclass
class WindowedConditional:
    """The conditional test run on the three windows of the 2015 test book (CR-03).

    ``selection`` (2015-01..06) alone decides KEEP/REJECT. ``holdout`` (2015-07..12) is the
    headline evidence, since nothing has been chosen on it. ``full`` is the continuity column.
    """

    selection: ConditionalResult
    holdout: ConditionalResult
    full: ConditionalResult

    @property
    def decision(self) -> str:
        return "KEEP" if self.selection.keep else "REJECT"

    @property
    def holdout_confirms(self) -> bool:
        return self.holdout.keep

    def summary(self) -> dict[str, dict]:
        out = {}
        for name, r in (("selection_h1", self.selection), ("holdout_h2", self.holdout),
                        ("full_2015", self.full)):
            out[name] = {
                "n_strata": r.n_strata, "n_strata_positive": r.n_strata_positive,
                "mean_gap_difference": r.mean_gap_difference,
                "sign_test_p_twosided": r.sign_test_p_twosided,
                "sign_test_p_onesided": r.sign_test_p_onesided,
                "odds_ratio_per_sd": r.odds_ratio_per_sd,
                "odds_ratio_ci_lower": r.odds_ratio_ci_lower,
                "odds_ratio_ci_upper": r.odds_ratio_ci_upper,
                "spearman_with_pd": r.spearman_with_pd,
                "keep": r.keep,
            }
        return out

    def markdown(self) -> str:
        rows = [
            "| window | strata with larger gap | mean gap difference | sign test (two-sided) | odds ratio / SD (95% CI) |",
            "|---|---|---|---|---|",
        ]
        from credit_risk.data.splits import WINDOW_LABELS
        for key, r in (("dev", self.selection), ("holdout", self.holdout), ("full", self.full)):
            rows.append(
                f"| {WINDOW_LABELS[key]} | {r.n_strata_positive}/{r.n_strata} | "
                f"{r.mean_gap_difference:+.4f} | {r.sign_test_p_twosided:.4f} | "
                f"{r.odds_ratio_per_sd:.3f} [{r.odds_ratio_ci_lower:.3f}, {r.odds_ratio_ci_upper:.3f}] |"
            )
        return "\n".join(rows)


def conditional_by_window(
    y_true: np.ndarray,
    pd_estimate: np.ndarray,
    signal: np.ndarray,
    issue_period: pd.Series,
    n_strata: int = 10,
    min_per_half: int = 200,
) -> WindowedConditional:
    """Run :func:`conditional_signal_table` on 2015-H1, 2015-H2 and the full test book.

    ``selection`` (which alone decides KEEP/REJECT) is built by ``select``, which asserts that the
    exact rows it reads end at the dev cutoff; ``holdout`` and ``full`` are built by ``report``, which
    is for reporting only. ``tests/test_selection_windows.py`` poisons the holdout and shows the
    decision does not move.
    """
    from credit_risk.data.splits import assert_selection_window, test_window_masks

    periods = pd.Series(issue_period).to_numpy()
    masks = test_window_masks(issue_period)
    y, p, s = (np.asarray(a) for a in (y_true, pd_estimate, signal))

    def report(mask: np.ndarray) -> ConditionalResult:
        return conditional_signal_table(
            y[mask], p[mask], s[mask], n_strata=n_strata, min_per_half=min_per_half
        )

    def select(mask: np.ndarray) -> ConditionalResult:
        assert_selection_window(periods[mask])  # the rows actually read, not a mask rebuilt for the check
        return report(mask)

    return WindowedConditional(
        selection=select(masks["dev"]), holdout=report(masks["holdout"]), full=report(masks["full"])
    )


# ---------------------------------------------------------------------------
# 2. Decision value: does acting on the signal improve the book?
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """One policy's book, measured on realised cashflows."""

    policy: str
    n_loans: int
    capital_deployed: float
    observed_profit: float
    return_on_capital: float
    default_rate: float
    mean_apr: float
    budget_utilisation: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SwapAnalysis:
    """Comparison of two capital-constrained books built from the same applicant pool."""

    portfolios: list[PortfolioResult]
    n_swapped_out: int
    n_swapped_in: int
    swapped_out_default_rate: float
    swapped_in_default_rate: float
    swapped_out_return: float
    swapped_in_return: float
    budget: float
    notes: list[str] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.portfolios])

    def verdict(self, treatment: str, baseline: str) -> str:
        a = next(p for p in self.portfolios if p.policy == baseline)
        b = next(p for p in self.portfolios if p.policy == treatment)
        d_profit = b.observed_profit - a.observed_profit
        d_roc = b.return_on_capital - a.return_on_capital
        d_capital = b.capital_deployed - a.capital_deployed
        material = abs(d_roc) >= 0.005  # 50bp of return on capital

        headline = (
            f"{treatment} changed realised profit by **${d_profit:+,.0f}** on a "
            f"${a.capital_deployed:,.0f} book, deploying ${d_capital:+,.0f} of capital "
            f"(a {d_roc:+.2%} change in return on capital)."
        )
        if material:
            reading = (
                "That is a material change in return on capital and the direction is the finding."
            )
        else:
            reading = (
                "That is **below 50bp of return on capital and should be read as no material "
                "difference in profitability**. The book default rate moved from "
                f"{a.default_rate:.2%} to {b.default_rate:.2%} at essentially unchanged return, "
                "so the defensible claim is risk reduction at constant profitability -- not a "
                "profit improvement."
            )
        return headline + " " + reading


def build_portfolio(
    df: pd.DataFrame,
    approve: pd.Series | np.ndarray,
    policy: str,
    budget: float,
    exposure_column: str = "funded_amnt",
) -> PortfolioResult:
    """Summarise an approved book using **observed** cashflows.

    Realised profit is read from the data, never modelled: it is
    ``total_pymnt - collection_recovery_fee - funded_amnt`` (total_pymnt already includes recoveries).
    Using an assumed LGD here is what inverted the original EXP11 conclusion.
    """
    mask = np.asarray(approve, dtype=bool)
    book = df.loc[mask]
    exposure = pd.to_numeric(book[exposure_column], errors="coerce")
    capital = float(exposure.sum())
    profit = float(realised_profit(book).sum())

    return PortfolioResult(
        policy=policy,
        n_loans=int(mask.sum()),
        capital_deployed=capital,
        observed_profit=profit,
        return_on_capital=profit / capital if capital > 0 else float("nan"),
        default_rate=float(pd.to_numeric(book["default"], errors="coerce").mean()),
        mean_apr=float(pd.to_numeric(book["int_rate"], errors="coerce").mean() / 100.0),
        budget_utilisation=capital / budget if budget > 0 else float("nan"),
    )


def greedy_allocate(
    density: pd.Series,
    exposure: pd.Series,
    budget: float,
    min_density: float = 0.0,
) -> pd.Series:
    """Fill a capital budget in descending profit-per-dollar order.

    The continuous-relaxation solution to the knapsack: with divisible capital, ranking by profit
    density and filling until the budget binds is optimal. Loans below ``min_density`` are never
    funded regardless of remaining budget, since they lose money in expectation.
    """
    order = density.sort_values(ascending=False, kind="mergesort").index
    cumulative = exposure.loc[order].cumsum()
    approve = pd.Series(False, index=density.index)
    approve.loc[order] = (cumulative.to_numpy() <= budget) & (
        density.loc[order].to_numpy() > min_density
    )
    return approve


def compare_capital_constrained(
    df: pd.DataFrame,
    pd_estimate: np.ndarray,
    flagged: np.ndarray,
    cashflow_model: CashflowModel,
    economics: EconomicsConfig | None = None,
    budget_fraction: float = 0.80,
    exposure_column: str = "funded_amnt",
) -> SwapAnalysis:
    """Compare a PD-ranked book against a blind-spot-aware book at **identical budget**.

    Both policies rank by expected profit per dollar and fill the same capital budget. The only
    difference is that the aware policy refuses to fund flagged borrowers.

    Three corrections relative to the original EXP11 implementation, each of which mattered:

    * profit is **observed**, not computed from an assumed LGD;
    * the comparison is between the two **portfolios**, not between the two swap sets, since the
      swap sets need not deploy the same capital;
    * **return on capital** is reported alongside profit, because a book that deploys more capital
      will usually earn more profit without being better.
    """
    economics = economics or EconomicsConfig()
    frame = df.reset_index(drop=True).copy()
    pd_hat = np.asarray(pd_estimate, dtype=float)
    flag = np.asarray(flagged, dtype=bool)

    exposure = pd.to_numeric(frame[exposure_column], errors="coerce").fillna(0.0)
    apr = pd.to_numeric(frame["int_rate"], errors="coerce").fillna(0.13) / 100.0

    profit = expected_profit(pd_hat, exposure.to_numpy(), apr.to_numpy(), cashflow_model, economics)
    density = pd.Series(
        np.divide(profit, exposure.to_numpy(), out=np.zeros_like(profit),
                  where=exposure.to_numpy() > 0),
        index=frame.index,
    )

    fundable = exposure[density > 0].sum()
    budget = float(fundable * budget_fraction)

    aware_density = density.copy()
    aware_density[flag] = -np.inf

    approve_a = greedy_allocate(density, exposure, budget)
    approve_b = greedy_allocate(aware_density, exposure, budget)

    swapped_out = approve_a & ~approve_b
    swapped_in = ~approve_a & approve_b

    def _return(mask: pd.Series) -> float:
        capital = exposure[mask].sum()
        return float(realised_profit(frame.loc[mask]).sum() / capital) if capital > 0 else float("nan")

    analysis = SwapAnalysis(
        portfolios=[
            build_portfolio(frame, approve_a, "A_pd_ranked", budget, exposure_column),
            build_portfolio(frame, approve_b, "B_blindspot_aware", budget, exposure_column),
        ],
        n_swapped_out=int(swapped_out.sum()),
        n_swapped_in=int(swapped_in.sum()),
        swapped_out_default_rate=float(
            pd.to_numeric(frame.loc[swapped_out, "default"], errors="coerce").mean()
        ),
        swapped_in_default_rate=float(
            pd.to_numeric(frame.loc[swapped_in, "default"], errors="coerce").mean()
        ),
        swapped_out_return=_return(swapped_out),
        swapped_in_return=_return(swapped_in),
        budget=budget,
    )

    for portfolio in analysis.portfolios:
        LOG.info(
            "%-20s n=%s capital=$%.0f profit=$%.0f RoC=%.4f default=%.4f",
            portfolio.policy,
            f"{portfolio.n_loans:,}",
            portfolio.capital_deployed,
            portfolio.observed_profit,
            portfolio.return_on_capital,
            portfolio.default_rate,
        )
    return analysis


def exposure_capped_policy(
    density: pd.Series,
    exposure: pd.Series,
    flagged: np.ndarray,
    budget: float,
    cap_multiplier: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Fund flagged borrowers at reduced exposure instead of refusing them.

    The alternative the original EXP11 conclusion pointed at without testing: if a flagged borrower
    is *mispriced* rather than simply bad, the response is to keep the yield and cut the exposure.
    Returns ``(approve, exposure_used)``.
    """
    adjusted = exposure.copy()
    adjusted[flagged] = adjusted[flagged] * cap_multiplier
    approve = greedy_allocate(density, adjusted, budget)
    return approve, adjusted.where(approve, 0.0)


@dataclass
class PlaceboComparison:
    random_default_mean: float
    random_default_ci: tuple[float, float]
    random_roc_mean: float
    random_roc_ci: tuple[float, float]
    apr_matched_default_mean: float
    apr_matched_default_ci: tuple[float, float]
    apr_matched_roc_mean: float
    apr_matched_roc_ci: tuple[float, float]
    signal_beats_random: bool
    signal_beats_apr_matched: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_placebo_controls(
    density: pd.Series,
    exposure: pd.Series,
    defaults: pd.Series,
    realised_profits: pd.Series,
    flagged: np.ndarray,
    apr: pd.Series,
    budget: float,
    signal_default_rate: float,
    n_draws: int = 200,
    seed: int = 42,
) -> PlaceboComparison:
    """Compare the signal-exclusion policy against random and APR-matched placebo exclusions.

    P0-3 Control:
    A claim that excluding flagged borrowers improves default rates requires ruling out the boring
    explanation: that refilling the portfolio to the same budget simply brings in lower-APR, lower-default
    loans.
    """
    rng = np.random.default_rng(seed)
    n = len(density)
    n_flag = int(np.sum(flagged))

    order = density.sort_values(ascending=False, kind="mergesort").index.to_numpy()
    exposure_arr = exposure.to_numpy()
    defaults_arr = pd.to_numeric(defaults, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    profit_arr = pd.to_numeric(realised_profits, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # 1. Random placebo
    rand_defs = np.empty(n_draws)
    rand_rocs = np.empty(n_draws)

    for i in range(n_draws):
        perm = rng.permutation(n)
        exclude_set = set(perm[:n_flag])
        valid_order = [idx for idx in order if idx not in exclude_set]
        cum_exp = exposure_arr[valid_order].cumsum()
        fits = cum_exp <= budget
        approved = [idx for idx, f in zip(valid_order, fits, strict=True) if f]
        rand_defs[i] = defaults_arr[approved].mean() if approved else float("nan")
        total_exp = exposure_arr[approved].sum()
        rand_rocs[i] = profit_arr[approved].sum() / total_exp if total_exp > 0 else float("nan")

    # 2. APR-decile matched placebo
    apr_clean = pd.to_numeric(apr, errors="coerce").fillna(0.13)
    deciles = pd.qcut(apr_clean, 10, labels=False, duplicates="drop")
    decile_indices = [np.where(deciles == d)[0] for d in range(10)]
    decile_n_flags = [int(flagged[idx].sum()) for idx in decile_indices]

    apr_defs = np.empty(n_draws)
    apr_rocs = np.empty(n_draws)

    for i in range(n_draws):
        flagged_apr = np.zeros(n, dtype=bool)
        for idxs, k in zip(decile_indices, decile_n_flags, strict=True):
            if len(idxs) > 0 and k > 0:
                chosen = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
                flagged_apr[chosen] = True
        exclude_set = set(np.where(flagged_apr)[0])
        valid_order = [idx for idx in order if idx not in exclude_set]
        cum_exp = exposure_arr[valid_order].cumsum()
        fits = cum_exp <= budget
        approved = [idx for idx, f in zip(valid_order, fits, strict=True) if f]
        apr_defs[i] = defaults_arr[approved].mean() if approved else float("nan")
        total_exp = exposure_arr[approved].sum()
        apr_rocs[i] = profit_arr[approved].sum() / total_exp if total_exp > 0 else float("nan")

    rand_ci = (float(np.percentile(rand_defs, 2.5)), float(np.percentile(rand_defs, 97.5)))
    rand_roc_ci = (float(np.percentile(rand_rocs, 2.5)), float(np.percentile(rand_rocs, 97.5)))
    apr_ci = (float(np.percentile(apr_defs, 2.5)), float(np.percentile(apr_defs, 97.5)))
    apr_roc_ci = (float(np.percentile(apr_rocs, 2.5)), float(np.percentile(apr_rocs, 97.5)))

    beats_rand = bool(signal_default_rate < rand_ci[0])
    beats_apr = bool(signal_default_rate < apr_ci[0])

    return PlaceboComparison(
        random_default_mean=float(np.mean(rand_defs)),
        random_default_ci=rand_ci,
        random_roc_mean=float(np.mean(rand_rocs)),
        random_roc_ci=rand_roc_ci,
        apr_matched_default_mean=float(np.mean(apr_defs)),
        apr_matched_default_ci=apr_ci,
        apr_matched_roc_mean=float(np.mean(apr_rocs)),
        apr_matched_roc_ci=apr_roc_ci,
        signal_beats_random=beats_rand,
        signal_beats_apr_matched=beats_apr,
    )


def flag_cutpoints(
    signals: dict[str, np.ndarray],
    keep: list[str],
    issue_period: pd.Series,
    quantile: float,
) -> dict[str, float]:
    """Flag cut-points: the ``quantile`` of each kept signal on the selection window (2015-01..06).

    Signal values from 2015-07..12 are never read, so the holdout cannot move a cut-point.
    """
    from credit_risk.data.splits import assert_selection_window, test_window_masks

    dev = test_window_masks(issue_period)["dev"]
    assert_selection_window(pd.Series(issue_period).to_numpy()[dev])  # the rows read below
    return {name: float(np.quantile(np.asarray(signals[name])[dev], quantile)) for name in keep}
