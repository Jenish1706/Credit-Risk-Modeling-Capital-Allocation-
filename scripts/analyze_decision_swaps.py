#!/usr/bin/env python
"""EXP11: does acting on blind-spot signals improve the book?

    python scripts/analyze_decision_swaps.py --config configs/experiments/exp11_swaps.yaml

Two capital-constrained books are built from the same applicant pool and the same budget:

* **Policy A** ranks by expected profit per dollar and funds until the budget binds.
* **Policy B** does the same but refuses to fund borrowers flagged by the blind-spot signals.
* **Policy C** funds flagged borrowers at reduced exposure instead of refusing them.

Profit is read from **observed cashflows** (``total_pymnt + recoveries -
collection_recovery_fee - funded_amnt``), never from an assumed loss-given-default, and the
comparison is between the two *portfolios* at equal budget rather than between the two swap sets.
Return on capital is reported alongside profit, because a book that deploys more capital will
usually earn more profit without being better.

**Windows (CR-03).** Which signals are used (the H1 KEEP/REJECT decisions of EXP08 and EXP10) and
where "flagged" starts (the ``--flag-quantile`` of each signal, measured on 2015-01..06 only) are
fixed on the selection window. The headline portfolios are built and scored on the untouched
2015-07..12 holdout; the full-2015 result is written alongside for continuity. ``--flag-quantile``,
``--budget-fraction`` and ``--exposure-cap`` are pre-specified, not tuned.

An earlier version of this experiment used a hardcoded ``LGD=0.5`` profit formula and compared the
swap sets. That understated book profit by a factor of 2.4 and **reversed the sign** of the
headline result. See docs/AUDIT_REPORT.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.data.splits import (  # noqa: E402
    WINDOW_LABELS,
    assert_selection_window,
    test_window_masks,
)
from credit_risk.decision.blindspot import (  # noqa: E402
    flag_cutpoints,
    build_portfolio,
    compare_capital_constrained,
    evaluate_placebo_controls,
    exposure_capped_policy,
)
from credit_risk.decision.expected_loss import (  # noqa: E402
    EconomicsConfig,
    expected_profit,
    fit_cashflow_model,
    realised_profit,
)
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    detect_source,
    load_prepared,
    resolve,
    save_json,
    save_table,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.exp11")


def _placebo_conclusion(default_rate: float, p) -> str:
    """State exactly which placebo the signal does and does not beat."""
    head = f"**Conclusion:** Policy B's default rate ({default_rate:.2%}) "
    if p.signal_beats_random and p.signal_beats_apr_matched:
        return head + "lies outside both placebo intervals, so the reduction is not explained by refilling the budget or by APR mix alone."
    if p.signal_beats_random:
        return (
            head + "lies outside the random-exclusion interval but inside the APR-matched interval: "
            "the reduction is what excluding the same share of loans with a matching APR profile "
            "produces, so it is **not attributable to the signal's selectivity** beyond the shift "
            "toward lower-APR loans."
        )
    return (
        head + "falls inside the placebo intervals, so the default reduction is **not attributable "
        "to the blind-spot signals**: refilling the budget after an arbitrary exclusion draws in "
        "lower-APR loans that naturally default less."
    )


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp11_swaps.yaml")
    parser.add_argument("--flag-quantile", type=float, default=0.90,
                        help="Signals above this quantile are treated as blind spots.")
    parser.add_argument("--budget-fraction", type=float, default=0.80,
                        help="Capital budget as a fraction of profitable demand.")
    parser.add_argument("--exposure-cap", type=float, default=0.5,
                        help="Exposure multiplier applied to flagged borrowers under Policy C.")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].reset_index(drop=True)

    # --- signals ------------------------------------------------------------
    required = {
        "pd": paths.artifacts / "boosting_pd_test.npy",
        "anomaly": paths.artifacts / "anomaly_test.npy",
        "disagreement": paths.artifacts / "disagreement_test.npy",
    }
    missing = [str(p.name) for p in required.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing signal files: " + ", ".join(missing) + "\nRun, in order:\n"
            "  python scripts/train_baseline.py --config configs/models/boosting.yaml\n"
            "  python scripts/analyze_blind_spots.py\n"
            "  python scripts/analyze_disagreement.py"
        )
    signals = {k: np.load(v) for k, v in required.items()}
    for name, values in signals.items():
        if len(values) != len(test):
            raise SystemExit(
                f"{name} has {len(values):,} rows but the test split has {len(test):,}. "
                "Re-run the upstream scripts against the current processed data."
            )

    pd_hat = signals["pd"]
    anomaly, disagreement = signals["anomaly"], signals["disagreement"]

    # --- economics, estimated from observed training cashflows --------------
    economics = EconomicsConfig(
        annual_cost_of_funds=float(config.get_path("economics.annual_cost_of_funds", 0.03)),
        opex_fixed=float(config.get_path("economics.opex_fixed", 150.0)),
        opex_variable_rate=float(config.get_path("economics.opex_variable_rate", 0.010)),
        weighted_average_life_years=float(
            config.get_path("economics.weighted_average_life_years", 1.55)
        ),
    )
    cashflow = fit_cashflow_model(train, economics)

    # --- selection (2015-01..06 only) ----------------------------------------
    masks = test_window_masks(test["issue_period"])
    assert_selection_window(test.loc[masks["dev"], "issue_period"])
    q = float(args.flag_quantile)
    candidate = {
        "anomaly": (anomaly, "exp08_blindspots"),
        "disagreement": (disagreement, "exp10_disagreement"),
    }
    used = []
    for name, (values, run) in candidate.items():
        record = json.loads((paths.results_tables / f"{run}.json").read_text(encoding="utf-8"))
        LOG.info("signal %-12s | H1 decision %s", name, record["decision"])
        if record["decision"] == "KEEP":
            used.append(name)
    cuts = flag_cutpoints({n: candidate[n][0] for n in used}, used, test["issue_period"], q)
    if not used:
        raise SystemExit("no signal survived the H1 selection; EXP11 has nothing to act on")
    flagged_all = np.zeros(len(test), dtype=bool)
    for name in used:
        flagged_all |= candidate[name][0] > cuts[name]

    def evaluate(mask: np.ndarray):
        sub = test[mask].reset_index(drop=True)
        pd_w, flagged = pd_hat[mask], flagged_all[mask]
        LOG.info(
            "flagged %s of %s borrowers (%.1f%%)",
            f"{flagged.sum():,}", f"{len(sub):,}", 100 * flagged.mean(),
        )
        analysis = compare_capital_constrained(
            sub, pd_w, flagged, cashflow, economics, budget_fraction=float(args.budget_fraction)
        )
        exposure = pd.to_numeric(sub["funded_amnt"], errors="coerce").fillna(0.0)
        apr = pd.to_numeric(sub["int_rate"], errors="coerce").fillna(13.0) / 100.0
        profit = expected_profit(pd_w, exposure.to_numpy(), apr.to_numpy(), cashflow, economics)
        density = pd.Series(
            np.divide(profit, exposure.to_numpy(), out=np.zeros_like(profit),
                      where=exposure.to_numpy() > 0),
            index=sub.index,
        )
        approve_c, exposure_c = exposure_capped_policy(
            density, exposure, flagged, analysis.budget, cap_multiplier=float(args.exposure_cap)
        )
        # Realised profit scales with the exposure actually taken, so a half-size loan earns half
        # the realised return. Same linear-scaling assumption the simulator makes for
        # counterfactual exposures; it is an assumption -- see docs/ASSUMPTIONS.md.
        scale = np.where(flagged, float(args.exposure_cap), 1.0)
        capped_profit = float((realised_profit(sub) * scale)[approve_c].sum())
        capped_capital = float(exposure_c[approve_c].sum())
        policy_c = build_portfolio(sub, approve_c, "C_exposure_capped", analysis.budget)
        policy_c.capital_deployed = capped_capital
        policy_c.observed_profit = capped_profit
        policy_c.return_on_capital = (
            capped_profit / capped_capital if capped_capital else float("nan")
        )
        policy_c.budget_utilisation = capped_capital / analysis.budget
        analysis.portfolios.append(policy_c)
        b_port = analysis.portfolios[1]
        placebo = evaluate_placebo_controls(
            density=density,
            exposure=exposure,
            defaults=pd.to_numeric(sub["default"], errors="coerce").fillna(0.0),
            realised_profits=realised_profit(sub),
            flagged=flagged,
            apr=apr,
            budget=analysis.budget,
            signal_default_rate=b_port.default_rate,
            n_draws=200,
            seed=seed,
        )
        return analysis, policy_c, placebo, int(flagged.sum()), len(sub)

    LOG.info("EXP11 headline: 2015-07..12 holdout")
    analysis, policy_c, placebo_comp, n_flagged, n_pool = evaluate(masks["holdout"])
    LOG.info("EXP11 continuity: full 2015")
    analysis_full, _, placebo_full, n_flagged_full, n_pool_full = evaluate(masks["full"])
    save_table(analysis.frame, paths, f"{run_name}_portfolios", index=False)
    save_table(analysis_full.frame, paths, f"{run_name}_portfolios_full", index=False)
    table = analysis.frame
    verdict_b = analysis.verdict("B_blindspot_aware", "A_pd_ranked")
    a, b = analysis.portfolios[0], analysis.portfolios[1]
    c_delta_roc = policy_c.return_on_capital - a.return_on_capital
    cut_text = ", ".join(f"{k} {v:.4g}" for k, v in cuts.items())

    lines = [
        "# EXP11: does acting on blind-spot signals improve the book?",
        "",
        f"**Headline window: {WINDOW_LABELS['holdout']}**, {n_pool:,} loans. Signals used: "
        f"{', '.join(used)} (KEEP on 2015-01..06). Flag cut-points are the {100 * q:.0f}th "
        f"percentile of each signal on 2015-01..06 ({cut_text}). Capital budget "
        f"${analysis.budget:,.0f}, {100 * float(args.budget_fraction):.0f}% of profitable demand. "
        f"Flagged: {n_flagged:,} borrowers ({100 * n_flagged / n_pool:.1f}%).",
        "",
        "## Portfolios at equal budget, measured on observed cashflows (2015-07..12)",
        "",
        table.round(4).to_markdown(index=False),
        "",
        f"Full-2015 continuity column ({n_pool_full:,} loans, flagged {n_flagged_full:,}):",
        "",
        analysis_full.frame.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        verdict_b,
        "",
        f"The swap involved {analysis.n_swapped_out:,} loans out "
        f"(realised return {analysis.swapped_out_return:+.2%}, default "
        f"{analysis.swapped_out_default_rate:.2%}) and {analysis.n_swapped_in:,} loans in "
        f"(realised return {analysis.swapped_in_return:+.2%}, default "
        f"{analysis.swapped_in_default_rate:.2%}).",
        "",
        f"Policy C, which caps flagged borrowers at {float(args.exposure_cap):.0%} exposure rather "
        f"than refusing them, changed return on capital by {c_delta_roc:+.2%} against Policy A.",
        "",
        "## Placebo Controls (Attribution Check)",
        "",
        "To test whether the default reduction is attributable to the blind-spot signals or merely "
        "the mechanical consequence of excluding loans and refilling the portfolio, 200 draws of "
        "two placebo arms were evaluated at the exact same capital budget (2015-07..12):",
        "",
        "1. **Random exclusion (same share flagged):**",
        f"   - Mean Default Rate: {placebo_comp.random_default_mean:.2%} (95% CI: "
        f"[{placebo_comp.random_default_ci[0]:.2%}, {placebo_comp.random_default_ci[1]:.2%}])",
        f"   - Mean RoC: {placebo_comp.random_roc_mean:.2%} (95% CI: "
        f"[{placebo_comp.random_roc_ci[0]:.2%}, {placebo_comp.random_roc_ci[1]:.2%}])",
        f"   - Signal beats random placebo: **{placebo_comp.signal_beats_random}**",
        "",
        "2. **APR-decile matched exclusion:**",
        f"   - Mean Default Rate: {placebo_comp.apr_matched_default_mean:.2%} (95% CI: "
        f"[{placebo_comp.apr_matched_default_ci[0]:.2%}, {placebo_comp.apr_matched_default_ci[1]:.2%}])",
        f"   - Mean RoC: {placebo_comp.apr_matched_roc_mean:.2%} (95% CI: "
        f"[{placebo_comp.apr_matched_roc_ci[0]:.2%}, {placebo_comp.apr_matched_roc_ci[1]:.2%}])",
        f"   - Signal beats APR-matched placebo: **{placebo_comp.signal_beats_apr_matched}**",
        "",
        _placebo_conclusion(b.default_rate, placebo_comp),
        "",
        "## What this does and does not show",
        "",
        "- Profit here is **observed**: `total_pymnt - collection_recovery_fee - funded_amnt` "
        "for every matured loan (total_pymnt already includes recoveries). No loss-given-default "
        "is assumed.",
        "- The comparison is between **whole portfolios at the same budget**, not between the two "
        "swap sets. Swap sets need not deploy equal capital, so differencing their profits is not "
        "a portfolio result.",
        "- **Return on capital is the comparable quantity.** Profit alone rewards whichever book "
        "happens to deploy more capital.",
        "- This is a *retrospective reallocation* on an already-approved population. It does not "
        "establish what would happen to borrowers LendingClub declined, and it is not evidence "
        "that the policy is safe to deploy.",
        "",
        "> **Correction.** An earlier version of this experiment computed profit as "
        "`amount x apr x 1.55 - amount x 0.5 x default` and compared the two swap sets rather "
        "than the portfolios. That formula understated realised book profit by roughly a factor "
        "of 2.4, and it reported a **$9.8M loss** as a *'net increase in profit'* -- the "
        "conclusion was asserted regardless of the sign of the number printed above it. See "
        "`docs/AUDIT_REPORT.md`.",
    ]
    write_report(lines, paths, f"{run_name}", source)
    save_json(
        {
            "budget": analysis.budget,
            "flag_quantile": q,
            "headline_window": WINDOW_LABELS["holdout"],
            "signals_used": used,
            "flag_cutpoints_h1": cuts,
            "n_flagged": n_flagged,
            "n_pool": n_pool,
            "full_2015": {
                "portfolios": [p.to_dict() for p in analysis_full.portfolios],
                "placebo_controls": placebo_full.to_dict(),
                "n_flagged": n_flagged_full, "n_pool": n_pool_full,
            },
            "portfolios": [p.to_dict() for p in analysis.portfolios],
            "n_swapped_out": analysis.n_swapped_out,
            "n_swapped_in": analysis.n_swapped_in,
            "swapped_out_return": analysis.swapped_out_return,
            "swapped_in_return": analysis.swapped_in_return,
            "cashflow_model": cashflow.to_dict(),
            "placebo_controls": placebo_comp.to_dict(),
        },
        paths, f"{run_name}",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
