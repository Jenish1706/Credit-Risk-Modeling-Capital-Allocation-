#!/usr/bin/env python
"""CR-02: does predicted expected profit match realised profit? (explains the EXP03/EXP06 control)

    python scripts/analysis/analyze_profit_calibration.py --config configs/experiments/exp06_rl.yaml

Writes, under ``results/tables/``:

* ``profit_calibration_deciles.csv``     predicted vs realised profit by decile, validation and test
* ``profit_calibration_decomposition.csv`` dollar gap split into PD / yield / severity effects
* ``profit_calibration_cashflow_fit.csv`` modelled vs realised good yield / bad loss by APR quintile
* ``profit_calibration_pd_by_apr.csv``   PD calibration (observed / expected) by APR quintile
* ``profit_calibration_rules.csv``       realised profit of accept/reject rules at the observed exposure
* ``profit_calibration_exposure_mix.csv`` how often simulator approvals earn a *realised* return

Every figure is at the exposure that was actually funded, so it is a measurement rather than a
simulation, except the last table, which audits the simulator itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.data.splits import test_window_masks  # noqa: E402
from credit_risk.decision.expected_loss import (  # noqa: E402
    EconomicsConfig,
    expected_profit,
    fit_cashflow_model,
    realised_return,
)
from credit_risk.decision.policies import (  # noqa: E402
    ActionSpace,
    ExpectedProfitPolicy,
    FixedThresholdPolicy,
)
from credit_risk.decision.profit_audit import (  # noqa: E402
    gap_decomposition,
    paired_bootstrap_ci,
    profit_decile_table,
    rule_realised_profit,
)
from credit_risk.simulation.environment import (  # noqa: E402
    LendingEnvironment,
    SimulationConfig,
    build_applicant_pool,
)
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    load_prepared,
    resolve,
    save_table,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402
from scripts.experiments.train_policy import _load_pd, _score_split  # noqa: E402

LOG = get_logger("scripts.profit_calibration")
THRESHOLD = 0.15


def _arrays(frame: pd.DataFrame, pd_hat: np.ndarray) -> dict[str, np.ndarray]:
    apr = pd.to_numeric(frame["int_rate"], errors="coerce").to_numpy(float) / 100.0
    funded = pd.to_numeric(frame["funded_amnt"], errors="coerce").to_numpy(float)
    return {
        "pd_hat": np.asarray(pd_hat, dtype=float),
        "exposure": funded,
        "apr": apr,
        "realised_return": realised_return(frame).to_numpy(float),
        "defaulted": pd.to_numeric(frame["default"], errors="coerce").to_numpy(float),
    }


def cashflow_fit_table(frames: dict[str, pd.DataFrame], cashflow) -> pd.DataFrame:
    """Modelled vs realised good yield and bad loss by APR quintile, for each window."""
    rows = []
    edges = None
    for name, frame in frames.items():
        a = _arrays(frame, np.zeros(len(frame)))
        df = pd.DataFrame({"apr": a["apr"], "ret": a["realised_return"], "d": a["defaulted"]}).dropna()
        df = df[df["ret"].between(-1.2, 1.2)]
        if edges is None:
            edges = np.quantile(df["apr"], np.linspace(0, 1, 6))
            edges[0], edges[-1] = 0.0, 1.0
        df["bin"] = pd.cut(df["apr"], edges, labels=False, include_lowest=True) + 1
        for b, g in df.groupby("bin"):
            good, bad = g[g["d"] < 0.5], g[g["d"] >= 0.5]
            rows.append(
                {
                    "window": name, "apr_quintile": int(b), "n_good": len(good), "n_bad": len(bad),
                    "mean_apr": g["apr"].mean(),
                    "modelled_good_yield": float(cashflow.good_yield(good["apr"].to_numpy()).mean()),
                    "realised_good_return": good["ret"].mean(),
                    "modelled_bad_loss": float(cashflow.bad_loss_rate(bad["apr"].to_numpy()).mean()),
                    "realised_bad_loss": -bad["ret"].mean(),
                }
            )
    return pd.DataFrame(rows)


def pd_by_apr_table(pools: dict[str, tuple[pd.DataFrame, np.ndarray]], n_bins: int = 5) -> pd.DataFrame:
    """Is the PD model calibrated *within* APR bands?

    The PD model deliberately excludes ``int_rate`` (it is LendingClub's own risk grade), while
    the expected-profit rule uses APR as the revenue lever. If PD is miscalibrated along APR, the
    rule systematically prefers the loans whose risk the model understates.
    """
    rows = []
    for name, (frame, p_hat) in pools.items():
        a = _arrays(frame, p_hat)
        df = pd.DataFrame({"apr": a["apr"], "p": a["pd_hat"], "d": a["defaulted"]}).dropna()
        df["bin"] = pd.qcut(df["apr"], n_bins, labels=False) + 1
        for b, g in df.groupby("bin"):
            rows.append(
                {
                    "pool": name, "apr_quintile": int(b), "n": len(g), "mean_apr": g["apr"].mean(),
                    "mean_pd_hat": g["p"].mean(), "observed_default_rate": g["d"].mean(),
                    "observed_over_expected": g["d"].mean() / g["p"].mean(),
                }
            )
    return pd.DataFrame(rows)


def exposure_mix_table(pool, cashflow, economics, space, n_steps, seeds) -> pd.DataFrame:
    """Audit the simulator: what share of each policy's approvals earn a realised return?"""
    env = LendingEnvironment(
        pool, cashflow, economics, space, SimulationConfig(n_steps=n_steps, capital_budget=None)
    )
    rows = []
    for policy in (FixedThresholdPolicy(THRESHOLD, space), ExpectedProfitPolicy(cashflow, economics, space)):
        shares, multipliers = [], []
        for seed in seeds:
            env.run(policy, seed=seed, keep_log=True, learn=False)
            log = pd.DataFrame(env._log)
            log = log[log["exposure"] > 0]
            observed = pool.observed_exposure[log["applicant"].to_numpy()]
            at_observed = np.abs(log["exposure"].to_numpy() - observed) < 1.0
            shares.append(float(at_observed.mean()))
            multipliers.append(float((log["exposure"].to_numpy() / pool.requested_amount[log["applicant"].to_numpy()]).mean()))
        rows.append(
            {
                "policy": policy.name,
                "share_of_approvals_at_observed_exposure": float(np.mean(shares)),
                "mean_exposure_multiplier_of_approvals": float(np.mean(multipliers)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp06_rl.yaml")
    parser.add_argument("--pd-run", default="boosting")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)

    df = load_prepared(paths)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    ecfg = config.get("economics", {})
    economics = EconomicsConfig(
        annual_cost_of_funds=float(ecfg.get("annual_cost_of_funds", 0.03)),
        opex_fixed=float(ecfg.get("opex_fixed", 150.0)),
        opex_variable_rate=float(ecfg.get("opex_variable_rate", 0.010)),
        manual_review_cost=float(ecfg.get("manual_review_cost", 75.0)),
        weighted_average_life_years=float(ecfg.get("weighted_average_life_years", 1.55)),
    )
    cashflow = fit_cashflow_model(train, economics)
    cashflow_valid = fit_cashflow_model(valid, economics)

    pd_test = _load_pd(paths, args.pd_run, test, train, valid, seed)
    pd_valid = _score_split(paths, args.pd_run, valid)
    if pd_valid is None:
        raise SystemExit("validation PDs unavailable: run scripts/experiments/train_baseline.py first")
    # CR-03: nothing below selects anything on the test book (rules are fixed a priori, the
    # valid-fit curves use validation only), so 2015-07..12 is the headline window and the
    # other test windows are reported alongside. The "oracle" refit is fitted on the pool it
    # is scored on: a diagnostic ceiling, never used to choose anything.
    masks = test_window_masks(test["issue_period"])
    pools = {
        "valid": (valid, pd_valid),
        "test_h1": (test[masks["dev"]], pd_test[masks["dev"]]),
        "test_h2": (test[masks["holdout"]], pd_test[masks["holdout"]]),
        "test_full": (test, pd_test),
    }

    # -- deciles + decomposition ------------------------------------------------------
    deciles, decomp = [], []
    for name, (frame, p_hat) in pools.items():
        a = _arrays(frame, p_hat)
        table = profit_decile_table(**a, cashflow=cashflow, economics=economics)
        table.insert(0, "pool", name)
        deciles.append(table)
        oracle = fit_cashflow_model(frame, economics)
        for source, model in (("train_fit", cashflow), ("valid_fit", cashflow_valid),
                              ("pool_fit_ORACLE_DIAGNOSTIC", oracle)):
            row = gap_decomposition(**a, cashflow=model)
            row.update({"pool": name, "cashflow_source": source,
                        "mean_pd_hat": float(np.nanmean(a["pd_hat"])),
                        "observed_default_rate": float(np.nanmean(a["defaulted"]))})
            decomp.append(row)
    save_table(pd.concat(deciles), paths, "profit_calibration_deciles", index=False)
    save_table(pd.DataFrame(decomp), paths, "profit_calibration_decomposition", index=False)
    save_table(cashflow_fit_table(
        {"train": train, "valid": valid, "test_h2": test[masks["holdout"]]}, cashflow),
               paths, "profit_calibration_cashflow_fit", index=False)

    save_table(pd_by_apr_table(pools), paths, "profit_calibration_pd_by_apr", index=False)

    # -- accept/reject rules at the observed exposure ---------------------------------
    n_boot = int(config.get_path("evaluation.n_bootstrap", 2000))
    rule_rows = []
    for name, (frame, p_hat) in pools.items():
        a = _arrays(frame, p_hat)
        ok = np.isfinite(a["realised_return"]) & np.isfinite(a["exposure"]) & np.isfinite(a["apr"])
        a = {k: v[ok] for k, v in a.items()}
        ep = {src: expected_profit(a["pd_hat"], a["exposure"], a["apr"], model, economics)
              for src, model in (("train_fit", cashflow), ("valid_fit", cashflow_valid))}
        rules = {
            "fixed_threshold@0.15": a["pd_hat"] < THRESHOLD,
            "expected_profit>0 (train-fit cashflow)": ep["train_fit"] > 0,
            "expected_profit>0 (valid-fit cashflow)": ep["valid_fit"] > 0,
            "approve_all": np.ones(len(a["pd_hat"]), dtype=bool),
        }
        per_loan = {k: rule_realised_profit(v, a["exposure"], a["apr"], a["realised_return"], economics)
                    for k, v in rules.items()}
        base = per_loan["fixed_threshold@0.15"]
        for rule, profit in per_loan.items():
            mean, lo, hi = paired_bootstrap_ci(profit - base, n_bootstrap=n_boot, seed=seed)
            rule_rows.append(
                {
                    "pool": name, "rule": rule, "n_accepted": int(rules[rule].sum()),
                    "realised_profit_total": float(profit.sum()),
                    "realised_profit_per_accepted": float(profit.sum() / max(rules[rule].sum(), 1)),
                    "diff_vs_fixed_threshold_per_loan": mean,
                    "diff_ci_low": lo, "diff_ci_high": hi,
                    "diff_vs_fixed_threshold_total": float((profit - base).sum()),
                }
            )
    save_table(pd.DataFrame(rule_rows), paths, "profit_calibration_rules", index=False)

    # -- audit of the simulator's exposure asymmetry ----------------------------------
    space = ActionSpace(max_exposure=float(config.get_path("actions.max_exposure", 40000.0))).without_review()
    pool = build_applicant_pool(test[masks["holdout"]], pd_test[masks["holdout"]], cashflow)
    steps = int(config.get("simulation", {}).get("n_steps", 2000))
    mix = pd.concat(
        [
            exposure_mix_table(pool, cashflow, economics, space.at_observed_exposure(), steps, list(range(5)))
            .assign(action_space="accept_reject (headline)"),
            exposure_mix_table(pool, cashflow, economics, space, steps, list(range(5)))
            .assign(action_space="multi_exposure (sensitivity)"),
        ],
        ignore_index=True,
    )
    save_table(mix, paths, "profit_calibration_exposure_mix", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
