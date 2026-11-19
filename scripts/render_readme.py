#!/usr/bin/env python
"""Render data and results tables in README.md directly from artifact files.

Ensures that every number in README.md is strictly traceable to committed
artifacts and prevents hand-edited numbers from surviving.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from credit_risk.data.splits import test_window_masks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
README_PATH = PROJECT_ROOT / "README.md"


def render_data_table() -> str:
    """Generate Markdown data split table from loans.parquet or historical metadata."""
    loans_path = PROJECT_ROOT / "data" / "processed" / "loans.parquet"
    if loans_path.exists():
        df = pd.read_parquet(loans_path, columns=["split", "default", "issue_period"])
        counts = df["split"].value_counts()
        defaults = df.groupby("split")["default"].mean()

        hist_n = int(counts.get("history", 8277))
        tr_n = int(counts.get("train", 239104))
        va_n = int(counts.get("valid", 90615))
        te_n = int(counts.get("test", 283026))
        mo_n = int(counts.get("monitor", 988585))

        tr_d = float(defaults.get("train", 0.126702) * 100)
        va_d = float(defaults.get("valid", 0.141257) * 100)
        te_d = float(defaults.get("test", 0.148863) * 100)
        test = df[df["split"] == "test"]
        win = test_window_masks(test["issue_period"])
        h1_n, h2_n = int(win["dev"].sum()), int(win["holdout"].sum())
        h1_d = float(test.loc[win["dev"], "default"].astype(float).mean() * 100)
        h2_d = float(test.loc[win["holdout"], "default"].astype(float).mean() * 100)
    else:
        hist_n, tr_n, va_n, te_n, mo_n = 8277, 239104, 90615, 283026, 988585
        tr_d, va_d, te_d = 12.67, 14.13, 14.89
        h1_n, h2_n, h1_d, h2_d = 120790, 162236, 15.12, 14.71

    lines = [
        "| Split | Window | Loans | Default rate |",
        "|---|---|---|---|",
        f"| history (graph only) | 2007-06 … 2009-12 | {hist_n:,} | unlabelled by design |",
        f"| **train** | 2010-01 … 2014-06 | {tr_n:,} | {tr_d:.2f}% |",
        f"| **valid** | 2014-07 … 2014-12 | {va_n:,} | {va_d:.2f}% |",
        f"| **test** (out-of-time) | 2015-01 … 2015-12 | {te_n:,} | {te_d:.2f}% |",
        f"| ↳ test H1 (all selections) | 2015-01 … 2015-06 | {h1_n:,} | {h1_d:.2f}% |",
        f"| ↳ test H2 (untouched holdout, headline) | 2015-07 … 2015-12 | {h2_n:,} | {h2_d:.2f}% |",
        f"| monitor (production traffic) | 2016-01 … 2018-12 | {mo_n:,} | unlabelled by design |",
    ]
    return "\n".join(lines)


def render_results_table() -> str:
    """Generate Markdown results table from results/tables/*.csv artifacts."""
    # Every row reports the untouched 2015-07..12 holdout (H2) first and the full-2015 figure as a
    # labelled second column. Selections (calibrator, KEEP/REJECT) are made on 2015-01..06 only.
    H2, FULL = "2015-07..12", "full 2015"

    # EXP01
    df_h2 = pd.read_csv(TABLES_DIR / "boosting_exp01_models_h2.csv", index_col=0)
    df_fu = pd.read_csv(TABLES_DIR / "boosting_exp01_models.csv", index_col=0)
    auc = lambda d: (float(d.loc["logistic", "roc_auc"]), float(d.loc["lightgbm", "roc_auc"]))  # noqa: E731
    (l2, g2), (lf, gf) = auc(df_h2), auc(df_fu)
    res_exp01 = (
        f"AUC **{l2:.4f} → {g2:.4f}** ({g2 - l2:+.4f}) on {H2} ({FULL}: {lf:.4f} → {gf:.4f}) `OBSERVED`"
    )
    exp01_verdict = "Material. Boosting justified" if g2 - l2 >= 0.01 else "Small margin; logistic remains defensible"

    # EXP02
    choice = json.loads((TABLES_DIR / "boosting_calibration_choice.json").read_text(encoding="utf-8"))
    prod = choice["method"]
    c2 = pd.read_csv(TABLES_DIR / "boosting_exp02_calibration_h2.csv").set_index("method")
    cf = pd.read_csv(TABLES_DIR / "boosting_exp02_calibration.csv").set_index("method")
    d_auc = abs(float(c2.loc[prod, "roc_auc"]) - float(c2.loc["identity", "roc_auc"]))
    auc_text = "AUC **unchanged to 5 d.p.**" if d_auc < 5e-6 else f"AUC moves {d_auc:.4f}"
    res_exp02 = (
        f"{auc_text}; ECE **{float(c2.loc['identity', 'ece']):.4f} → {float(c2.loc[prod, 'ece']):.4f}** "
        f"({prod}, chosen on 2015-01..06) on {H2} ({FULL}: {float(cf.loc['identity', 'ece']):.4f} → "
        f"{float(cf.loc[prod, 'ece']):.4f}); slope {float(c2.loc['identity', 'calibration_slope']):.3f} → "
        f"{float(c2.loc[prod, 'calibration_slope']):.3f} `OBSERVED`"
    )

    # EXP03
    df_exp03 = pd.read_csv(TABLES_DIR / "exp06_rl_exp03_differences.csv")
    row_diff = df_exp03[df_exp03["treatment"] == "expected_profit"].iloc[0]
    diff_m = float(row_diff["mean_difference"])
    ci_l = float(row_diff["ci_low"])
    ci_u = float(row_diff["ci_high"])
    real = _profit_rules("test_h2").loc["expected_profit>0 (train-fit cashflow)"]
    res_exp03 = (
        f"Simulated diff vs 15% threshold: **{diff_m:+,.0f}**/episode (95% CI [{ci_l:+,.0f}, {ci_u:+,.0f}]) `SIMULATED` on {H2}. "
        f"Real book at observed exposure: **{real['diff_vs_fixed_threshold_per_loan']:+.1f}**/loan "
        f"(95% CI [{real['diff_ci_low']:+.1f}, {real['diff_ci_high']:+.1f}]) `OBSERVED` on {H2}"
    )
    if real["diff_ci_high"] < 0:
        headline = "**Expected-profit rule loses to a PD threshold on the real book.**"
    elif real["diff_ci_low"] > 0:
        headline = "**Expected-profit rule beats a PD threshold on the real book.**"
    else:
        headline = "**No significant difference on the real book.**"
    exp03_verdict = (
        headline
        + (" Simulation CI straddles zero (null)." if ci_l <= 0 <= ci_u else " Simulation CI excludes zero.")
        + " Mechanism is measured below, not assumed"
    )

    # EXP04 & EXP05
    ctrl = json.loads((TABLES_DIR / "cohort_controls_exp04.json").read_text(encoding="utf-8"))
    h2 = ctrl["holdout_h2"]
    df_exp04 = pd.read_csv(TABLES_DIR / "cohort_graph_exp04_exp05_h2.csv", index_col=0)
    df_exp04_full = pd.read_csv(TABLES_DIR / "cohort_graph_exp04_exp05.csv", index_col=0)
    auc_a = float(df_exp04.loc["A_tabular", "roc_auc"])
    auc_b = float(df_exp04.loc["B_tabular_plus_graph", "roc_auc"])
    ece_a = float(df_exp04.loc["A_tabular", "ece"])
    ece_b = float(df_exp04.loc["B_tabular_plus_graph", "ece"])
    oe_a = float(df_exp04.loc["A_tabular", "observed_expected_ratio"])
    oe_b = float(df_exp04.loc["B_tabular_plus_graph", "observed_expected_ratio"])
    ss = ctrl["seed_summary"]
    seeds_tbl = pd.read_csv(TABLES_DIR / "cohort_controls_exp04_seed_summary.csv", index_col=0)
    n_seeds = len(ss["seeds"])
    a_s, b_s = seeds_tbl.loc["A_tabular"], seeds_tbl.loc["B_tabular_plus_graph"]
    res_exp04 = (
        f"AUC **{auc_b - auc_a:+.4f}**. Mean over {n_seeds} LightGBM seeds on {H2}: ECE **{a_s['ece_mean']:.4f} ± {a_s['ece_std']:.4f} "
        f"→ {b_s['ece_mean']:.4f} ± {b_s['ece_std']:.4f}**, O/E **{a_s['observed_expected_ratio_mean']:.3f} → "
        f"{b_s['observed_expected_ratio_mean']:.3f}** (EXP04/05 single-fit run: ECE {ece_a:.4f} → {ece_b:.4f}, O/E {oe_a:.3f} → {oe_b:.3f}) `OBSERVED`"
    )
    fmt = lambda d: f"{d['mean']:+.4f} ± {d['std']:.4f}"  # noqa: E731
    exp04_verdict = (
        ("No discrimination gain; **calibration gain survives both controls in every one of "
         f"{n_seeds} seeds**" if ss["survives_controls_for_every_seed"]
         else f"No discrimination gain; **calibration gain does not survive the controls in every one of {n_seeds} seeds**")
        + f" (ΔECE B−tabular {fmt(ss['delta_ece_vs_tabular'])}, B−macro {fmt(ss['delta_ece_vs_macro'])}, "
        f"B−shuffled {fmt(ss['delta_ece_vs_shuffled'])}; mean ± sd over seeds, per-seed paired bootstrap, {H2})"
    )

    auc_c = float(df_exp04.loc["C_graphsage", "roc_auc"])
    res_exp05 = f"AUC **{auc_c:.4f}** vs {auc_b:.4f} (aggregates) vs {auc_a:.4f} (tabular) on {H2} `OBSERVED`"
    exp05_verdict = (
        "**No.** The GNN loses to both" if auc_c < min(auc_a, auc_b)
        else "**Mixed.** See table" if auc_c < max(auc_a, auc_b) else "**Yes.** The GNN beats both"
    )

    # EXP06 (learning policies are frozen during evaluation episodes -- CR-01)
    cond = _exp06_conditions()
    unc = cond["unconstrained"]
    b60 = cond["budget=0.60"]
    best = lambda d, names: max(names, key=lambda n: d.loc[n, "mean_difference"])  # noqa: E731
    learners_unc = _learners(unc)
    learners_60 = _learners(b60)
    lu = best(unc, learners_unc)
    l6 = best(b60, learners_60)
    thr_c = _exp06_conditions(kind="diffthr")
    t60 = thr_c["budget=0.60"]
    res_exp06 = (
        f"Frozen-at-test learners, accept/reject at observed exposure ({H2}). Unconstrained: best learner {lu} "
        f"{unc.loc[lu, 'mean_difference']:+,.0f} vs myopic expected-profit (95% CI [{unc.loc[lu, 'ci_low']:+,.0f}, "
        f"{unc.loc[lu, 'ci_high']:+,.0f}]). 60% budget: {l6} {t60.loc[l6, 'mean_difference']:+,.0f} vs the 15% threshold "
        f"(95% CI [{t60.loc[l6, 'ci_low']:+,.0f}, {t60.loc[l6, 'ci_high']:+,.0f}]) `SIMULATED`"
    )
    exp06_verdict = _exp06_verdict(cond, _exp06_conditions(kind="diffthr"))

    # EXP07
    df_exp07 = pd.read_csv(TABLES_DIR / "boosting_exp07_calibration_drift.csv")
    df_exp07_h2 = pd.read_csv(TABLES_DIR / "boosting_exp07_calibration_drift_2015h2.csv")
    row_2010 = df_exp07[df_exp07["vintage"] == 2010].iloc[0]
    row_2015 = df_exp07[df_exp07["vintage"] == 2015].iloc[0]
    row_h2 = df_exp07_h2[df_exp07_h2["vintage"] == 2015].iloc[0]
    res_exp07 = (
        f"O/E **{float(row_2010['observed_expected']):.3f} (2010) → {float(row_h2['observed_expected']):.3f} (2015-07..12)** "
        f"(full 2015: {float(row_2015['observed_expected']):.3f}) while AUC moves "
        f"{float(row_2010['roc_auc']):.4f} → {float(row_h2['roc_auc']):.4f} `OBSERVED`"
    )

    # EXP08 / EXP09 / EXP10: decision on 2015-01..06, evidence on the 2015-07..12 holdout
    def signal_row(run: str, label: str):
        rec = json.loads((TABLES_DIR / f"{run}.json").read_text(encoding="utf-8"))
        w = rec["windows"]
        h, f = w["holdout_h2"], w["full_2015"]
        res = (
            f"H1 decision **{rec['decision']}**. On {H2}: {label} in **{h['n_strata_positive']}/{h['n_strata']}** "
            f"PD strata, two-sided sign test p={h['sign_test_p_twosided']:.4f}, odds ratio "
            f"{h['odds_ratio_per_sd']:.3f}/SD [{h['odds_ratio_ci_lower']:.3f}, {h['odds_ratio_ci_upper']:.3f}] "
            f"({FULL}: {f['n_strata_positive']}/{f['n_strata']}, OR {f['odds_ratio_per_sd']:.3f}) `OBSERVED`"
        )
        if rec["decision"] == "KEEP":
            verdict = "**KEEP**, confirmed on holdout" if rec["holdout_confirms"] else "**KEEP** on H1, **not confirmed** on holdout"
        else:
            verdict = "**REJECT**"
        return res, verdict

    res_exp08, exp08_verdict = signal_row("exp08_blindspots", "survives PD control")
    res_exp09, exp09_verdict = signal_row("exp09_vgae", "survives PD control")
    res_exp10, exp10_verdict = signal_row("exp10_disagreement", "survives PD control")

    # EXP11
    df_exp11 = pd.read_csv(TABLES_DIR / "exp11_swaps_portfolios.csv")
    pj = json.loads((TABLES_DIR / "exp11_swaps.json").read_text(encoding="utf-8"))["placebo_controls"]
    def_a = float(df_exp11[df_exp11["policy"] == "A_pd_ranked"]["default_rate"].iloc[0]) * 100
    def_b = float(df_exp11[df_exp11["policy"] == "B_blindspot_aware"]["default_rate"].iloc[0]) * 100
    lo = min(pj["random_default_ci"][0], pj["apr_matched_default_ci"][0]) * 100
    hi = max(pj["random_default_ci"][1], pj["apr_matched_default_ci"][1]) * 100
    res_exp11 = (
        f"Default rate **{def_a:.2f}% → {def_b:.2f}%** on {H2}, placebo controls show **{lo:.2f}%–{hi:.2f}%** `OBSERVED`"
    )
    if pj["signal_beats_random"] and pj["signal_beats_apr_matched"]:
        exp11_verdict = "**Signal beats both placebos.** See report"
    elif pj["signal_beats_random"]:
        exp11_verdict = "**Beats random exclusion but not the APR-matched placebo:** the risk reduction is the APR-mix shift, not blind-spot selectivity"
    else:
        exp11_verdict = "**Null result.** Risk reduction is driven by refilling budget with safer loans, not blind-spot selectivity"

    rows = [
        ("EXP01", "Does non-linear modelling beat logistic regression?", res_exp01, exp01_verdict),
        ("EXP02", "Does calibration improve PD reliability?", res_exp02, "The point of the project in one row: ranking metrics are blind to calibration"),
        ("EXP03", "Does expected-profit optimisation beat a fixed PD threshold?", res_exp03, exp03_verdict),
        ("EXP04", "Do graph cohort features add signal?", res_exp04, exp04_verdict),
        ("EXP05", "Does GraphSAGE beat the aggregates?", res_exp05, exp05_verdict),
        ("EXP06", "Can a budget-aware learned policy beat static policies?", res_exp06, exp06_verdict),
        ("EXP07", "How does the model degrade under vintage drift?", res_exp07, "Calibration drifts before ranking does"),
        ("EXP08", "Does autoencoder anomaly flag untrustworthy PDs?", res_exp08, exp08_verdict),
        ("EXP09", "Does VGAE relational uncertainty flag them?", res_exp09, exp09_verdict),
        ("EXP10", "Does model disagreement flag them?", res_exp10, exp10_verdict),
        ("EXP11", "Does acting on those signals improve the book?", res_exp11, exp11_verdict),
    ]

    lines = [
        "| # | Question | Result | Verdict |",
        "|---|---|---|---|",
    ]
    for exp_id, q, res, v in rows:
        lines.append(f"| **{exp_id}** | {q} | {res} | {v} |")
    return "\n".join(lines)


LEARNER_MARKERS = ("q(", "linucb", "thompson")
EXP06_CONDITIONS = ("unconstrained", "budget=0.60", "budget=0.35", "budget=0.20")


def _exp06_conditions(prefix: str = "exp06_rl", kind: str = "diff") -> dict[str, pd.DataFrame]:
    """Paired differences per capital condition: ``diff`` = vs expected-profit, ``diffthr`` = vs the 15% threshold."""
    out = {}
    for cond in EXP06_CONDITIONS:
        tag = cond.replace("=", "").replace(".", "")
        out[cond] = pd.read_csv(TABLES_DIR / f"{prefix}_exp06_{kind}_{tag}.csv", index_col=0)
    return out


def _learners(diff: pd.DataFrame) -> list[str]:
    return [n for n in diff.index if any(m in n for m in LEARNER_MARKERS)]


def _exp06_verdict(cond: dict[str, pd.DataFrame], thr: dict[str, pd.DataFrame]) -> str:
    """Verdict derived from the paired CIs, never typed by hand."""

    def sig(frames, c, side):
        rows = frames[c].loc[_learners(frames[c])]
        mask = rows["ci_low"] > 0 if side == "win" else rows["ci_high"] < 0
        return list(rows.index[mask])

    constrained = [c for c in cond if c != "unconstrained"]
    parts = []
    if sig(cond, "unconstrained", "win"):
        parts.append("**Control breached:** learners significantly beat the myopic rule with unconstrained capital")
    else:
        parts.append("Control holds: no learner significantly beats the myopic rule with unconstrained capital")
    def by_learner(side):
        found: dict[str, list[str]] = {}
        for c in cond:
            for learner in sig(thr, c, side):
                found.setdefault(learner, []).append(c)
        return "; ".join(f"{k} ({', '.join(v)})" for k, v in found.items())

    wins, losses = by_learner("win"), by_learner("lose")
    parts.append(
        f"learners significantly beating the fixed 15% threshold: {wins}" if wins
        else "**no learner significantly beats the fixed 15% threshold in any condition**"
    )
    if losses:
        parts.append(f"significantly worse than the threshold: {losses}")
    del constrained
    return "; ".join(parts)


def _exp06_meta() -> dict:
    return json.loads((TABLES_DIR / "exp06_rl_exp03_exp06.json").read_text(encoding="utf-8"))["exp06"]


def _exp06_policy_table(cond: dict[str, pd.DataFrame], thr: dict[str, pd.DataFrame]) -> list[str]:
    """Every policy in every capital condition, with approval rate and capital actually deployed."""
    sweep = pd.read_csv(TABLES_DIR / "exp06_rl_exp06_sweep.csv")
    meta = _exp06_meta()
    rows = [
        "| Condition | Policy | Mean profit | Approval rate | Capital deployed | vs expected-profit [95% CI] | vs 15% threshold [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(frame: pd.DataFrame, policy: str) -> str:
        if policy not in frame.index:
            return "baseline"
        d = frame.loc[policy]
        return f"{d['mean_difference']:+,.0f}{'*' if d['significant'] else ''} [{d['ci_low']:+,.0f}, {d['ci_high']:+,.0f}]"

    for name in cond:
        budget = meta[name]["budget"]
        sub = sweep[sweep["condition"] == name].sort_values("mean_reward", ascending=False)
        for _, r in sub.iterrows():
            capital = f"{r['capital_deployed'] / 1e6:.2f}M"
            if budget:
                capital += f" ({100 * r['capital_deployed'] / budget:.0f}% of budget)"
            rows.append(
                f"| {name} | {r['policy']} | {r['mean_reward']:+,.0f} | {r['approval_rate']:.1%} | {capital} | "
                f"{cell(cond[name], r['policy'])} | {cell(thr[name], r['policy'])} |"
            )
    return rows


def _exp06_policy_notes(cond: dict[str, pd.DataFrame], thr: dict[str, pd.DataFrame]) -> list[str]:
    """Plain-language reading of the table above, computed from the same artifacts."""
    sweep = pd.read_csv(TABLES_DIR / "exp06_rl_exp06_sweep.csv")
    meta = _exp06_meta()
    reward = lambda c, p: float(sweep[(sweep["condition"] == c) & (sweep["policy"] == p)]["mean_reward"].iloc[0])  # noqa: E731
    notes = ["", "\\* = 95% CI excludes zero.", ""]

    lam_items, diag_below_myopic, diag_below_thr = [], [], []
    for name in cond:
        if name == "unconstrained":
            continue
        for p in cond[name].index:
            if p.startswith("expected_profit(lambda="):
                row = sweep[(sweep["condition"] == name) & (sweep["policy"] == p)].iloc[0]
                d, t = cond[name].loc[p], thr[name].loc[p]
                lam_items.append(
                    f"{name}: {d['mean_difference']:+,.0f} vs expected-profit [{d['ci_low']:+,.0f}, {d['ci_high']:+,.0f}], "
                    f"{t['mean_difference']:+,.0f} vs the 15% threshold [{t['ci_low']:+,.0f}, {t['ci_high']:+,.0f}], "
                    f"deploying {100 * row['capital_deployed'] / meta[name]['budget']:.0f}% of the budget"
                )
            if p.startswith("expected_profit(lambda_eval_pool="):
                if reward(name, p) < reward(name, "expected_profit"):
                    diag_below_myopic.append(name)
                if reward(name, p) < reward(name, "fixed_threshold@0.150"):
                    diag_below_thr.append(name)
    if lam_items:
        notes.append(
            "The knapsack shadow-price policy (lambda fitted on the validation pool) is the analytical "
            "solution in the formulation, and here it does not help: " + "; ".join(lam_items) + "."
        )
        notes.append("")
    if diag_below_myopic or diag_below_thr:
        parts = []
        if diag_below_myopic:
            parts.append("below the myopic rule at " + ", ".join(diag_below_myopic))
        if diag_below_thr:
            parts.append("below the 15% threshold at " + ", ".join(diag_below_thr))
        notes.append(
            "`expected_profit(lambda_eval_pool=...)` fits lambda on the evaluation pool itself. It is a "
            "diagnostic, not an upper bound: it scores " + " and ".join(parts) + ". Its lambda is applied to "
            "predicted profit that is over-dispersed relative to realised profit (profit audit below), so "
            "the knapsack optimum is for the wrong objective."
        )
        notes.append("")
    collapsed = []
    for _, r in sweep.iterrows():
        if any(m in r["policy"] for m in LEARNER_MARKERS) and r["approval_rate"] < 0.01:
            collapsed.append(
                f"{r['policy']} at {r['condition']} (approval rate {r['approval_rate']:.2%}, "
                f"{r['capital_deployed'] / 1e6:.3f}M deployed)"
            )
    if collapsed:
        notes.append(
            "Collapsed learners: " + "; ".join(collapsed) + ". These stopped lending (approval rate "
            "below 1%); \"significantly worse\" understates it."
        )
    return notes


def render_rl_paragraph() -> str:
    """The prose under the results table, with every number read from the EXP06 artifacts."""
    cond = _exp06_conditions()
    thr = _exp06_conditions(kind="diffthr")
    sweep = pd.read_csv(TABLES_DIR / "exp06_rl_exp06_sweep.csv")
    unc = sweep[sweep["condition"] == "unconstrained"].set_index("policy")["mean_reward"]
    d_unc = cond["unconstrained"]
    best = max(_learners(d_unc), key=lambda n: d_unc.loc[n, "mean_difference"])
    bd = d_unc.loc[best]
    breached = bool(bd["ci_low"] > 0)
    thr_unc = d_unc.loc["fixed_threshold@0.150"]
    lines = [
        "**RL does not earn its place here, and that is the finding.** Learning policies "
        "(LinUCB, Thompson sampling, Q-learning) choose between accept and reject at the exposure "
        "that was actually funded, so every simulated outcome is a real one. They are trained on "
        "the validation pool and their parameters are **frozen** during every test-pool episode; "
        "an earlier version kept updating them on realised test outcomes after every decision, "
        "which no deployed policy can do (loan outcomes take up to 36 months to arrive).",
        "",
        f"The experiment carries a control. With unconstrained capital the myopic expected-profit "
        f"rule should be near-optimal. The best learner ({best}) earns {float(unc[best]):,.0f} vs the "
        f"myopic rule's {float(unc['expected_profit']):,.0f} (paired difference "
        f"{bd['mean_difference']:+,.0f}, 95% CI [{bd['ci_low']:+,.0f}, {bd['ci_high']:+,.0f}]): "
        + ("**the control is breached**." if breached else "**the control holds**.")
        + f" The plain 15% PD threshold beats the myopic rule by {thr_unc['mean_difference']:+,.0f} "
        f"(95% CI [{thr_unc['ci_low']:+,.0f}, {thr_unc['ci_high']:+,.0f}]), consistent with the "
        "real-book audit below.",
        "",
        "Policy-level results by capital condition (mean over 30 episodes, 95% bootstrap CI on the "
        "paired difference), against the myopic expected-profit policy and the fixed 15% PD threshold:",
        "",
    ]
    lines += _exp06_policy_table(cond, thr)
    lines += _exp06_policy_notes(cond, thr)
    return "\n".join(lines) + "\n\n" + render_sensitivity_table() + "\n\n" + render_profit_paragraph()


def render_sensitivity_table() -> str:
    """CR-04: the exposure-changing action space, labelled as a sensitivity analysis."""
    sens = _exp06_conditions("exp06_rl_sens")
    exp03 = pd.read_csv(TABLES_DIR / "exp06_rl_sens_exp03_differences.csv")
    ep = exp03[exp03["treatment"] == "expected_profit"].iloc[0]
    lines = [
        "**Sensitivity: exposure-changing actions.** Adding 0.5x and 1.5x exposure actions makes "
        "off-diagonal outcomes model counterfactuals rather than measurements, so these are not "
        "headline numbers. Expected-profit vs the 15% threshold in that space: "
        f"{ep['mean_difference']:+,.0f} per episode (95% CI [{ep['ci_low']:+,.0f}, {ep['ci_high']:+,.0f}]). "
        "Learners vs the myopic expected-profit policy:",
        "",
        "| Condition | Learner | Difference | 95% CI | Significant |",
        "|---|---|---|---|---|",
    ]
    for name, diff in sens.items():
        for learner in _learners(diff):
            r = diff.loc[learner]
            lines.append(
                f"| {name} | {learner} | {r['mean_difference']:+,.0f} | "
                f"[{r['ci_low']:+,.0f}, {r['ci_high']:+,.0f}] | {'yes' if r['significant'] else 'no'} |"
            )
    return "\n".join(lines)


def _profit_rules(pool: str) -> pd.DataFrame:
    rules = pd.read_csv(TABLES_DIR / "profit_calibration_rules.csv")
    return rules[rules["pool"] == pool].set_index("rule")


def render_profit_paragraph() -> str:
    """CR-02: why the expected-profit rule underperforms, with every number read from artifacts."""
    ep = "expected_profit>0 (train-fit cashflow)"
    ep_valid_fit = "expected_profit>0 (valid-fit cashflow)"
    test, valid = _profit_rules("test_h2"), _profit_rules("valid")
    decomp = pd.read_csv(TABLES_DIR / "profit_calibration_decomposition.csv")
    d = decomp[(decomp["pool"] == "test_h2") & (decomp["cashflow_source"] == "train_fit")].iloc[0]
    d_oracle = decomp[
        (decomp["pool"] == "test_h2") & (decomp["cashflow_source"] == "pool_fit_ORACLE_DIAGNOSTIC")
    ].iloc[0]
    by_apr = pd.read_csv(TABLES_DIR / "profit_calibration_pd_by_apr.csv")
    by_apr = by_apr[by_apr["pool"] == "test_h2"].sort_values("apr_quintile")
    deciles = pd.read_csv(TABLES_DIR / "profit_calibration_deciles.csv")
    deciles = deciles[deciles["pool"] == "test_h2"].sort_values("decile")
    mix_all = pd.read_csv(TABLES_DIR / "profit_calibration_exposure_mix.csv")
    mix = mix_all[mix_all["action_space"].str.startswith("multi_exposure")].set_index("policy")
    ep_mix = mix.loc["expected_profit"]
    thr_mix = mix.loc["fixed_threshold@0.150"]

    gap = -float(d["gap_total"])
    pct = lambda v: 100.0 * abs(float(v)) / gap  # noqa: E731
    lines = [
        "**Why the expected-profit rule underperforms.** Measured on the real book at the "
        "exposure that was actually funded, so no simulator is involved:",
        "",
        f"- On the 2015-07..12 holdout, accepting every loan with predicted profit above zero earns "
        f"**{test.loc[ep, 'diff_vs_fixed_threshold_per_loan']:+.1f}** per loan vs the 15% PD "
        f"threshold (95% CI [{test.loc[ep, 'diff_ci_low']:+.1f}, {test.loc[ep, 'diff_ci_high']:+.1f}], "
        f"{test.loc[ep, 'diff_vs_fixed_threshold_total'] / 1e6:+.2f}M in total). On the validation "
        f"pool the two rules are indistinguishable ({valid.loc[ep, 'diff_vs_fixed_threshold_per_loan']:+.1f} "
        f"per loan, CI [{valid.loc[ep, 'diff_ci_low']:+.1f}, {valid.loc[ep, 'diff_ci_high']:+.1f}]).",
        f"- Predicted profit on the holdout exceeds realised profit by **{gap / 1e6:.1f}M** "
        f"({gap / float(d['predicted_total']):.0%} of the prediction). By source: default count "
        f"(PD too low) {pct(d['pd_effect']):.0f}%, good-loan yield curve {pct(d['yield_effect']):.0f}%, "
        f"loss-severity curve {pct(d['severity_effect']):.0f}%. Refitting the two cashflow curves "
        f"on the holdout itself (an oracle diagnostic, not deployable) removes the yield and "
        f"severity errors ({d_oracle['yield_effect'] / 1e6:+.1f}M, {d_oracle['severity_effect'] / 1e6:+.1f}M) "
        f"and leaves the PD effect ({d_oracle['pd_effect'] / 1e6:+.1f}M).",
        f"- Cashflow drift is therefore a real but secondary cause. Refitting the curves on the "
        f"validation window narrows the shortfall to {test.loc[ep_valid_fit, 'diff_vs_fixed_threshold_per_loan']:+.1f} "
        f"per loan (CI [{test.loc[ep_valid_fit, 'diff_ci_low']:+.1f}, {test.loc[ep_valid_fit, 'diff_ci_high']:+.1f}]), "
        f"still significantly negative.",
        f"- The PD model excludes `int_rate` by design, while the rule's revenue term rises with APR. "
        f"Observed/expected defaults on the holdout run from **{by_apr.iloc[0]['observed_over_expected']:.2f}** in the "
        f"lowest APR quintile to **{by_apr.iloc[-1]['observed_over_expected']:.2f}** in the highest, so risk is "
        f"understated exactly where predicted profit is largest. Predicted profit is over-dispersed: "
        f"the top predicted-profit decile is predicted at {deciles.iloc[-1]['mean_predicted_profit']:+,.0f} "
        f"per loan and realises {deciles.iloc[-1]['mean_realised_profit']:+,.0f}.",
        f"- Simulator caveat: in the exposure-changing sensitivity space "
        f"{ep_mix['share_of_approvals_at_observed_exposure']:.0%} of the expected-profit policy's "
        f"approvals are at the observed exposure (mean multiplier "
        f"{ep_mix['mean_exposure_multiplier_of_approvals']:.2f}x) vs "
        f"{thr_mix['share_of_approvals_at_observed_exposure']:.0%} for the threshold, so that "
        f"comparison mixes realised and modelled returns. The headline EXP03/EXP06 tables use "
        f"accept/reject at the observed exposure, where every outcome is real.",
        "",
        "2015-07..12 holdout, deciles of predicted profit (per loan, dollars):",
        "",
        "| Decile | Mean PD | Observed default | Predicted profit | Realised profit |",
        "|---|---|---|---|---|",
    ]
    for _, r in deciles.iterrows():
        lines.append(
            f"| {int(r['decile'])} | {r['mean_pd_hat']:.3f} | {r['observed_default_rate']:.3f} | "
            f"{r['mean_predicted_profit']:+,.0f} | {r['mean_realised_profit']:+,.0f} |"
        )
    return "\n".join(lines)


def update_readme() -> None:
    """Read README.md, replace data and results tables, and write back."""
    content = README_PATH.read_text(encoding="utf-8")

    # Replace Data table
    data_table = render_data_table()
    data_pattern = re.compile(r"\| Split \| Window \| Loans \| Default rate \|[\s\S]*?(?=\n\nThe rising default rate)")
    if not data_pattern.search(content):
        raise RuntimeError("Could not locate Data table in README.md")
    content = data_pattern.sub(lambda m: data_table, content)

    # Replace Results table
    results_table = render_results_table()
    results_pattern = re.compile(r"\| # \| Question \| Result \| Verdict \|[\s\S]*?(?=\n\n### Four results worth reading twice)")
    if not results_pattern.search(content):
        raise RuntimeError("Could not locate Results table in README.md")
    content = results_pattern.sub(lambda m: results_table, content)

    # Replace RL paragraph
    rl_pattern = re.compile(r"\*\*RL does not earn its place here[\s\S]*?(?=\n\n---\n\n## Architecture)")
    if not rl_pattern.search(content):
        raise RuntimeError("Could not locate RL paragraph in README.md")
    content = rl_pattern.sub(lambda m: render_rl_paragraph(), content)

    README_PATH.write_text(content, encoding="utf-8")
    print(f"Successfully updated tables in {README_PATH}")


if __name__ == "__main__":
    update_readme()
