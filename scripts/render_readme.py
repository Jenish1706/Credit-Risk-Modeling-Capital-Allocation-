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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
README_PATH = PROJECT_ROOT / "README.md"


def render_data_table() -> str:
    """Split table, read only from the tracked data profile (no raw or processed data needed)."""
    d = json.loads((TABLES_DIR / "data_profile.json").read_text(encoding="utf-8"))
    counts, rates, windows = d["split_counts"], d["default_rate_by_split"], d["test_window_counts"]
    hist_n, tr_n, va_n = int(counts["history"]), int(counts["train"]), int(counts["valid"])
    te_n, mo_n = int(counts["test"]), int(counts["monitor"])
    tr_d, va_d, te_d = (100 * float(rates[k]) for k in ("train", "valid", "test"))
    h1_n, h2_n = int(windows["test_h1"]["rows"]), int(windows["test_h2"]["rows"])
    h1_d, h2_d = (100 * float(windows[k]["default_rate"]) for k in ("test_h1", "test_h2"))

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
    ci1 = json.loads((TABLES_DIR / "exp01_auc_difference.json").read_text(encoding="utf-8"))
    res_exp01 = (
        f"AUC **{l2:.4f} → {g2:.4f}** ({g2 - l2:+.4f}, paired-bootstrap 95% CI [{ci1['ci_low']:+.4f}, "
        f"{ci1['ci_high']:+.4f}], {ci1['n_bootstrap']:,} draws) on {H2} ({FULL}: {lf:.4f} → {gf:.4f}) `OBSERVED`"
    )
    if ci1["ci_low"] >= 0.01:
        exp01_verdict = "Material: the whole interval is above the +0.01 AUC threshold. Boosting justified"
    elif ci1["ci_low"] > 0:
        exp01_verdict = "Real but small: the interval excludes zero but reaches below +0.01 AUC; logistic remains defensible"
    else:
        exp01_verdict = "No significant difference; logistic regression is enough"

    # EXP02
    choice = json.loads((TABLES_DIR / "boosting_calibration_choice.json").read_text(encoding="utf-8"))
    prod = choice["method"]
    c2 = pd.read_csv(TABLES_DIR / "boosting_exp02_calibration_h2.csv").set_index("method")
    cf = pd.read_csv(TABLES_DIR / "boosting_exp02_calibration.csv").set_index("method")
    d_auc = abs(float(c2.loc[prod, "roc_auc"]) - float(c2.loc["identity", "roc_auc"]))
    auc_text = "AUC **unchanged to 5 d.p.**" if d_auc < 5e-6 else f"AUC moves {d_auc:.4f}"
    others = [m for m in c2.index if m != "identity"]
    best_gain = float(c2.loc["identity", "ece"]) - min(float(c2.loc[m, "ece"]) for m in others)
    if "intercept_only" in c2.index and best_gain > 0:
        shift_share = (float(c2.loc["identity", "ece"]) - float(c2.loc["intercept_only", "ece"])) / best_gain
        exp02_verdict = (
            f"Ranking is unchanged and ECE falls, but this is mostly a base-rate shift: a one-parameter "
            f"intercept shift recovers {shift_share:.0%} of the best calibrator's ECE gain"
        )
    else:
        exp02_verdict = "Ranking is unchanged and ECE falls; no intercept-only baseline was run"
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
        f"Real book at observed exposure: **{real['diff_vs_fixed_threshold_per_loan']:+.1f}**/pool applicant "
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
    df_exp04 = pd.read_csv(TABLES_DIR / "cohort_graph_exp04_exp05_h2.csv", index_col=0)
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
        f"AUC **{b_s['roc_auc_mean'] - a_s['roc_auc_mean']:+.4f}** (mean over {n_seeds} LightGBM seeds). On {H2}: ECE **{a_s['ece_mean']:.4f} ± {a_s['ece_std']:.4f} "
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
    res_exp05 = f"AUC **{auc_c:.4f}** vs {auc_b:.4f} (aggregates) vs {auc_a:.4f} (tabular) on {H2}; the GNN is one training run `OBSERVED`"
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
    quarterly = pd.read_csv(TABLES_DIR / "boosting_exp07_calibration_drift_quarterly.csv")
    in_sample = quarterly[quarterly["split"] == "train"]["roc_auc"]
    out_sample = quarterly[quarterly["split"].isin(["valid", "test"])]["roc_auc"]
    res_exp07 = (
        f"O/E **{float(row_2010['observed_expected']):.3f} (2010, a training vintage) → {float(row_h2['observed_expected']):.3f} (2015-07..12)** "
        f"(full 2015: {float(row_2015['observed_expected']):.3f}). AUC by quarter: in-sample training vintages "
        f"{in_sample.min():.3f}–{in_sample.max():.3f}, out-of-sample (validation and test) "
        f"{out_sample.min():.3f}–{out_sample.max():.3f} `OBSERVED`"
    )
    oos_flat = float(out_sample.max() - out_sample.min()) < 0.02
    oos_months = 3 * len(out_sample)
    exp07_verdict = (
        "Calibration drifts across vintages (O/E rises from below 1 to above 1). Out-of-sample ranking is "
        f"flat over the {oos_months} months observed, so the fall in AUC from the training vintages to the "
        "out-of-sample quarters is mostly in-sample optimism. Every out-of-sample quarter is a late one, so "
        "this cannot rule out drift in ranking between earlier vintages"
        if oos_flat and float(in_sample.mean() - out_sample.mean()) > 0.02
        else "Calibration and ranking both move across vintages; see the quarterly table"
    )

    # CR-12: deployment realism (labels that were not yet knowable at the deployment date)
    dr = pd.read_csv(TABLES_DIR / "deployment_realism_p1_5.csv")
    std, real = dr.iloc[0], dr.iloc[1]
    sel = pd.read_csv(TABLES_DIR / "deployment_realism_selection.csv")
    tr_s, va_s = sel.iloc[0], sel.iloc[1]
    res_dr = (
        f"Trained only on loans resolved by 2014-12, on {H2}: AUC **{std['h2_test_auc']:.4f} → {real['h2_test_auc']:.4f}**, "
        f"ECE **{std['h2_test_ece']:.4f} → {real['h2_test_ece']:.4f}**, O/E **{std['h2_observed_expected_ratio']:.3f} → "
        f"{real['h2_observed_expected_ratio']:.3f}** ({FULL}: AUC {std['test_auc']:.4f} → {real['test_auc']:.4f}, "
        f"O/E {std['observed_expected_ratio']:.3f} → {real['observed_expected_ratio']:.3f}). The resolved loans are "
        f"a selected sample: default rate {tr_s['default_rate_resolved_loans']:.1%} against "
        f"{tr_s['default_rate_all_loans']:.1%} for all loans issued in the training window, "
        f"{va_s['default_rate_resolved_loans']:.1%} against {va_s['default_rate_all_loans']:.1%} in the calibration "
        "window `OBSERVED`"
    )
    worse = real["h2_test_auc"] < std["h2_test_auc"] and real["h2_test_ece"] > std["h2_test_ece"]
    dr_verdict = (
        "**A naive 'train on what has resolved' model is clearly worse on day one:** ranking and calibration both "
        "fall. The comparison mixes staleness with selection on resolution, so it does not isolate staleness, and "
        "no arm here trains on matured vintages only"
        if worse else "Training only on resolved loans does not make the model worse"
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
            sel_h1 = w["selection_h1"]
            verdict = (
                f"**REJECT** on H1 ({sel_h1['n_strata_positive']}/{sel_h1['n_strata']} strata, two-sided sign-test "
                f"p={sel_h1['sign_test_p_twosided']:.4f}; the rule needs p<0.05 with the gap in the right direction). "
                "The odds-ratio interval is reported but is not part of the rule"
            )
        return res, verdict

    res_exp08, exp08_verdict = signal_row("exp08_blindspots", "survives PD control")
    res_exp09, exp09_verdict = signal_row("exp09_vgae", "survives PD control")
    res_exp10, exp10_verdict = signal_row("exp10_disagreement", "survives PD control")

    # EXP11
    df_exp11 = pd.read_csv(TABLES_DIR / "exp11_swaps_portfolios.csv")
    pj = json.loads((TABLES_DIR / "exp11_swaps.json").read_text(encoding="utf-8"))["placebo_controls"]
    def_a = float(df_exp11[df_exp11["policy"] == "A_pd_ranked"]["default_rate"].iloc[0]) * 100
    def_b = float(df_exp11[df_exp11["policy"] == "B_blindspot_aware"]["default_rate"].iloc[0]) * 100
    rnd_lo, rnd_hi = (x * 100 for x in pj["random_default_ci"])
    apr_lo, apr_hi = (x * 100 for x in pj["apr_matched_default_ci"])
    apr_mean = pj["apr_matched_default_mean"] * 100
    roc_a = float(df_exp11[df_exp11["policy"] == "A_pd_ranked"]["return_on_capital"].iloc[0]) * 100
    roc_b = float(df_exp11[df_exp11["policy"] == "B_blindspot_aware"]["return_on_capital"].iloc[0]) * 100
    rr_lo, rr_hi = (x * 100 for x in pj["random_roc_ci"])
    ar_lo, ar_hi = (x * 100 for x in pj["apr_matched_roc_ci"])
    res_exp11 = (
        f"Default rate **{def_a:.2f}% → {def_b:.2f}%** on {H2}. Placebo exclusion (200 draws, 95% intervals): "
        f"random {rnd_lo:.2f}%–{rnd_hi:.2f}%, APR-matched {apr_lo:.2f}%–{apr_hi:.2f}%. Return on capital "
        f"{roc_a:.2f}% → {roc_b:.2f}% (random {rr_lo:.2f}%–{rr_hi:.2f}%, APR-matched {ar_lo:.2f}%–{ar_hi:.2f}%) `OBSERVED`"
    )
    roc_inside = rr_lo <= roc_b <= rr_hi and ar_lo <= roc_b <= ar_hi
    below_mean = apr_mean - def_b
    share = below_mean / (def_a - def_b) if def_a > def_b else float("nan")
    if pj["signal_beats_random"] and pj["signal_beats_apr_matched"]:
        exp11_verdict = "**Signal beats both placebos.** See report"
    elif pj["signal_beats_random"]:
        exp11_verdict = (
            "**Beats random exclusion but not the APR-matched placebo.** The signal book is "
            f"{below_mean:.2f} points below the APR-matched placebo mean, {share:.0%} of its {def_a - def_b:.2f}-point "
            f"reduction, and {def_b - apr_lo:+.3f} points from that interval's lower edge, so most of the "
            "reduction is the lower-APR refill; the margin is borderline"
        )
    else:
        exp11_verdict = "**Null result.** Risk reduction is driven by refilling budget with safer loans, not blind-spot selectivity"
    if roc_inside:
        exp11_verdict += "; return on capital sits inside both placebo intervals, so there is no evidence of a profit gain"

    rows = [
        ("EXP01", "Does non-linear modelling beat logistic regression?", res_exp01, exp01_verdict),
        ("EXP02", "Does calibration improve PD reliability?", res_exp02, exp02_verdict),
        ("EXP03", "Does expected-profit optimisation beat a fixed PD threshold?", res_exp03, exp03_verdict),
        ("EXP04", "Do graph cohort features add signal?", res_exp04, exp04_verdict),
        ("EXP05", "Does GraphSAGE beat the aggregates?", res_exp05, exp05_verdict),
        ("EXP06", "Can a budget-aware learned policy beat static policies?", res_exp06, exp06_verdict),
        ("EXP07", "How does the model degrade under vintage drift?", res_exp07, exp07_verdict),
        ("Deployment realism", "What if the model only knows the labels available at the deployment date?", res_dr, dr_verdict),
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
        unc = cond["unconstrained"].loc[_learners(cond["unconstrained"])]
        parts.append(
            "Control: no learner significantly beats the myopic rule with unconstrained capital (not significant "
            f"is not equal: the best interval reaches {unc['ci_high'].max():+,.0f})"
        )
    beat_myopic = {c: sig(cond, c, "win") for c in constrained}
    beat_myopic = {c: v for c, v in beat_myopic.items() if v}
    if beat_myopic:
        parts.append(
            "under a binding budget some learners do significantly beat the myopic rule: "
            + "; ".join(f"{', '.join(v)} ({c})" for c, v in beat_myopic.items())
        )
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
    sweep = pd.read_csv(TABLES_DIR / "exp06_rl_exp06_sweep.csv")
    collapsed = [
        f"{r['policy']} ({r['condition']})" for _, r in sweep.iterrows()
        if any(m in r["policy"] for m in LEARNER_MARKERS) and r["approval_rate"] < 0.01
    ]
    if collapsed:
        parts.append("learners that stopped lending (approval below 1%): " + "; ".join(collapsed))
    lam_loss = [
        c for c in cond
        if any(p.startswith("expected_profit(lambda=") and cond[c].loc[p, "ci_high"] < 0 for p in cond[c].index)
    ]
    if lam_loss:
        parts.append("the validation-fitted shadow-price policy is significantly worse than the myopic rule at " + ", ".join(lam_loss))
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


def _binding_budget_notes(cond: dict[str, pd.DataFrame], thr: dict[str, pd.DataFrame]) -> list[str]:
    """Where learners significantly beat the myopic rule under a budget, say so and say what that is worth."""
    notes = []
    for c in cond:
        if c == "unconstrained":
            continue
        d = cond[c].loc[_learners(cond[c])]
        wins = d[d["ci_low"] > 0]
        if wins.empty:
            continue
        t = thr[c]
        listed = "; ".join(
            f"{n} {r['mean_difference']:+,.0f} [{r['ci_low']:+,.0f}, {r['ci_high']:+,.0f}]" for n, r in wins.iterrows()
        )
        best = max(wins.index, key=lambda n: t.loc[n, "mean_difference"])
        myopic = (
            f" (the myopic rule is itself {t.loc['expected_profit', 'mean_difference']:+,.0f} against the threshold there)"
            if "expected_profit" in t.index else ""
        )
        notes.append(
            f"Under {c}, learners do significantly beat the myopic rule ({listed}){myopic}; against the 15% threshold "
            f"the best of them, {best}, is {t.loc[best, 'mean_difference']:+,.0f} "
            f"[{t.loc[best, 'ci_low']:+,.0f}, {t.loc[best, 'ci_high']:+,.0f}]."
        )
        notes.append("")
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
        + ("**the control is breached**." if breached else
           "**the control is not breached** (the difference is not significant, though the interval is wide).")
        + f" The plain 15% PD threshold beats the myopic rule by {thr_unc['mean_difference']:+,.0f} "
        f"(95% CI [{thr_unc['ci_low']:+,.0f}, {thr_unc['ci_high']:+,.0f}]), consistent with the "
        "real-book audit below.",
        "",
        *_binding_budget_notes(cond, thr),
        "Policy-level results by capital condition (mean over 30 episodes, 95% bootstrap CI on the "
        "paired difference), against the myopic expected-profit policy and the fixed 15% PD threshold:",
        "",
    ]
    lines += _exp06_policy_table(cond, thr)
    lines += _exp06_policy_notes(cond, thr)
    return "\n".join(lines) + "\n\n" + render_sensitivity_table() + "\n\n" + render_profit_paragraph() + "\n\n" + render_apr_ablation_paragraph()


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
        f"**{test.loc[ep, 'diff_vs_fixed_threshold_per_loan']:+.1f}** per pool applicant vs the 15% PD "
        f"threshold (95% CI [{test.loc[ep, 'diff_ci_low']:+.1f}, {test.loc[ep, 'diff_ci_high']:+.1f}], "
        f"{test.loc[ep, 'diff_vs_fixed_threshold_total'] / 1e6:+.2f}M in total). On the validation "
        f"pool the two rules are indistinguishable ({valid.loc[ep, 'diff_vs_fixed_threshold_per_loan']:+.1f} "
        f"per pool applicant, CI [{valid.loc[ep, 'diff_ci_low']:+.1f}, {valid.loc[ep, 'diff_ci_high']:+.1f}]).",
        f"- Predicted profit on the holdout exceeds realised profit by **{gap / 1e6:.1f}M** "
        f"({gap / float(d['predicted_total']):.0%} of the prediction). By source: default count "
        f"(PD too low) {pct(d['pd_effect']):.0f}%, good-loan yield curve {pct(d['yield_effect']):.0f}%, "
        f"loss-severity curve {pct(d['severity_effect']):.0f}%. Refitting the two cashflow curves "
        f"on the holdout itself (an oracle diagnostic, not deployable) removes the yield and "
        f"severity errors ({d_oracle['yield_effect'] / 1e6:+.1f}M, {d_oracle['severity_effect'] / 1e6:+.1f}M) "
        f"and leaves the PD effect ({d_oracle['pd_effect'] / 1e6:+.1f}M).",
        f"- Cashflow drift is therefore a real but secondary cause. Refitting the curves on the "
        f"validation window narrows the shortfall to {test.loc[ep_valid_fit, 'diff_vs_fixed_threshold_per_loan']:+.1f} "
        f"per pool applicant (CI [{test.loc[ep_valid_fit, 'diff_ci_low']:+.1f}, {test.loc[ep_valid_fit, 'diff_ci_high']:+.1f}]), "
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


PD_SOURCE_LABELS = {
    "primary_pd": "Production PD (primary features)",
    "with_lc_grade_pd": "+ LendingClub grade and APR (ablation)",
    "cohort_features_pd": "+ cohort features (EXP04 arm B)",
}


def _rule_phrase(row: pd.Series) -> str:
    if row["same_ci_high"] < 0:
        return "still loses to"
    if row["same_ci_low"] > 0:
        return "beats"
    return "is statistically indistinguishable from"


def render_apr_ablation_paragraph() -> str:
    """CR-10 and CR-11: the profit audit repeated with three PDs, in one table."""
    m = pd.read_csv(TABLES_DIR / "apr_in_pd_metrics.csv").set_index("pd_source")
    by_apr = pd.read_csv(TABLES_DIR / "apr_in_pd_by_apr.csv")
    rules = pd.read_csv(TABLES_DIR / "apr_in_pd_rules.csv")
    sources = [k for k in PD_SOURCE_LABELS if k in m.index]

    def gradient(source: str) -> tuple[float, float]:
        sub = by_apr[by_apr["pool"] == source].sort_values("apr_quintile")
        return float(sub.iloc[0]["observed_over_expected"]), float(sub.iloc[-1]["observed_over_expected"])

    def rule_row(source: str) -> pd.Series:
        return rules[(rules["pd_source"] == source) & (rules["rule"] == "expected_profit>0")].iloc[0]

    lines = [
        "**PD ablations: LendingClub's pricing and the cohort features.** The production PD excludes "
        "`int_rate`, `grade` and `sub_grade` by design (they are LendingClub's own risk score). Two "
        "refits are audited on the same real book (calibrator chosen on 2015-01..06, every number on "
        "2015-07..12). Neither is a replacement for the production PD: the first consumes the "
        "incumbent's pricing, and the second is the EXP04 comparison arm.",
        "",
        "| PD | ROC-AUC | ECE | O/E | O/E, lowest → highest APR quintile | Expected-profit rule vs 15% threshold, same PD (per pool applicant, 95% CI) | vs production-PD threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for source in sources:
        lo, hi = gradient(source)
        r = rule_row(source)
        lines.append(
            f"| {PD_SOURCE_LABELS[source]} | {m.loc[source, 'roc_auc']:.4f} | {m.loc[source, 'ece']:.4f} | "
            f"{m.loc[source, 'observed_expected_ratio']:.3f} | {lo:.2f} → {hi:.2f} | "
            f"{r['diff_vs_same_pd_threshold_per_loan']:+.1f} [{r['same_ci_low']:+.1f}, {r['same_ci_high']:+.1f}] | "
            f"{r['diff_vs_primary_pd_threshold_per_loan']:+.1f} [{r['ref_ci_low']:+.1f}, {r['ref_ci_high']:+.1f}] |"
        )
    lines.append("")
    base = rule_row("primary_pd")
    if "with_lc_grade_pd" in sources:
        g = rule_row("with_lc_grade_pd")
        glo, ghi = gradient("with_lc_grade_pd")
        lines.append(
            f"- With LendingClub's grade and APR in the PD, the observed/expected gradient across APR quintiles "
            f"flattens to {glo:.2f}–{ghi:.2f} and the expected-profit rule {_rule_phrase(g)} the 15% threshold under "
            f"that PD ({g['diff_vs_same_pd_threshold_per_loan']:+.1f} per pool applicant, against "
            f"{base['diff_vs_same_pd_threshold_per_loan']:+.1f} with the production PD)."
        )
    if "cohort_features_pd" in sources:
        c = rule_row("cohort_features_pd")
        clo, chi = gradient("cohort_features_pd")
        level = float(m.loc["cohort_features_pd", "observed_expected_ratio"])
        level_fixed = abs(level - 1.0) < 0.05
        gradient_left = (chi - clo) > 0.2
        lines.append(
            f"- With the cohort features, overall observed/expected is {level:.3f} "
            f"({'level fixed' if level_fixed else 'level not fixed'}), the gradient across APR quintiles is "
            f"{clo:.2f}–{chi:.2f}, and the expected-profit rule {_rule_phrase(c)} the 15% threshold under that PD "
            f"({c['diff_vs_same_pd_threshold_per_loan']:+.1f} per pool applicant)."
        )
        if level_fixed and gradient_left and c["same_ci_high"] < 0:
            sentence = (
                "- Fixing the level of the PD did not fix the decision: the rule still loses while an APR gradient "
                "in the PD's errors remains."
            )
            if "with_lc_grade_pd" in sources and rule_row("with_lc_grade_pd")["diff_vs_same_pd_threshold_per_loan"] > c["diff_vs_same_pd_threshold_per_loan"]:
                sentence += (
                    " The refit that flattens that gradient (LendingClub's pricing) narrows the shortfall more, "
                    "though it does not close it, so pricing information along APR matters more here than "
                    "calibration in the large."
                )
            lines.append(sentence)
        elif level_fixed and not (c["same_ci_high"] < 0):
            lines.append("- Fixing the level of the PD was enough to remove the rule's shortfall on this book.")
    lines += _cutoff_comparison(rules, m, sources)
    return "\n".join(lines)


def _signed_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def _cutoff_comparison(rules: pd.DataFrame, metrics: pd.DataFrame, sources: list[str]) -> list[str]:
    """Is the cohort PD's gain at a 15% cut-off a better PD, or just a stricter cut? (volume-matched control)"""
    vm_rows = rules[rules["rule"] == "volume_matched_to_cohort_15"]
    cut = rules[rules["rule"] == "fixed_threshold@0.15"].set_index("pd_source")
    if vm_rows.empty or "cohort_features_pd" not in cut.index:
        return []
    vm = vm_rows.iloc[0]
    c = cut.loc["cohort_features_pd"]
    prod = cut.loc["primary_pd"]
    labels = {
        "primary_pd": "Production PD, 15% cut-off",
        "with_lc_grade_pd": "+ LendingClub grade and APR, 15% cut-off",
        "cohort_features_pd": "+ cohort features, 15% cut-off",
    }
    lines = [
        "",
        "**Is the cohort PD's gain a better PD or a stricter cut?** At a 15% cut-off the cohort-feature PD accepts "
        "a different number of loans than the production PD because its predictions sit at a different level "
        f"(observed/expected {metrics.loc['cohort_features_pd', 'observed_expected_ratio']:.3f} against "
        f"{metrics.loc['primary_pd', 'observed_expected_ratio']:.3f}). "
        "The control tightens the production PD's own cut-off until it accepts the same number of loans as the "
        "cohort PD at 15% (chosen from predicted PDs and a loan count only; no outcome is read). "
        "Realised profit on 2015-07..12, per pool applicant, paired bootstrap 95% CI:",
        "",
        "| PD and rule | Loans accepted | PD cut-off | vs production PD at 15% | vs the volume-matched control |",
        "|---|---|---|---|---|",
    ]
    for source in sources:
        if source not in cut.index:
            continue
        r = cut.loc[source]
        ref = "reference" if source == "primary_pd" else _signed_ci(
            r["diff_vs_primary_pd_threshold_per_loan"], r["ref_ci_low"], r["ref_ci_high"]
        )
        lines.append(
            f"| {labels[source]} | {int(r['n_accepted']):,} | {r['pd_cutoff']:.3f} | {ref} | "
            f"{_signed_ci(r['diff_vs_volume_matched_per_loan'], r['vm_ci_low'], r['vm_ci_high'])} |"
        )
    lines.append(
        f"| Production PD, cut-off tightened to match the cohort PD's volume (control) | {int(vm['n_accepted']):,} | "
        f"{vm['pd_cutoff']:.3f} | {_signed_ci(vm['diff_vs_primary_pd_threshold_per_loan'], vm['ref_ci_low'], vm['ref_ci_high'])} | reference |"
    )
    lines.append("")
    gain_real = c["ref_ci_low"] > 0
    d_lo, d_hi = c["vm_ci_low"], c["vm_ci_high"]
    auc_c, auc_p = metrics.loc["cohort_features_pd", "roc_auc"], metrics.loc["primary_pd", "roc_auc"]
    if not gain_real:
        reading = "The cohort PD's 15% cut-off is not significantly better than the production PD's."
    elif d_lo <= 0 <= d_hi:
        reading = (
            "The gain is matched by a stricter cut on the production PD: at equal volume the two PDs are not "
            f"distinguishable (the interval includes zero), and the cohort PD's ROC-AUC is {auc_c:.4f} against "
            f"{auc_p:.4f}. On this book the cohort features buy a better-levelled cut-off, with no detectable "
            "ranking advantage."
        )
    elif d_lo > 0:
        reading = (
            "The gain is not matched by a stricter cut on the production PD: at equal volume the cohort-feature PD "
            "still earns more."
        )
    else:
        reading = "At equal volume the production PD earns more than the cohort-feature PD."
    lines.append(
        f"The 15% cut-off on the cohort-feature PD accepts {int(c['n_accepted']):,} loans against "
        f"{int(prod['n_accepted']):,} and earns {_signed_ci(c['diff_vs_primary_pd_threshold_per_loan'], c['ref_ci_low'], c['ref_ci_high'])} "
        f"per pool applicant relative to the production PD's cut-off. The production PD tightened to the same volume "
        f"earns {_signed_ci(vm['diff_vs_primary_pd_threshold_per_loan'], vm['ref_ci_low'], vm['ref_ci_high'])}, and the "
        f"cohort PD against that control earns {_signed_ci(c['diff_vs_volume_matched_per_loan'], d_lo, d_hi)}. {reading}"
    )
    return lines


def render_how_built() -> str:
    """CR-14: the experiments as one logical chain. Figures come from artifacts, none from the author."""
    decomp = pd.read_csv(TABLES_DIR / "profit_calibration_decomposition.csv")
    d = decomp[(decomp["pool"] == "test_h2") & (decomp["cashflow_source"] == "train_fit")].iloc[0]
    pd_share = abs(float(d["pd_effect"])) / abs(float(d["gap_total"]))
    by_apr = pd.read_csv(TABLES_DIR / "apr_in_pd_by_apr.csv")
    prim = by_apr[by_apr["pool"] == "primary_pd"].sort_values("apr_quintile")
    lo, hi = float(prim.iloc[0]["observed_over_expected"]), float(prim.iloc[-1]["observed_over_expected"])
    cond6, thr6 = _exp06_conditions(), _exp06_conditions(kind="diffthr")
    learner_wins = [c for c in thr6 if (thr6[c].loc[_learners(thr6[c]), "ci_low"] > 0).any()]
    lam_worse = [
        c for c in cond6
        if any(p.startswith("expected_profit(lambda=") and cond6[c].loc[p, "ci_high"] < 0 for p in cond6[c].index)
    ]
    step4_tail = (
        "no learned policy significantly beats the threshold in any capital condition"
        if not learner_wins else f"learned policies beat the threshold in {', '.join(learner_wins)}"
    ) + ("; the validation-fitted shadow-price policy is worse than the myopic rule" if lam_worse else "")
    lines = [
        "Each step answers a question the previous result made necessary. The experiment ids refer to the "
        "table below, which keeps them as references.",
        "",
        "1. **Predict (EXP01).** Can a model rank borrowers out of time? Yes, modestly: gradient boosting beats "
        "a logistic baseline on a window no selection touched.",
        "2. **Trust the probability (EXP02, EXP07, deployment realism).** A decision multiplies the probability "
        "by dollars, so ranking is not enough. Calibration gains are mostly a base-rate correction, the PD's "
        "level drifts across vintages while out-of-sample ranking is flat over the months observed, and a model "
        "trained only on the loans already resolved at the deployment date does clearly worse than the usual "
        "benchmark, though that comparison mixes staleness with selection on resolution.",
        "3. **Add relational information (EXP04, EXP05).** Do cohorts help? Not ranking. They fix the *level* of "
        "the PD in every training seed tested, though shuffled cohort keys reproduce part of that gain; GraphSAGE is "
        "the weakest arm.",
        "4. **Decide (EXP03, EXP06).** Does a profit rule built on the PD beat a plain PD threshold, and does "
        f"learning or capital pricing help? The profit rule loses to the threshold on the real book; {step4_tail}.",
        "5. **Diagnose the failure.** Where does predicted profit go wrong? A dollar decomposition of predicted "
        "versus realised profit, a PD refit with LendingClub's pricing, and a PD refit with the cohort features "
        "(correct level, remaining APR gradient). The PD's default-count error is the largest single source, it "
        "varies systematically with APR, and correcting the PD's level alone does not fix the decision.",
        "6. **Find where the PD is unreliable (EXP08, EXP09, EXP10).** Blind-spot signals, judged conditional on "
        "the PD level and decided on a selection window; only one passes.",
        "7. **Act on it (EXP11).** Excluding borrowers flagged by the kept signal, judged against random and "
        "APR-matched placebos.",
        "",
        f"**Root cause, measured.** The decision layers inherit the PD's errors. The production PD under-predicts "
        f"defaults where APR is high (observed/expected {lo:.2f} in the lowest APR quintile to {hi:.2f} in the "
        f"highest), so a rule that rewards APR selects understated risk: the PD's default-count error is "
        f"{pd_share:.0%} of the gap between predicted and realised profit (EXP03; the rest is drift in the cashflow "
        "curves), refitting the PD for level alone does not fix the decision, and the risk reduction in EXP11 is "
        "matched by excluding loans of the same APR mix.",
    ]
    return "\n".join(lines)


def render_label_gate_paragraph() -> str:
    """G4: the same-month rule for neighbour labels versus a strictly-before rule."""
    g = pd.read_csv(TABLES_DIR / "label_gate_sensitivity.csv").set_index("arm")
    labels = {
        "A_tabular": "Tabular only",
        "B_cohort_current_rule": "+ cohort features, current rule (`resolved <= issue` month)",
        "B_cohort_strict_rule": "+ cohort features, strict rule (`resolved < issue` month)",
    }
    lines = [
        "**Same-month label rule.** Cohort label aggregates admit a neighbour when its resolution month is on or "
        "before the applicant's issue month, so a neighbour resolved in the same month is admitted. A strictly-before "
        "rule is more conservative. Rebuilding the cohort features under the strict rule and refitting (three LightGBM "
        "seeds, 2015-07..12, mean ± sd):",
        "",
        "| Arm | ROC-AUC | ECE | O/E |",
        "|---|---|---|---|",
    ]
    for arm, label in labels.items():
        r = g.loc[arm]
        lines.append(
            f"| {label} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | {r['ece_mean']:.4f} ± {r['ece_std']:.4f} | "
            f"{r['observed_expected_ratio_mean']:.3f} ± {r['observed_expected_ratio_std']:.3f} |"
        )
    cur, strict, base = g.loc["B_cohort_current_rule"], g.loc["B_cohort_strict_rule"], g.loc["A_tabular"]
    shift = float(strict["ece_mean"] - cur["ece_mean"])
    noise = max(float(cur["ece_std"]), float(strict["ece_std"]), 1e-4)
    still_helps = float(strict["ece_mean"]) < float(base["ece_mean"]) - 2 * float(base["ece_std"])
    lines.append("")
    lines.append(
        f"Under the strict rule ECE moves by {shift:+.4f} against the current rule "
        + ("(within seed noise), " if abs(shift) <= 2 * noise else "(larger than seed noise), ")
        + ("and the cohort features still lower ECE below the tabular arm. " if still_helps
           else "and the cohort features no longer lower ECE below the tabular arm. ")
        + "The headline cohort results use the current rule; this table states what the stricter rule would change."
    )
    return "\n".join(lines)


def render_data_profile() -> str:
    """Row funnel and default rate by vintage, read from the measured data profile."""
    d = json.loads((TABLES_DIR / "data_profile.json").read_text(encoding="utf-8"))
    final = d["rows_after_window_filter"] - d["rows_dropped_non_terminal"] - d["rows_dropped_unmapped_status"]
    lines = [
        "| Stage | Rows |",
        "|---|---|",
        f"| Read from the raw accepted-loans file | {d['rows_read']:,} |",
        f"| After the 36-month term filter | {d['rows_after_term_filter']:,} |",
        f"| After dropping loans issued outside every split window | {d['rows_after_window_filter']:,} |",
        f"| Dropped: non-terminal or unmapped status inside a matured window | {d['rows_dropped_non_terminal'] + d['rows_dropped_unmapped_status']:,} |",
        f"| **Modelling table** | **{final:,}** |",
        "",
        "Rows with a malformed id are dropped while reading: " + f"{d['rows_dropped_bad_id']:,}.",
        "",
        "| Issue year | Default rate |",
        "|---|---|",
    ]
    for year, rate in d["default_rate_by_vintage"].items():
        lines.append(f"| {year} | {float(rate):.2%} |")
    return "\n".join(lines)


GUIDE_DIR = PROJECT_ROOT / "docs" / "guide"


def _blocks() -> dict:
    """Named blocks a guide page may embed between BEGIN/END markers.

    A page writes ``<!-- BEGIN:results-table -->`` ... ``<!-- END:results-table -->`` and this
    script fills the space, so no result number in ``docs/guide`` is ever typed by hand.
    """
    return {
        "data-table": render_data_table,
        "how-built": render_how_built,
        "label-gate": render_label_gate_paragraph,
        "data-profile": render_data_profile,
        "results-table": render_results_table,
        "rl-section": render_rl_paragraph,
        "profit-audit": render_profit_paragraph,
        "apr-ablation": render_apr_ablation_paragraph,
    }


def _block_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        r"(<!-- BEGIN:" + re.escape(name) + r" -->\n)[\s\S]*?(<!-- END:" + re.escape(name) + r" -->)"
    )


def render_guide_page(text: str) -> str:
    """Return ``text`` with every known block between markers regenerated from the artifacts."""
    for name, fn in _blocks().items():
        pattern = _block_pattern(name)
        if pattern.search(text):
            body = fn()
            text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    unknown = set(re.findall(r"<!-- BEGIN:([a-z0-9-]+) -->", text)) - set(_blocks())
    if unknown:
        raise RuntimeError(f"unknown guide block(s): {sorted(unknown)}")
    return text


def update_guide() -> None:
    """Regenerate the embedded blocks in every docs/guide page that has any."""
    for path in sorted(GUIDE_DIR.glob("*.md")):
        old = path.read_text(encoding="utf-8")
        new = render_guide_page(old.replace("\r\n", "\n"))
        if new != old.replace("\r\n", "\n"):
            path.write_text(new, encoding="utf-8", newline="\n")
            print(f"Updated blocks in {path}")


def update_readme() -> None:
    """Regenerate the BEGIN/END blocks in README.md, then in every guide page."""
    old = README_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    new = render_guide_page(old)
    if new != old:
        README_PATH.write_text(new, encoding="utf-8", newline="\n")
        print(f"Successfully updated tables in {README_PATH}")
    else:
        print(f"{README_PATH} is up to date")
    update_guide()


if __name__ == "__main__":
    update_readme()
