"""README and guide pages are rendered from artifacts; a hand-edited number cannot survive.

``scripts/render_readme.py`` fills named BEGIN/END blocks from ``results/tables/``. These tests
re-render every block and fail on any difference, and check the specific results the README and the
results page must not drop.
"""

from __future__ import annotations

import json

import pandas as pd

from scripts.render_readme import (
    GUIDE_DIR,
    PROJECT_ROOT,
    README_PATH,
    TABLES_DIR,
    render_guide_page,
)

RESULTS_PAGE = GUIDE_DIR / "09_RESULTS_AND_LIMITATIONS.md"


def _read(path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_readme_blocks_match_rendered_artifacts():
    text = _read(README_PATH)
    assert "<!-- BEGIN:results-table -->" in text and "<!-- BEGIN:data-table -->" in text
    assert render_guide_page(text) == text, (
        "README.md has a stale embedded block; run `python scripts/render_readme.py`"
    )


def test_guide_blocks_match_rendered_artifacts():
    pages = sorted(GUIDE_DIR.glob("*.md"))
    assert pages, "docs/guide has no pages"
    for page in pages:
        text = _read(page)
        assert render_guide_page(text) == text, (
            f"{page.name} has a stale embedded block; run `python scripts/render_readme.py`"
        )


def test_readme_is_a_short_landing_page():
    assert len(_read(README_PATH).splitlines()) < 200
    assert "docs/guide/00_START_HERE.md" in _read(README_PATH)


def test_calibration_ratios_use_one_convention_and_match_artifacts():
    """CR-05: observed/expected (O/E) only; P/O must not appear in public text."""
    texts = {p: _read(p) for p in [README_PATH, *sorted((PROJECT_ROOT / "docs").rglob("*.md"))]}
    for path, text in texts.items():
        assert "P/O" not in text, f"{path} uses the P/O convention; use O/E"

    h2 = pd.read_csv(TABLES_DIR / "cohort_graph_exp04_exp05_h2.csv", index_col=0)
    a = float(h2.loc["A_tabular", "observed_expected_ratio"])
    b = float(h2.loc["B_tabular_plus_graph", "observed_expected_ratio"])
    # O/E is observed / mean predicted, i.e. the reciprocal of the predicted/observed column
    assert abs(a - 1.0 / float(h2.loc["A_tabular", "predicted_observed_ratio"])) < 1e-3
    seeds = pd.read_csv(TABLES_DIR / "cohort_controls_exp04_seed_summary.csv", index_col=0)
    sa = float(seeds.loc["A_tabular", "observed_expected_ratio_mean"])
    sb = float(seeds.loc["B_tabular_plus_graph", "observed_expected_ratio_mean"])
    assert f"O/E **{sa:.3f} → {sb:.3f}**" in texts[README_PATH]
    assert f"O/E {a:.3f} → {b:.3f}" in texts[README_PATH]  # single-fit run, unbolded


def test_exp04_is_reported_over_at_least_three_seeds():
    """CR-06: the calibration claim must rest on several model fits, not one."""
    summary = json.loads((TABLES_DIR / "cohort_controls_exp04.json").read_text(encoding="utf-8"))["seed_summary"]
    assert len(summary["seeds"]) >= 3
    deltas = pd.read_csv(TABLES_DIR / "cohort_controls_exp04_seed_deltas.csv")
    assert len(deltas) == len(summary["seeds"])
    every_seed = bool(
        (deltas["delta_ece_vs_macro_ci_high"] < 0).all() and (deltas["delta_ece_vs_shuffled_ci_high"] < 0).all()
    )
    assert summary["survives_controls_for_every_seed"] == every_seed


def test_exp06_table_reports_lambda_rows_approval_rates_and_collapsed_learners():
    """CR-08: nothing the EXP06 artifacts contain may be dropped from the results page."""
    from scripts.render_readme import LEARNER_MARKERS

    text = _read(RESULTS_PAGE)
    sweep = pd.read_csv(TABLES_DIR / "exp06_rl_exp06_sweep.csv")
    assert "oracle_ceiling" not in text and "oracle_ceiling" not in " ".join(sweep["policy"])
    lam = sweep[sweep["policy"].str.startswith("expected_profit(lambda=")]
    assert len(lam) >= 3
    for _, r in lam.iterrows():
        assert f"| {r['condition']} | {r['policy']} |" in text
        assert f"{r['approval_rate']:.1%}" in text
    assert sweep["policy"].str.startswith("expected_profit(lambda_eval_pool=").any()
    collapsed = sweep[
        sweep["policy"].apply(lambda p: any(m in p for m in LEARNER_MARKERS)) & (sweep["approval_rate"] < 0.01)
    ]
    for _, r in collapsed.iterrows():
        assert f"{r['policy']} at {r['condition']} (approval rate {r['approval_rate']:.2%}" in text


def test_superseded_banners_and_stale_claims_are_handled():
    """CR-09: pre-v3 documents are marked superseded; false README claims are gone."""
    readme = _read(README_PATH)
    assert "Every policy result comes from a simulator" not in readme
    assert "Platt/isotonic calibration" not in readme
    audit = _read(PROJECT_ROOT / "docs" / "AUDIT_REPORT.md")
    assert audit.count("Superseded by the v3 audit") >= 4
    broke = _read(PROJECT_ROOT / "docs" / "WHAT_BROKE.md")
    assert "not an upper bound" in broke


def test_apr_ablation_is_rendered_from_its_artifacts():
    """CR-10 and CR-11: the PD ablations (LendingClub pricing, cohort features) match their CSVs."""
    text = _read(RESULTS_PAGE)
    assert "**PD ablations: LendingClub's pricing and the cohort features.**" in text
    rules = pd.read_csv(TABLES_DIR / "apr_in_pd_rules.csv")
    metrics = pd.read_csv(TABLES_DIR / "apr_in_pd_metrics.csv").set_index("pd_source")
    assert {"primary_pd", "with_lc_grade_pd", "cohort_features_pd"} <= set(metrics.index)
    for source in ("primary_pd", "with_lc_grade_pd", "cohort_features_pd"):
        row = rules[(rules["pd_source"] == source) & (rules["rule"] == "expected_profit>0")].iloc[0]
        cell = (
            f"{row['diff_vs_same_pd_threshold_per_loan']:+.1f} "
            f"[{row['same_ci_low']:+.1f}, {row['same_ci_high']:+.1f}]"
        )
        assert cell in text, f"{source}: rule-vs-threshold cell missing from the results page"
        assert f"{metrics.loc[source, 'roc_auc']:.4f}" in text
        assert f"{metrics.loc[source, 'observed_expected_ratio']:.3f}" in text



def test_readme_prose_states_no_result_numbers():
    """G4: outside the rendered blocks and code fences, the README carries no result figures."""
    import re

    text = _read(README_PATH)
    text = re.sub(r"<!-- BEGIN:([a-z0-9-]+) -->.*?<!-- END:\1 -->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    protocol_parameters = {"15%"}  # the fixed PD-threshold parameter, not a result
    found = re.findall(r"\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?%", text)
    unexpected = [t for t in found if t not in protocol_parameters]
    assert not unexpected, f"README prose contains figures that are not rendered from artifacts: {unexpected}"


def test_cutoff_comparison_and_volume_matched_control_are_rendered_from_the_rules_csv():
    """The 15% cut-off on each PD vs the production PD's, and the volume-matched control, match their artifact."""
    text = _read(RESULTS_PAGE)
    rules = pd.read_csv(TABLES_DIR / "apr_in_pd_rules.csv")
    cut = rules[rules["rule"] == "fixed_threshold@0.15"].set_index("pd_source")
    vm = rules[rules["rule"] == "volume_matched_to_cohort_15"].iloc[0]
    assert "**Is the cohort PD's gain a better PD or a stricter cut?**" in text

    # the control accepts exactly as many loans as the cohort-feature PD at 15%, with a cut-off tighter than 15%
    assert int(vm["n_accepted"]) == int(cut.loc["cohort_features_pd", "n_accepted"])
    assert vm["pd_cutoff"] < 0.15

    for source, row in cut.iterrows():
        assert f"| {int(row['n_accepted']):,} | {row['pd_cutoff']:.3f} |" in text, source
        if source != "primary_pd":
            cell = (
                f"{row['diff_vs_primary_pd_threshold_per_loan']:+.1f} "
                f"[{row['ref_ci_low']:+.1f}, {row['ref_ci_high']:+.1f}]"
            )
            assert cell in text, f"{source}: cut-off comparison cell missing"
    control = (
        f"{vm['diff_vs_primary_pd_threshold_per_loan']:+.1f} [{vm['ref_ci_low']:+.1f}, {vm['ref_ci_high']:+.1f}]"
    )
    assert f"| {int(vm['n_accepted']):,} | {vm['pd_cutoff']:.3f} | {control} | reference |" in text

    # the stated reading follows the interval of cohort-PD-vs-control, not the other way round
    lo, hi = cut.loc["cohort_features_pd", "vm_ci_low"], cut.loc["cohort_features_pd", "vm_ci_high"]
    if cut.loc["cohort_features_pd", "ref_ci_low"] > 0 and lo <= 0 <= hi:
        assert "The gain is matched by a stricter cut on the production PD" in text
    elif lo > 0:
        assert "The gain is not matched by a stricter cut on the production PD" in text
