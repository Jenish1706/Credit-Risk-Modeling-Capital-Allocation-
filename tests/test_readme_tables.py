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
    """CR-10: the with-LendingClub-pricing ablation is reported, labelled, and matches its CSVs."""
    text = _read(RESULTS_PAGE)
    assert "**Ablation: a PD that sees LendingClub's own pricing.**" in text
    rules = pd.read_csv(TABLES_DIR / "apr_in_pd_rules.csv")
    row = rules[(rules["pd_source"] == "with_lc_grade_pd") & (rules["rule"] == "expected_profit>0")].iloc[0]
    assert f"{row['diff_vs_same_pd_threshold_per_loan']:+.1f} per loan vs the 15% threshold under the same PD" in text
    metrics = pd.read_csv(TABLES_DIR / "apr_in_pd_metrics.csv").set_index("pd_source")
    assert f"{metrics.loc['with_lc_grade_pd', 'roc_auc']:.4f}" in text
