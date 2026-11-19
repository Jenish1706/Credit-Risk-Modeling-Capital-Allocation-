"""Test that README.md tables match rendered artifact outputs exactly.

Enforces BRIEF_A §7 step S4.4:
'Render the README tables from artifacts with a script. Add a test that
re-renders them and fails on any difference, so a hand-edited number can't survive.'
"""

import re
from pathlib import Path

from scripts.render_readme import (
    README_PATH,
    render_data_table,
    render_results_table,
    render_rl_paragraph,
)


def test_readme_data_table_matches_rendered():
    content = README_PATH.read_text(encoding="utf-8")
    expected = render_data_table()
    data_pattern = re.compile(
        r"\| Split \| Window \| Loans \| Default rate \|[\s\S]*?(?=\n\nThe rising default rate)"
    )
    match = data_pattern.search(content)
    assert match is not None, "Data table not found in README.md"
    assert match.group(0).strip() == expected.strip(), (
        "README.md data table differs from rendered artifact values! "
        "Run `python scripts/render_readme.py` to synchronize."
    )


def test_readme_results_table_matches_rendered():
    content = README_PATH.read_text(encoding="utf-8")
    expected = render_results_table()
    results_pattern = re.compile(
        r"\| # \| Question \| Result \| Verdict \|[\s\S]*?(?=\n\n### Four results worth reading twice)"
    )
    match = results_pattern.search(content)
    assert match is not None, "Results table not found in README.md"
    assert match.group(0).strip() == expected.strip(), (
        "README.md results table differs from rendered artifact values! "
        "Run `python scripts/render_readme.py` to synchronize."
    )


def test_readme_rl_paragraph_matches_rendered():
    content = README_PATH.read_text(encoding="utf-8")
    expected = render_rl_paragraph()
    pattern = re.compile(r"\*\*RL does not earn its place here[\s\S]*?(?=\n\n---\n\n## Architecture)")
    match = pattern.search(content)
    assert match is not None, "RL paragraph not found in README.md"
    assert match.group(0).strip() == expected.strip(), (
        "README.md RL paragraph differs from rendered artifact values! "
        "Run `python scripts/render_readme.py` to synchronize."
    )


def test_calibration_ratios_use_one_convention_and_match_artifacts():
    """CR-05: observed/expected (O/E) only; P/O must not appear in public text."""
    import pandas as pd

    from scripts.render_readme import PROJECT_ROOT, TABLES_DIR

    texts = {p: p.read_text(encoding="utf-8")
             for p in [README_PATH, *sorted((PROJECT_ROOT / "docs").rglob("*.md"))]}
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
    import json

    import pandas as pd

    from scripts.render_readme import TABLES_DIR

    summary = json.loads((TABLES_DIR / "cohort_controls_exp04.json").read_text(encoding="utf-8"))["seed_summary"]
    assert len(summary["seeds"]) >= 3
    deltas = pd.read_csv(TABLES_DIR / "cohort_controls_exp04_seed_deltas.csv")
    assert len(deltas) == len(summary["seeds"])
    every_seed = bool(
        (deltas["delta_ece_vs_macro_ci_high"] < 0).all() and (deltas["delta_ece_vs_shuffled_ci_high"] < 0).all()
    )
    assert summary["survives_controls_for_every_seed"] == every_seed


def test_exp06_table_reports_lambda_rows_approval_rates_and_collapsed_learners():
    """CR-08: nothing the EXP06 artifacts contain may be dropped from the README."""
    import pandas as pd

    from scripts.render_readme import LEARNER_MARKERS, TABLES_DIR

    text = README_PATH.read_text(encoding="utf-8")
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
    from scripts.render_readme import PROJECT_ROOT

    readme = README_PATH.read_text(encoding="utf-8")
    assert "Every policy result comes from a simulator" not in readme
    assert "Platt/isotonic calibration" not in readme
    audit = (PROJECT_ROOT / "docs" / "AUDIT_REPORT.md").read_text(encoding="utf-8")
    assert audit.count("Superseded by the v3 audit") >= 4
    broke = (PROJECT_ROOT / "docs" / "WHAT_BROKE.md").read_text(encoding="utf-8")
    assert "not an upper bound" in broke


def test_apr_ablation_is_rendered_from_its_artifacts():
    """CR-10: the with-LendingClub-pricing ablation is reported, labelled, and matches its CSVs."""
    import pandas as pd

    from scripts.render_readme import TABLES_DIR

    text = README_PATH.read_text(encoding="utf-8")
    assert "**Ablation: a PD that sees LendingClub's own pricing.**" in text
    rules = pd.read_csv(TABLES_DIR / "apr_in_pd_rules.csv")
    row = rules[(rules["pd_source"] == "with_lc_grade_pd") & (rules["rule"] == "expected_profit>0")].iloc[0]
    assert f"{row['diff_vs_same_pd_threshold_per_loan']:+.1f} per loan vs the 15% threshold under the same PD" in text
    metrics = pd.read_csv(TABLES_DIR / "apr_in_pd_metrics.csv").set_index("pd_source")
    assert f"{metrics.loc['with_lc_grade_pd', 'roc_auc']:.4f}" in text
