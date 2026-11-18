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
    assert f"O/E **{a:.3f} → {b:.3f}**" in texts[README_PATH]
