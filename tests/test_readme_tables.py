"""Test that README.md tables match rendered artifact outputs exactly.

Enforces BRIEF_A §7 step S4.4:
'Render the README tables from artifacts with a script. Add a test that
re-renders them and fails on any difference, so a hand-edited number can't survive.'
"""

import re
from pathlib import Path

from scripts.render_readme import README_PATH, render_data_table, render_results_table


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
