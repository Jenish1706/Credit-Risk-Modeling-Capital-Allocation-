"""D4: the documentation's links, paths and named functions all exist."""

from __future__ import annotations

from scripts import check_docs


def test_every_relative_link_resolves():
    assert check_docs.broken_links() == []


def test_every_path_named_in_the_code_walkthrough_exists():
    assert check_docs.missing_paths() == []


def test_every_function_named_in_the_code_pages_is_defined():
    assert check_docs.missing_functions() == []


def test_the_checker_can_actually_fail():
    names = check_docs.defined_names()
    assert "compute_metrics" in names and "conditional_by_window" in names
    assert "definitely_not_a_function" not in names
