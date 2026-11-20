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


def test_the_checker_catches_planted_bad_references(tmp_path, monkeypatch):
    """G4-r2: a bogus function, a path with a flag, a bare test name and a bad path::test all fail."""
    (tmp_path / "docs" / "guide").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "real.py").write_text("def real_function():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_a.py").write_text("def test_real():\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("`scripts/real.py --any-flag` and `tests/test_a.py::test_real`.\n", encoding="utf-8")
    page = tmp_path / "docs" / "guide" / "09.md"
    page.write_text("`real_function()` is fine.\n", encoding="utf-8")
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.missing_functions() == []
    assert check_docs.missing_repository_references() == []

    page.write_text(
        "`ghost_function(x)` `scripts/nope.py --bogus-flag` `test_does_not_exist` `tests/test_a.py::test_gone`\n",
        encoding="utf-8",
    )
    functions = check_docs.missing_functions()
    references = check_docs.missing_repository_references()
    assert any("ghost_function" in p for p in functions)
    assert any("scripts/nope.py" in p for p in references)
    assert any("test_does_not_exist" in p for p in references)
    assert any("test_gone" in p for p in references)
