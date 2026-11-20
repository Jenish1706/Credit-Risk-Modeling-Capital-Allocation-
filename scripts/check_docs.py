#!/usr/bin/env python
"""Documentation checks (gate G3).

    python scripts/check_docs.py

1. Every relative link in README.md and docs/ resolves to a file or directory.
2. Every repository path named in backticks in docs/guide/05_CODE_WALKTHROUGH.md exists.
3. Every path or test name in backticks in any markdown file (except the historical audit report) exists.
4. Every function, method or class named in backticks with a call or class shape in the guide pages
   that describe code (01, 05, 07) is defined somewhere in src/ or scripts/.

Exit status is non-zero when anything is missing, and each problem is printed with its file.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s]+)\)")
BACKTICK = re.compile(r"`([^`\n]+)`")
PATH_ROOTS = ("scripts/", "src/", "configs/", "tests/", "docs/")
CODE_PAGES = ("01_CONCEPTS.md", "05_CODE_WALKTHROUGH.md", "07_TESTING.md")

#: Names that appear as `name(...)` in prose but are not defined in this repository.
EXTERNAL = {
    "pytest", "python", "pip", "conda", "git", "argmax", "sqrt", "ln", "mean", "ReLU", "act", "N",
    "N_r", "importorskip", "qcut", "searchsorted", "pd", "np", "torch", "lgb", "bool", "int", "float", "len",
}


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def broken_links() -> list[str]:
    problems = []
    for md in markdown_files():
        text = md.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0].split(":", 1)[0] if not re.match(r"^[A-Za-z]:", target) else target
            if not path:
                continue
            if not (md.parent / path).resolve().exists():
                problems.append(f"{md.relative_to(ROOT)}: broken link -> {target}")
    return problems


def missing_paths() -> list[str]:
    page = ROOT / "docs" / "guide" / "05_CODE_WALKTHROUGH.md"
    problems = []
    for token in BACKTICK.findall(page.read_text(encoding="utf-8")):
        token = token.strip()
        if token.startswith(PATH_ROOTS) and re.fullmatch(r"[\w./\-*]+", token) and "*" not in token:
            if not (ROOT / token).exists():
                problems.append(f"05_CODE_WALKTHROUGH.md: path does not exist -> {token}")
    return problems


REPO_PATH = re.compile(r"^(?:scripts|src|tests|configs|docs|results/tables|results/reports)/[\w./\-]+$")
#: Documents that describe superseded history may name files that no longer exist.
HISTORICAL = {"AUDIT_REPORT.md"}


def missing_repository_references() -> list[str]:
    """Every path-like or test-name token in backticks, in every markdown file, must exist."""
    test_names = _test_function_names()
    problems = []
    for md in markdown_files():
        if md.name in HISTORICAL:
            continue
        for token in BACKTICK.findall(md.read_text(encoding="utf-8")):
            token = token.strip()
            path_part, _, func = token.partition("::")
            if REPO_PATH.match(path_part) and "*" not in path_part and "{" not in path_part:
                if not (ROOT / path_part).exists():
                    problems.append(f"{md.relative_to(ROOT)}: path does not exist -> {token}")
                    continue
                if func and func.startswith("test_") and func not in test_names:
                    problems.append(f"{md.relative_to(ROOT)}: test does not exist -> {token}")
    return problems


def _test_function_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "tests").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef):
                names.add(node.name)
    return names


def defined_names() -> set[str]:
    names: set[str] = set()
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py"), *(ROOT / "tests").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
    return names


def missing_functions() -> list[str]:
    defined = defined_names()
    problems = []
    for name in CODE_PAGES:
        text = (ROOT / "docs" / "guide" / name).read_text(encoding="utf-8")
        for token in BACKTICK.findall(text):
            match = re.match(r"^(?:[A-Za-z_][\w]*\.)*([A-Za-z_][\w]*)\(", token.strip())
            if not match:
                continue
            fn = match.group(1)
            if fn in defined or fn in EXTERNAL:
                continue
            problems.append(f"{name}: `{fn}(` is not defined in src/, scripts/ or tests/")
    return problems


def main() -> int:
    problems = [*broken_links(), *missing_paths(), *missing_functions(), *missing_repository_references()]
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s) found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
