# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A Subject's aliases change only through ``_set_aliases``.

The subject alias index is derived from ``subjects.aliases_json``, and it stays
in step only because every write of that column goes through the one helper
that rewrites the index beside it. A static scan of ``particles/`` fails on a
write anywhere else, the way ``tests/test_provenance_append_only.py`` guards
provenance. A "write" is an assignment to an attribute named ``aliases_json``,
an ``aliases_json=`` keyword (the ORM constructor, ``update(...).values(...)``),
a dict literal keyed ``"aliases_json"``, a ``setattr`` with that literal name,
or a string literal that looks like ``UPDATE subjects … aliases_json``.

The scan cannot see a dynamic attribute name, SQL assembled at run time, or a
write outside ``particles/``. The behavioural half lives in
``tests/test_subjects.py::TestAliasIndex``, which checks the index against a
rebuild after every writer and deleter.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "particles"
COLUMN = "aliases_json"

# (path relative to particles/, enclosing function qualname), each with its reason.
ALLOWLIST = {
    # Row creation; insert_subject indexes the same aliases beside it.
    ("store/subject_store.py", "SubjectRow.from_model"),
    # The one writer after creation.
    ("store/subject_store.py", "_set_aliases"),
    # Bulk-inserts Subject rows with a literal "[]": nothing to index.
    ("benchmark/query_latency/generator.py", "build_store"),
    # The taxonomy table's own column of the same name, not a Subject's.
    ("store/taxonomy_store.py", "insert_taxonomy"),
}

_RAW_UPDATE = re.compile(r"\bUPDATE\s+subjects\b.*\baliases_json\b", re.I | re.S)


class _WriteFinder(ast.NodeVisitor):
    """Collect ``(qualname, lineno)`` for every write to the column."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.writes: list[tuple[str, int]] = []

    def _record(self, node: ast.AST) -> None:
        self.writes.append((".".join(self.scope) or "<module>", getattr(node, "lineno", 0)))

    def _scoped(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_ClassDef = _scoped
    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped

    def _check_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Attribute) and target.attr == COLUMN:
            self._record(target)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._check_target(elt)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # A bare annotation (the ORM column declaration) is not a write.
        if node.value is not None:
            self._check_target(node.target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if any(kw.arg == COLUMN for kw in node.keywords):
            self._record(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == COLUMN
        ):
            self._record(node)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if any(isinstance(k, ast.Constant) and k.value == COLUMN for k in node.keys):
            self._record(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _RAW_UPDATE.search(node.value):
            self._record(node)


def _writes_in(source: str) -> list[tuple[str, int]]:
    finder = _WriteFinder()
    finder.visit(ast.parse(source))
    return finder.writes


def _package_writes() -> list[tuple[str, str, int]]:
    out = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        for qualname, lineno in _writes_in(path.read_text(encoding="utf-8")):
            out.append((rel, qualname, lineno))
    return out


def test_aliases_json_written_only_at_sanctioned_sites() -> None:
    offenders = [
        f"particles/{rel}:{lineno} in {qualname}"
        for rel, qualname, lineno in _package_writes()
        if (rel, qualname) not in ALLOWLIST
    ]
    assert not offenders, (
        "A Subject's aliases change only through subject_store._set_aliases, which"
        " keeps the alias index in step. Offending writes: " + ", ".join(offenders)
    )


def test_allowlist_is_not_stale() -> None:
    # An entry whose site no longer writes the column would silently widen the
    # exception to whatever next lands under that name.
    present = {(rel, qualname) for rel, qualname, _ in _package_writes()}
    assert present >= ALLOWLIST


def test_scan_is_not_vacuous() -> None:
    source = """
class Row:
    aliases_json: str

def ok_read(row):
    return row.aliases_json

def attr(row):
    row.aliases_json = "[]"

def aug(row):
    row.aliases_json += "x"

def kw(Row):
    return Row(aliases_json="[]")

def dict_key(stmt):
    return stmt.values({"aliases_json": "[]"})

def via_setattr(row):
    setattr(row, "aliases_json", "[]")

def raw():
    return "UPDATE subjects SET aliases_json = '[]'"
"""
    assert [q for q, _ in _writes_in(source)] == [
        "attr",
        "aug",
        "kw",
        "dict_key",
        "via_setattr",
        "raw",
    ]
