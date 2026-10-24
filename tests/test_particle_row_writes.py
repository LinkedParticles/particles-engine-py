# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The store never rewrites a particle row from a model (D2).

An SDK older than a row may write to the store (a minor ahead of it), and is
safe only because no write path rebuilds an existing row from the SDK's own
model: a column, or a key inside a JSON column, that the older model does not
know would be silently erased. A particle is inserted once from its model and
every later change is a column-targeted update; JSON-valued columns are edited
as raw JSON. These static checks keep a new write path from breaking that
without anyone noticing. The audit they encode is the table D2.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "particles"
STORE = PACKAGE / "store" / "particle_store.py"

#: Where ``ParticleRow.from_model`` may be called: the one insert path, and the
#: synthetic-store generator, which writes fresh rows into an empty file.
FROM_MODEL_ALLOWED = {
    ("store/particle_store.py", "insert_particle"),
    ("benchmark/query_latency/generator.py", "_row"),
}


def _modules() -> list[tuple[str, ast.Module]]:
    return [
        (str(path.relative_to(PACKAGE)), ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(PACKAGE.rglob("*.py"))
    ]


def _json_columns() -> set[str]:
    tree = ast.parse(STORE.read_text(encoding="utf-8"))
    (row,) = (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ParticleRow")
    return {
        stmt.target.id
        for stmt in row.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id.endswith("_json")
    }


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, str]:
    owner: dict[ast.AST, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            for node in ast.walk(fn):
                owner.setdefault(node, fn.name)
    return owner


def test_the_audit_sees_the_json_columns() -> None:
    # Guards the gate itself: an empty set would make the next test vacuous.
    assert {"provenance_json", "properties_json", "tags_json"} <= _json_columns()


def test_from_model_is_called_only_where_a_row_is_born() -> None:
    callers: set[tuple[str, str]] = set()
    for rel, tree in _modules():
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_model"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ParticleRow"
            ):
                callers.add((rel, owner.get(node, "<module>")))
    assert callers <= FROM_MODEL_ALLOWED, sorted(callers - FROM_MODEL_ALLOWED)


def test_no_json_column_is_written_from_a_model_dump() -> None:
    columns = _json_columns()
    offenders: list[str] = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Attribute) and t.attr in columns for t in node.targets):
                continue
            for call in ast.walk(node.value):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in {"model_dump", "model_dump_json"}
                ):
                    offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_no_write_path_merges_a_detached_row() -> None:
    # ``session.merge(row)`` copies every mapped column from the object onto
    # the stored row, which is a whole-row rewrite by another name.
    offenders = [
        f"{rel}:{node.lineno}"
        for rel, tree in _modules()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "merge"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "session"
    ]
    assert offenders == []
