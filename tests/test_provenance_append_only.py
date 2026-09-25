# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Provenance is append-only (D1, gated per D6).

A static scan of every module under ``particles/`` fails on any write to the
``provenance_json`` column outside an explicit allowlist of three sites:

- ``ParticleRow.from_model`` — row creation, reached through
  ``insert_particle``;
- ``append_provenance_ref`` — the append-only, idempotent writer;
- ``strip_entry_provenance_refs`` — the operator hard-delete exception, called
  only by ``particles corpus delete``.

A "write" is any of:

- an assignment, augmented assignment or annotated assignment whose target is
  an attribute named ``provenance_json`` (``row.provenance_json = …``);
- a call passing a ``provenance_json=`` keyword, which covers both the ORM
  constructor and ``update(ParticleRow).values(provenance_json=…)``;
- a dict literal keyed ``"provenance_json"`` (``.values({"provenance_json": …})``);
- ``setattr(obj, "provenance_json", …)`` with a literal name;
- a string literal that looks like ``UPDATE particles … provenance_json``.

Behavioural tests below pin what the two allowlisted mutators do: the append
never reorders or displaces ``provenance[0]``, and the strip removes only the
named entry's ``SOURCE`` refs in place.

What the scan **cannot** see, which is why D6 records this principle as partly
checked:

- a dynamic attribute name (``setattr(row, name, …)``, ``row.__dict__[…]``);
- raw SQL assembled at run time, or held anywhere other than a single string
  literal (an f-string, a concatenation, a file);
- a whole-row overwrite such as ``session.merge(ParticleRow.from_model(p))``
  onto an existing id, which reaches the column through the sanctioned creation
  site;
- a write in code outside ``particles/`` (migrations, scripts).
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.store.particle_store import (
    ParticleRow,
    append_provenance_ref,
    insert_particle,
    strip_entry_provenance_refs,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "particles"
COLUMN = "provenance_json"

# (path relative to particles/, enclosing function qualname). Each site carries
# a comment in the source citing the rule it is sanctioned under.
ALLOWLIST = {
    ("store/particle_store.py", "ParticleRow.from_model"),
    ("store/particle_store.py", "append_provenance_ref"),
    ("store/particle_store.py", "strip_entry_provenance_refs"),
}

_RAW_UPDATE = re.compile(r"\bUPDATE\s+particles\b.*\bprovenance_json\b", re.I | re.S)


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


# --- the gate ----------------------------------------------------------------


def test_provenance_json_written_only_at_sanctioned_sites() -> None:
    offenders = [
        f"particles/{rel}:{lineno} in {qualname}"
        for rel, qualname, lineno in _package_writes()
        if (rel, qualname) not in ALLOWLIST
    ]
    assert not offenders, (
        "provenance_json is append-only (D1): only row creation,"
        " append_provenance_ref and the hard-delete stripper may write it."
        " Supersede the claim instead of rewriting its basis. Offending writes: "
        + ", ".join(offenders)
    )


def test_allowlist_is_not_stale() -> None:
    # An entry whose site no longer writes the column would silently widen the
    # exception to whatever next lands under that name.
    present = {(rel, qualname) for rel, qualname, _ in _package_writes()}
    assert present >= ALLOWLIST


def test_scan_is_not_vacuous() -> None:
    source = """
class Row:
    provenance_json: str

def ok_read(row):
    return row.provenance_json

def attr(row):
    row.provenance_json = "[]"

def aug(row):
    row.provenance_json += "x"

def ann(row):
    row.provenance_json: str = "[]"

def unpack(a, b):
    a.provenance_json, b.x = "[]", 1

async def orm_update(session):
    await session.execute(update(ParticleRow).values(provenance_json="[]"))

def dict_values(stmt):
    return stmt.values({"provenance_json": "[]"})

def literal_setattr(row):
    setattr(row, "provenance_json", "[]")

def raw_sql(conn):
    conn.execute("UPDATE particles SET provenance_json = :p WHERE id = :id")

def raw_sql_other_column(conn):
    conn.execute("UPDATE particles SET properties_json = :p WHERE id = :id")

def dynamic(row, name):
    setattr(row, name, "[]")
"""
    assert [q for q, _ in _writes_in(source)] == [
        "attr",
        "aug",
        "ann",
        "unpack",
        "orm_update",
        "dict_values",
        "literal_setattr",
        "raw_sql",
    ]


# --- behaviour of the two sanctioned mutators --------------------------------


def _src(entry: str, snapshot: str, *, location: str | None = None) -> ProvenanceRef:
    return ProvenanceRef(
        type=ProvenanceRefType.SOURCE,
        corpus_entry_id=entry,
        snapshot_id=snapshot,
        location=location,
    )


def _premise(particle_id: str) -> ProvenanceRef:
    return ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=particle_id)


def _particle(provenance: list[ProvenanceRef]) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content="A claim with several sources.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        asserted_at=datetime.now(UTC),
        provenance=provenance,
    )


async def _stored_refs(session: Any, particle_id: str) -> list[dict[str, Any]]:
    row = await session.get(ParticleRow, particle_id)
    assert row is not None
    refs: list[dict[str, Any]] = json.loads(row.provenance_json)
    return refs


def _keys(refs: list[dict[str, Any]]) -> list[tuple[str, str | None, str | None]]:
    return [(r["type"], r["corpus_entry_id"], r.get("snapshot_id")) for r in refs]


async def test_append_never_reorders_or_displaces_provenance_zero(db_session: Any) -> None:
    p = _particle([_src("E1", "s1"), _src("E2", "s2")])
    await insert_particle(db_session, p)
    await db_session.flush()
    before = await _stored_refs(db_session, p.id)

    offered = [
        _src("E3", "s3"),
        _src("E1", "s1"),  # identical to provenance[0]: a no-op, never moved
        _src("E1", "s9"),  # same entry, new snapshot: appended at the end
        _src("E2", "s2"),
        _src("E0", "s0"),  # sorts "earlier" by id, still goes last
    ]
    for ref in offered:
        await append_provenance_ref(db_session, p.id, ref)
        after = await _stored_refs(db_session, p.id)
        # Every earlier ref is still there, unchanged and in the same order.
        assert after[: len(before)] == before
        assert after[0] == before[0]
        before = after

    assert _keys(before) == [
        ("SOURCE", "E1", "s1"),
        ("SOURCE", "E2", "s2"),
        ("SOURCE", "E3", "s3"),
        ("SOURCE", "E1", "s9"),
        ("SOURCE", "E0", "s0"),
    ]


async def test_strip_removes_only_the_named_entrys_source_refs(db_session: Any) -> None:
    premise_id = "E-shared-id"
    p = _particle(
        [
            _src("E1", "s1"),
            _src("DEL", "d1", location="chunk-a"),
            _src("E2", "s2"),
            _premise(premise_id),
        ]
    )
    await insert_particle(db_session, p)
    # A second ref to one entry arrives by append, as in production (the edge
    # index holds one row per entry, so insert cannot carry two).
    for ref in (
        _src("DEL", "d2", location="chunk-b"),
        _premise("DEL"),  # a PARTICLE ref whose id equals the entry id
        _src("E3", "s3"),
    ):
        await append_provenance_ref(db_session, p.id, ref)
    await db_session.flush()
    before = await _stored_refs(db_session, p.id)

    removed = await strip_entry_provenance_refs(db_session, p.id, "DEL")

    after = await _stored_refs(db_session, p.id)
    assert removed == 2
    # Exactly the survivors, as the same dicts, in their original order.
    assert after == [
        r for r in before if not (r["type"] == "SOURCE" and r["corpus_entry_id"] == "DEL")
    ]
    assert _keys(after) == [
        ("SOURCE", "E1", "s1"),
        ("SOURCE", "E2", "s2"),
        ("PARTICLE", premise_id, None),
        ("PARTICLE", "DEL", None),
        ("SOURCE", "E3", "s3"),
    ]
