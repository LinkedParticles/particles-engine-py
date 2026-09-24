# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-key reviewer-derived trust statements onto corpus entries.

Until 1.141.14 a ``PREFER_A`` / ``PREFER_B`` review wrote its
``REVIEWER_DERIVED`` ``SourceTrustStatement`` with ``source_ref_type =
'CORPUS_ENTRY'`` and ``source_ref_value = <the preferred *particle's* id>``.
Every consumer of that column — the §6.4 layered cascade the ingest ladder and
the query path consult, and the reviewer-confirmation gate —
matches it against a *corpus-entry* id, so each such statement was inert: it
could never raise the preferred source's rank, and two reviews preferring the
same source could never count as two confirmations.

**The rewrite.** For every ``REVIEWER_DERIVED`` / ``CORPUS_ENTRY`` statement
whose value is the id of a row in ``particles``, resolve that particle's first
``SOURCE``-typed provenance ref and replace the value with its
``corpus_entry_id`` — the key the review would have written had it been
correct. A statement whose particle has no SOURCE ref (agent-asserted, derived,
or already purged) is left exactly as found: it stays inert, which is what it
always was, and deleting it would erase the record that a review happened. The
original particle id is preserved in ``basis`` so the rewrite is inspectable
and reversible.

``downgrade()`` restores the particle-id key from that ``basis`` note — a
faithful inverse, not a no-op, so a rollback reproduces the pre-037 table.

``SCHEMA_VERSION`` is unchanged: the statement table is Extension B storage,
never serialized to the schema artifacts or interchange.

Revision ID: 037
Revises: 036
"""

from __future__ import annotations

import json
import re

import sqlalchemy as sa

from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None

_REKEY_NOTE = " [migration 037: re-keyed from particle {particle_id}]"
_REKEY_RE = re.compile(r" \[migration 037: re-keyed from particle ([^\]]+)\]$")


def rekey_reviewer_statements(bind: sa.engine.Connection) -> int:
    """Re-key mis-keyed reviewer statements onto their particle's source entry.

    Takes the connection rather than calling ``op.get_bind()`` so the rewrite is
    exercisable outside an Alembic context — the unit suite runs ``create_all``
    on an in-memory DB and never runs a migration, and this one edits operator
    data. See ``tests/test_migration_037_reviewer_statement_key.py``.

    Returns the number of statements rewritten.
    """
    statements = bind.execute(
        sa.text(
            "SELECT statement_id, source_ref_value, basis FROM trust_statements "
            "WHERE policy_provenance = 'REVIEWER_DERIVED' "
            "AND source_ref_type = 'CORPUS_ENTRY'"
        )
    ).fetchall()
    rewritten = 0
    for statement_id, value, basis in statements:
        particle = bind.execute(
            sa.text("SELECT provenance_json FROM particles WHERE id = :id"),
            {"id": value},
        ).fetchone()
        if particle is None:
            # Not a particle id: either already a corpus-entry key, or the
            # particle is gone. Either way there is nothing to resolve.
            continue
        try:
            refs = json.loads(particle[0] or "[]")
        except (TypeError, ValueError):
            continue
        entry_id = next(
            (
                ref.get("corpus_entry_id")
                for ref in refs
                if isinstance(ref, dict) and ref.get("type") == "SOURCE"
            ),
            None,
        )
        if not entry_id:
            continue
        bind.execute(
            sa.text(
                "UPDATE trust_statements SET source_ref_value = :entry, basis = :basis "
                "WHERE statement_id = :sid"
            ),
            {
                "entry": entry_id,
                "basis": (basis or "") + _REKEY_NOTE.format(particle_id=value),
                "sid": statement_id,
            },
        )
        rewritten += 1
    return rewritten


def restore_particle_keys(bind: sa.engine.Connection) -> int:
    """Inverse of :func:`rekey_reviewer_statements`, driven by the basis note."""
    statements = bind.execute(
        sa.text(
            "SELECT statement_id, basis FROM trust_statements "
            "WHERE policy_provenance = 'REVIEWER_DERIVED' "
            "AND source_ref_type = 'CORPUS_ENTRY' AND basis LIKE '%[migration 037:%'"
        )
    ).fetchall()
    restored = 0
    for statement_id, basis in statements:
        match = _REKEY_RE.search(basis or "")
        if match is None:
            continue
        bind.execute(
            sa.text(
                "UPDATE trust_statements SET source_ref_value = :pid, basis = :basis "
                "WHERE statement_id = :sid"
            ),
            {
                "pid": match.group(1),
                "basis": _REKEY_RE.sub("", basis or ""),
                "sid": statement_id,
            },
        )
        restored += 1
    return restored


def upgrade() -> None:
    rekey_reviewer_statements(op.get_bind())


def downgrade() -> None:
    restore_particle_keys(op.get_bind())
