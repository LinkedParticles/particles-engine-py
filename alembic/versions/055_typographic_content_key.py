# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-key stored content hashes for the typographic fold.

``particles.core.duplicate_key.normalize_content`` now folds typographic
variants: double quotation marks are dropped, curly apostrophes fold to
``'``, dashes and the minus sign to ``-``, the ellipsis to ``...``,
invisible zero-width marks are removed, and the text is NFC-composed. Two
stored values are keyed on that function, and both are re-keyed here so a
lookup computed with the new fold finds what an older run wrote:

1. **``particles.content_norm_hash``**, the indexed probe behind extract-time
   suppression and the retired-value index. A row left on
   the old key would be invisible to both: a claim already held with a curly
   quote would be minted again, and a retired value restated with one would
   escape quarantine. Every row whose key changes is rewritten (on the owner's
   store, measured 2026-10-22: 4,691 of 73,103 rows, almost all for a straight
   double quote, an em or en dash, or an ellipsis).
2. **``probe_verdicts.hash_a`` / ``hash_b``**, the verdict ledger,
   keyed on ``content_hash`` of each claim. Without
   the re-key a recorded verdict on a pair with a quoted claim would be missed
   once and paid for again, and a shadow rung's row would no longer join its
   label. The old-to-new mapping is built from the stored claims, so only
   hashes of claims the store still holds are rewritten; the keys of the three
   symmetric kinds (and of the census's rung rows, the only rung rows written
   as of this revision, which use the symmetric contradiction key) are
   re-sorted after the rewrite. A subject-link row is keyed on the hash of a
   name and a claim together, which no stored value reproduces: it is left
   alone, and a link judged for a claim the fold changed is asked once more.

The hash is **derived, never authoritative**: ``content`` remains the source
of truth and ``ParticleRow.from_model`` writes the current key on every write.

**Downgrade** restores ``content_norm_hash`` under the old function. The ledger
is not reverse-mapped, because two old keys may have folded into one; a
downgraded install misses those verdicts once and re-records them.

Both functions are vendored so this migration stays a frozen artifact, as in
migration 031.

Revision ID: 055
Revises: 054
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable

import sqlalchemy as sa

from alembic import op

revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None

_TRAILING_PUNCT = ".,;:!?\"'"
_DROPPED_QUOTES = '"\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a\u300c\u300d\u300e\u300f\uff02'
_SINGLE_QUOTES = "\u2018\u2019\u201a\u201b"
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_INVISIBLE = "\u200b\u2060\ufeff\u00ad"
_FOLD = str.maketrans(
    {
        **dict.fromkeys(_DROPPED_QUOTES + _INVISIBLE),
        **dict.fromkeys(_SINGLE_QUOTES, "'"),
        **dict.fromkeys(_DASHES, "-"),
        "\u2026": "...",
    }
)

#: The ledger kinds whose key is the sorted pair (``ProbeKind.symmetric``), plus
#: the census's rung rows, which record under the contradiction key.
_SYMMETRIC_KINDS = frozenset({"contradiction", "second_reading", "reconcile_contradiction"})
_RUNG_PREFIX = "rung:"
_CHUNK = 500


def _hash_v1(content: str) -> str:
    normalized = " ".join(content.split()).rstrip(_TRAILING_PUNCT).rstrip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_v2(content: str) -> str:
    folded = unicodedata.normalize("NFC", content).translate(_FOLD)
    normalized = " ".join(folded.split()).rstrip(_TRAILING_PUNCT).rstrip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _rekey_particles(
    bind: sa.engine.Connection, key: Callable[[str], str]
) -> tuple[int, dict[str, str]]:
    """Rewrite every row whose stored key differs from ``key``.

    Returns how many rows were rewritten and the old-to-new key mapping.
    """
    rows = bind.execute(sa.text("SELECT id, content, content_norm_hash FROM particles")).fetchall()
    remap: dict[str, str] = {}
    updates: list[dict[str, str]] = []
    for pid, content, stored in rows:
        new = key(content or "")
        if stored != new:
            updates.append({"pid": pid, "key": new})
            if stored is not None:
                remap[stored] = new
    for start in range(0, len(updates), _CHUNK):
        bind.execute(
            sa.text("UPDATE particles SET content_norm_hash = :key WHERE id = :pid"),
            updates[start : start + _CHUNK],
        )
    return len(updates), remap


def _rekey_ledger(bind: sa.engine.Connection, remap: dict[str, str]) -> int:
    if not remap:
        return 0
    rows = bind.execute(sa.text("SELECT id, probe_kind, hash_a, hash_b FROM probe_verdicts"))
    updates: list[dict[str, object]] = []
    for row_id, kind, hash_a, hash_b in rows.fetchall():
        if hash_a not in remap and hash_b not in remap:
            continue
        a, b = remap.get(hash_a, hash_a), remap.get(hash_b, hash_b)
        if (kind in _SYMMETRIC_KINDS or kind.startswith(_RUNG_PREFIX)) and b < a:
            a, b = b, a
        updates.append({"rid": row_id, "a": a, "b": b})
    for start in range(0, len(updates), _CHUNK):
        bind.execute(
            sa.text("UPDATE probe_verdicts SET hash_a = :a, hash_b = :b WHERE id = :rid"),
            updates[start : start + _CHUNK],
        )
    return len(updates)


def rekey(bind: sa.engine.Connection) -> tuple[int, int]:
    """Re-key both tables under the typographic fold; return ``(particles, verdicts)``.

    Takes the connection rather than calling ``op.get_bind()`` so it is
    exercisable outside an Alembic context, as in migration 053. See
    ``tests/test_migration_055_typographic_content_key.py``. Idempotent: a
    second run finds every row already on the new key and rewrites nothing.
    """
    rewritten, remap = _rekey_particles(bind, _hash_v2)
    return rewritten, _rekey_ledger(bind, remap)


def upgrade() -> None:
    rekey(op.get_bind())


def downgrade() -> None:
    _rekey_particles(op.get_bind(), _hash_v1)
