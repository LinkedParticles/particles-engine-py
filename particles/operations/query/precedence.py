"""Document-precedence tie-break among detected conflicts.

*"The later authored decision wins."* When two **ACTIVE** particles a
contradiction probe has flagged as mutually conflicting share **no authored
`supersedes:` edge** (an authored edge already demotes its loser off ACTIVE at
reconcile time, so it never reaches here), an open query retrieves
both incompatible "current" answers with no signal to prefer the later one.
This module supplies that signal as a **rank-time tie-break**: the conflict's
recency-loser is demoted in ordering only — its reported
``effective_confidence`` and stored ``confidence.value`` are untouched
, and no status changes (``spec_impact: implementation``).

The tie-break is **conflict-gated** (it fires only between particles the
L-SEM-01 detector confirmed in conflict — never a global reorder)
and
**default-safe** (inert outside a detected conflict, and inert when neither side
exposes a comparable precedence key). It composes multiplicatively in
``_rank_score`` with the NARRATIVE demotion and the code-symbol demotion.

Two halves:

* :func:`build_precedence_keys` — the runtime key builder. For each candidate
  particle it reads the **provenance document's authored precedence key**: the
  ADR ``date`` + id ordinal via the genre-adapter seam (the id alone is
  a total monotonic order among ADRs — a higher id is the later decision),
  falling back to the snapshot's ``content_published_at``. Returns ``None`` for
  a particle with no comparable key (default-safe).
* :func:`precedence_demotions` — the **pure** tie-break. Given the candidate
  set, the detected-conflict pairs, and the per-particle precedence keys, it
  returns the set of particle ids to demote (the recency-loser of each
  conflict). Pure and deterministic, so it is unit-tested with the conflict set
  and keys injected directly (the LLM probe itself is out of scope per
  tests/AGENTS.md).
"""

from __future__ import annotations

from datetime import UTC, datetime

from particles.core.schema import Particle, ProvenanceRefType

#: A document-level precedence ordinal: ``(authored_datetime, adr_id_ordinal)``.
#: Compared lexicographically, **later wins**. Either component may be absent
#: (``datetime.min`` / ``-1`` sentinels) — a particle whose *whole* key is
#: absent is excluded from the tie-break by :func:`build_precedence_keys`.
PrecedenceKey = tuple[datetime, int]

_MIN_DT = datetime.min.replace(tzinfo=UTC)


def _adr_id_ordinal(supersession_json: str | None) -> int | None:
    """Recover the ADR id ordinal from a stored ``document_supersession_json``.

    The cap. 2 genre adapter stamps ``{"key": "adr:0166", …}`` on every
    ADR corpus entry; the integer after ``adr:`` is the monotonic decision
    ordinal (a higher id is the later decision). Returns ``None`` for any entry
    that is not a recognised ADR genre or whose key is unparseable.
    """
    if not supersession_json:
        return None
    from particles.corpus.supersession import DocumentRelation

    rel = DocumentRelation.from_json(supersession_json)
    if rel is None:
        return None
    prefix = "adr:"
    if not rel.key.startswith(prefix):
        return None
    token = rel.key[len(prefix) :].strip()
    try:
        return int(token)
    except ValueError:
        return None


def _source_entry_id(p: Particle) -> str | None:
    """corpus_entry_id of the particle's first SOURCE provenance ref, or None."""
    for ref in p.provenance:
        if ref.type == ProvenanceRefType.SOURCE and ref.corpus_entry_id:
            return ref.corpus_entry_id
    return None


def build_precedence_keys(
    particles: list[Particle],
    *,
    pub_at_by_id: dict[str, datetime | None],
    supersession_by_entry: dict[str, str | None],
) -> dict[str, PrecedenceKey]:
    """Map each particle id to its document precedence key.

    The key is ``(authored_datetime, adr_id_ordinal)``, compared **later-wins**.
    The ADR genre supplies the id ordinal via the stamped supersession key; the
    general fallback supplies the ``content_published_at`` datetime. A particle
    is **omitted** from the result (no comparable key) when *both* components are
    absent — the default-safe "do nothing rather than guess" rule. Naive
    datetimes are coerced to UTC so the keys compare cleanly.
    """
    keys: dict[str, PrecedenceKey] = {}
    for p in particles:
        entry_id = _source_entry_id(p)
        ordinal = (
            _adr_id_ordinal(supersession_by_entry.get(entry_id)) if entry_id is not None else None
        )
        pub_at = pub_at_by_id.get(p.id)
        if pub_at is None and ordinal is None:
            continue  # no comparable precedence key — inert for this particle
        dt = pub_at if pub_at is not None else _MIN_DT
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        keys[p.id] = (dt, ordinal if ordinal is not None else -1)
    return keys


def precedence_demotions(
    particles: list[Particle],
    conflict_pairs: set[frozenset[str]],
    precedence_keys: dict[str, PrecedenceKey],
) -> set[str]:
    """Return the particle ids to demote — the recency-loser of each conflict.

    Pure and deterministic. ``conflict_pairs`` is the set of detected-conflict
    id pairs (each a 2-element ``frozenset``); only pairs where **both** sides
    expose a comparable :data:`PrecedenceKey` are broken. Within such a pair the
    particle with the *strictly smaller* key (the earlier authored decision) is
    demoted; an exact tie (equal keys) is left alone (no defensible winner). A
    particle is demoted if it loses **any** conflict it participates in.
    """
    present_ids = {p.id for p in particles}
    demote: set[str] = set()
    for pair in conflict_pairs:
        if len(pair) != 2:
            continue
        a, b = tuple(pair)
        if a not in present_ids or b not in present_ids:
            continue
        ka = precedence_keys.get(a)
        kb = precedence_keys.get(b)
        if ka is None or kb is None or ka == kb:
            continue  # no comparable key on one side, or a true tie — inert
        demote.add(a if ka < kb else b)
    return demote
