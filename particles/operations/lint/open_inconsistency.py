# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""One ``OPEN_INCONSISTENCY`` finding per open INCONSISTENCY record.

An open record is a question the store has asked and not yet answered, so it
is reported as a ``WARNING`` whatever state its members are in: both ACTIVE,
one quarantined, both demoted, or one no longer in the store. The finding is
the one source of the ``INCONSISTENCY`` curation card, keyed by the record.

The records come from the INCONSISTENCY status scan the contested finder
already runs; the lint pass hands them over rather than scanning twice.

A record the release pass examined and found still blocked
gains a line: which claim retired its stored side, and which ACTIVE claims
still contradict the held one. It is read from the record's latest
``HELD_CLAIM_EXAMINED`` event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.conflict_review import (
    conflict_pair_ids,
    is_retired_value,
    retirement_kind,
)
from particles.core.contradiction_disclosure import census_sides
from particles.core.schema import LintFinding, Particle
from particles.core.status import Status
from particles.store.event_store import OperatorEventType, list_particle_events
from particles.store.particle_store import get_inconsistency_particles, get_particles_by_ids

FINDING_TYPE = "OPEN_INCONSISTENCY"


def record_sides(record: Particle) -> tuple[list[str], list[str]]:
    """The record's members by side, ``(a, b)``, in the order ``review`` names them.

    A census record carries its sides explicitly and may name several claims
    per side. Every other record names claim A and claim B as
    its first two PARTICLE provenance refs; a missing ref yields an empty side.
    """
    sides = census_sides(record)
    if sides is not None:
        return list(sides.a), list(sides.b)
    a_id, b_id = conflict_pair_ids(record)
    return ([a_id] if a_id else []), ([b_id] if b_id else [])


def _quote(pid: str, members: dict[str, Particle]) -> str:
    p = members.get(pid)
    if p is None:
        return "(no longer in the store)"
    return f"“{p.content}”{_state(p)}"


def _state(p: Particle) -> str:
    """`` [STATUS / REASON]`` for a claim that is not ACTIVE, else empty."""
    if p.status is Status.ACTIVE:
        return ""
    reason = f" / {p.status_reason.value}" if p.status_reason is not None else ""
    return f" [{p.status.value}{reason}]"


def _side_text(ids: list[str], members: dict[str, Particle]) -> str:
    if not ids:
        return "(no claim recorded)"
    return " + ".join(_quote(pid, members) for pid in ids)


def _examination_ids(examination: Mapping[str, Any] | None) -> list[str]:
    """The claims an examination names beyond the record's sides: retirer and blockers."""
    if not examination:
        return []
    ids = [examination.get("retired_by"), *(examination.get("blocking_ids") or [])]
    return [pid for pid in ids if isinstance(pid, str) and pid]


def examination_line(examination: Mapping[str, Any], members: Mapping[str, Particle]) -> str:
    """The card's line for a record the release pass found still blocked (pure)."""
    reason = str(examination.get("partner_reason") or "")
    how = {
        "SUPERSEDED_BY_UPDATE": "by update",
        "SUPERSEDED_BY_REANCHOR": "by re-anchoring",
        "DOCUMENT_SUPERSEDED": "by the document sweep",
    }.get(reason, "by an autonomous pass")
    winner = examination.get("retired_by")
    retired = f"Claim A was retired {how}"
    if isinstance(winner, str) and winner:
        retired += f" in favour of {_quote(winner, dict(members))}"
    blocking = [str(pid) for pid in examination.get("blocking_ids") or [] if isinstance(pid, str)]
    still = " + ".join(_quote(pid, dict(members)) for pid in blocking) or "(none recorded)"
    return f"{retired}; the held claim B is still contradicted by {still}."


#: How a retired-value record's claim A was retired, as the card's title and
#: detail line name it. ``None`` is a claim A the store no longer
#: holds, or one in a state a hold does not expect.
RETIREMENT_TITLES: dict[str | None, str] = {
    "retracted": "Restated after retraction",
    "superseded": "Restated after replacement",
    "reviewed": "Restated after review",
    None: "Restated after retirement",
}

#: What happened to a retired-value record's claim A, as a sentence's verb.
RETIRED_VERBS: dict[str | None, str] = {
    "retracted": "was retracted",
    "superseded": "was replaced",
    "reviewed": "lost a conflict review",
    None: "was retired",
}

#: The ruling ``PREFER_A`` keeps on a retired-value record, by retirement.
RETIREMENT_RULINGS: dict[str | None, str] = {
    "retracted": "the retraction",
    "superseded": "the replacement",
    "reviewed": "that ruling",
    None: "the earlier ruling",
}


def _retired_value_detail(a: list[str], b: list[str], members: dict[str, Particle]) -> str:
    """The detail line of a record that holds a restated retired claim (pure).

    Claim A is the retired claim and claim B the restatement held behind it.
    The two usually read the same, so B is quoted only when its text differs.
    """
    a_id = a[0] if a else None
    b_id = b[0] if b else None
    claim_a = members.get(a_id) if a_id else None
    claim_b = members.get(b_id) if b_id else None
    kind = retirement_kind(claim_a)
    head = f"{RETIREMENT_TITLES[kind]}: {_side_text(a, members)} {RETIRED_VERBS[kind]}"
    if claim_b is None:
        return f"{head}, and its restatement is no longer in the store"
    if claim_a is not None and claim_a.content == claim_b.content:
        again = f"a later source states it again{_state(claim_b)}"
    else:
        again = f"a later source restates it as {_quote(claim_b.id, members)}"
    return f"{head}, and {again}, held for review"


def build_finding(
    record: Particle,
    members: dict[str, Particle],
    examination: Mapping[str, Any] | None = None,
) -> LintFinding:
    """The finding for one open record, given its loaded members (pure).

    ``examination`` is the record's latest ``HELD_CLAIM_EXAMINED`` payload, when
    the release pass has examined it; its line is appended to the detail. A
    retired-value record is not a conflict between sources, so its
    detail and recommended action say what it holds instead.
    """
    a, b = record_sides(record)
    retired = is_retired_value(record)
    if retired:
        detail = _retired_value_detail(a, b, members)
    else:
        detail = f"Open conflict: {_side_text(a, members)} vs {_side_text(b, members)}"
    note = examination_line(examination, members) if examination else ""
    if note:
        detail += ". " + note
    apply = f"particles curate apply resolve inconsistency:{record.id}"
    if retired:
        kind = retirement_kind(members.get(a[0]) if a else None)
        action = (
            f"Keep {RETIREMENT_RULINGS[kind]}: {apply} --action PREFER_A. Restore the "
            f"claim: {apply} --action PREFER_B (or particles review {record.id})"
        )
    else:
        action = (
            f"Resolve it: {apply} "
            f"--action PREFER_A|PREFER_B|BOTH_VALID|DISCARD (or particles review {record.id})"
        )
    return LintFinding(
        particle_id=record.id,
        finding_type=FINDING_TYPE,
        severity="WARNING",
        detail=detail,
        recommended_action=action,
        particle_content=record.content,
        inconsistency_id=record.id,
        conflict_sides=[a, b],
        conflict_note=note,
    )


async def open_inconsistency_findings(
    session: AsyncSession, records: Sequence[Particle] | None = None
) -> list[LintFinding]:
    """One finding per open INCONSISTENCY record, sorted by record id.

    ``records`` is the status scan the caller already holds; ``None`` runs it
    here. Records not at status INCONSISTENCY are skipped, so a caller may pass
    any list. Members are loaded in one batch.
    """
    if records is None:
        records = await get_inconsistency_particles(session)
    open_records = [r for r in records if r.status is Status.INCONSISTENCY]
    if not open_records:
        return []
    examinations = await latest_held_examinations(session)
    member_ids = sorted(
        {pid for r in open_records for side in record_sides(r) for pid in side}
        | {pid for r in open_records for pid in _examination_ids(examinations.get(r.id))}
    )
    members = await get_particles_by_ids(session, member_ids)
    return [
        build_finding(r, members, examinations.get(r.id))
        for r in sorted(open_records, key=lambda r: r.id)
    ]


async def latest_held_examinations(session: AsyncSession) -> dict[str, dict[str, Any]]:
    """Each record's latest ``HELD_CLAIM_EXAMINED`` payload, by record id.

    The release pass's memory, which decides whether a record is examined
    again, and the source of the card's line.
    """
    latest: dict[str, dict[str, Any]] = {}
    for pid, _at, payload in await list_particle_events(
        session, OperatorEventType.HELD_CLAIM_EXAMINED
    ):
        if isinstance(payload, dict) and payload.get("record_id") == pid:
            latest[pid] = payload  # oldest-first, so the last one wins
    return latest
