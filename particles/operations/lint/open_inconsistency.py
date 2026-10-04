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
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.conflict_review import conflict_pair_ids
from particles.core.contradiction_disclosure import census_sides
from particles.core.schema import LintFinding, Particle
from particles.core.status import Status
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
    text = f"“{p.content}”"
    if p.status is not Status.ACTIVE:
        reason = f" / {p.status_reason.value}" if p.status_reason is not None else ""
        text += f" [{p.status.value}{reason}]"
    return text


def _side_text(ids: list[str], members: dict[str, Particle]) -> str:
    if not ids:
        return "(no claim recorded)"
    return " + ".join(_quote(pid, members) for pid in ids)


def build_finding(record: Particle, members: dict[str, Particle]) -> LintFinding:
    """The finding for one open record, given its loaded members (pure)."""
    a, b = record_sides(record)
    return LintFinding(
        particle_id=record.id,
        finding_type=FINDING_TYPE,
        severity="WARNING",
        detail=f"Open conflict: {_side_text(a, members)} vs {_side_text(b, members)}",
        recommended_action=(
            f"Resolve it: particles curate apply resolve inconsistency:{record.id} "
            f"--action PREFER_A|PREFER_B|BOTH_VALID|DISCARD (or particles review {record.id})"
        ),
        particle_content=record.content,
        inconsistency_id=record.id,
        conflict_sides=[a, b],
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
    member_ids = sorted({pid for r in open_records for side in record_sides(r) for pid in side})
    members = await get_particles_by_ids(session, member_ids)
    return [build_finding(r, members) for r in sorted(open_records, key=lambda r: r.id)]
