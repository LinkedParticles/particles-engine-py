"""Query-time stance (agreement) distribution assembly.

For each result particle, gather the ``ENDORSES`` / ``DISPUTES`` edges pointing
into its CO_EVIDENTIAL group and render them as a per-claim agreement
distribution: the holders of each position, attributed and cited, each with the
stance particle's own effective confidence and optional magnitude. Computed at
query time, never stored (substrate-plus-lens).

The aggregate is surfaced *alongside* the target claim's confidence and MUST NOT
feed any term of ``effective_confidence`` (the §4 MUST). The holder grouping is
the raw, unverified ``stance:holder`` key (§3 / M6): a count of keys,
not of verified agents — see :data:`AGREEMENT_CAVEAT`.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, RelationType, StancePosition
from particles.core.stance import stance_holder, stance_magnitude
from particles.core.status import Status
from particles.store.particle_store import get_particle
from particles.store.relation_store import get_co_evidential_group, get_incoming

from .effective_confidence import score_effective_confidence
from .source_trust import TrustPolicy

#: Surfaced whenever a non-empty distribution is returned (M6).
AGREEMENT_CAVEAT = (
    "Holders are grouped by raw stance:holder key (unverified): two keys may be "
    "one agent and a key's owner is not verified. This is a count "
    "of keys, not of verified agents."
)


async def _stance_ids_by_kind(session: AsyncSession, target: Particle) -> dict[str, RelationType]:
    """Map each incident stance particle id → its kind, over the target's group.

    Collapses the target over its CO_EVIDENTIAL group and gathers the
    ``ENDORSES`` / ``DISPUTES`` edges into any group member. A stance incident to
    several group members is counted once.
    """
    group = await get_co_evidential_group(session, target.id)
    by_kind: dict[str, RelationType] = {}
    for member in group:
        for sid in await get_incoming(session, member, RelationType.ENDORSES):
            by_kind[sid] = RelationType.ENDORSES
        for sid in await get_incoming(session, member, RelationType.DISPUTES):
            by_kind[sid] = RelationType.DISPUTES
    return by_kind


async def compute_stance_distribution(
    session: AsyncSession,
    targets: list[Particle],
    trust_policy: TrustPolicy,
) -> tuple[list[list[StancePosition]], bool]:
    """Return the per-target agreement distributions and whether any is non-empty.

    ``distributions[i]`` is the list of :class:`StancePosition` for ``targets[i]``
    — the ACTIVE stances endorsing / disputing that claim (or a co-evidential
    twin), each with the stance particle's own query-time effective confidence
    and optional magnitude. A holder who both endorses and disputes the group
    appears in *both* positions
    (within-group same-holder split at M1) — positions are
    never netted. The boolean is True when at least one distribution is
    non-empty, so the caller can attach the M6 caveat.
    """
    per_target: list[dict[str, RelationType]] = [
        await _stance_ids_by_kind(session, t) for t in targets
    ]
    all_ids = {sid for m in per_target for sid in m}
    if not all_ids:
        return [[] for _ in targets], False

    # Load the stance particles; keep only ACTIVE ones — a dangling edge to a
    # retracted/absent stance contributes no position.
    stance_particles: dict[str, Particle] = {}
    for sid in all_ids:
        p = await get_particle(session, sid)
        if p is not None and p.status == Status.ACTIVE:
            stance_particles[sid] = p

    eff_conf_by_id = await score_effective_confidence(
        session, list(stance_particles.values()), trust_policy
    )

    distributions: list[list[StancePosition]] = []
    any_positions = False
    for ids_by_kind in per_target:
        positions: list[StancePosition] = []
        for sid, kind in ids_by_kind.items():
            p = stance_particles.get(sid)
            if p is None:
                continue
            holder = stance_holder(p)
            if holder is None:
                continue  # an edge without a holder marker has nothing to attribute
            positions.append(
                StancePosition(
                    kind=kind,
                    holder=holder,
                    stance_particle_id=sid,
                    effective_confidence=eff_conf_by_id.get(sid, p.confidence.value),
                    magnitude=stance_magnitude(p),
                )
            )
        positions.sort(key=lambda sp: (sp.kind.value, sp.holder, sp.stance_particle_id))
        any_positions = any_positions or bool(positions)
        distributions.append(positions)
    return distributions, any_positions
