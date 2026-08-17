"""Query-time per-claim contestedness — lens-divergence of effective confidence.

Contestedness is the spread (max − min) of a claim's ``effective_confidence``
evaluated separately under each policy in the viewer's policy set — the store's
**local** policy plus **each adopted lens, standalone** (§2). Each member is a
complete standalone snapshot with resolve-or-None neutrality; each is
applied *in full* — source trust rank, URL rules, *and* extractor-weight
overrides (§1). The metric is computed at read time from existing
machinery, surfaced as disclosure, and never fed into
confidence or ranking (the
§5 invariants). With fewer than two policies the metric is **absent** (§3) —
absence of measurement is not measured invariance.

Why range, not variance (§1): the policy set is a complete, small population
whose extremes are *nameable policies* — the rendering attributes them
("local: 0.43; acme-numismatics: 0.81") — and the founding thesis is a sup-norm
claim ("invariant across every credible trust policy"), not an average-deviation
one; variance would dilute one dissenting community among many agreeing ones.

This deliberately differs from ranking. Ranking uses the single composed
policy (local-wins, min-across-lenses); contestedness re-evaluates the same
candidate under each member **separately**, because composition would collapse
the very spread being measured (a local row would mask every lens's view of that
key). See ``source_trust.load_trust_policy`` for the composed (ranking) policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ContestednessReading, Particle, PolicyRendering
from particles.core.scoring.confidence import (
    compute_effective_confidence,
    merge_co_evidential_confidence,
)
from particles.core.scoring.decay import recency_factor
from particles.core.status import Status
from particles.store.extractor_store import get_all_records
from particles.store.lens_store import get_adopted_lenses
from particles.store.particle_store import get_particle
from particles.store.relation_store import get_co_evidential_group

from .rank import _first_source_key
from .source_info import SourceRow, load_source_rows
from .source_trust import TrustPolicy, lens_to_trust_policy, load_local_trust_policy

#: Neutral source row for a particle whose provenance resolved to nothing —
#: matches ``main._gather_scored``'s ``(None, "", None, None, None)`` fallback.
_EMPTY_ROW: SourceRow = (None, "", None, None, None)


@dataclass(frozen=True)
class MemberPolicy:
    """One nameable member of the viewer's contestedness policy set.

    ``name`` is the attribution label — ``"local"`` for the store's own policy or
    an adopted lens's name. ``trust`` is the member's standalone
    ``TrustPolicy``; ``extractor_weights`` is the member's extractor-weight
    overrides (absence ⇒ neutral 1.0). The recency factor is policy-invariant
    today (§6) so it is not carried here.
    """

    name: str
    trust: TrustPolicy
    extractor_weights: dict[str, float]

    def effective_confidence(self, particle: Particle, source_row: SourceRow) -> float:
        """Effective confidence of one particle under this member alone.

        Mirrors ``main._gather_scored`` term-for-term, but with this member's
        extractor weight and trust policy instead of the composed ranking policy.
        """
        extractor_id = (
            particle.extractor_ref.name if particle.extractor_ref else "general-extractor"
        )
        weight = self.extractor_weights.get(extractor_id, 1.0)
        pub_at, source_type, entry_id, uri_r, author_id = source_row
        rank = self.trust.evaluate(entry_id, source_type, uri_r, author_id)
        return compute_effective_confidence(
            particle.confidence.value,
            extractor_trust_weight=weight,
            source_trust_rank=1.0 if rank is None else rank,
            recency_factor=recency_factor(pub_at, source_type),
            calibration_source=particle.confidence.calibration_source,
        )


async def load_member_policies(session: AsyncSession) -> list[MemberPolicy]:
    """Snapshot the viewer's policy set: local standalone + each adopted lens.

    The local policy is **always** a member (even when empty — the neutral policy
    is the viewer's operative rendering); each adopted lens contributes one
    standalone member with no local overlay and no cross-lens min. The list has
    one member with no lenses adopted, so the caller's degeneracy check (§3 —
    fewer than two members ⇒ no metric) lights up at the first lens adoption.

    In federation this is loaded from the **viewer's** store, the same shape as
    an earlier model — the viewer's policy set applies to every store's candidates.
    """
    local = MemberPolicy(
        name="local",
        trust=await load_local_trust_policy(session),
        extractor_weights={r.extractor_id: r.trust_weight for r in await get_all_records(session)},
    )
    members = [local]
    for lens in await get_adopted_lenses(session):
        members.append(
            MemberPolicy(
                name=lens.name,
                trust=lens_to_trust_policy(lens),
                extractor_weights=dict(lens.extractor_weights),
            )
        )
    return members


def spread_for_group(
    members: list[MemberPolicy],
    group: list[tuple[Particle, str]],
    source_rows: dict[str, SourceRow],
) -> ContestednessReading | None:
    """Merge-then-spread over one co-evidential group under each member.

    Per the §1 rule: for each member policy, apply the §6.9 noisy-OR merge over
    the group first (each particle scored under that member, throttled by its
    ``source_key``), then take the spread of the per-member merged values. For a
    singleton group the merge is the single particle's value. ``group`` is a list
    of ``(particle, source_key)`` pairs; ``source_rows`` maps particle id → its
    provenance row.

    Returns ``None`` when there are fewer than two members (§3 degeneracy) or the
    group is empty — the caller renders absence, never 0.0. Renderings preserve
    member order (local first), so the extremes stay attributable.
    """
    if len(members) < 2 or not group:
        return None
    renderings: list[PolicyRendering] = []
    for m in members:
        entries = [
            (m.effective_confidence(p, source_rows.get(p.id, _EMPTY_ROW)), source_key)
            for p, source_key in group
        ]
        merged = merge_co_evidential_confidence(entries)
        renderings.append(PolicyRendering(policy=m.name, effective_confidence=merged))
    values = [r.effective_confidence for r in renderings]
    return ContestednessReading(spread=max(values) - min(values), renderings=renderings)


async def compute_contestedness(
    session: AsyncSession,
    targets: list[Particle],
    members: list[MemberPolicy],
) -> list[ContestednessReading]:
    """Per-target contestedness readings over the viewer's policy set.

    Returns ``[]`` when fewer than two members (§3 — absent, never 0.0).
    Otherwise one :class:`ContestednessReading` per target: the §6.9
    merge-then-spread (§1) over the target's CO_EVIDENTIAL group, evaluated under
    each member separately. The group is the threshold-gated co-evidential
    closure (``query.equivalence_threshold``, matching the §9.3 collapse), with
    only ACTIVE members contributing.

    Co-evidential twins are loaded through a shared cache so a group is fetched
    once; source rows are batch-loaded for every participating particle in one
    pass (the already-loaded-rows cost shape).
    """
    if len(members) < 2 or not targets:
        return []

    min_equivalence = get_config().query.equivalence_threshold
    cache: dict[str, Particle | None] = {t.id: t for t in targets}
    groups: list[list[Particle]] = []
    for t in targets:
        group_ids = await get_co_evidential_group(session, t.id, min_confidence=min_equivalence)
        group_particles: list[Particle] = []
        for pid in group_ids:
            if pid not in cache:
                cache[pid] = await get_particle(session, pid)
            p = cache[pid]
            if p is not None and p.status == Status.ACTIVE:
                group_particles.append(p)
        # The target is an ACTIVE query result, so it is always present; the
        # fallback only guards a degenerate empty closure.
        groups.append(group_particles or [t])

    all_particles = {p.id: p for g in groups for p in g}
    source_rows = await load_source_rows(session, list(all_particles.values()))

    readings: list[ContestednessReading] = []
    for g in groups:
        group_with_keys = [(p, _first_source_key(p)) for p in g]
        reading = spread_for_group(members, group_with_keys, source_rows)
        assert reading is not None  # len(members) >= 2 and group non-empty
        readings.append(reading)
    return readings
