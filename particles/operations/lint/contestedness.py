"""Store-wide composed-contested finder + the distribution.

One evaluation, two renderings. The badge composes three bases —
``stance`` (a live ``DISPUTES`` edge into the claim's co-evidential group),
``divergence`` (the lens spread at or above
``contestedness.callout_threshold``), and ``inconsistency`` (an open
INCONSISTENCY particle referencing the claim). Before the recall
surfaces composed all three while the hygiene surfaces each keyed on one, so
they disagreed about what "contested" meant. This module is the single finder
they now share: it emits

- one **per-claim** ``CONTESTED`` INFO finding per contested belief — the card source for ``CardKind.CONTESTED``, and through it the census bucket and the run record; and
- the **store-level** ``CONTESTEDNESS_DISTRIBUTION`` INFO finding
  — the divergence drill-down histogram, unchanged in shape.

Like every contestedness surface this is **disclosure, not discount**
: it changes no status and feeds no confidence or
ranking term. Severity is ``INFO`` for that reason — an unresolved
contradiction is a defect (``CONTRADICTION``, ERROR); a lens disagreement is
not.

Absence: with fewer than two policies the divergence basis is
*absent*, not a non-firing vote, so a single-policy store mints no fact-like
divergence badge and the histogram is not emitted at all.

Cost — the reason this pass can be exhaustive where the
contradiction probe must be capped: it makes **no LLM call**, and each basis is
inverted so the work scales with the contested substrate rather than the store.
The relation table is asked *once* for every ``DISPUTES`` edge and once for
every ``CO_EVIDENTIAL`` edge, co-evidential components are built in memory, and
the INCONSISTENCY backref map is one status scan. A claim outside a
co-evidential group is its own group, so its singleton spread *is* the §1
merge-then-spread — which is why the store-wide reading is exact here, unlike
the pre-0215 approximation this module used to document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import (
    ContestedBadge,
    ContestednessReading,
    LintFinding,
    Particle,
    ParticleRelation,
    RelationType,
)
from particles.core.stance import stance_holder
from particles.core.status import Status
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta
from particles.operations.query.contested import compose_badge
from particles.operations.query.contestedness import (
    MemberPolicy,
    load_member_policies,
    spread_for_group,
)
from particles.operations.query.rank import _first_source_key
from particles.operations.query.source_info import load_source_rows
from particles.store.particle_store import get_active_particles, get_inconsistency_backrefs
from particles.store.relation_store import get_all_relations

#: Spread histogram edges — half-open buckets [lo, hi); the last is closed at 1.0.
_BUCKET_EDGES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)

#: How many most-contested claims to name in the store-level report.
_TOP_N = 5

#: Per-basis drill-down verb, surfaced as the finding's recommended action so a
#: contested card is a door into the loop that can actually resolve it. A
#: divergence disagreement has no INCONSISTENCY for `review` to resolve.
_BASIS_VERB: dict[str, str] = {
    "inconsistency": "`particles review` to resolve the INCONSISTENCY",
    "divergence": "`particles query --contestedness` to inspect the lens readings",
    "stance": "`particles links list <id>` to inspect the declared positions",
}


def _bucket_label(lo: float, hi: float, last: bool) -> str:
    return f"[{lo:.2f},{hi:.2f}{']' if last else ')'}"


def _histogram(spreads: list[float]) -> dict[str, int]:
    counts = {
        _bucket_label(_BUCKET_EDGES[i], _BUCKET_EDGES[i + 1], i + 2 == len(_BUCKET_EDGES)): 0
        for i in range(len(_BUCKET_EDGES) - 1)
    }
    for s in spreads:
        for i in range(len(_BUCKET_EDGES) - 1):
            lo, hi = _BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]
            last = i + 2 == len(_BUCKET_EDGES)
            if (lo <= s < hi) or (last and s == hi):
                counts[_bucket_label(lo, hi, last)] += 1
                break
    return counts


def _components(edges: list[ParticleRelation], min_confidence: float) -> dict[str, frozenset[str]]:
    """Group ids into CO_EVIDENTIAL components from the whole edge list at once.

    The in-memory equivalent of running ``get_co_evidential_group`` per particle,
    honouring the same ``effective_equivalence >= min_confidence`` gate. Ids with
    no qualifying edge are simply absent — their component is the singleton the
    caller supplies.
    """
    from particles.core.equivalence import effective_equivalence

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if effective_equivalence(e.confidence) < min_confidence:
            continue
        a, b = find(e.particle_a), find(e.particle_b)
        if a != b:
            parent[a] = b

    groups: dict[str, set[str]] = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return {node: frozenset(members) for members in groups.values() for node in members}


@dataclass(frozen=True)
class ContestedCensus:
    """The store's composed-contested state, computed in one pass."""

    #: particle id → its badge, for every belief where a basis fired.
    badges: dict[str, ContestedBadge] = field(default_factory=dict)
    #: particle id → its divergence reading, when the basis is measurable.
    readings: dict[str, ContestednessReading] = field(default_factory=dict)
    #: Names of the member policies, in order (empty when divergence is absent).
    policy_names: list[str] = field(default_factory=list)
    #: Spreads of every claim the histogram evaluated (empty when absent).
    spreads: list[float] = field(default_factory=list)


async def compute_store_contested(session: AsyncSession) -> ContestedCensus:
    """Compose the badge for every ACTIVE belief in the store.

    Each basis is inverted so the pass costs a fixed handful of queries rather
    than a walk per belief:

    - ``inconsistency`` — one INCONSISTENCY status scan (``get_inconsistency_backrefs``).
    - ``stance`` — one ``DISPUTES`` edge query, then only those edges' targets are
      expanded across their co-evidential component. Cost is O(#DISPUTES).
    - ``divergence`` — absent (and free) below two policies; otherwise one
      ``CO_EVIDENTIAL`` edge query plus the batched source-row load, with the
      component map giving each claim its exact merge-then-spread group.

    When ``contestedness.badge_enabled`` is off the composition falls back to the
    inconsistency basis alone — promise that the switch restores
    the pre-badge behaviour exactly, honoured on the hygiene surfaces too. The
    divergence readings are still computed for the histogram, which
    predates the badge and is not gated by it.
    """
    cfg = get_config()
    badge_enabled = cfg.contestedness.badge_enabled

    actives = await get_active_particles(session)
    by_id: dict[str, Particle] = {p.id: p for p in actives}
    backrefs = await get_inconsistency_backrefs(session)

    # --- divergence: absent below two policies (§3), and then free ---
    members: list[MemberPolicy] = await load_member_policies(session)
    measurable = [
        p
        for p in actives
        if not is_excluded_document_meta(p.properties) and not is_non_asserted(p.properties)
    ]
    readings: dict[str, ContestednessReading] = {}
    policy_names: list[str] = []
    if len(members) >= 2 and measurable:
        policy_names = [m.name for m in members]
        coev = _components(
            await get_all_relations(session, RelationType.CO_EVIDENTIAL),
            cfg.query.equivalence_threshold,
        )
        groups = {
            p.id: [by_id[pid] for pid in sorted(coev.get(p.id, frozenset({p.id}))) if pid in by_id]
            or [p]
            for p in measurable
        }
        source_rows = await load_source_rows(
            session, list({m.id: m for g in groups.values() for m in g}.values())
        )
        for p in measurable:
            reading = spread_for_group(
                members, [(m, _first_source_key(m)) for m in groups[p.id]], source_rows
            )
            if reading is not None:
                readings[p.id] = reading

    # --- stance: ask the relation table once, expand only the disputed ---
    disputed: set[str] = set()
    if badge_enabled:
        edges = await get_all_relations(session, RelationType.DISPUTES)
        # A dangling or holder-less stance contributes no position.
        live = {
            e.particle_b
            for e in edges
            if (s := by_id.get(e.particle_a)) is not None
            and s.status == Status.ACTIVE
            and stance_holder(s) is not None
        }
        if live:
            # The dispute travels across the target's co-evidential group, the
            # same closure `_dispute_presence` walks (ungated, min_confidence=0).
            groups_all = _components(
                await get_all_relations(session, RelationType.CO_EVIDENTIAL), 0.0
            )
            for target in live:
                disputed |= set(groups_all.get(target, frozenset({target})))

    badges: dict[str, ContestedBadge] = {}
    for pid in by_id:
        badge = compose_badge(
            has_dispute=pid in disputed,
            reading=readings.get(pid) if badge_enabled else None,
            inconsistency_id=backrefs.get(pid),
        )
        if badge is not None:
            badges[pid] = badge

    return ContestedCensus(
        badges=badges,
        readings=readings,
        policy_names=policy_names,
        spreads=[r.spread for r in readings.values()],
    )


def _describe(badge: ContestedBadge, reading: ContestednessReading | None) -> str:
    """The finding's detail: the fired bases plus each basis's drill-down."""
    parts: list[str] = []
    if "stance" in badge.bases:
        parts.append("at least one declared DISPUTES position")
    if "divergence" in badge.bases and reading is not None:
        hi = max(reading.renderings, key=lambda r: r.effective_confidence)
        lo = min(reading.renderings, key=lambda r: r.effective_confidence)
        parts.append(
            f"lens spread {reading.spread:.2f} "
            f"({hi.policy} {hi.effective_confidence:.2f} vs {lo.policy} "
            f"{lo.effective_confidence:.2f})"
        )
    if "inconsistency" in badge.bases and badge.inconsistency_id is not None:
        parts.append(f"open INCONSISTENCY {badge.inconsistency_id[:8]}…")
    detail = f"Contested ({', '.join(badge.bases)}) — {'; '.join(parts)}"
    if badge.caveat:
        detail += f". {badge.caveat}"
    return detail


async def _check_contested(session: AsyncSession) -> list[LintFinding]:
    """Emit the per-claim CONTESTED findings and the store-level distribution.

    Both renderings come off one :func:`compute_store_contested` pass, so the
    claims the histogram counts above the threshold are exactly the claims whose
    badge fires the ``divergence`` basis — the pre-0215 mismatch (singleton
    spread in lint vs merge-then-spread in query, sharing one threshold) cannot
    recur by construction.
    """
    census = await compute_store_contested(session)
    findings: list[LintFinding] = [
        LintFinding(
            particle_id=pid,
            finding_type="CONTESTED",
            severity="INFO",
            detail=_describe(badge, census.readings.get(pid)),
            recommended_action="Contested is disclosure, not a quality defect; "
            + "; ".join(_BASIS_VERB[b] for b in badge.bases),
            contested_bases=list(badge.bases),
            inconsistency_id=badge.inconsistency_id,
        )
        for pid, badge in sorted(census.badges.items())
    ]
    findings += _report_distribution(census)
    return findings


def _report_distribution(census: ContestedCensus) -> list[LintFinding]:
    """The store-level histogram, rendered from the shared census."""
    if not census.spreads:
        return []
    threshold = get_config().contestedness.callout_threshold
    contested = sorted(
        (
            (
                r.spread,
                pid,
                max(r.renderings, key=lambda x: x.effective_confidence),
                min(r.renderings, key=lambda x: x.effective_confidence),
            )
            for pid, r in census.readings.items()
            if r.spread >= threshold
        ),
        key=lambda t: -t[0],
    )
    top_str = "; ".join(
        f"{pid[:8]}… {sp:.2f} ({hi.policy}:{hi.effective_confidence:.2f} vs "
        f"{lo.policy}:{lo.effective_confidence:.2f})"
        for sp, pid, hi, lo in contested[:_TOP_N]
    )
    detail = (
        f"Contestedness across {len(census.policy_names)} policies "
        f"({', '.join(census.policy_names)}): {len(census.spreads)} claim(s) evaluated, "
        f"{len(contested)} with spread ≥ {threshold:.2f}. "
        f"Spread histogram: {json.dumps(_histogram(census.spreads))}."
    )
    if top_str:
        detail += f" Most contested: {top_str}."
    return [
        LintFinding(
            finding_type="CONTESTEDNESS_DISTRIBUTION",
            severity="INFO",
            detail=detail,
            recommended_action=(
                "Inspect the most-contested claims with `query --contestedness`; "
                "contestedness is disclosure, not a quality defect"
                if contested
                else None
            ),
        )
    ]


async def _report_contestedness_distribution(session: AsyncSession) -> list[LintFinding]:
    """Back-compat seam: the store-level finding alone.

    ``_check_contested`` is the orchestrator's entry point and emits this
    finding as part of the shared pass; this wrapper keeps the surface
    callable on its own.
    """
    return _report_distribution(await compute_store_contested(session))
