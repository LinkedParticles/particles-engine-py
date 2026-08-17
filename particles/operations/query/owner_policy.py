"""Read-time owner-relevance policy — the aboutness lens.

The aboutness analogue of the ``UtilityPolicy`` (``utility_policy.py``)
and the ``DecayPolicy`` (``decay_policy.py``): a once-per-render
snapshot resolving *who the viewer is* into a set of local Subject ids, which
turns each belief's subject links into an additive rank bonus
(``core/scoring/relevance.py``).

**The "owner" is the viewer.** ``A(p)`` is not defined against a special
principal; it is defined against the party whose lenses are in effect for this
read, and today that party is bound to the reader's own config. This is the
single-viewer binding of the viewer seam ("the viewer is the
querying store's owner; single-viewer, no auth"), which is what keeps the lens
inside substrate-plus-lens invariant rather than beside it: a
per-observer quantity, computed at query time, never stored.

Viewer identity lives in the **reader's config**, not in the store, and the
shared-store case is what decides it rather than what it concedes: several
contributors sharing one store each run their own SDK and so each get their own
viewer, whereas a store-resident field would burn *one* viewpoint into the
substrate the spec makes the unit of sharing. The binding holds exactly up to
multi-tenant line — one engine serving several readers from one
config is a separate multi-tenant layer, which injects a per-request viewer at
this same seam without changing the term.

**Resolve-or-inert neutrality** (mirrored): entries that do not
resolve to a local Subject are *reported*, never guessed. If none resolve the
policy is :data:`EMPTY_OWNER_POLICY` and the ordering is byte-identical to the
pre-0220 ranking. Resolution is **local-only** — ``find_by_name`` / by-id
against the Subject store, never the live-authority rungs of the cascade — because the digest is a zero-LLM, zero-network surface
and must stay one.

**Applied only on the recall (projection / digest / graph-node) ranking path** —
never the semantic-search retrieval rank. The boundary is the one
drawn for utility, and the reasoning is sharper here: this lens exists to fix the
*unqueried* surfaces, where the projection sets ``query: null`` and the "section" is
simply the top of the store. A caller who wants the viewer's beliefs
specifically already has ``QueryRequest.subject_id``.

The bonus is added to form a **ranking score**; neither the stored
``confidence.value`` nor the read-time ``effective_confidence`` is ever changed
. With ``owner_lens.enabled = False``, no configured subjects, or a
belief not about the viewer, the bonus is ``+0``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.scoring.relevance import owner_rank_bonus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OwnerPolicy:
    """One reader's resolved owner-relevance policy, snapshotted for a render.

    Attributes:
        viewer_subject_ids: local Subject ids the configured viewer resolved to.
            Empty ⇒ the lens is inert.
        rank_lift: ``ω``, the additive lift a viewer-relevant belief receives.
        unresolved: configured entries that matched no local Subject, kept for
            disclosure — a silently-inert lens is the failure mode this exists
            to prevent.
    """

    viewer_subject_ids: frozenset[str]
    rank_lift: float
    unresolved: tuple[str, ...] = field(default=())

    def is_owner_relevant(self, subject_ids: list[str] | frozenset[str]) -> bool:
        """``A(p)`` — whether a belief's subjects intersect the viewer's.

        Tier 1 aboutness: the viewer is among the things the claim is *about*.
        Tier 2 (owner-*endorsed* / *asserted* beliefs, via per-turn
        ``contributors`` / ``stance:holder``) is blocked and widens
        this predicate rather than replacing it.
        """
        if not self.viewer_subject_ids:
            return False
        return not self.viewer_subject_ids.isdisjoint(subject_ids)

    def bonus(self, subject_ids: list[str] | frozenset[str]) -> float:
        """Additive recall rank-lift for one belief under this policy.

        ``0.0`` when the lens is inert or the belief is not about the viewer —
        the belief keeps its base position (promotion-only).
        """
        return owner_rank_bonus(self.is_owner_relevant(subject_ids), self.rank_lift)


#: Neutral policy — lens disabled, unconfigured, or fully unresolved (bonus +0).
EMPTY_OWNER_POLICY = OwnerPolicy(viewer_subject_ids=frozenset(), rank_lift=0.0)


async def load_owner_policy(
    session: AsyncSession, *, require_rank_lift: bool = True
) -> OwnerPolicy:
    """Snapshot the reader's owner-relevance policy, resolving the viewer locally.

    Returns :data:`EMPTY_OWNER_POLICY` when the lens is disabled, no viewer is
    configured, ``ω`` is zero, or nothing resolves — in every one of those cases
    the caller adds bonus ``0.0`` and the ranking is byte-for-byte the pre-0220
    order. Called once per render; the result is reused across every belief.

    Unresolved entries are logged and carried on the policy rather than raising:
    requiring *every* alias to resolve would let one speculative entry silently
    disable the whole lens, which is the opposite of the intended failure mode.

    Args:
        session: Active store session.
        require_rank_lift: Short-circuit to the empty policy when ``ω`` is zero.
            ``True`` (the render path) skips the subject lookups entirely when
            the lens can have no effect. The **calibration sweep passes
            ``False``**: ``ω`` ships at ``0.0``, so an operator's first
            ``sweep-owner-lift`` would otherwise resolve no viewer and report an
            empty cohort — you would have to set the value before you could
            calibrate it. The resolved cohort is the sweep's input; the bonus is
            zero either way, since ``owner_rank_bonus`` floors at ``ω <= 0``.
    """
    from particles.config import get_config
    from particles.store.subject_store import find_by_name, get_subject

    cfg = get_config().owner_lens
    if not cfg.enabled or not cfg.subjects:
        return EMPTY_OWNER_POLICY
    if require_rank_lift and cfg.rank_lift <= 0.0:
        return EMPTY_OWNER_POLICY

    resolved: set[str] = set()
    unresolved: list[str] = []
    for entry in cfg.subjects:
        name = entry.strip()
        if not name:
            continue
        # Ids and names share one config field, so try both — **name first**,
        # since names are the documented common case and an id-shaped name is
        # the rarer collision. Local lookups only: no authority rung, no
        # network.
        subject = await find_by_name(session, name)
        if subject is None:
            try:
                subject = await get_subject(session, name)
            except ValueError:
                # Ambiguous id prefix. Treat as unresolved rather than letting
                # it abort the render: a misconfigured viewer must degrade to an
                # inert lens, never to a failed digest.
                logger.warning(
                    "owner_lens: viewer entry %r is an ambiguous subject-id prefix; "
                    "use more characters or the canonical name",
                    name,
                )
                subject = None
        if subject is None:
            unresolved.append(name)
        else:
            resolved.add(subject.id)

    if unresolved:
        logger.warning(
            "owner_lens: %d of %d configured viewer subject(s) did not resolve "
            "locally and contribute nothing: %s",
            len(unresolved),
            len(cfg.subjects),
            ", ".join(unresolved),
        )
    if not resolved:
        logger.warning(
            "owner_lens: no configured viewer subject resolved; the lens is inert "
            "and ranking is unchanged"
        )
        return EMPTY_OWNER_POLICY

    return OwnerPolicy(
        viewer_subject_ids=frozenset(resolved),
        rank_lift=cfg.rank_lift,
        unresolved=tuple(unresolved),
    )


async def apply_owner(
    session: AsyncSession,
    eff_by_id: dict[str, float],
    subject_ids_by_id: dict[str, list[str]],
    *,
    policy: OwnerPolicy | None = None,
) -> dict[str, float]:
    """Add the owner-relevance rank-lift to a score map.

    ``eff_by_id`` maps ``particle_id → score`` — the effective confidence from
    the trust / decay formula, already carrying the utility lift when
    the caller applied it; ``subject_ids_by_id`` maps ``particle_id →
    subject_ids`` for the aboutness test. Returns a new map of
    ``… + ω·A(p)``. A belief that is not about the viewer keeps its score
    unchanged (bonus ``+0``).

    The returned values are **ranking scores, not confidences** — the additive
    bonus is unbounded above with respect to ``[0, 1]``, so a caller that
    *displays* a confidence must score separately with the lens off (see
    ``operations/digest.py``, which keeps the two maps apart).

    This is the single seam that keeps owner-relevance on the recall path only:
    the semantic-search ``query`` path never calls it.
    """
    if not eff_by_id:
        return dict(eff_by_id)
    if policy is None:
        policy = await load_owner_policy(session)
    if policy is EMPTY_OWNER_POLICY or not policy.viewer_subject_ids:
        return dict(eff_by_id)

    return {
        pid: eff + policy.bonus(subject_ids_by_id.get(pid, [])) for pid, eff in eff_by_id.items()
    }
