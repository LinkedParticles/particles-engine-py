"""Effective-confidence scoring for a particle list.

The scoring kernel shared by the read surfaces that compute effective confidence
outside a semantic query: the stance distribution and the session-start digest. Given a set of particles and a trust policy, it computes
each particle's effective confidence with the **full** query-path formula —
``confidence.value × extractor_trust_weight × source_trust_rank ×
recency_factor`` — minus the embedding/similarity step. Lifting it here keeps
those surfaces and ``query`` from ever disagreeing on a particle's effective
confidence (the formula lives in exactly one place).

Resolve-or-None neutrality is preserved: a particle with no
applicable trust policy resolves at ``source_trust_rank = 1.0``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle
from particles.core.scoring.confidence import compute_effective_confidence
from particles.operations.abstraction import stale_support_discounts
from particles.store.extractor_store import (
    get_cached_trust_weight,
    get_trust_weight_map,
    populate_trust_cache,
)

from .decay_policy import DecayPolicy, load_decay_policy
from .owner_policy import apply_owner
from .source_info import load_source_rows
from .source_trust import TrustPolicy, load_trust_policy
from .utility_policy import apply_utility


async def score_effective_confidence(
    session: AsyncSession,
    particles: list[Particle],
    trust_policy: TrustPolicy | None = None,
    *,
    decay_policy: DecayPolicy | None = None,
    populate_cache: bool = False,
    apply_utility_factor: bool = False,
    apply_owner_lens: bool = False,
    now: datetime | None = None,
) -> dict[str, float]:
    """Return ``{particle_id: effective_confidence}`` for ``particles``.

    Mirrors ``main._gather_scored`` scoring exactly, without the
    embedding/similarity step, so a read surface that uses this and ``query``
    cannot disagree on a particle's effective confidence.

    Args:
        session: Active store session.
        particles: The particles to score (any status; the caller filters).
        trust_policy: A pre-loaded snapshot to reuse (the query orchestrator and
            stance share one per query). ``None`` loads the store's own policy
            via :func:`load_trust_policy`.
        populate_cache: Seed the in-process extractor trust-weight cache from
            this store before scoring. Set ``True`` when calling **outside** the
            query orchestrator (e.g. the digest), which has not already
            populated it; leave ``False`` inside a query where the orchestrator
            has (avoids a redundant fetch).
        apply_utility_factor: Add the usefulness rank-lift, returning
            ``rank_score = effective_confidence + λ·ln(1 + R)`` instead of the
            truth-only score. Set ``True`` **only** on the projection / digest
            ranking path (the digest projection) — never the
            semantic-search ``query`` path. Default ``False``
            preserves the truth-only score for ``query`` / stance / curation
            callers. **The lifted value is an ordering key, not a confidence:**
            it is unbounded above and may exceed ``1.0``, so a caller that
            *displays* the number must use :func:`score_confidence_and_rank`
            and render the truth-only half.
        apply_owner_lens: Add the owner-relevance rank-lift
            (``+ ω·A``). Same rule and same caveat as ``apply_utility_factor``:
            recall path only, ordering key only, never displayed. Default
            ``False`` — a caller opts in per surface, so the graph view
             and the hygiene surfaces stay on the truth-only
            score.
        now: The decay reference instant (an as-of consumer
            evaluates decay at T; trust/utility stay current). ``None`` — the
            default and every present-time caller — evaluates decay at the
            wall clock, exactly the prior behaviour.

    Returns:
        A map for every input particle. A particle with no applicable trust
        policy is scored at the neutral ``source_trust_rank = 1.0``.
    """
    eff, rank = await score_confidence_and_rank(
        session,
        particles,
        trust_policy,
        decay_policy=decay_policy,
        populate_cache=populate_cache,
        with_utility=apply_utility_factor,
        with_owner=apply_owner_lens,
        now=now,
    )
    return rank if (apply_utility_factor or apply_owner_lens) else eff


async def score_confidence_and_rank(
    session: AsyncSession,
    particles: list[Particle],
    trust_policy: TrustPolicy | None = None,
    *,
    decay_policy: DecayPolicy | None = None,
    populate_cache: bool = False,
    with_utility: bool = True,
    with_owner: bool = True,
    now: datetime | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(effective_confidence, rank_score)`` in a single scoring pass.

    The split made explicit: ``effective_confidence`` is the truth-axis
    quantity (what a surface *displays*), while ``rank_score`` is the
    recall *ordering* key, which is not a probability and may exceed ``1.0``.
    Under the superseded multiplier these were conflated in one
    returned number, so the digest rendered a utility-scaled value under a
    "confidence" label.

    The ordering key composes the three read-time axes as separate addends::

        rank_score = effective_confidence + λ·ln(1 + R) + ω·A
                     \\___ truth ________/   \\__ use __/   \\_ aboutness _/

    — truth, use (``with_utility``), and aboutness
    (``with_owner``). Both addends are promotion-only, so either flag
    can be turned off independently and the remaining order is unchanged.

    A projection / digest caller needs both maps — rank by the second, render
    the first — and getting them from one call avoids re-running the trust,
    decay, source-row and stale-support lookups a second time over the same
    particles.

    With both flags ``False`` (or both lenses disabled / no evidence / no viewer
    configured) the two maps are equal, so a caller can use this
    unconditionally.
    """
    if not particles:
        return {}, {}
    if populate_cache:
        populate_trust_cache(await get_trust_weight_map(session))
    if trust_policy is None:
        trust_policy = await load_trust_policy(session)
    if decay_policy is None:
        decay_policy = await load_decay_policy(session)
    source_info = await load_source_rows(session, particles)
    # a derived particle with a non-ACTIVE premise is discounted
    # at read time (keep-ACTIVE-and-discount) until the next revalidation.
    support_discounts = await stale_support_discounts(session, particles)
    out: dict[str, float] = {}
    for p in particles:
        extractor_id = p.extractor_ref.name if p.extractor_ref else "general-extractor"
        trust_weight = get_cached_trust_weight(extractor_id)
        pub_at, source_type, entry_id, uri_r, author_id = source_info.get(
            p.id, (None, "", None, None, None)
        )
        rank = trust_policy.evaluate(entry_id, source_type, uri_r, author_id)
        out[p.id] = compute_effective_confidence(
            p.confidence.value,
            extractor_trust_weight=trust_weight,
            source_trust_rank=1.0 if rank is None else rank,
            recency_factor=decay_policy.recency_factor(pub_at, source_type, uri_r, now=now),
            calibration_source=p.confidence.calibration_source,
        ) * support_discounts.get(p.id, 1.0)
    if not with_utility and not with_owner:
        return out, dict(out)
    rank_out = out
    if with_utility:
        util_source_info: dict[str, tuple[str, str | None]] = {}
        for p in particles:
            _pub, st, _eid, uri, _auth = source_info.get(p.id, (None, "", None, None, None))
            util_source_info[p.id] = (st, uri)
        rank_out = await apply_utility(session, rank_out, util_source_info)
    if with_owner:
        # the aboutness addend, applied *after* the utility addend and
        # onto the same ordering score. Two promotion-only terms on orthogonal
        # axes — neither reads the other, and both leave `out` (the truth-axis
        # map a surface displays) untouched.
        rank_out = await apply_owner(session, rank_out, {p.id: p.subject_ids for p in particles})
    return out, rank_out
