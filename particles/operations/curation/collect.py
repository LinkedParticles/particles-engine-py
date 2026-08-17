"""Union the existing finders into one card list.

Calls the finders that **already exist** and normalizes each one's native
output into a :class:`CurationCard`. Writes no new detection logic. ``quality``
is the session *header*, not a card source (its outputs are store-level counts,
not per-record findings) — the one exception is ``snapshots_failed``, which no
lint finding emits per-particle, so it becomes a single batch card.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import SuggestMode
from particles.operations.abstraction import pending_candidate_events
from particles.operations.deposit_suggest import suggest_deposits
from particles.operations.links_suggest import suggest_co_evidential
from particles.operations.lint import ContradictionProbeControl, run_lint
from particles.operations.quality import get_quality_report

from .cards import (
    CardKind,
    CurationCard,
    DuplicateVerdict,
    contested_gestures,
    gestures_for,
)

# Lint finding_type → the card kind it produces. Only per-record findings (those
# carrying a particle_id) become cards; the rest of lint's INFO/aggregate
# findings are not curation tasks.
_LINT_KIND: dict[str, CardKind] = {
    "STALENESS": CardKind.STALE,
    "RETRACTION_CASCADE": CardKind.RETRACTION_CASCADE,
    "CORPUS_LINK_INTEGRITY": CardKind.BROKEN_PROVENANCE,
    "CONFIDENCE_DECAY": CardKind.CONFIDENCE_DECAY,
    # postdated the earlier design; the audit added the mapping
    # age-discount finding is a card for both the queue and the audit census.
    "RECENCY_DECAY": CardKind.RECENCY_DECAY,
    "CONTRADICTION": CardKind.CONTRADICTION,
    # the most common finding becomes a card now that assign-subject
    # (a provenance-preserving operator-supersede) is the resolving write-op.
    "NO_SUBJECT": CardKind.NO_SUBJECT,
    # the contested class moves onto the composed finder, so
    # the card covers all three bases instead of the inconsistency one, and the
    # hygiene surfaces stop disagreeing with recall about what "contested"
    # means. This replaces a bespoke get_inconsistency_backrefs branch — one
    # fewer hand-rolled finder, and the one-kind ↔ one-finder rule holds.
    "CONTESTED": CardKind.CONTESTED,
}


async def collect_cards(
    session: AsyncSession,
    *,
    semantic: bool,
    duplicate_mode: SuggestMode | None = None,
    contradiction_probe: ContradictionProbeControl | None = None,
    duplicate_scope_ids: frozenset[str] | None = None,
) -> list[CurationCard]:
    """Gather every curation card from the existing finders.

    ``semantic`` gates the LLM-assisted lint finders (the CONTRADICTION probe);
    the structural finders always run. Lint is invoked read-only (``fix=False``)
    — a curation card never auto-mutates.

    ``duplicate_mode`` overrides the mode the duplicate finder runs in. The
    default (``None``) keeps the coupling — ``LLM_JUDGE`` when
    ``semantic`` is on, ``REPORT`` otherwise. The audit decouples
    them: its contradiction probe runs semantic while duplicates stay
    ``REPORT`` unless the operator passes ``--judge``.

    ``contradiction_probe`` is passed through to the probe: the audit uses it to cap / scope the probe's LLM cost and
    to read back the candidate-pair census for the "probed X of Y" disclosure.

    ``duplicate_scope_ids`` is passed through to the co-evidential
    finder as its ``scope_particle_ids``: enumeration stays store-wide, but the
    ``LLM_JUDGE`` verdict pass is bounded to pairs touching this harvest. The
    audit sets it to harvest-scope the ``--judge`` cost; it is ``None`` (no
    bound) for ``particles curate`` and re-audits.
    """
    cards: list[CurationCard] = []

    # --- lint (read-only): per-record structural + optional semantic findings ---
    # granularity_probe=False: GRANULARITY_VIOLATION has no CardKind, so the
    # per-particle LLM granularity loop would burn one call per long particle
    # and every finding would be dropped by the mapping below.
    report = await run_lint(
        session,
        fix=False,
        semantic=semantic,
        contradiction_probe=contradiction_probe,
        granularity_probe=False,
    )
    for f in report.findings:
        kind = _LINT_KIND.get(f.finding_type)
        if kind is None or f.particle_id is None:
            continue
        # a contested card offers `comment` only where an
        # INCONSISTENCY exists for `review` to resolve, so its gestures follow
        # the fired bases rather than the kind alone.
        bases = f.contested_bases if kind is CardKind.CONTESTED else None
        cards.append(
            CurationCard(
                kind=kind,
                particle_ids=[f.particle_id],
                diagnostic=f.detail,
                suggested_gestures=(
                    contested_gestures(bases) if bases is not None else gestures_for(kind)
                ),
                contested_bases=list(bases) if bases is not None else None,
                inconsistency_id=(f.inconsistency_id if kind is CardKind.CONTESTED else None),
            )
        )

    # --- duplicate pairs: co-evidential candidates within a Subject ---
    # with semantic finders on, run the duplicate finder in LLM_JUDGE
    # mode so each candidate carries the model's same-claim verdict (advisory) —
    # not raw cosine alone; with semantic off it stays REPORT (similarity only),
    # exactly as before. The judge degrades gracefully: an open
    # breaker / unavailable LLM leaves the verdict UNSURE, never DISTINCT.
    dup_mode = (
        duplicate_mode
        if duplicate_mode is not None
        else (SuggestMode.LLM_JUDGE if semantic else SuggestMode.REPORT)
    )
    suggest = await suggest_co_evidential(
        session, mode=dup_mode, scope_particle_ids=duplicate_scope_ids
    )
    for cluster in suggest.clusters:
        name = cluster.subject_name or cluster.subject_id
        for c in cluster.candidates:
            verdict = (
                DuplicateVerdict(
                    verdict=c.verdict,
                    rationale=getattr(c, "rationale", None),
                )
                if c.verdict is not None
                else None
            )
            cards.append(
                CurationCard(
                    kind=CardKind.DUPLICATE_PAIR,
                    particle_ids=[c.particle_a, c.particle_b],
                    subject_ids=[cluster.subject_id],
                    diagnostic=f"Possible duplicate in '{name}' (similarity {c.similarity:.2f})",
                    suggested_gestures=gestures_for(CardKind.DUPLICATE_PAIR),
                    verdict=verdict,
                )
            )

    # --- uncited URLs: undeposited-but-frequently-cited (already snooze-filtered) ---
    deposits = await suggest_deposits(session)
    for s in deposits.suggestions:
        cards.append(
            CurationCard(
                kind=CardKind.UNCITED_URL,
                corpus_url=s.canonical_url,
                diagnostic=(
                    f"{s.distinct_sources} distinct source(s) cite this undeposited "
                    f"URL (score {s.score:.2f})"
                ),
                suggested_gestures=gestures_for(CardKind.UNCITED_URL),
            )
        )

    # --- proposed abstractions: pending propose-mode candidates ---
    # The candidate's persistence is its ABSTRACTION_CANDIDATE event; the card
    # fronts the event and the accept / reject gestures re-read it by id.
    for event in await pending_candidate_events(session):
        payload = event.payload or {}
        claim = str(payload.get("claim") or "")
        premise_ids = [str(p) for p in payload.get("premise_ids") or []]
        if not claim or not premise_ids:
            continue
        rationale = str(payload.get("rationale") or "")
        cards.append(
            CurationCard(
                kind=CardKind.PROPOSED_ABSTRACTION,
                particle_ids=premise_ids,
                subject_ids=[str(s) for s in payload.get("subject_ids") or []],
                diagnostic=(
                    f"Proposed abstraction over {len(premise_ids)} specifics: "
                    f"“{claim}”" + (f" — {rationale}" if rationale else "")
                ),
                suggested_gestures=gestures_for(CardKind.PROPOSED_ABSTRACTION),
                candidate_event_id=event.event_id,
            )
        )

    # --- failed snapshots: the one aggregate the quality dashboard owns ---
    quality = await get_quality_report(session)
    if quality.snapshots_failed > 0:
        cards.append(
            CurationCard(
                kind=CardKind.FAILED_SNAPSHOTS,
                diagnostic=(
                    f"{quality.snapshots_failed} snapshot(s) failed extraction — "
                    "re-extract to recover them"
                ),
                suggested_gestures=gestures_for(CardKind.FAILED_SNAPSHOTS),
            )
        )

    return cards
