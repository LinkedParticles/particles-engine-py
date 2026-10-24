# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Union the existing finders into one card list.

Calls the finders that **already exist** and normalizes each one's native
output into a :class:`CurationCard`. Writes no new detection logic. ``quality``
is the session *header*, not a card source (its outputs are store-level counts,
not per-record findings) — the one exception is ``snapshots_failed``, which no
lint finding emits per-particle, so it becomes a single batch card.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import LintFinding, Particle, ProvenanceRefType, SuggestMode
from particles.corpus.store import get_entry_uri_map
from particles.operations.abstraction import pending_candidate_events
from particles.operations.contradiction_disclosure import covered_pair_set
from particles.operations.deposit_suggest import suggest_deposits
from particles.operations.links_suggest import suggest_co_evidential
from particles.operations.lint import ContradictionProbeControl, run_lint
from particles.operations.quality import get_quality_report
from particles.operations.rank_lift_monitor import (
    RankLiftMonitorReport,
    latest_monitor_block,
    render_rank_lift_lines,
)
from particles.operations.reanchor import pending_unrestated_events
from particles.operations.subject_relink import plan_gated_relink
from particles.store.particle_store import get_particles_by_ids, get_recorded_demotions

from .cards import (
    CardKind,
    CurationCard,
    DuplicateVerdict,
    gestures_for,
)
from .suppression import dismissed_members, suppressed_keys

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
    # one card per open INCONSISTENCY record, keyed by the record.
    "OPEN_INCONSISTENCY": CardKind.INCONSISTENCY,
}


def cards_from_findings(findings: Sequence[LintFinding]) -> list[CurationCard]:
    """Normalize per-record lint findings into curation cards.

    Findings with no card kind, or with no particle, are not curation tasks
    and are dropped. Shared with the nightly disclosure pass, which rebuilds
    the cards of the records and claims it touched (as amended).
    """
    cards: list[CurationCard] = []
    for f in findings:
        kind = _LINT_KIND.get(f.finding_type)
        if kind is None or f.particle_id is None:
            continue
        if kind is CardKind.INCONSISTENCY:
            cards.append(_conflict_card(f))
            continue
        bases = f.contested_bases if kind is CardKind.CONTESTED else None
        diagnostic = f.detail
        if bases is not None:
            # a belief contested only because it sits in an open
            # INCONSISTENCY gets no CONTESTED card; each of its conflicts has
            # its own card, which is where that work is.
            if not [b for b in bases if b != "inconsistency"]:
                continue
            if "inconsistency" in bases and f.inconsistency_id:
                diagnostic += f". Settle the conflict on card inconsistency:{f.inconsistency_id}"
        cards.append(
            CurationCard(
                kind=kind,
                particle_ids=[f.particle_id],
                diagnostic=diagnostic,
                suggested_gestures=gestures_for(kind),
                contested_bases=list(bases) if bases is not None else None,
                inconsistency_id=(f.inconsistency_id if kind is CardKind.CONTESTED else None),
            )
        )
    return cards


def _conflict_card(f: LintFinding) -> CurationCard:
    """The ``INCONSISTENCY`` card for one ``OPEN_INCONSISTENCY`` finding.

    ``particle_ids`` lists the members A, B, then any further census members by
    side, so ``particle_ids[0]`` / ``[1]`` are the claims ``review`` calls A
    and B. The card's identity is the record, never its members.
    """
    sides = f.conflict_sides or []
    a_side = sides[0] if len(sides) > 0 else []
    b_side = sides[1] if len(sides) > 1 else []
    members = a_side[:1] + b_side[:1] + a_side[1:] + b_side[1:]
    return CurationCard(
        kind=CardKind.INCONSISTENCY,
        particle_ids=list(dict.fromkeys(members)),
        diagnostic=f.detail,
        suggested_gestures=gestures_for(CardKind.INCONSISTENCY),
        inconsistency_id=f.particle_id,
    )


async def _fold_recoverable_orphans(
    session: AsyncSession, cards: list[CurationCard]
) -> list[CurationCard]:
    """Replace the NO_SUBJECT cards a relink can resolve with one batch card.

    An orphan whose gated subject names the accepted relink tiers recover (and
    whose project is known) is not a per-belief question: the relink answers
    it. Those orphans become one ``gated_subjects`` card that names them all,
    so its leverage is scored over its beliefs. Every other orphan keeps its
    own ``no_subject`` card.
    """
    orphan_ids = {c.particle_ids[0] for c in cards if c.kind is CardKind.NO_SUBJECT}
    if not orphan_ids:
        return cards
    plan = await plan_gated_relink(session)
    recoverable = [item.particle_id for item in plan.items if item.particle_id in orphan_ids]
    if not recoverable:
        return cards
    covered = set(recoverable)
    kept = [
        c for c in cards if c.kind is not CardKind.NO_SUBJECT or c.particle_ids[0] not in covered
    ]
    tiers = ", ".join(str(t) for t in plan.tiers)
    kept.append(
        CurationCard(
            kind=CardKind.GATED_SUBJECTS,
            particle_ids=sorted(covered),
            diagnostic=(
                f"{len(covered)} belief(s) with no subject name a file, record, identifier "
                f"or command the extraction gate withheld; relink tiers {tiers} recover them"
            ),
            suggested_gestures=gestures_for(CardKind.GATED_SUBJECTS),
        )
    )
    return kept


#: The fewest subjectless beliefs one source needs for a card of its own.
#: A lone orphan keeps its one-gesture ``no_subject`` card.
_SOURCE_GROUP_MIN = 2

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _first_source_entry(p: Particle) -> str | None:
    """The corpus entry of the belief's first SOURCE ref, the one its brief names."""
    ref = next((r for r in p.provenance if r.type == ProvenanceRefType.SOURCE), None)
    return ref.corpus_entry_id if ref is not None and ref.corpus_entry_id else None


def group_orphans_by_source(
    cards: list[CurationCard],
    *,
    sources: Mapping[str, str],
    asserted_at: Mapping[str, datetime],
    uris: Mapping[str, str],
    suppressed: set[str],
    dismissed: set[str],
) -> list[CurationCard]:
    """Fold the per-belief NO_SUBJECT cards into one card per source (pure).

    ``sources`` maps a belief to its first SOURCE corpus entry. A source with
    at least :data:`_SOURCE_GROUP_MIN` orphans becomes one ``subjectless_source``
    card naming them in assertion order; every other orphan keeps its own card.
    A belief a source dismissal named (``dismissed``) has been ruled on and
    leaves the collection. A belief whose own ``no_subject`` key is snoozed or
    dismissed (``suppressed``) is not grouped: it keeps its card, which the
    live half hides by key as before (§4).
    """
    groups: dict[str, list[str]] = {}
    for card in cards:
        if card.kind is not CardKind.NO_SUBJECT:
            continue
        pid = card.particle_ids[0]
        entry = sources.get(pid)
        if pid in dismissed or card.key in suppressed or entry is None:
            continue
        groups.setdefault(entry, []).append(pid)
    grouped = {e: ids for e, ids in groups.items() if len(ids) >= _SOURCE_GROUP_MIN}
    gone = dismissed | {pid for ids in grouped.values() for pid in ids}
    out = [c for c in cards if c.kind is not CardKind.NO_SUBJECT or c.particle_ids[0] not in gone]
    for entry, ids in sorted(grouped.items()):
        ids.sort(key=lambda pid: (asserted_at.get(pid, _EPOCH), pid))
        out.append(
            CurationCard(
                kind=CardKind.SUBJECTLESS_SOURCE,
                particle_ids=ids,
                corpus_entry_id=entry,
                diagnostic=f"{len(ids)} belief(s) with no subject from {uris.get(entry, entry)}",
                suggested_gestures=gestures_for(CardKind.SUBJECTLESS_SOURCE),
            )
        )
    return out


async def _group_orphans(session: AsyncSession, cards: list[CurationCard]) -> list[CurationCard]:
    """Gather what :func:`group_orphans_by_source` decides over."""
    orphan_ids = [c.particle_ids[0] for c in cards if c.kind is CardKind.NO_SUBJECT]
    if not orphan_ids:
        return cards
    particles = await get_particles_by_ids(session, orphan_ids)
    sources = {
        pid: entry for pid, p in particles.items() if (entry := _first_source_entry(p)) is not None
    }
    asserted_at = {
        pid: (p.asserted_at if p.asserted_at.tzinfo else p.asserted_at.replace(tzinfo=UTC))
        for pid, p in particles.items()
    }
    return group_orphans_by_source(
        cards,
        sources=sources,
        asserted_at=asserted_at,
        uris={
            e: uri
            for e, uri in (await get_entry_uri_map(session, set(sources.values()))).items()
            if uri
        },
        suppressed=await suppressed_keys(session),
        dismissed=await dismissed_members(session),
    )


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

    ``contradiction_probe`` is passed through to the
    probe: the audit uses it to cap / scope the probe's LLM cost and
    to read back the candidate-pair census for the "probed X of Y" disclosure.
    Without one, a semantic collection (``particles curate --semantic``,
    ``GET /curation``, a rebuild) still reads every flag a second time when
    ``audit.verify_contradictions`` is on, uncapped like the probe itself, so a
    card surface counts the same confirmed contradictions the audit and the
    nightly census do.

    ``duplicate_scope_ids`` is passed through to the co-evidential
    finder as its ``scope_particle_ids``: enumeration stays store-wide, but the
    ``LLM_JUDGE`` verdict pass is bounded to pairs touching this harvest. The
    audit sets it to harvest-scope the ``--judge`` cost; it is ``None`` (no
    bound) for ``particles curate`` and re-audits.
    """
    cards: list[CurationCard] = []

    if semantic and contradiction_probe is None:
        contradiction_probe = ContradictionProbeControl(
            verify=get_config().audit.verify_contradictions
        )
    if semantic and contradiction_probe is not None and contradiction_probe.exclude_pairs is None:
        # a disagreement a census record already discloses is
        # reported through the record, not probed and paid for again.
        contradiction_probe.exclude_pairs = await covered_pair_set(session)

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
    orphans_folded = await _fold_recoverable_orphans(session, cards_from_findings(report.findings))
    cards.extend(await _group_orphans(session, orphans_folded))

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

    # --- stale basis: dependents the re-anchor pass kept for review ---
    # The unrestated event is the card's persistence; the card is open while
    # the belief is ACTIVE, and affirm / snooze act on its key as for any card.
    for event in await pending_unrestated_events(session):
        payload = event.payload or {}
        pid = str(payload.get("particle_id") or "")
        if not pid:
            continue
        retired = str(payload.get("retired_content") or payload.get("retired_id") or "")
        why = str(payload.get("why") or "")
        restatement = str(payload.get("restatement") or "")
        cards.append(
            CurationCard(
                kind=CardKind.STALE_BASIS,
                particle_ids=[pid],
                diagnostic=(
                    f"Relied on “{retired}”, which a later update replaced; {why}"
                    + (f". Proposed restatement: “{restatement}”" if restatement else "")
                ),
                suggested_gestures=gestures_for(CardKind.STALE_BASIS),
            )
        )

    # --- demotions: claims a later claim retired as its replacement ---
    # A listing of what the ladder and the sweeps already decided, not a new
    # detection. particle_ids is (retired, replacement); the gestures record a
    # ruling and never change either status.
    for demoted, replacement in await get_recorded_demotions(session):
        reason = demoted.status_reason.value if demoted.status_reason else "demoted"
        cards.append(
            CurationCard(
                kind=CardKind.DEMOTION,
                particle_ids=[demoted.id, replacement.id],
                subject_ids=list(demoted.subject_ids),
                diagnostic=(
                    f"“{demoted.content}” was retired ({reason}) in favour of "
                    f"“{replacement.content}”"
                ),
                suggested_gestures=gestures_for(CardKind.DEMOTION),
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

    # --- the rank-lift monitor's alert, from the run record ---
    card = await rank_lift_card(session)
    if card is not None:
        cards.append(card)
    # --- refused sources: held because the extraction model declined them ---
    refused = await refused_snapshots_card(session)
    if refused is not None:
        cards.append(refused)

    return cards


async def rank_lift_card(session: AsyncSession) -> CurationCard | None:
    """One card while the latest monitor reading carries an alert.

    The alert is raised only on a transition into an out-of-band verdict, so a
    standing verdict known at activation never becomes a card. The card names
    the reading and the sweep that reproduces it; choosing a new ``λ`` is the
    operator's act, outside the queue.
    """
    block = await latest_monitor_block(session)
    if not block or not block.get("alert"):
        return None
    report = RankLiftMonitorReport.model_validate(block)
    lines = render_rank_lift_lines(report)
    diagnostic = " ".join(line.strip() for line in lines).removeprefix("rank-lift").strip()
    return CurationCard(
        kind=CardKind.RANK_LIFT,
        rank_lift_alert=report.alert,
        diagnostic=f"Usefulness rank-lift: {diagnostic}",
        suggested_gestures=gestures_for(CardKind.RANK_LIFT),
    )


#: How many held sources the refused-sources card names before "and N more".
_NAMED_REFUSED = 3


async def refused_snapshots_card(session: AsyncSession) -> CurationCard | None:
    """The one batch card naming the sources a current refusal holds.

    Each named source carries the projects it is in view for, read from its
    entry's ``project:`` tags: like every card it displays observer scope and
    is never filtered by it.
    """
    from particles.core.observer_scope import project_keys
    from particles.ingest.refusal_hold import held_snapshots

    held = await held_snapshots(session)
    if not held:
        return None
    named = []
    for item in held[:_NAMED_REFUSED]:
        label = item.uri_r.rstrip("/").rsplit("/", 1)[-1] if item.uri_r else item.entry_id[:8]
        details = [item.category] if item.category else []
        projects = sorted(project_keys(item.tags))
        if projects:
            details.append("in view for " + ", ".join(projects))
        named.append(f"{label} ({'; '.join(details)})" if details else label)
    rest = len(held) - len(named)
    names = ", ".join(named) + (f" and {rest} more" if rest > 0 else "")
    return CurationCard(
        kind=CardKind.REFUSED_SNAPSHOTS,
        diagnostic=(
            f"{len(held)} source(s) were declined by the extraction model "
            f"({held[0].provider_model}) and are not in memory: {names}"
        ),
        suggested_gestures=gestures_for(CardKind.REFUSED_SNAPSHOTS),
    )
