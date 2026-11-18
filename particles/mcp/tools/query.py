# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``query`` — tag-aware semantic search + NL response.

Routed through the ``Backend`` seam: with no engine configured the
local backend runs the query in-process (today's behaviour); with
``engine.base_url`` set it runs on the canonical engine. The
contested marker is fetched through the same backend so it reflects whichever
store served the query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from particles.config import get_config
from particles.core.schema import (
    AudienceHint,
    ContestedBadge,
    HitSource,
    Particle,
    ProvenanceRefType,
    QueryRequest,
    QueryResponse,
    StructuralGroupBy,
    TimeWindow,
)
from particles.core.validity import window_end_of_day, window_for_day
from particles.mcp.limits import admit
from particles.mcp.observer import observer_for, scope_disclosure
from particles.operations.query.grounding import strip_attribution_tags

#: The response lists parallel to ``particles``: entry ``i`` annotates hit ``i``
#: (or the list is empty when its feature is off). Cut with the hits.
_PARALLEL = (
    "effective_confidences",
    "content_published_ats",
    "sources",
    "agreement_distributions",
    "contestedness",
    "contested",
    "as_of_notes",
    "window_hits",
)

#: Parallel lists a compact hit carries itself, so the compact response drops them.
#: The rest stay, cut to the listed hits.
_CARRIED_BY_HIT = ("effective_confidences", "content_published_ats", "sources", "contested")

_DETAILS = ("compact", "summary", "full")


async def query(
    question: str | None = None,
    tags: list[str] | None = None,
    subject_id: str | None = None,
    min_confidence: float = 0.0,
    top_k: int | None = None,
    audience: str = "GENERAL",
    summary: bool | None = None,
    as_of: str | None = None,
    during_start: str | None = None,
    during_end: str | None = None,
    on: str | None = None,
    predicate: str | None = None,
    object_eq: str | None = None,
    object_gt: str | None = None,
    object_lt: str | None = None,
    object_contains: str | None = None,
    count: bool = False,
    group_by: str | None = None,
    min_effective_confidence: float | None = None,
    list_predicates: bool = False,
    all_projects: bool = False,
    grounded: bool | None = None,
    include_refused_candidates: bool = False,
    detail: Literal["compact", "summary", "full"] | None = None,
    hits: int | None = None,
) -> dict[str, Any]:
    """Run a tag-aware semantic query against the particle store.

    Args:
        question: Natural-language question. Optional since: omit it
            and set structural claim filters (or an aggregate / the predicate
            listing) for the deterministic modes — no embedding, no LLM call.
        tags: Subtree-expanded tag filters, repeatable.
        subject_id: Restrict to particles about this subject.
        min_confidence: Minimum confidence value (0.0–1.0).
        top_k: Number of particles to retrieve, which the answer reads (1 to
            200). Default: the server's ``query.default_top_k`` (40).
        audience: "GENERAL" | "EXPERT" | "REGULATORY".
        summary: Deprecated alias of ``detail``: ``True`` means
            ``"summary"`` and ``False`` means ``"full"``. ``detail`` wins
            when both are given.
        detail: The shape of the result. ``"compact"`` (the
            server default, ``mcp.query_detail``) lists slim hits and drops
            every null, empty, zero or false key: a key that is absent is
            unset. ``"summary"`` is the older slim shape. ``"full"`` returns
            the whole ``QueryResponse`` with full particle records.
        hits: The most hits the result lists (0 or more). Default: under
            ``"compact"`` the server's ``mcp.query_hits`` (10); under
            ``"summary"`` and ``"full"`` every retrieved hit. The answer still
            reads all ``top_k`` hits; ``hits=0`` returns the answer and its
            disclosures only.
        as_of: ISO-8601 past instant (a bare date means start of that
            day, UTC): answer what the store believed at T.
            Retired hits carry an ``as_of_notes`` entry (current status,
            retirement instant + basis, and the superseding belief);
            ``as_of_excluded_undatable`` discloses retirements the store
            cannot date. A future instant is rejected.
        during_start: With ``during_end``, a window of world time (ISO-8601
            dates, inclusive; either may be omitted for an open end): answer
            about what began in, or was in force during, that period.
            The window prefilters the candidate set before ranking;
            with no question it returns the claims in date order, each with a
            ``window_hits`` reading (``BEGAN``, ``IN_FORCE``, ``HELD_AT``,
            ``WITHIN``). A retired claim is admitted for the period it held,
            marked with its retirement and its close. Reads ``valid_from``;
            ``window_coverage`` discloses the undated and unkinded claims the
            window could not judge. Resolve the anchor first ("when did the
            user move?"), then ask with dates: the anchor step is the caller's.
        during_end: The inclusive end date of the window; see ``during_start``.
        on: The point form: what held or happened on this date (ISO-8601).
            Alternative to ``during_start`` / ``during_end``.
        predicate: Structural filter — claims whose predicate term
            equals this string (case-insensitive, exact: a CURIE and its
            expanded IRI are different strings; discover terms with
            ``list_predicates``). With a question it prefilters the semantic
            candidate set; without one it selects the deterministic listing.
        object_eq: Claims whose object equals this value (typed comparison
            when both sides normalize — numbers and ISO dates — else
            case-insensitive text).
        object_gt: Claims whose object is greater than this number or ISO
            date; claims whose object would not normalize are excluded and
            disclosed in ``claim_coverage.not_normalizable_excluded``.
        object_lt: Claims whose object is less than this number or ISO date
            (same normalization and disclosure as ``object_gt``).
        object_contains: Claims whose object contains this substring
            (case-insensitive; every term kind).
        count: Deterministic aggregate — the number of matching claims with
            their effective-confidence distribution. Rejects a simultaneous
            question.
        group_by: Deterministic aggregate — bucket matching claims by
            "subject", "predicate", or "object" with per-bucket counts and
            confidence distribution. Rejects a simultaneous question.
        min_effective_confidence: Explicit confidence floor for the aggregate
            modes; excluded rows are disclosed. There is no default floor.
        list_predicates: List the distinct predicate terms with kind and
            claim count (``predicate_vocabulary`` in the response) — the
            discovery surface for the exact-string predicate filter.
        all_projects: On a server bound to one project, read the whole
            store instead of that project's view. The result says so. No effect
            on an unbound server.
        grounded: Compose a grounded answer: every sentence ends
            with the ids of the retrieved particles it rests on (``[p-1a2b3c4d]``,
            the form ``particle_show`` accepts), or with ``[inference]`` (the
            model's own inference over cited claims) or ``[background]`` (from
            nothing in the store). Cited ids are checked against the retrieved
            set; a sentence with no valid citation and no label is
            ``[unattributed]``. Nothing is dropped and nothing is stored.
            ``answer_attribution`` carries the per-sentence labels and full
            cited ids. Omit to use the server's ``query.grounded_answers``.
        include_refused_candidates: When the answer refuses (the store holds
            nothing relevant), the nearest beliefs are withheld by default:
            they are likely unrelated, and listing them costs the
            caller's context. ``withheld_candidates`` then says how many were
            withheld. Set ``True`` to receive them anyway, still marked by
            ``answer_refused``. No effect when the answer did not refuse.

    Returns:
        By default (``detail="compact"``) the answer, its disclosures, and the
        hits the answer cites, in the order it first cites them: the facts the
        answer rests on. The answer is composed grounded for this, and its
        citation tags are removed from the text. When the answer cites nothing
        (all inference, a failed answer, ``grounded=False``) the top ``hits``
        ranked hits are listed instead. Each compact hit carries ``id`` (for
        ``particle_show``), ``content``, ``effective_confidence``, ``source``
        (the source type, URI and publication date of its first source, with
        ``(+N more)`` when it has several), ``valid_from``, and, when the claim
        is contested, ``contested`` (the open conflict's id, which
        ``particle_show`` opens) and ``contested_bases``. A listing without a
        question adds ``predicate`` and ``object``. ``hits_omitted`` counts the
        retrieved hits the answer read that the list leaves out; ask with
        ``detail="full"`` and a larger ``hits`` to see them.
        ``sources_unavailable`` means the engine sent no source lines.
        ``coverage_gaps_count`` counts the corpus entries anywhere in the store
        still PENDING or FAILED extraction (not scoped to the question); their
        ids are under ``coverage_gaps`` with ``detail="full"``. When
        the answer refuses, nothing is listed and ``withheld_candidates`` says
        how many were withheld.

        With ``detail="full"``: the full ``QueryResponse`` as JSON, answer
        string plus the ranked particles, effective
        confidences, coverage gaps, and any truncation warning. A structural
        call returns rows
        or counts, not generated prose: the deterministic listing fills
        ``particles`` ordered by effective confidence, the aggregates fill
        ``structural_aggregate``, the vocabulary listing fills
        ``predicate_vocabulary``, and every structural result carries
        ``claim_coverage`` (the annotated-fraction footer plus the
        non-normalizable exclusion disclosure). Each particle's ``contested`` key is
        the id of an open INCONSISTENCY referencing it,
        else null; the composed contested badge rides
        beside it — the response-level ``contested`` list carries one
        badge (fired bases + drill-downs) or null per ranked particle,
        and with ``summary=True`` each slim hit carries
        ``contested_bases`` (the fired basis labels, else null) beside
        its ``contested`` id. When ``summary=True``, each particle
        entry is a small dict instead of the full Pydantic model.
    """
    # Routed through the backend seam. The deferred-import test seam is
    # preserved one layer down: ``LocalBackend.query`` defers
    # ``from particles.operations.query import query as query_op``, so
    # tests/test_mcp_server.py patching ``particles.operations.query.query`` still
    # reaches the call. See tests/AGENTS.md § Mocking strategy.
    from particles.api.client import get_backend

    # the shape, the list and the count, each from config unless the
    # call names it.
    mcp_cfg = get_config().mcp
    if detail is None:
        detail = mcp_cfg.query_detail if summary is None else ("summary" if summary else "full")
    if detail not in _DETAILS:
        raise ValueError(f"Unknown detail {detail!r}. Valid: {', '.join(_DETAILS)}.")
    if hits is not None and hits < 0:
        raise ValueError(f"hits must be 0 or more, got {hits}.")
    cited_list = detail == "compact" and mcp_cfg.query_list == "cited" and bool(question)

    as_of_dt: datetime | None = None
    if as_of is not None:
        try:
            as_of_dt = datetime.fromisoformat(as_of)
        except ValueError:
            raise ValueError(
                f"Invalid as_of value {as_of!r}: expected an ISO-8601 date or "
                "datetime (e.g. 2000-01-01 or 2006-08-24T12:00:00+00:00)."
            ) from None
    group_by_val: StructuralGroupBy | None = None
    if group_by is not None:
        try:
            group_by_val = StructuralGroupBy(group_by.strip().lower())
        except ValueError:
            valid = ", ".join(g.value for g in StructuralGroupBy)
            raise ValueError(f"Unknown group_by axis {group_by!r}. Valid: {valid}.") from None
    window: TimeWindow | None = None
    if on is not None and (during_start is not None or during_end is not None):
        raise ValueError("on and during_start / during_end are alternatives; pass one form.")
    try:
        if on is not None:
            window = window_for_day(datetime.fromisoformat(on))
        elif during_start is not None or during_end is not None:
            window = TimeWindow(
                start=datetime.fromisoformat(during_start) if during_start else None,
                end=window_end_of_day(datetime.fromisoformat(during_end)) if during_end else None,
            )
    except ValueError as exc:
        raise ValueError(
            f"Invalid window: {exc}. Expected ISO-8601 dates (e.g. 2026-10-10)."
        ) from None
    req = QueryRequest(
        question=question,
        tags=list(tags or []),
        subject_id=subject_id,
        min_confidence=min_confidence,
        top_k=top_k if top_k is not None else get_config().query.default_top_k,
        audience=AudienceHint(audience),
        as_of=as_of_dt,
        window=window,
        predicate=predicate,
        object_eq=object_eq,
        object_gt=object_gt,
        object_lt=object_lt,
        object_contains=object_contains,
        count=count,
        group_by=group_by_val,
        min_effective_confidence=min_effective_confidence,
        list_predicates=list_predicates,
        observer_project=observer_for(all_projects),
        # the cited list needs the answer's citations, so it asks
        # for a grounded answer unless the caller chose.
        grounded=grounded if grounded is not None else (True if cited_list else None),
        include_refused_candidates=include_refused_candidates,
    )
    if question:
        admit("query")  # the NL response is a model call (security review D30)
    backend = get_backend()
    resp = await backend.query(req)
    # mark each returned ACTIVE belief contested when an open
    # INCONSISTENCY references it, so the agent sees the §6/§6b conflict at
    # recall (query is otherwise ACTIVE-only and hides the ledger).
    backrefs = await backend.inconsistency_backrefs()
    for particle in resp.particles:
        particle.contested = backrefs.get(particle.id)
    disclosure = scope_disclosure(all_projects)
    # ``mcp.query_hits`` bounds the compact list; the older shapes
    # list every retrieved hit unless the call names a count, so they return
    # what they returned before.
    if hits is not None:
        max_hits = hits
    elif detail == "compact":
        max_hits = mcp_cfg.query_hits
    else:
        max_hits = len(resp.particles)
    listed = _listed(resp, max_hits, cited=cited_list)
    omitted = len(resp.particles) - len(listed)
    if detail == "compact":
        out = _compact(resp, listed, structural=not question)
    else:
        out = _cut(resp.model_dump(mode="json"), listed, len(resp.particles))
        if detail == "summary":
            out["particles"] = [_summary_hit(resp, i) for i in listed]
    # a cut list says so; absent means nothing was cut.
    if omitted > 0:
        out["hits_omitted"] = omitted
    if disclosure is not None:
        out["observer"] = disclosure
    return out


def _listed(resp: QueryResponse, max_hits: int, *, cited: bool) -> list[int]:
    """The indices of the hits to list, in listing order (§3).

    The cited list is the hits the answer cites, in the order it first cites
    them, up to ``max_hits``; when it cites nothing it falls back to the top
    ``max_hits`` ranked hits, which is also the uncited list.
    """
    if cited and resp.answer_attribution is not None:
        index = {p.id: i for i, p in enumerate(resp.particles)}
        order: list[int] = []
        for sentence in resp.answer_attribution.sentences:
            for pid in sentence.cited_ids:
                i = index.get(pid)
                if i is not None and i not in order:
                    order.append(i)
        if order:
            return order[:max_hits]
    return list(range(min(max_hits, len(resp.particles))))


def _cut(out: dict[str, Any], listed: list[int], n: int) -> dict[str, Any]:
    """Cut ``particles`` and every parallel list to the listed hits, in order."""
    for key in ("particles", *_PARALLEL):
        values = out.get(key)
        if isinstance(values, list) and len(values) == n:
            out[key] = [values[i] for i in listed]
    constituents = out.get("narrative_constituents")
    if constituents:
        kept = {p["id"] for p in out["particles"]}
        out["narrative_constituents"] = {k: v for k, v in constituents.items() if k in kept}
    return out


def _badge(resp: QueryResponse, i: int) -> ContestedBadge | None:
    return resp.contested[i] if len(resp.contested) == len(resp.particles) else None


def _summary_hit(resp: QueryResponse, i: int) -> dict[str, Any]:
    """One hit of the older ``summary`` shape, unchanged."""
    particle = resp.particles[i]
    hit: dict[str, Any] = {
        "id": particle.id,
        "content": particle.content,
        "confidence": particle.confidence.value,
        "effective_confidence": resp.effective_confidences[i],
        "subject_ids": list(particle.subject_ids),
        "contested": particle.contested,
    }
    # no contested_bases key when the badge is disabled, so the
    # pre-badge slim shape is restored exactly.
    if len(resp.contested) == len(resp.particles):
        badge = resp.contested[i]
        hit["contested_bases"] = list(badge.bases) if badge is not None else None
    return hit


def _source_line(source: HitSource | None, particle: Particle) -> str | None:
    """``TYPE uri date``, plus ``(+N more)`` when the hit has more than one source."""
    if source is None:
        return None
    parts = [source.source_type] if source.source_type else []
    if source.uri:
        parts.append(source.uri)
    if source.published is not None:
        parts.append(source.published.date().isoformat())
    if not parts:
        return None
    more = sum(1 for ref in particle.provenance if ref.type == ProvenanceRefType.SOURCE) - 1
    line = " ".join(parts)
    return f"{line} (+{more} more)" if more > 0 else line


def _compact_hit(resp: QueryResponse, i: int, *, structural: bool) -> dict[str, Any]:
    """One compact hit: only the fields that are set."""
    particle = resp.particles[i]
    hit: dict[str, Any] = {
        "id": particle.id,
        "content": particle.content,
        "effective_confidence": resp.effective_confidences[i],
    }
    sources = resp.sources if len(resp.sources) == len(resp.particles) else []
    line = _source_line(sources[i], particle) if sources else None
    if line is not None:
        hit["source"] = line
    if particle.valid_from is not None:
        hit["valid_from"] = particle.valid_from.isoformat()
    badge = _badge(resp, i)
    conflict = particle.contested or (badge.inconsistency_id if badge is not None else None)
    if conflict:
        hit["contested"] = conflict
    if badge is not None:
        hit["contested_bases"] = list(badge.bases)
    claim = particle.structured_claim
    if structural and claim is not None:
        hit["predicate"] = claim.predicate.value
        hit["object"] = claim.object.value
    return hit


def _compact(resp: QueryResponse, listed: list[int], *, structural: bool) -> dict[str, Any]:
    """The compact result (§2).

    The answer as plain prose, every disclosure that is set, the listed hits in
    the compact shape, and the parallel lists those hits do not carry, cut to
    them. Null, empty, zero and false keys are dropped.
    """
    n = len(resp.particles)
    out = _cut(resp.model_dump(mode="json"), listed, n)
    for key in (*_CARRIED_BY_HIT, "answer_attribution"):
        out.pop(key, None)
    # ``coverage_gaps`` is every PENDING or FAILED entry in the store,
    # whatever the question, so its ids can outweigh the rest of the result.
    # Compact keeps the signal as a count; ``detail="full"`` keeps the ids.
    out["coverage_gaps_count"] = len(out.pop("coverage_gaps", None) or [])
    if resp.answer_attribution is not None:
        out["answer"] = strip_attribution_tags(resp.answer)
    out["particles"] = [_compact_hit(resp, i, structural=structural) for i in listed]
    # An engine that predates ``sources`` sends none; say so rather than let a
    # missing ``source`` read as a hit with no source.
    if listed and len(resp.sources) != n:
        out["sources_unavailable"] = True
    return {
        k: v
        for k, v in out.items()
        if k in ("answer", "particles") or v not in (None, [], {}, 0, False, "")
    }
