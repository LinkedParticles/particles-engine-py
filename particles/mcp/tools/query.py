"""``query`` — tag-aware semantic search + NL response.

Routed through the ``Backend`` seam: with no engine configured the
local backend runs the query in-process (today's behaviour); with
``engine.base_url`` set it runs on the canonical engine. The contested marker is fetched through the same backend so it reflects whichever
store served the query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from particles.core.schema import AudienceHint, QueryRequest, StructuralGroupBy


async def query(
    question: str | None = None,
    tags: list[str] | None = None,
    subject_id: str | None = None,
    min_confidence: float = 0.0,
    top_k: int = 40,
    audience: str = "GENERAL",
    summary: bool = False,
    as_of: str | None = None,
    predicate: str | None = None,
    object_eq: str | None = None,
    object_gt: str | None = None,
    object_lt: str | None = None,
    object_contains: str | None = None,
    count: bool = False,
    group_by: str | None = None,
    min_effective_confidence: float | None = None,
    list_predicates: bool = False,
) -> dict[str, Any]:
    """Run a tag-aware semantic query against the particle store.

    Args:
        question: Natural-language question. Optional since: omit it
            and set structural claim filters (or an aggregate / the predicate
            listing) for the deterministic modes — no embedding, no LLM call.
        tags: Subtree-expanded tag filters, repeatable.
        subject_id: Restrict to particles about this subject.
        min_confidence: Minimum confidence value (0.0–1.0).
        top_k: Number of particles to retrieve (1–200).
        audience: "GENERAL" | "EXPERT" | "REGULATORY".
        summary: When ``True``, each hit in ``particles`` is reduced to
            ``{id, content, confidence, effective_confidence,
            subject_ids}`` — provenance, tags, properties, and
            embeddings are dropped. The NL ``answer`` string and
            coverage gaps are unchanged. Use this when the agent only
            needs to surface the answer and pick particles to
            ``particle_show`` later. Default ``False`` preserves the
            full ``QueryResponse`` shape for backward compatibility.
        as_of: ISO-8601 past instant (a bare date means start of that
            day, UTC): answer what the store believed at T.
            Retired hits carry an ``as_of_notes`` entry (current status,
            retirement instant + basis, and the superseding belief);
            ``as_of_excluded_undatable`` discloses retirements the store
            cannot date. A future instant is rejected.
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

    Returns:
        The full ``QueryResponse`` as JSON — answer string plus the
        ranked particles, effective confidences, coverage gaps, and
        any truncation warning. A structural call returns rows
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
    req = QueryRequest(
        question=question,
        tags=list(tags or []),
        subject_id=subject_id,
        min_confidence=min_confidence,
        top_k=top_k,
        audience=AudienceHint(audience),
        as_of=as_of_dt,
        predicate=predicate,
        object_eq=object_eq,
        object_gt=object_gt,
        object_lt=object_lt,
        object_contains=object_contains,
        count=count,
        group_by=group_by_val,
        min_effective_confidence=min_effective_confidence,
        list_predicates=list_predicates,
    )
    backend = get_backend()
    resp = await backend.query(req)
    # mark each returned ACTIVE belief contested when an open
    # INCONSISTENCY references it, so the agent sees the §6/§6b conflict at
    # recall (query is otherwise ACTIVE-only and hides the ledger).
    backrefs = await backend.inconsistency_backrefs()
    for particle in resp.particles:
        particle.contested = backrefs.get(particle.id)
    if not summary:
        return resp.model_dump(mode="json")

    # Build a slim response: drop full Particle bodies, keep the
    # ranked-hit essentials plus the NL answer and coverage info.
    out = resp.model_dump(mode="json")
    # the composed badge list is parallel to particles when the
    # badge is enabled ([] when disabled — then no contested_bases key, so
    # the pre-badge slim shape is restored exactly).
    badges = resp.contested if len(resp.contested) == len(resp.particles) else []
    slim_hits: list[dict[str, Any]] = []
    for i, (particle, eff) in enumerate(
        zip(resp.particles, resp.effective_confidences, strict=True)
    ):
        hit: dict[str, Any] = {
            "id": particle.id,
            "content": particle.content,
            "confidence": particle.confidence.value,
            "effective_confidence": eff,
            "subject_ids": list(particle.subject_ids),
            "contested": particle.contested,
        }
        if badges:
            badge = badges[i]
            hit["contested_bases"] = list(badge.bases) if badge is not None else None
        slim_hits.append(hit)
    out["particles"] = slim_hits
    return out
