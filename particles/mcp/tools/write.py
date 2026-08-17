"""Write tools for the read-write MCP surface.

The epistemic write verbs — ``particle_assert`` (the flagship), ``particle_supersede``,
``particle_retract``, the standalone cheap ``deposit_text``, and the ``link`` / ``tag``
mirrors. Since every tool routes through the ``Backend`` seam
(``get_backend()``): with no engine configured the local backend constructs and
reconciles in-process exactly as before; with ``engine.base_url`` set the write
is performed on the **canonical engine** over HTTP, so the laptop agent reads and
writes one store — the split-brain is closed. The §6.6-reconciling
orchestration lives in :mod:`particles.operations.agent_write` (the convergence
point both transports reach); this module is the thin tool layer:

* it **allowlist-checks** the target store locally (``mcp.write.enabled_stores``,
  default-deny, §5) via :func:`_resolve_store` — the gate that decides whether the
  local server *offers* the write at all (the engine independently gates
  server-side);
* it shapes the backend's :class:`~particles.operations.agent_write.AgentWriteResult`
  into the tool response dict — the §6.6 verdict is a first-class response, not an
  error.

Trust-, status-, and identity-bearing fields are constructed server-side (§4a) —
the caller never supplies them. Mutating another principal's beliefs
(supersede/retract) is own-beliefs-only unless ``mcp.write.allow_cross_asserter``;
operator-asserted (``HUMAN_REVIEW``) particles are never agent-mutable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from particles.config import get_config

if TYPE_CHECKING:
    from particles.operations.agent_write import AgentWriteResult


def _resolve_store(store: str | None) -> str:
    """Validate the target store against the write allowlist (§5, default-deny)."""
    allowed = get_config().mcp.write.enabled_stores
    if not allowed:
        raise ValueError(
            "MCP writes are disabled: mcp.write.enabled_stores is empty "
            "(default-deny). Add a store handle to enable."
        )
    if store is None:
        if len(allowed) == 1:
            return allowed[0]
        raise ValueError(
            f"Multiple write stores are enabled {allowed}; pass `store` to choose one."
        )
    if store not in allowed:
        raise ValueError(f"Store {store!r} is not write-enabled. Allowed: {allowed}.")
    return store


def _assert_result(result: AgentWriteResult, store: str) -> dict[str, Any]:
    """Shape the §6.6 verdict into the tool response — a conflict is first-class.

    ``asserted_particle_id`` is always the agent's belief id, never the
    INCONSISTENCY meta-particle's id; on a conflict that belief is the quarantined
    ``CONFLICT_PENDING`` particle and ``inconsistency_id`` names the separate
    INCONSISTENCY record. Keys are emitted conditionally to match the historical
    response shape exactly.
    """
    if result.asserted_particle_id is None:
        # Lower-trust drop — unreachable on a consensus write store.
        return {"asserted_particle_id": None, "store": store, "verdict": result.verdict}
    out: dict[str, Any] = {
        "asserted_particle_id": result.asserted_particle_id,
        "store": store,
        "verdict": result.verdict,
    }
    if result.status is not None:
        out["status"] = result.status
    if result.inconsistency_id is not None:
        out["inconsistency_id"] = result.inconsistency_id
    return out


async def particle_assert(
    content: str,
    subject_names: list[str],
    confidence: float,
    source_excerpt: str | None = None,
    corpus_entry_id: str | None = None,
    uncertainty_nature: str = "EPISTEMIC",
    tags: list[str] | None = None,
    store: str | None = None,
) -> dict[str, Any]:
    """Assert a belief into a write-enabled store (the flagship).

    One MCP call deposits the ``source_excerpt`` as the belief's provenance and
    asserts one particle through the §6.6 ladder. Trust-, status-, and
    identity-bearing fields are constructed server-side and cannot be supplied by
    the caller (§4a). A confirmed contradiction surfaces as an INCONSISTENCY
    (consensus mode) rather than replacing the existing belief.

    Args:
        content: The belief, claim-granular (§3.3).
        subject_names: Subject names (resolved to subjects server-side; agents
            speak names, not UUIDs).
        confidence: Self-reported confidence in [0, 1]; clamped to
            ``mcp.write.max_asserted_confidence``.
        source_excerpt: The conversational excerpt establishing the belief,
            deposited as its CONVERSATION provenance. Required unless
            ``corpus_entry_id`` is given.
        corpus_entry_id: Reuse an existing corpus entry as provenance instead of
            depositing a new excerpt (N beliefs, one entry).
        uncertainty_nature: "EPISTEMIC" (default) or "ALEATORY".
        tags: Optional tags.
        store: Target store handle; required only when several are write-enabled.

    Returns:
        ``{asserted_particle_id, store, verdict, status, [inconsistency_id]}`` —
        ``verdict`` is ``ASSERTED`` or ``INCONSISTENCY_RAISED``.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    result = await get_backend().particle_assert(
        content=content,
        subject_names=subject_names,
        confidence=confidence,
        source_excerpt=source_excerpt,
        corpus_entry_id=corpus_entry_id,
        uncertainty_nature=uncertainty_nature,
        tags=tags,
        store=handle,
    )
    return _assert_result(result, handle)


async def particle_supersede(
    supersedes_id: str,
    content: str,
    subject_names: list[str],
    confidence: float,
    source_excerpt: str | None = None,
    corpus_entry_id: str | None = None,
    uncertainty_nature: str = "EPISTEMIC",
    tags: list[str] | None = None,
    store: str | None = None,
) -> dict[str, Any]:
    """Revise a belief: assert a successor and move the prior particle to SUPERSEDED.

    The deliberate-revision path (the ledger, not an edit). The prior particle
    transitions ACTIVE → SUPERSEDED with reason ``EXPLICIT_SUPERSESSION`` first
    (so the successor does not spuriously conflict with the claim it replaces),
    then the successor is asserted with ``supersedes`` set. Own-beliefs-only;
    operator (HUMAN_REVIEW) particles are not agent-mutable (§6).

    Args:
        supersedes_id: Id of the prior ACTIVE belief to revise (own-beliefs-only).
        content: The successor belief, claim-granular (§3.3).
        subject_names: Subject names for the successor (resolved server-side).
        confidence: Self-reported confidence in [0, 1]; clamped to
            ``mcp.write.max_asserted_confidence``.
        source_excerpt: Conversational excerpt establishing the successor,
            deposited as its provenance. Required unless ``corpus_entry_id`` is given.
        corpus_entry_id: Reuse an existing corpus entry as provenance instead.
        uncertainty_nature: "EPISTEMIC" (default) or "ALEATORY".
        tags: Optional tags on the successor.
        store: Target store handle; required only when several are write-enabled.

    Returns the assertion result plus ``superseded_id``.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    result = await get_backend().particle_supersede(
        supersedes_id=supersedes_id,
        content=content,
        subject_names=subject_names,
        confidence=confidence,
        source_excerpt=source_excerpt,
        corpus_entry_id=corpus_entry_id,
        uncertainty_nature=uncertainty_nature,
        tags=tags,
        store=handle,
    )
    out = _assert_result(result, handle)
    out["superseded_id"] = supersedes_id
    return out


async def particle_retract(
    particle_id: str,
    reason: str,
    store: str | None = None,
) -> dict[str, Any]:
    """Retract a belief: ACTIVE → RETRACTED with reason ``EXPLICIT_RETRACTION``.

    Own-beliefs-only; operator (HUMAN_REVIEW) particles are not agent-mutable
    (§6). The free-text ``reason`` is recorded on the audit event.

    Args:
        particle_id: Id of the own ACTIVE belief to retract.
        reason: Free-text reason, recorded on the audit event (required, non-empty).
        store: Target store handle; required only when several are write-enabled.
    """
    from particles.api.client import get_backend

    if not reason.strip():
        raise ValueError("particle_retract requires a non-empty reason.")
    handle = _resolve_store(store)
    await get_backend().particle_retract(particle_id=particle_id, reason=reason, store=handle)
    return {"particle_id": particle_id, "store": handle, "verdict": "RETRACTED"}


async def deposit_text(
    text: str,
    tags: list[str] | None = None,
    store: str | None = None,
) -> dict[str, Any]:
    """Deposit conversational material into the corpus without asserting a belief.

    The standalone cheap deposit (CONVERSATION, zero extraction) for material
    worth archiving that does not yet warrant a belief. Attributed to the
    server-bound asserter identity. Returns the created ``corpus_entry_id`` /
    ``snapshot_id`` so a later ``particle_assert`` can cite it.

    Args:
        text: The conversational material to archive as a CONVERSATION entry.
        tags: Optional tags on the corpus entry.
        store: Target store handle; required only when several are write-enabled.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    entry_id, snapshot_id = await get_backend().deposit_text(text=text, tags=tags, store=handle)
    return {"corpus_entry_id": entry_id, "snapshot_id": snapshot_id, "store": handle}


def _parse_relation(relation_type: str) -> Any:
    """Parse a relation-kind string; only CO_EVIDENTIAL is active over MCP."""
    from particles.core.schema import RelationType

    try:
        rt = RelationType(relation_type)
    except ValueError as exc:
        raise ValueError(f"Unknown relation_type {relation_type!r}.") from exc
    if rt != RelationType.CO_EVIDENTIAL:
        raise ValueError(
            f"Only CO_EVIDENTIAL relations are emittable over MCP today; "
            f"got {relation_type!r}."
        )
    return rt


async def link_add(
    particle_a: str,
    particle_b: str,
    relation_type: str = "CO_EVIDENTIAL",
    store: str | None = None,
) -> dict[str, Any]:
    """Link two particles with a typed relation (CO_EVIDENTIAL today).

    A thin mirror of ``particles links add``; particles are addressed by full id.

    Args:
        particle_a: Full id of the first particle.
        particle_b: Full id of the second particle.
        relation_type: Relation kind; only ``CO_EVIDENTIAL`` is emittable over MCP today.
        store: Target store handle; required only when several are write-enabled.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    rt = _parse_relation(relation_type)
    rel = await get_backend().links_add(
        particle_a, particle_b, relation_type=rt.value, confidence=1.0
    )
    return {
        "store": handle,
        "particle_a": rel.particle_a,
        "particle_b": rel.particle_b,
        "relation_type": rel.relation_type.value,
        "verdict": "LINKED",
    }


async def link_remove(
    particle_a: str,
    particle_b: str,
    relation_type: str = "CO_EVIDENTIAL",
    store: str | None = None,
) -> dict[str, Any]:
    """Remove a typed relation between two particles. Mirror of ``particles links remove``.

    Args:
        particle_a: Full id of the first particle.
        particle_b: Full id of the second particle.
        relation_type: Relation kind; only ``CO_EVIDENTIAL`` is emittable over MCP today.
        store: Target store handle; required only when several are write-enabled.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    rt = _parse_relation(relation_type)
    removed = await get_backend().links_remove(particle_a, particle_b, relation_type=rt.value)
    return {
        "store": handle,
        "removed": removed,
        "verdict": "UNLINKED" if removed else "NO_SUCH_LINK",
    }


async def particle_tag(
    particle_id: str,
    tags: list[str],
    store: str | None = None,
) -> dict[str, Any]:
    """Add tags to a particle (idempotent). Mirror of ``particles particle tag``.

    Args:
        particle_id: Full id of the particle to tag.
        tags: Tags to add (idempotent — re-adding an existing tag is a no-op).
        store: Target store handle; required only when several are write-enabled.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    added = await get_backend().particle_tag(particle_id, list(tags))
    return {"store": handle, "particle_id": particle_id, "added": added}


async def particle_untag(
    particle_id: str,
    tags: list[str],
    store: str | None = None,
) -> dict[str, Any]:
    """Remove tags from a particle (idempotent). Mirror of ``particles particle untag``.

    Args:
        particle_id: Full id of the particle to untag.
        tags: Tags to remove (idempotent — removing an absent tag is a no-op).
        store: Target store handle; required only when several are write-enabled.
    """
    from particles.api.client import get_backend

    handle = _resolve_store(store)
    removed = await get_backend().particle_untag(particle_id, list(tags))
    return {"store": handle, "particle_id": particle_id, "removed": removed}
