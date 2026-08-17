"""``particle_show`` / ``particles_list`` / ``particle_search``.

Routed through the ``Backend`` seam: in-process locally, against
the canonical engine when one is configured. The subject-name / provenance-URI
enrichment of ``particle_show`` is store-only, so over a remote engine it
degrades gracefully (empty subjects, ``None`` URIs) — the same boundary the
CLI's ``particle show`` documents.
"""

from __future__ import annotations

from typing import Any

from particles.core.schema import ContestedBadge
from particles.core.status import Status


async def particle_show(particle_id: str) -> dict[str, Any]:
    """Show one particle with its provenance, subjects, and tags.

    Args:
        particle_id: Full UUID or unambiguous prefix (≥ 8 chars).

    Returns:
        Dict with ``particle`` (Pydantic JSON), ``subjects`` (list of
        ``{id, canonical_name}``), and ``provenance`` (list of
        ``{type, corpus_entry_id, uri_r, source_type, snapshot_id}``).
        Raises if the prefix matches zero or multiple particles.
    """
    from particles.api.client import get_backend

    detail = await get_backend().particle_detail(particle_id)
    return {
        "particle": detail.particle.model_dump(mode="json"),
        "subjects": detail.subjects,
        "provenance": detail.provenance,
    }


async def particles_list(
    status: str | None = None,
    subject_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List particles, optionally filtered by status or subject.

    Lightweight alternative to ``query`` and ``lint`` when an agent
    just needs the IDs of particles in a particular state — e.g. the
    open INCONSISTENCY queue, RETRACTED particles for cleanup, or
    every ACTIVE particle linked to a subject. Embeddings are never
    included; responses are designed to stay within MCP tool-result
    size caps.

    Args:
        status: Optional status filter — one of
            ``ACTIVE``, ``SUPERSEDED``, ``RETRACTED``,
            ``PROVENANCE_STALE``, ``INCONSISTENCY``. Unset returns
            particles in any status.
        subject_id: Optional subject UUID. When set, only
            particles linked to this subject via the
            ``particle_subjects`` join table are returned.
        limit: Maximum particles to return (default 50). Must be > 0.
        offset: Number of particles to skip before returning results
            (default 0). Combine with ``limit`` to page through the
            full set.

    Returns:
        Dict with ``particles`` — a list of summary dicts containing
        ``id``, ``status``, ``content``, ``confidence``,
        ``subject_ids``, ``asserted_at``, and ``status_reason``. Each
        entry's ``contested`` key is the id of an open INCONSISTENCY
        referencing it, else null; ``contested_bases``
         rides beside it with the fired basis labels —
        composed by the same composer the ``query`` tool uses, so all
        three bases (stance / divergence / inconsistency) are evaluated
        and the two surfaces on one server always agree. Plus
        ``limit``/``offset`` echoed back so the agent can drive
        pagination. Embeddings are intentionally omitted.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if offset < 0:
        raise ValueError("offset must be zero or a positive integer.")

    # Validate the status filter here so the error message is identical across
    # transports; the backend re-validates server-side.
    if status is not None:
        try:
            Status(status)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in Status)
            raise ValueError(f"Unknown status {status!r}. Allowed: {allowed}.") from exc

    from particles.api.client import get_backend
    from particles.config import get_config

    backend = get_backend()
    particles = await backend.particles_list(
        status=status, subject_id=subject_id, limit=limit, offset=offset
    )
    # contested marker — an open INCONSISTENCY references this id.
    backrefs = await backend.inconsistency_backrefs()
    # contested_bases rides beside the id-valued key (no key
    # repurposed), composed by the one composer every other recall
    # surface calls — this listing hand-rolled a one-basis version of its own
    # until, which made it disagree with `query` on the same server.
    # Gated by the §7 kill switch (off restores the pre-badge shape exactly)
    # and skipped entirely when off, so the composition is never paid for.
    badge_enabled = get_config().contestedness.badge_enabled
    badges: list[ContestedBadge | None] = (
        await backend.contested_badges([p.id for p in particles]) if badge_enabled else []
    )

    entries: list[dict[str, Any]] = []
    for i, p in enumerate(particles):
        entry: dict[str, Any] = {
            "id": p.id,
            "status": p.status.value,
            "status_reason": p.status_reason.value if p.status_reason else None,
            "content": p.content,
            "confidence": p.confidence.value,
            "subject_ids": list(p.subject_ids),
            "asserted_at": p.asserted_at.isoformat(),
            "contested": backrefs.get(p.id),
        }
        if badge_enabled:
            badge = badges[i]
            entry["contested_bases"] = list(badge.bases) if badge is not None else None
        entries.append(entry)

    return {
        "limit": limit,
        "offset": offset,
        "particles": entries,
    }


async def particle_search(fingerprint: str, limit: int = 50) -> dict[str, Any]:
    """List particles sharing a context fingerprint.

    Args:
        fingerprint: Full 64-char SHA-256 hex or a prefix (≥ 8 chars).
        limit: Maximum particles to return (default 50, capped at 200 to
            keep responses within MCP per-tool-result token caps —
            fingerprint matches share full ``content`` strings).

    Returns:
        Dict with ``fingerprint`` (the queried value) and ``particles``
        (list of ``{id, status, content}`` summaries).
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if limit > 200:
        raise ValueError("limit must be 200 or less.")

    fp = fingerprint.lower().strip()
    if len(fp) < 8:
        raise ValueError("Fingerprint prefix must be at least 8 hex characters.")
    if any(ch not in "0123456789abcdef" for ch in fp):
        raise ValueError("Fingerprint must be hexadecimal.")

    from particles.api.client import get_backend

    rows = await get_backend().particles_by_fingerprint(fp, limit=limit)

    return {
        "fingerprint": fp,
        "particles": [{"id": r.id, "status": r.status.value, "content": r.content} for r in rows],
    }
