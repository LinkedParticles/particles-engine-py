"""Subject authority plugin registry.

An **authority** owns how one external-ID namespace (``wikidata``, ``numista``,
``isbn``, …) is populated during subject resolution. This mirrors the extractor
 and exporter registries: a new ID source is added by
implementing :class:`SubjectAuthority` and registering it in
``_make_authorities()`` — not by editing ``subject_resolver``.

The resolver (``subject_resolver.resolve_subject``) drives the cascade and owns
every write (insert / alias-merge / cache); authorities only *recognize* an
in-name reference and optionally *resolve* a live lookup into a neutral
:class:`AuthorityResolution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from particles.config import get_config
from particles.core.schema import ApplicabilityClause, ExternalRef, Subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class AuthorityResolution:
    """Neutral result of a live authority lookup.

    Either ``existing`` is set — the ``(namespace, id)`` is already stored, and
    the resolver merges the searched name as an alias — or the new-subject
    fields (``external_ref`` + ``canonical_name`` + ``aliases`` +
    ``description``) describe a Subject the resolver builds and inserts. Keeping
    this authority-agnostic is what lets Wikidata stop being special-cased.
    """

    existing: Subject | None = None
    external_ref: ExternalRef | None = None
    canonical_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    description: str | None = None


@runtime_checkable
class SubjectAuthority(Protocol):
    """Protocol every subject authority must satisfy."""

    NAMESPACE: str
    PRIORITY: int  # arbitration: lower wins; registry order breaks ties
    LIVE: bool  # True if resolve() performs a live lookup
    DEFAULT_LINK_CONFIDENCE: float  # feeds ExternalRef.confidence
    APPLICABILITY: list[ApplicabilityClause]  # domain gating

    def uri_for(self, external_id: str) -> str | None:
        """Canonical IRI for an id in this namespace, or None if none exists."""
        ...

    def recognize(self, name: str) -> ExternalRef | None:
        """Recognise an in-name reference (regex / pattern step), or None."""
        ...

    async def resolve(
        self,
        session: AsyncSession,
        name: str,
        *,
        particle_content: str | None,
        domain: str | None,
    ) -> AuthorityResolution | None:
        """Live lookup. Recognize-only authorities return None."""
        ...

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        """Best canonical name for a recognised id (bare-local enrichment)."""
        ...


# ---------------------------------------------------------------------------
# Lazy singleton — mirrors particles.extraction.registry.get_extractors
# ---------------------------------------------------------------------------

_authorities: list[SubjectAuthority] | None = None


def get_authorities() -> list[SubjectAuthority]:
    """Return the registered authorities, ordered by PRIORITY (cached)."""
    global _authorities
    if _authorities is None:
        _authorities = _make_authorities()
    return _authorities


def clear_authorities() -> None:
    """Drop the cached registry and all rate limiters (config reload / tests)."""
    global _authorities
    _authorities = None
    from particles.ingest.authorities._shared import reset_limiters

    reset_limiters()


def _make_authorities() -> list[SubjectAuthority]:
    # defer: lazy-init — authority classes imported inside the factory so a
    # plugin file's own imports cannot break registry import. Mirrors
    # extraction.registry._make_extractors (root AGENTS.md § Deferred imports).
    import re

    from particles.ingest.authorities._shared import PatternAuthority
    from particles.ingest.authorities.wikidata import WikidataAuthority

    # Built in the historical _NAMESPACE_PATTERNS order; PRIORITY encodes that
    # order so recognize() arbitration is deterministic.
    raw: list[SubjectAuthority] = [
        PatternAuthority(
            namespace="numista",
            pattern=re.compile(r"\bN#\s*(\d+)\b", re.I),
            priority=10,
            uri_template="https://en.numista.com/catalogue/pieces{id}.html",
        ),
        PatternAuthority(
            namespace="km_catalog",
            pattern=re.compile(r"\bKM#\s*([\w\-]+)\b", re.I),
            priority=20,
        ),
        WikidataAuthority(),  # PRIORITY = 30
        PatternAuthority(
            namespace="isbn",
            pattern=re.compile(r"\bISBN[-\s]?([\d\-X]{10,17})\b", re.I),
            priority=40,
        ),
        PatternAuthority(
            namespace="doi",
            pattern=re.compile(r"\bDOI:\s*(\S+)", re.I),
            priority=50,
            uri_template="https://doi.org/{id}",
        ),
    ]

    # Per-authority config: enable/disable + priority override.
    cfg = get_config().authorities
    enabled: list[SubjectAuthority] = []
    for auth in raw:
        ac = cfg.get(auth.NAMESPACE)
        if ac is not None:
            if not ac.enabled:
                continue
            if ac.priority is not None:
                auth.PRIORITY = ac.priority
        enabled.append(auth)

    enabled.sort(key=lambda a: a.PRIORITY)
    return enabled


def is_applicable(authority: SubjectAuthority, domain: str | None) -> bool:
    """Whether an authority's live lookup should run for a claim's domain.

    Broad authorities (empty APPLICABILITY) and unknown domain (None) always
    apply — preserving today's "Wikidata runs everywhere". A MUST_NOT clause on
    the domain excludes; otherwise any MUST/SHOULD clause must match.
    """
    clauses = authority.APPLICABILITY
    if not clauses or domain is None:
        return True
    if any(c.keyword == "MUST_NOT" and c.domain_label == domain for c in clauses):
        return False
    positive = [c for c in clauses if c.keyword in ("MUST", "SHOULD")]
    if positive:
        return any(c.domain_label == domain for c in positive)
    return True
