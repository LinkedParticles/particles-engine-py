"""Shared helpers for subject authorities.

Hosts the per-namespace rate limiter (moved verbatim from
``subject_resolver``) and ``PatternAuthority`` — the generic recognize-only
authority used for namespaces whose only resolution step is an in-name regex
(``numista``, ``km_catalog``, ``isbn``, ``doi``).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from particles.core.schema import ApplicabilityClause, ExternalRef

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.ingest.authorities.registry import AuthorityResolution


# ---------------------------------------------------------------------------
# Rate limiter (per namespace) — moved verbatim from subject_resolver
# ---------------------------------------------------------------------------


@dataclass
class _RateLimiter:
    requests_per_second: float
    _min_interval: float = field(init=False)
    _last: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._min_interval = 1.0 / self.requests_per_second

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


_limiters: dict[str, _RateLimiter] = {}


def get_limiter(namespace: str, requests_per_second: float) -> _RateLimiter:
    """Return the rate limiter for ``namespace``, creating it on first use.

    Each live authority owns one limiter keyed by its ``NAMESPACE`` at its
    configured rate. Generalizes the old wikidata-only ``_limiter`` helper.
    """
    lim = _limiters.get(namespace)
    if lim is None:
        lim = _RateLimiter(requests_per_second=requests_per_second)
        _limiters[namespace] = lim
    return lim


def reset_limiters() -> None:
    """Drop all rate limiters so they are re-created with fresh config.

    Called by ``subject_resolver.clear_cache`` (tests, module reload).
    """
    _limiters.clear()


# ---------------------------------------------------------------------------
# PatternAuthority — recognize-only authority (regex over the name)
# ---------------------------------------------------------------------------


class PatternAuthority:
    """A recognize-only :class:`SubjectAuthority` backed by one regex.

    Replaces a single ``_NAMESPACE_PATTERNS`` row: ``recognize`` runs the
    pattern over the extracted name and emits ``ExternalRef(namespace, id)``.
    There is no live lookup (``LIVE = False``; ``resolve`` returns ``None``).

    For parity with the old ``_detect_namespace_pattern`` the recognized
    ``ExternalRef`` carries **no** ``uri`` (``uri`` defaults to ``None``); the
    canonical URI is available separately via :meth:`uri_for` (the IRI-template capability).
    """

    LIVE = False

    def __init__(
        self,
        *,
        namespace: str,
        pattern: re.Pattern[str],
        priority: int,
        uri_template: str | None = None,
        applicability: list[ApplicabilityClause] | None = None,
        default_link_confidence: float = 1.0,
    ) -> None:
        self.NAMESPACE = namespace
        self.PRIORITY = priority
        self.DEFAULT_LINK_CONFIDENCE = default_link_confidence
        self.APPLICABILITY = applicability or []
        self._pattern = pattern
        self._uri_template = uri_template

    def uri_for(self, external_id: str) -> str | None:
        if self._uri_template is None:
            return None
        return self._uri_template.format(id=external_id)

    def recognize(self, name: str) -> ExternalRef | None:
        m = self._pattern.search(name)
        if m:
            # No uri (parity with the old _detect_namespace_pattern).
            return ExternalRef(namespace=self.NAMESPACE, id=m.group(1))
        return None

    async def resolve(
        self,
        session: AsyncSession,
        name: str,
        *,
        particle_content: str | None,
        domain: str | None,
    ) -> AuthorityResolution | None:
        return None

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        return None
