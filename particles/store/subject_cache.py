"""In-memory cache for subject resolution.

Two disjoint caches:

* **Positive resolutions** — ``(store scope, name) → Subject``. Scoped per store
  so a Subject row resolved against store A is never handed back to store B
  (which would leave a dangling subject id in store B's ``particle_subjects``).
  Invalidated on subject mutation.
* **Negative live-lookup misses** — ``name → expiry``. A live-authority search
  miss (e.g. Wikidata has no entity for "the user's hamster") is a fact about
  the *authority*, not about any store, so these are process-global and
  unscoped — which is what lets concurrent scratch stores share one miss instead
  of each paying the fruitless call. They **survive** subject mutations (only a
  positive resolution can go stale) and expire only by TTL.

Lives in ``store/`` so both ``store/subject_store.py`` (which invalidates on
mutation) and ``ingest/subject_resolver.py`` (which populates on resolution)
depend on it as a shared leaf — ``store`` never reaches up into ``ingest``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from particles.config import get_config
from particles.core.schema import Subject

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class CacheEntry:
    subject: Subject | None
    expires_at: float


#: Positive resolutions, keyed by :func:`make_key` (store scope + name).
_cache: dict[str, CacheEntry] = {}

#: Negative live-lookup misses, keyed by lower-cased name; value is the expiry.
_negative: dict[str, float] = {}


def store_scope(session: AsyncSession) -> str:
    """Stable per-store identity for scoping positive cache entries.

    Keyed by the session's bound engine: there is exactly one engine object per
    store handle (cached in ``particles.db._engines``), so every session of a
    store shares a scope and two distinct stores never collide — including two
    in-memory SQLite stores that share the ``:memory:`` URL but are separate
    databases behind separate engine objects.
    """
    bind = session.get_bind()
    url = getattr(bind, "url", None)
    return f"{url}#{id(bind)}" if url is not None else str(id(bind))


def make_key(session: AsyncSession, name: str) -> str:
    """Positive-cache key for ``name`` resolved in ``session``'s store."""
    return f"subject:{store_scope(session)}:{name.lower()}"


def cache_get(key: str) -> CacheEntry | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    if time.monotonic() > entry.expires_at:
        del _cache[key]
        return None
    return entry


def cache_set(key: str, subject: Subject | None) -> None:
    ttl = get_config().subjects.wikidata_cache_ttl_seconds
    _cache[key] = CacheEntry(subject=subject, expires_at=time.monotonic() + ttl)


def negative_get(name: str) -> bool:
    """True if ``name`` is a cached, unexpired live-lookup miss."""
    key = name.lower()
    expires_at = _negative.get(key)
    if expires_at is None:
        return False
    if time.monotonic() > expires_at:
        del _negative[key]
        return False
    return True


def negative_set(name: str) -> None:
    """Record that every applicable live authority searched ``name`` and missed."""
    ttl = get_config().subjects.wikidata_cache_ttl_seconds
    _negative[name.lower()] = time.monotonic() + ttl


def clear(*, keep_negative: bool = False) -> None:
    """Drop cached resolutions. Called by store mutations and from tests.

    Subject mutations (alias merge, ref removal, merge / split / delete) pass
    ``keep_negative=True``: a mutation can change which Subject a name resolves
    to — invalidating a positive entry — but it cannot turn a live-authority
    miss into a hit, and the resolver consults the local store (cascade Step 1)
    *before* the negative cache, so a surviving negative entry can never mask a
    name that a mutation just made resolvable locally. The resolver reset
    (tests, module reload) drops everything.
    """
    _cache.clear()
    if not keep_negative:
        _negative.clear()
