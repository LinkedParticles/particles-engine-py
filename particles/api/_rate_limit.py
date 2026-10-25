# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""In-process token-bucket rate limiting for the LLM/embedding-driving endpoints.

Security review F6: ``/query``, ``/extract``, ``/reindex`` and the semantic
``/lint`` path each drive a *paid* Anthropic completion and/or an embedding per
request, with no per-caller limit anywhere in ``particles/api/``. An
unauthenticated (``/query`` carries no bearer) or compromised caller could
otherwise burn tokens unbounded. ``enforce_rate_limit`` applies a per-client
token bucket keyed on the *real* client host (proxy-aware via
``particles.api.auth._real_client_host``) and raises ``429`` once the bucket is
empty.

This is a process-local *second* line of defense, not a substitute for the
documented reverse-proxy / fail-closed-bind posture. The cap is read
from ``api.rate_limit_per_minute`` at request time, so ``0`` (or any value
``≤ 0``) disables it cleanly and a reload takes effect without a restart. Local
CLI / MCP tools run in-process (``LocalBackend``) and never cross the HTTP
boundary unless a remote engine is configured, so enabling the limiter does not
throttle the default single-process workflow.

Bucket state is reset on every :func:`particles.config.reset_config` (registered
below) so a config reload — and every test that calls ``reset_config()`` — starts
from a clean limiter.
"""

from __future__ import annotations

import ipaddress
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from particles.config import get_config, register_reset_hook

# Per-client bucket state: key -> (tokens, last_refill_monotonic). Module-global
# and lock-free: ``enforce_rate_limit`` mutates it synchronously (no ``await``
# in the critical section), so the asyncio event loop cannot interleave two
# updates for the same key. Bounded by ``_MAX_BUCKETS`` (security review
# 2026-10-24, C9): a caller rotating source addresses grew it without limit.
_buckets: dict[str, tuple[float, float]] = {}

#: The most clients tracked at once. A bucket idle for a minute has refilled to
#: full and is the same as no bucket, so those are dropped first.
_MAX_BUCKETS = 10_000


def _bucket_key(host: str) -> str:
    """The rate-limit key for a client address.

    An IPv6 client is keyed by its ``/64``: one host is routinely handed a whole
    ``/64``, so keying by address let a caller take a fresh bucket per request
    (security review 2026-10-24, C9). An IPv4-mapped address is keyed as IPv4.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network(f"{ip}/64", strict=False))
    return host


def _make_room(now: float) -> None:
    """Drop buckets until a new one fits under ``_MAX_BUCKETS``.

    Full buckets go first, since forgetting them changes nothing (any bucket
    refills completely in a minute); then the least recently seen.
    """
    for key in [k for k, (_, last) in _buckets.items() if now - last >= 60.0]:
        del _buckets[key]
    if len(_buckets) < _MAX_BUCKETS:
        return
    for key, _ in sorted(_buckets.items(), key=lambda item: item[1][1])[
        : len(_buckets) - _MAX_BUCKETS + 1
    ]:
        del _buckets[key]


def reset_rate_limiter() -> None:
    """Drop all token-bucket state. Wired to ``reset_config`` (and used by tests)."""
    _buckets.clear()


def _allow(host: str, rate_per_minute: int, now: float) -> bool:
    """Token-bucket admission for ``host``: refill, then spend one token.

    Capacity and refill rate are both ``rate_per_minute`` (a one-minute burst,
    refilling to full over a minute). Returns ``True`` when a token was spent,
    ``False`` when the bucket is empty (caller should reject with 429).
    """
    capacity = float(rate_per_minute)
    if host not in _buckets and len(_buckets) >= _MAX_BUCKETS:
        _make_room(now)
    tokens, last = _buckets.get(host, (capacity, now))
    # Refill proportional to elapsed time, capped at capacity.
    tokens = min(capacity, tokens + (now - last) * rate_per_minute / 60.0)
    if tokens < 1.0:
        _buckets[host] = (tokens, now)
        return False
    _buckets[host] = (tokens - 1.0, now)
    return True


def enforce_rate_limit(request: Request) -> None:
    """Apply the per-client token bucket to ``request``; raise 429 when exhausted.

    No-op when ``api.rate_limit_per_minute`` is ``≤ 0`` (disabled) or when the
    request has no identifiable network peer (in-process / ASGI transports
    without a client tuple — treated as local, like the loopback gate).
    """
    rate = get_config().api.rate_limit_per_minute
    if rate <= 0:
        return
    # Deferred import: avoids a module-import cycle with auth.py and keeps the
    # proxy-aware client resolution in one place (auth._real_client_host).
    from particles.api.auth import _real_client_host

    host = _real_client_host(request)
    if host is None:
        return
    if not _allow(_bucket_key(host), rate, time.monotonic()):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Rate limit exceeded for this client. This endpoint drives a "
                "paid model call per request; retry after a short wait or raise "
                "api.rate_limit_per_minute."
            ),
            headers={"Retry-After": "60"},
        )


#: Dependency form for the unconditionally rate-limited endpoints (``/query``,
#: ``/extract``, ``/reindex``). The semantic ``/lint`` path calls
#: ``enforce_rate_limit`` inline instead, since only ``semantic=True`` drives the
#: LLM.
RateLimitDep = Annotated[None, Depends(enforce_rate_limit)]


# Clear bucket state whenever the config singleton is reset (CLI reload / tests).
register_reset_hook(reset_rate_limiter)
