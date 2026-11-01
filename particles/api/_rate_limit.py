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

**The limiter counts model calls, not only requests** (security review
2026-10-25-2, R2). One admitted request could run a model pass over the whole
store (a semantic lint, a semantic curation queue, an ``LLM_JUDGE`` link
suggestion, a reconcile dry run). :class:`ModelCallMeterMiddleware` opens a
:func:`particles.llm.call_meter.metered_calls` scope for every request that
does not hold an operator key: the request's own token pays for its first
completion, and each further one spends a token from the same bucket. Once the
bucket is empty the rest of the pass degrades as for any failed call, and the
response says how many calls were refused (``X-Particles-Model-Calls-Refused``).
An operator key, which carries the operator's authority, is not metered by the
call; since an agent's or a reader's key is, like any other caller.
The same scope suspends the probe-verdict ledger, so a report served to a
request without the operator's authority leaves the store unchanged (R5).
Every bucket is a principal's when the request's key resolves to one, and an
address's otherwise (:func:`_client_bucket`).
"""

from __future__ import annotations

import ipaddress
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from particles.api.auth import (
    _real_client_host,
    principal_for_request,
    request_holds_operator_credential,
)
from particles.config import get_config, register_reset_hook
from particles.llm.call_meter import ModelCallBudgetSpent, current_meter, metered_calls
from particles.operations.probe_ledger import recording_suspended

# Per-client bucket state: key -> (tokens, last_refill_monotonic). Module-global
# and lock-free: ``enforce_rate_limit`` mutates it synchronously (no ``await``
# in the critical section), so the asyncio event loop cannot interleave two
# updates for the same key. Bounded by ``_MAX_BUCKETS`` (security review
# 2026-10-24, C9): a caller rotating source addresses grew it without limit.
_buckets: dict[str, tuple[float, float]] = {}

#: The most clients tracked at once. A bucket idle for a minute has refilled to
#: full and is the same as no bucket, so those are dropped first.
_MAX_BUCKETS = 10_000

#: The one bucket every loopback client shares.
_LOOPBACK_BUCKET = "loopback"

#: The one bucket every request with no network peer shares (a UNIX-socket
#: connection, an in-process transport): no address tells two of them apart.
_PEERLESS_BUCKET = "peerless"


def _bucket_key(host: str) -> str:
    """The rate-limit key for a client address.

    An IPv6 client is keyed by its ``/64``: one host is routinely handed a whole
    ``/64``, so keying by address let a caller take a fresh bucket per request
    (security review 2026-10-24, C9). An IPv4-mapped address is keyed as IPv4.
    Every loopback address (``127.0.0.0/8``, ``::1``) is one bucket: they are
    all this host, and a local process can bind or claim any of them, so keying
    each one apart gave every request a fresh allowance (security review
    2026-10-24-2, P8). A host that is not an address is prefixed ``host:``, so
    no client's spelling can equal a key principal's ``principal:`` bucket
    (security review 2026-10-26-1, W78).
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return f"host:{host}"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return _LOOPBACK_BUCKET
    if isinstance(ip, ipaddress.IPv6Address):
        return str(ipaddress.IPv6Network(f"{ip}/64", strict=False))
    return str(ip)


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


def _client_bucket(request: Request) -> str:
    """The bucket ``request`` spends from: its key's principal, else its real client.

    A request whose bearer resolves to a principal spends from that principal's
    bucket, wherever it comes from, so a key cannot multiply its allowance by
    changing address, and two keys of one principal share one bucket. On one
    host an agent's key no longer drains the bucket the operator's web app
    spends from. Every other request keys on its address, as
    before. The principal is resolved from the request itself, never from a
    gate's verdict, and a bearer that does not resolve reads as no principal,
    so a burst of bad keys spends from its address's bucket and never touches
    a principal's.
    """
    principal = principal_for_request(request)
    if principal is not None:
        return f"principal:{principal.id}"
    host = _real_client_host(request)
    return _PEERLESS_BUCKET if host is None else _bucket_key(host)


def enforce_rate_limit(request: Request) -> None:
    """Apply the per-client token bucket to ``request``; raise 429 when exhausted.

    No-op when ``api.rate_limit_per_minute`` is ``≤ 0`` (disabled). A request
    with no identifiable network peer spends from the one peerless bucket. The
    admitted token also pays for the request's first model call
    (:class:`_RequestMeter`).
    """
    rate = get_config().api.rate_limit_per_minute
    if rate <= 0:
        return
    if not _allow(_client_bucket(request), rate, time.monotonic()):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Rate limit exceeded for this client. This endpoint drives a "
                "paid model call per request; retry after a short wait or raise "
                "api.rate_limit_per_minute."
            ),
            headers={"Retry-After": "60"},
        )
    meter = current_meter()
    if isinstance(meter, _RequestMeter):
        meter.prepaid = True


async def _enforce_rate_limit_dep(request: Request) -> None:
    """:func:`enforce_rate_limit` as an async dependency.

    Run on the event loop rather than in the threadpool, so the bucket is only
    ever touched from one thread, the one the meter charges from.
    """
    enforce_rate_limit(request)


#: Dependency form for the unconditionally rate-limited endpoints (``/query``,
#: ``/extract``, ``/reindex``). The semantic ``/lint`` path calls
#: ``enforce_rate_limit`` inline instead, since only ``semantic=True`` drives the
#: LLM.
RateLimitDep = Annotated[None, Depends(_enforce_rate_limit_dep)]


class _RequestMeter:
    """Charges one request's model calls to its client's rate-limit bucket (R2)."""

    def __init__(self, request: Request) -> None:
        self._request = request
        #: Set when :func:`enforce_rate_limit` admitted the request: that token
        #: pays for the first completion.
        self.prepaid = False
        self.refused = 0

    def charge(self) -> None:
        if self.prepaid:
            self.prepaid = False
            return
        rate = get_config().api.rate_limit_per_minute
        if rate <= 0:
            return
        if _allow(_client_bucket(self._request), rate, time.monotonic()):
            return
        self.refused += 1
        raise ModelCallBudgetSpent(
            "this client's api.rate_limit_per_minute allowance is spent; the "
            "model call was not sent"
        )


#: The response header naming how many of a request's model calls were refused.
REFUSED_CALLS_HEADER = "X-Particles-Model-Calls-Refused"


class ModelCallMeterMiddleware:
    """Meter the model calls, and suspend the ledger, of a request without an operator key.

    A pure ASGI middleware, so the scope it opens is the context every handler
    and dependency of the request runs in.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if request_holds_operator_credential(request):
            await self.app(scope, receive, send)
            return
        meter = _RequestMeter(request)

        async def disclosing_send(message: Message) -> None:
            if message["type"] == "http.response.start" and meter.refused:
                headers = list(message.get("headers", []))
                headers.append((REFUSED_CALLS_HEADER.lower().encode(), str(meter.refused).encode()))
                message = {**message, "headers": headers}
            await send(message)

        with metered_calls(meter), recording_suspended():
            await self.app(scope, receive, disclosing_send)


# Clear bucket state whenever the config singleton is reset (CLI reload / tests).
register_reset_hook(reset_rate_limiter)
