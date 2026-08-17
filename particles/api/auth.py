"""Bearer-token auth for the FastAPI app.

``PARTICLES_API_KEY`` is read on every request (not captured at module
import), so it can be rotated without restarting the process — and tests
can set it via ``monkeypatch.setenv``. ``dev-key`` (the default) disables
bearer auth for local development — but, since, that affordance
is **bounded to loopback**: the app refuses to start when bearer auth is
disabled and ``api.bind_host`` is not a loopback address
(``enforce_fail_closed_on_startup``), and ``_verify_key`` refuses any
request from a non-loopback peer while the dev-key skip is active. The
key comparison is constant-time (``hmac.compare_digest``) to avoid the
byte-by-byte timing side channel the previous ``!=`` exposed.

The per-request loopback gate is **proxy-aware when configured**:
behind a reverse proxy the network peer is the proxy's address, so by default a
remote client would read as local. Set ``api.trusted_proxies`` to the proxy
IPs/CIDRs you trust and the gate then honours ``X-Forwarded-For`` (the nearest
untrusted hop) to identify the real client. With ``trusted_proxies`` empty (the
default) the header is ignored and the raw peer is used, exactly as before. The
gate remains a backstop for the authoritative ``bind_host`` startup check, not a
substitute for it (§(a)).
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from particles.config import get_config
from particles.secrets import get_particles_api_key

log = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)
_DEV_KEY = "dev-key"


def _is_loopback_host(host: str) -> bool:
    """True iff ``host`` is a loopback address (``127.0.0.0/8``, ``::1``, ``localhost``).

    Fail-closed: anything that does not *provably* resolve to a loopback IP
    returns False — non-IP hostnames, the all-interfaces wildcards
    (``0.0.0.0`` / ``::``), and the empty string. We do not perform DNS
    resolution, so a custom hostname that happens to point at loopback is
    treated as non-loopback (the safe direction for a security gate).
    """
    h = host.strip().lower()
    if h == "localhost":
        return True
    # Tolerate a bracketed IPv6 literal ("[::1]").
    h = h.removeprefix("[").removesuffix("]")
    # Drop an IPv6 zone id ("fe80::1%eth0") — irrelevant to loopback classification.
    h = h.split("%", 1)[0]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _ip_in_networks(host: str, networks: list[str]) -> bool:
    """True iff ``host`` is, or falls within, any of the IP / CIDR ``networks``.

    Plain IPs and CIDR ranges both work (``"127.0.0.1"``, ``"10.0.0.0/8"``).
    A non-IP ``host`` and any unparseable network string are treated as no
    match — the safe direction for a trust check.
    """
    try:
        ip = ipaddress.ip_address(host.strip())
    except ValueError:
        return False
    for net in networks:
        try:
            if ip in ipaddress.ip_network(net.strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def _real_client_host(request: Request) -> str | None:
    """The originating client's address, trusting ``X-Forwarded-For`` only behind a trusted proxy.

    Returns the raw network peer's host by default. When ``api.trusted_proxies``
    is configured **and** the immediate peer is one of those proxies, walk
    ``X-Forwarded-For`` right-to-left, skipping further trusted-proxy hops, and
    return the nearest untrusted hop — the real external client. Returns ``None``
    when there is no network peer. With no trusted proxies configured (the
    default) this is exactly ``request.client.host``, so a spoofable
    ``X-Forwarded-For`` is never consulted unless the operator opts in.
    """
    client = request.client
    if client is None:
        return None
    peer = client.host
    trusted = get_config().api.trusted_proxies
    if not trusted or not _ip_in_networks(peer, trusted):
        return peer
    forwarded = request.headers.get("X-Forwarded-For", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        if not _ip_in_networks(hop, trusted):
            return hop
    # Every forwarded hop is itself trusted → the request originated within the
    # trusted chain; treat the peer (a trusted proxy) as the client.
    return peer


def _client_is_loopback(request: Request) -> bool:
    """True iff the request's real client is loopback (or there is no peer).

    "Real client" honours ``api.trusted_proxies`` / ``X-Forwarded-For`` via
    ``_real_client_host``. A ``None`` peer means no network peer at all
    (in-process / ASGI transports without a client tuple) — not a remote
    caller, so it is treated as local.
    """
    host = _real_client_host(request)
    if host is None:
        return True
    return _is_loopback_host(host)


def _verify_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    api_key = get_particles_api_key()
    if api_key == _DEV_KEY:
        # Bearer auth is disabled (local-dev affordance). Bound to loopback
        #: never serve an unauthenticated request from a
        # non-loopback peer, even if the bind-host startup check was somehow
        # bypassed (e.g. uvicorn's --host drifted from api.bind_host). This is
        # the backstop; the startup check is authoritative. Refuse-to-serve
        # semantics → 503: in dev-key mode there is no credential the caller
        # could present to succeed, so 401/403 would mislead.
        if not _client_is_loopback(request):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Refusing to serve an unauthenticated request from a "
                    "non-loopback client. Set PARTICLES_API_KEY to enable "
                    "authenticated access, or bind the API to loopback for "
                    "local development."
                ),
            )
        return  # skip auth in dev mode (loopback only)
    # Constant-time compare (§(b)): encode to bytes so a non-ASCII
    # credential can't raise from compare_digest's str (ASCII-only) path.
    if creds is None or not hmac.compare_digest(
        creds.credentials.encode("utf-8"), api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


AuthDep = Annotated[None, Depends(_verify_key)]


def _verify_read_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Bearer gate for read routes — enforced only when ``api.require_auth_for_reads``.

    The bearer gates the *write* verbs; the read surface is
    unauthenticated by default, so once a real ``PARTICLES_API_KEY`` is set the
    reads stay open (security review F2). When an operator opts in via
    ``api.require_auth_for_reads=true`` this delegates to :func:`_verify_key`, so
    every read route then carries the same fail-closed bearer posture as the
    writes (including the dev-key loopback skip and constant-time compare). With
    the flag ``False`` (the default) it is a no-op, preserving the historical
    read posture. The three highest-value reads (``/query``, ``/events``,
    ``/digest``) take the unconditional :data:`AuthDep` instead, so they are
    gated by the bearer regardless of this flag.
    """
    if not get_config().api.require_auth_for_reads:
        return
    _verify_key(request, creds)


#: Conditional bearer gate for read routes (security review F2). Enforces the
#: bearer only when ``api.require_auth_for_reads`` is set; a no-op otherwise.
ReadAuthDep = Annotated[None, Depends(_verify_read_key)]


def verify_request_bearer(request: Request) -> None:
    """Apply the bearer gate to a raw Starlette request.

    The same fail-closed posture as ``_verify_key`` (the ``Depends`` gate the
    route handlers use), but callable outside FastAPI's dependency-injection
    machinery — for the curation-PWA static-bundle mount, which is
    a mounted Starlette sub-application, not a path operation, so it cannot take
    an ``AuthDep``. Parses the ``Authorization: Bearer`` header off the raw
    request and delegates to ``_verify_key`` so the dev-key loopback skip, the
    constant-time compare, and the 401/503 semantics live in one place. Raises
    ``HTTPException`` (401 / 503) exactly like ``_verify_key``.
    """
    header = request.headers.get("Authorization", "")
    creds: HTTPAuthorizationCredentials | None = None
    scheme, _, token = header.partition(" ")
    if scheme.lower() == "bearer" and token:
        creds = HTTPAuthorizationCredentials(scheme=scheme, credentials=token)
    _verify_key(request, creds)


def enforce_fail_closed_on_startup() -> None:
    """Refuse to start when bearer auth is disabled and the bind is not loopback.

    The fail-closed contract, clause (a): the ``"dev-key"`` skip is a
    deliberate local-development affordance, but it must not silently serve
    unauthenticated traffic beyond loopback. When bearer auth is disabled
    (``PARTICLES_API_KEY`` unset/``dev-key``) **and** ``api.bind_host`` is
    not a loopback address, raise — aborting startup with a non-zero exit —
    rather than boot open.

    Called from the FastAPI ``lifespan`` startup. The loopback dev case
    (auth disabled, bound to loopback) is allowed and only *warned* about
    (``warn_if_dev_auth_in_use``); this function returns silently for it.
    """
    if get_particles_api_key() != _DEV_KEY:
        return
    bind_host = get_config().api.bind_host
    if _is_loopback_host(bind_host):
        return
    raise RuntimeError(
        "Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is "
        f"unset or 'dev-key') and api.bind_host={bind_host!r} is not a "
        "loopback address, so the API would serve every endpoint — including "
        "the write verbs — unauthenticated beyond loopback. Set "
        "PARTICLES_API_KEY to a secret value, or bind the API to a loopback "
        "interface (api.bind_host=127.0.0.1) for local development."
    )


def warn_if_dev_auth_in_use() -> bool:
    """Log a WARNING when bearer auth is disabled (PARTICLES_API_KEY unset/dev-key).

    Called from the FastAPI startup event so operators see this in server
    logs at boot — easy to miss in a long-running dev session, easy to
    surface in any deployment that doesn't ship its log lines through grep.

    Returns True iff the warning fired (kept as a return value for testability).
    """
    if get_particles_api_key() != _DEV_KEY:
        return False
    log.warning(
        "PARTICLES_API_KEY is unset or set to %r — bearer-token authentication "
        "is DISABLED. This is intended for local development only. Set "
        "PARTICLES_API_KEY to a secret value before deploying or exposing "
        "the API on a non-loopback interface.",
        _DEV_KEY,
    )
    return True
