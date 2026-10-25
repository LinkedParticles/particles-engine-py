# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

The operator-scope writes are stricter: ``OperatorAuthDep`` and
``require_operator_credential`` refuse them under the dev-key skip even from
loopback, because a producing agent with a shell on the engine host is a
loopback caller too, and those routes act with the operator's authority.

**A browser on the same host is a loopback peer too** (security review
2026-10-23, the completeness critic's gap). Any web page the operator opens can
send a form or multipart POST to ``http://127.0.0.1:8000`` with no preflight, and
under the dev-key skip nothing checked who sent it. Every gate here therefore
refuses a *cross-site browser request* (``Sec-Fetch-Site: cross-site`` /
``same-site``, or an ``Origin`` that is not the engine's own) unless it carries a
bearer that verified against a real key, which a cross-site page cannot attach
without a CORS preflight this engine never approves. Requests with neither header
are not from a browser (the CLI, the MCP backend, ``curl``); the same-origin web
UI sends ``Sec-Fetch-Site: same-origin``. ``api.allowed_origins`` names any other
browser origin the operator trusts.

**Every request names an allowed ``Host``** (security review 2026-10-24, N1). A
DNS-rebinding page reaches 127.0.0.1 under its own hostname, and a same-origin
``GET`` from it carries neither ``Origin`` nor, on a plain-http origin, any
fetch metadata, so it cannot be told from the CLI by its headers. Its ``Host``
header still names the page's hostname. Under the dev-key skip only the
loopback names and ``api.allowed_hosts`` are served; with a real key the list,
when set, bounds every path. ``HostBoundMiddleware`` applies the same check to
the routes that carry no gate (``/health``, ``/openapi.json``, ``/app``).

The dev-key loopback bounds apply to the read routes as well, whether or not
``api.require_auth_for_reads`` is set (security review D5), and the startup
gate reads the uvicorn command line's own bind when the engine is launched by a
raw ``uvicorn`` rather than ``engine serve`` (``_uvicorn_cli_bind``).
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import sys
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

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


#: ``Sec-Fetch-Site`` values a browser sends for a request the engine's own page
#: made (``same-origin``) or the user made directly, by typing the URL or opening
#: a bookmark (``none``). ``same-site`` is refused: another port on localhost is
#: the same site as the engine.
_SAFE_FETCH_SITES = frozenset({"same-origin", "none"})


def _host_without_port(host: str) -> str:
    """The hostname part of a ``Host`` header value (``[::1]:8000`` → ``[::1]``)."""
    h = host.strip()
    if h.startswith("["):
        return h.split("]", 1)[0] + "]"
    if h.count(":") == 1:
        return h.split(":", 1)[0]
    return h


def _is_browser_request(request: Request) -> bool:
    """True when the request carries a header only a browser sets on its own.

    ``Sec-Fetch-Site`` is a forbidden header name, so page script cannot set or
    remove it, and every current browser sends it. ``Origin`` covers an older
    browser's POST. The CLI, the MCP HTTP backend and ``curl`` send neither.
    """
    return "sec-fetch-site" in request.headers or "origin" in request.headers


def _cross_site_reason(request: Request) -> str | None:
    """Why ``request`` is a cross-site browser request, or ``None`` when it is not.

    ``Sec-Fetch-Site`` decides when present, since the browser sets it and a
    TLS-terminating proxy cannot make it disagree with what the user saw. Without
    it, an ``Origin`` must name the same host and port as the ``Host`` header. An
    origin listed in ``api.allowed_origins`` is always accepted.
    """
    origin = request.headers.get("origin")
    if origin is not None:
        allowed = {o.strip().rstrip("/").lower() for o in get_config().api.allowed_origins}
        if origin.strip().rstrip("/").lower() in allowed:
            return None
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        if site.strip().lower() in _SAFE_FETCH_SITES:
            return None
        return f"Sec-Fetch-Site: {site.strip()}"
    if origin is None:
        return None
    netloc = urlsplit(origin.strip()).netloc.lower()
    if netloc and netloc == request.headers.get("host", "").strip().lower():
        return None
    return f"Origin: {origin.strip()}"


def _host_is_allowed(host: str, *, dev_key: bool) -> bool:
    """True iff the ``Host`` header value names a host this engine answers on.

    The loopback names always pass. Beyond them, ``api.allowed_hosts`` decides
    (exact names, or a ``*.`` suffix). With the list empty, any name passes
    under a real key and none does under the dev-key skip, because there the
    ``Host`` is the only thing that tells a rebinding page from a local client.
    """
    name = _host_without_port(host).lower()
    if _is_loopback_host(name):
        return True
    allowed = get_config().api.allowed_hosts
    if not allowed:
        return not dev_key
    for entry in allowed:
        if entry.startswith("*.") and name.endswith(entry[1:]):
            return True
        if name == entry:
            return True
    return False


#: The 503 every dev-key bound gives a non-loopback client.
_NON_LOOPBACK_REFUSAL = (
    "Refusing to serve an unauthenticated request from a "
    "non-loopback client. Set PARTICLES_API_KEY to enable "
    "authenticated access, or bind the API to loopback for "
    "local development."
)


def request_bound_refusal(request: Request) -> tuple[int, str] | None:
    """The status and reason a request is refused before any route runs, else ``None``.

    Applied to every path by :class:`particles.api._middleware.HostBoundMiddleware`
    and again inside each gate. Under the dev-key skip the real client must be
    loopback (503) and the ``Host`` must be a loopback name or an
    ``api.allowed_hosts`` entry (403). With a real key only a set
    ``api.allowed_hosts`` is enforced. A request with no ``Host`` header passes:
    every browser sends one, so its absence is not a rebinding page.
    """
    dev_key = get_particles_api_key() == _DEV_KEY
    if dev_key and not _client_is_loopback(request):
        return status.HTTP_503_SERVICE_UNAVAILABLE, _NON_LOOPBACK_REFUSAL
    host = request.headers.get("host")
    if host is not None and not _host_is_allowed(host, dev_key=dev_key):
        if dev_key:
            reason = (
                f"Refusing a request for host {host!r} while bearer auth is "
                "disabled. Open the engine at http://127.0.0.1 or "
                "http://localhost, list the name in api.allowed_hosts, or set "
                "PARTICLES_API_KEY."
            )
        else:
            reason = f"Refusing a request for host {host!r}: it is not in api.allowed_hosts."
        return status.HTTP_403_FORBIDDEN, reason
    return None


def _refuse_cross_site(request: Request) -> None:
    """403 a cross-site browser request (security review 2026-10-23, critic's gap)."""
    reason = _cross_site_reason(request)
    if reason is None:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"Refusing a cross-site browser request ({reason}). A web page on "
            "another site cannot drive this engine. Open the web UI from the "
            "engine itself (/app), or list a trusted browser origin in "
            "api.allowed_origins."
        ),
    )


def _bound_dev_key_request(request: Request) -> None:
    """The bounds on a request served under the dev-key skip.

    1. The real client must be loopback. This is the backstop for a bind the
       startup gate could not see; refuse-to-serve semantics → 503, because in
       dev-key mode there is no credential the caller could present to succeed,
       so 401/403 would mislead.
    2. The request must name a loopback ``Host`` or an ``api.allowed_hosts``
       entry, browser markers or not (:func:`request_bound_refusal`). A
       DNS-rebinding page reaches 127.0.0.1 under its own hostname, and a
       same-origin ``GET`` from it carries no browser marker at all.
    3. A browser request must not be cross-site (:func:`_refuse_cross_site`).
    """
    refusal = request_bound_refusal(request)
    if refusal is not None:
        raise HTTPException(status_code=refusal[0], detail=refusal[1])
    if _is_browser_request(request):
        _refuse_cross_site(request)


def _bearer_matches(api_key: str, creds: HTTPAuthorizationCredentials | None) -> bool:
    """True iff ``creds`` carries ``api_key``.

    Constant-time compare (§(b)): encode to bytes so a non-ASCII
    credential can't raise from compare_digest's str (ASCII-only) path.
    """
    return creds is not None and hmac.compare_digest(
        creds.credentials.encode("utf-8"), api_key.encode("utf-8")
    )


def _verify_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    api_key = get_particles_api_key()
    if api_key == _DEV_KEY:
        # Bearer auth is disabled (local-dev affordance), so the request is
        # served only within the loopback and same-origin bounds.
        _bound_dev_key_request(request)
        return
    refusal = request_bound_refusal(request)
    if refusal is not None:
        raise HTTPException(status_code=refusal[0], detail=refusal[1])
    if not _bearer_matches(api_key, creds):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


AuthDep = Annotated[None, Depends(_verify_key)]


#: The refusal an operator-scope route gives while the dev-key skip is active.
OPERATOR_DEV_KEY_REFUSAL = (
    "Refusing an operator write without a credential. This route acts with the "
    "operator's authority (a review resolution, an operator supersede or "
    "retract, a curation gesture, a subject merge or split, a link or tag "
    "edit, a lint fix), so it is never served under the dev-key "
    "skip, not even from loopback. Restart the engine "
    "with PARTICLES_API_KEY set to a secret value and send that value as the "
    "bearer token (PARTICLES_ENGINE_TOKEN for the CLI's remote backend, the "
    "token field in the web app or the Obsidian plugin). The local CLI calls "
    "these operations in-process and needs no key."
)


def require_operator_credential() -> None:
    """Refuse an operator-scope write while bearer auth is the ``dev-key`` skip.

    The dev-key skip lets any loopback process call the write verbs
    with no credential, which is a fair trade for an agent's own writes but not
    for writes that act with the operator's authority: an operator supersede or
    retract can retire any belief, and a review resolution decides between two
    claims and may write trust policy. A producing agent with a shell on the
    same host reaches loopback, so on those routes loopback is no evidence of
    the operator. They require a real ``PARTICLES_API_KEY`` they
    never ride the loopback skip.

    503, not 401/403, for the reason ``_verify_key`` gives for its own dev-key
    refusal: no credential the caller could present would succeed, since the
    engine has no key to check it against. It also keeps the clients from
    misreading it, since they treat 401 as a bad token and 403 as a read-only
    store. Callable inline for a route that writes only when ``dry_run`` is
    false; :data:`OperatorAuthDep` is the dependency form.
    """
    if get_particles_api_key() == _DEV_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=OPERATOR_DEV_KEY_REFUSAL,
        )


def _verify_operator_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Bearer gate for operator-scope writes: :func:`_verify_key` without the dev-key skip."""
    require_operator_credential()
    _verify_key(request, creds)


#: Bearer gate for operator-scope belief writes and review resolution.
#: Like :data:`AuthDep`, except that the dev-key loopback skip does not apply.
OperatorAuthDep = Annotated[None, Depends(_verify_operator_key)]


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
    the flag ``False`` (the default) no bearer is asked for, but two bounds
    still hold. Under the dev-key skip the read gets the same loopback and
    same-origin bounds as a write (security review D5: a raw ``uvicorn --host
    0.0.0.0`` launch otherwise served every read to the network). With a real
    key a cross-site browser read is refused unless it carries the bearer. The
    three highest-value reads (``/query``, ``/events``, ``/digest``) take the
    unconditional :data:`AuthDep` instead, so they are gated by the bearer
    regardless of this flag.
    """
    if get_config().api.require_auth_for_reads:
        _verify_key(request, creds)
        return
    api_key = get_particles_api_key()
    if api_key == _DEV_KEY:
        _bound_dev_key_request(request)
        return
    refusal = request_bound_refusal(request)
    if refusal is not None:
        raise HTTPException(status_code=refusal[0], detail=refusal[1])
    if not _bearer_matches(api_key, creds):
        _refuse_cross_site(request)


#: Conditional bearer gate for read routes (security review F2). Enforces the
#: bearer only when ``api.require_auth_for_reads`` is set; otherwise it applies
#: the dev-key loopback bounds or, with a real key, the cross-site refusal.
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


#: What :func:`_uvicorn_cli_bind` returns for a bind it cannot classify (``--fd``).
_UNPROVEN_BIND = "<inherited socket>"
#: What :func:`_uvicorn_cli_bind` returns for a UNIX-domain socket, which only
#: local processes can reach.
_UNIX_SOCKET_BIND = "<unix socket>"


#: ASGI servers other than uvicorn whose command line the gate does not parse.
#: Each can bind every interface (``fastapi run`` does by default), so under the
#: dev-key skip a launch by one of them is refused (security review 2026-10-24,
#: N11). ``engine serve`` and ``uvicorn`` are the launchers the gate can read.
_OPAQUE_LAUNCHERS = frozenset({"gunicorn", "hypercorn", "granian", "daphne", "fastapi"})


def _launcher_name() -> str | None:
    """The program that launched this process (``python -m X`` names ``X``)."""
    if not sys.argv:
        return None
    launcher = Path(sys.argv[0])
    if launcher.name == "__main__.py":
        return launcher.parent.name
    return launcher.name


def _uvicorn_cli_bind() -> str | None:
    """The interface a ``uvicorn`` command line binds, or ``None`` when this is not one.

    The ASGI app cannot see uvicorn's socket, and uvicorn runs the lifespan
    before it binds, so ``api.bind_host`` was the only input the startup gate
    had, and a raw ``uvicorn particles.api.app:app --host 0.0.0.0`` left it at
    the loopback default (security review D5). When this process is the uvicorn
    CLI (including the ``--reload`` / ``--workers`` children, which inherit its
    ``sys.argv``), parse that command line with uvicorn's own parser, so its
    ``--host``, ``--uds``, ``--fd`` and ``UVICORN_*`` environment variables
    resolve exactly as uvicorn resolves them. ``engine serve`` is not the uvicorn
    CLI: it sets ``api.bind_host`` from the address it hands to uvicorn.
    """
    if _launcher_name() != "uvicorn":
        return None
    try:
        # Branch-local: only a uvicorn CLI launch needs uvicorn's parser.
        from uvicorn.main import main as uvicorn_main

        ctx = uvicorn_main.make_context("uvicorn", list(sys.argv[1:]), resilient_parsing=True)
    except Exception:  # noqa: BLE001 - an unparseable command line proves nothing
        return _UNPROVEN_BIND
    if ctx.params.get("uds"):
        return _UNIX_SOCKET_BIND
    if ctx.params.get("fd") is not None:
        return _UNPROVEN_BIND
    # An empty --host is not loopback: uvicorn hands "" to create_server, which
    # binds every interface (security review 2026-10-24, N11).
    host = ctx.params.get("host")
    return "127.0.0.1" if host is None else str(host)


def enforce_fail_closed_on_startup() -> None:
    """Refuse to start when bearer auth is disabled and the bind is not loopback.

    The fail-closed contract, clause (a): the ``"dev-key"`` skip is a
    deliberate local-development affordance, but it must not silently serve
    unauthenticated traffic beyond loopback. When bearer auth is disabled
    (``PARTICLES_API_KEY`` unset/``dev-key``) **and** the bind is not a
    loopback address, raise — aborting startup with a non-zero exit — rather
    than boot open.

    The bind is the uvicorn command line's own when the engine was launched
    by ``uvicorn`` (:func:`_uvicorn_cli_bind`), and ``api.bind_host`` otherwise
    (``engine serve`` sets it from the address it binds). A bind the gate cannot
    classify (``uvicorn --fd``) is refused under the dev-key skip, since nothing
    proves it is loopback, and so is a launch by another ASGI server
    (``_OPAQUE_LAUNCHERS``), whose command line this does not parse. The
    per-request bounds (:func:`request_bound_refusal`, applied on every path by
    the middleware) remain the backstop for a launcher this cannot see.

    Called from the FastAPI ``lifespan`` startup. The loopback dev case
    (auth disabled, bound to loopback) is allowed and only *warned* about
    (``warn_if_dev_auth_in_use``); this function returns silently for it.
    """
    if get_particles_api_key() != _DEV_KEY:
        return
    cli_bind = _uvicorn_cli_bind()
    launcher = _launcher_name()
    if cli_bind is None and launcher in _OPAQUE_LAUNCHERS:
        raise RuntimeError(
            f"Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is unset "
            f"or 'dev-key') and the engine was launched by {launcher!r}, whose bind "
            "this check cannot read, so nothing proves the API is reachable from "
            "loopback only. Start it with `particles engine serve` or `uvicorn`, or "
            "set PARTICLES_API_KEY to a secret value."
        )
    if cli_bind is None:
        bind_host, source = get_config().api.bind_host, "api.bind_host"
    else:
        bind_host, source = cli_bind, "the uvicorn command line"
    if bind_host == _UNIX_SOCKET_BIND or _is_loopback_host(bind_host):
        return
    raise RuntimeError(
        "Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is "
        f"unset or 'dev-key') and the bind from {source}, {bind_host!r}, is not "
        "a loopback address, so the API would serve every endpoint — including "
        "the write verbs — unauthenticated beyond loopback. Set "
        "PARTICLES_API_KEY to a secret value, or bind the API to a loopback "
        "interface (api.bind_host=127.0.0.1, uvicorn --host 127.0.0.1) for "
        "local development."
    )


def open_reads_warning(bind_host: str) -> str | None:
    """Why a non-loopback engine with a real key still serves its reads openly, else ``None``.

    With ``api.require_auth_for_reads`` false the read routes, raw corpus bytes
    among them, answer anyone who reaches the port. A mounted config that
    replaces the container's baked one turns the flag off silently (security
    review 2026-10-24, N9), so ``engine serve`` says so when it binds wider than
    loopback. The dev-key skip never gets here: it cannot start on such a bind.
    """
    if get_particles_api_key() == _DEV_KEY or _is_loopback_host(bind_host):
        return None
    if get_config().api.require_auth_for_reads:
        return None
    return (
        f"the engine binds {bind_host!r} and api.require_auth_for_reads is false, so "
        "every read route, raw corpus bytes included, is served without the bearer "
        "to anything that reaches the port. Set api.require_auth_for_reads: true "
        "(or PARTICLES_API_REQUIRE_AUTH_FOR_READS=true) unless that is intended."
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
