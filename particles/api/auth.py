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

**A keyless engine serves reads only**. Under the dev-key skip every
route that changes the store refuses (503, :data:`KEYLESS_WRITE_REFUSAL`) even
from loopback and whatever token arrives, because loopback is no evidence of
who is calling: another local user, a sandboxed process and an agent with a
shell all reach it. ``AGENT_GATE`` and ``OPERATOR_GATE`` refuse it alike.

**Each key stands for a principal with a role**. A keyed engine
accepts ``PARTICLES_API_KEY``, which is an operator key for
``curation.operator_identity``, and the per-client keys in ``api.keys_file``
(:mod:`particles.api._keys`). Every gate resolves the request to a
:class:`Principal` and binds it (:func:`current_principal`, and the event
actor). ``AGENT_GATE`` admits an agent or operator key, ``OPERATOR_GATE`` only
an operator key, and a valid key of the wrong role gets 403 naming the role it
needs. Gates and handlers ask :meth:`Principal.can` for a :class:`Capability`,
never for a role, so a later permission model replaces one table.
:func:`require_operator_credential` is the inline form for a route that writes
on one branch only.

**Every gate resolves before any session.** A route names its gate in the
decorator (``dependencies=[AGENT_GATE]``), which FastAPI resolves before the
handler's parameter dependencies, so a refused request opens no session and
takes no store lock (security review 2026-10-25, Q12).

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
header still names the page's hostname. Under the dev-key skip, and on any
engine bound to loopback, only the loopback names and ``api.allowed_hosts`` are
served; on a wider bind with a real key the list, when set, bounds every path
(security review 2026-10-26, W10). ``HostBoundMiddleware`` applies the same
check to the routes that carry no gate (``/health``, ``/openapi.json``,
``/app``), and refuses before the body is read what a route's gate would refuse
whatever the body says (:func:`pre_body_refusal`, W8).

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
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from particles.api import _keys
from particles.api._keys import Role
from particles.config import get_config
from particles.corpus.harvest_marks import HARVEST_DEPOSITORS
from particles.secrets import get_particles_api_key
from particles.store.event_store import bind_http_actor

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


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


#: The client a forwarded hop that is not an address stands for: never loopback,
#: never an address, and one rate-limit bucket (security review 2026-10-27-1, G45).
_UNREADABLE_HOP = "unreadable-forwarded-hop"


def _forwarded_hop_address(hop: str) -> str | None:
    """The address an ``X-Forwarded-For`` hop names, or ``None`` when it names none.

    A proxy may write the client with its port (``203.0.113.7:4711``) or an IPv6
    client in brackets (``[2001:db8::7]``, ``[2001:db8::7]:443``); each names the
    address inside. Anything else that is not a bare address names no client.
    """
    candidate = hop
    if hop.startswith("["):
        inside, sep, rest = hop[1:].partition("]")
        if not sep or (rest and not (rest.startswith(":") and rest[1:].isdigit())):
            return None
        candidate = inside
    elif hop.count(":") == 1:
        host, _, port = hop.partition(":")
        if not port.isdigit():
            return None
        candidate = host
    return candidate if _is_ip(candidate) else None


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
    # Every field line, in order, combined as RFC 9110 §5.3 combines them: a
    # proxy that adds its own line rather than extending the client's left the
    # client's line first, and reading only that one let the client name the
    # hop (security review 2026-10-28-1, O49).
    forwarded = ",".join(request.headers.getlist("X-Forwarded-For"))
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        address = _forwarded_hop_address(hop)
        if address is None:
            # A hop that is not an address names no client the engine can
            # judge. Skipping it read the next hop to its left, which the
            # client wrote, so a ported hop let a remote caller pass as
            # loopback (security review 2026-10-27-1, G45). Used verbatim it
            # was a bucket any caller could spell (W78), so it is the one
            # fixed, non-loopback sentinel.
            return _UNREADABLE_HOP
        if not _ip_in_networks(address, trusted):
            return address
    # Every forwarded hop is itself trusted → the request originated within the
    # trusted chain; treat the peer (a trusted proxy) as the client.
    return peer


def _client_is_loopback(request: Request) -> bool:
    """True iff the request's real client is provably loopback.

    "Real client" honours ``api.trusted_proxies`` / ``X-Forwarded-For`` via
    ``_real_client_host``. A request with no peer is not provably loopback:
    uvicorn reports none for a UNIX-domain socket connection, so a reverse
    proxy on that socket forwarded every caller as peerless, and the dev-key
    bound read each one as local (security review 2026-10-25-2, R1). It now
    reads as remote; an in-process caller with no client tuple holds no
    network position worth trusting either.
    """
    host = _real_client_host(request)
    if host is None:
        return False
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


def _served_on_loopback(server: object) -> bool | None:
    """Whether the local address a request arrived on is loopback; ``None`` when unknown.

    The ASGI scope's ``server`` names it. A UNIX-domain socket, an in-process
    caller and a non-IP name (a test client's ``testserver``) give no address.
    """
    if not isinstance(server, tuple | list) or not server:
        return None
    try:
        address = ipaddress.ip_address(str(server[0]).split("%", 1)[0])
    except ValueError:
        return None
    return address.is_loopback


def _host_rule_is_loopback(server: object) -> bool:
    """Whether a keyed engine answers only the loopback names for this request.

    True when the bind the startup checks judge is loopback, and also when the
    request itself arrived on a loopback address, since a launch the checks
    cannot read (``uvicorn --fd``, or a script calling ``uvicorn.run`` under a
    wider ``api.bind_host``) is still a loopback listener a rebinding page can
    reach (security review 2026-10-27-1, G1). An inherited socket or a UNIX
    socket whose request names no address counts as loopback too: a browser
    reaches one only through a local proxy.
    """
    bind_host, _ = _startup_bind(_uvicorn_cli_bind())
    if _is_loopback_host(bind_host):
        return True
    served = _served_on_loopback(server)
    if served is not None:
        return served
    return bind_host in (_UNPROVEN_BIND, _UNIX_SOCKET_BIND)


def _host_is_allowed(host: str, *, dev_key: bool, server: object = None) -> bool:
    """True iff the ``Host`` header value names a host this engine answers on.

    The loopback names always pass. Beyond them, ``api.allowed_hosts`` decides
    (exact names, or a ``*.`` suffix). With the list empty, none passes under
    the dev-key skip, because there the ``Host`` is the only thing that tells a
    rebinding page from a local client. Under a real key none passes either
    while the engine listens on loopback, since minting a key at runtime must
    not open the reads to a rebinding page (security review 2026-10-26, W10);
    any name passes only on a wider listener (:func:`_host_rule_is_loopback`,
    judged from ``server``, the local address the request arrived on).
    """
    name = _host_without_port(host).lower()
    if _is_loopback_host(name):
        return True
    allowed = get_config().api.allowed_hosts
    if not allowed:
        return not dev_key and not _host_rule_is_loopback(server)
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


def engine_is_keyless() -> bool:
    """No credential is configured: no ``PARTICLES_API_KEY`` and no live per-client key.

    The one definition of a keyless engine. A key file that exists
    but cannot be used counts as holding keys (:func:`_keys.has_live_keys`), so
    it never relaxes the startup bind gate or reopens the read-only mode. The
    per-request bounds read :func:`_no_usable_key` instead, so the same file
    never lifts them either (security review 2026-10-26, W9).
    """
    return get_particles_api_key() == _DEV_KEY and not _keys.has_live_keys()


def _no_usable_key() -> bool:
    """No credential the engine can check: the keyless request bounds apply.

    Unlike :func:`engine_is_keyless`, a key file that cannot be used counts as
    no key here, so a damaged file never lifts the loopback-client and
    loopback-``Host`` bounds from an engine with no usable key, while writes
    stay refused (security review 2026-10-26, W9).
    """
    return get_particles_api_key() == _DEV_KEY and not _keys.has_usable_live_keys()


class Capability(StrEnum):
    """What a principal may do; gates ask for one, never for a role."""

    READ = "read"
    AGENT_WRITE = "agent_write"
    OPERATOR_WRITE = "operator_write"
    #: Name a depositor, author, URI, source type or project other than one's own.
    ATTRIBUTE_OTHERS = "attribute_others"
    #: Put the reserved memory-harvest tags on one's own deposits.
    HARVEST_MARKS = "harvest_marks"


#: The closed role table. A later permission model replaces this table, and only
#: this table: every gate reads :meth:`Principal.can`.
_ROLE_CAPABILITIES: dict[Role, frozenset[Capability]] = {
    Role.READER: frozenset({Capability.READ}),
    Role.AGENT: frozenset({Capability.READ, Capability.AGENT_WRITE}),
    Role.OPERATOR: frozenset(Capability),
}


@dataclass(frozen=True)
class Principal:
    """The caller a request's key stands for.

    ``id`` is the label the engine records for the caller: depositor, author,
    asserter, event actor. Handlers pass the Engine that string, never this
    object, so no Engine module imports the Surface.
    """

    id: str
    role: Role
    key_id: str
    project: str | None = None
    #: A harvester key's recorded client directories; ``None`` on
    #: every other principal and on a harvester key that records none.
    memory_roots: tuple[str, ...] | None = None
    archive_roots: tuple[str, ...] | None = None

    def can(self, capability: Capability) -> bool:
        if capability in _ROLE_CAPABILITIES[self.role]:
            return True
        # A harvest is defined by its depositor, so the capability follows from
        # the principal's name, which minting it needed --reserved-principal for
        # (rule 5).
        return (
            capability is Capability.HARVEST_MARKS
            and self.role is Role.AGENT
            and self.id in HARVEST_DEPOSITORS
        )


#: The key id the legacy ``PARTICLES_API_KEY`` resolves under.
LEGACY_KEY_ID = "env"


def _legacy_principal() -> Principal:
    """``PARTICLES_API_KEY`` is an operator key for the operator principal."""
    return Principal(
        id=get_config().curation.operator_identity, role=Role.OPERATOR, key_id=LEGACY_KEY_ID
    )


#: The principal the current request's gate resolved; ``None`` for an
#: unauthenticated request. Set by every gate, so inline checks and handlers in
#: the same request read it.
_current_principal: ContextVar[Principal | None] = ContextVar(
    "particles_http_principal", default=None
)


def current_principal() -> Principal | None:
    """The principal the current request's gate resolved, or ``None``."""
    return _current_principal.get()


def _bind_principal(principal: Principal | None) -> None:
    _current_principal.set(principal)
    if principal is None:
        bind_http_actor(None, None)
    else:
        bind_http_actor(principal.id, principal.key_id)


def _resolve_principal(creds: HTTPAuthorizationCredentials | None) -> Principal:
    """The principal ``creds`` authenticates on a keyed engine; 401 for anything else.

    The legacy key is compared first, in constant time, as before. A per-client
    key is looked up by its id and its secret's digest compared (:mod:`_keys`).
    A revoked key gets the same 401 as an unknown one: telling the caller
    "revoked" would confirm that a stolen key once worked. A key file that
    cannot be used answers 503 for per-client keys only; the legacy key keeps
    working, so a damaged file never locks the operator out.
    """
    api_key = get_particles_api_key()
    if api_key != _DEV_KEY and _bearer_matches(api_key, creds):
        return _legacy_principal()
    token = creds.credentials if creds is not None else ""
    if token.startswith(_keys.KEY_PREFIX):
        try:
            record = _keys.lookup(token)
        except _keys.KeyFileUnusable as exc:
            # The reason names the file's path, its owner's uid and the fix,
            # which are the operator's to read, never an unauthenticated
            # caller's (security review 2026-10-27-1, G43).
            log.error("per-client keys are unavailable: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Per-client keys are unavailable; the engine's log says why. "
                    "PARTICLES_API_KEY still works while the key file is repaired."
                ),
            ) from exc
        if record is not None:
            return Principal(
                id=record.principal,
                role=record.role,
                key_id=record.key_id,
                project=record.project,
                memory_roots=record.memory_roots,
                archive_roots=record.archive_roots,
            )
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def request_holds_a_key(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> bool:
    """Whether the request's bearer authenticates a principal. Never refuses.

    For an ungated route that shows more to a caller holding a key (``/health``'s
    daemon block; security review 2026-10-28-2, AA50). An unusable key file
    counts as no key here rather than a 503, so the liveness probe still answers.
    """
    if creds is None:
        return False
    try:
        _resolve_principal(creds)
    except HTTPException:
        return False
    return True


def _role_refusal(principal: Principal, action: str, needs: str) -> HTTPException:
    """403 for a valid key whose role cannot do ``action``.

    403, not 401: the key is good and the request is still refused. The detail
    names the fix; the web app and the plugin show it.
    """
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"This key ({principal.role.value}, principal {principal.id}) cannot {action}, "
            f"which needs {needs}."
        ),
    )


def request_bound_refusal(request: Request) -> tuple[int, str] | None:
    """The status and reason a request is refused before any route runs, else ``None``.

    Applied to every path by :class:`particles.api._middleware.HostBoundMiddleware`
    and again inside each gate. With no usable key (:func:`_no_usable_key`, an
    unusable key file included) the real client must be loopback (503) and the
    ``Host`` must be a loopback name or an ``api.allowed_hosts`` entry (403).
    With a real key a set ``api.allowed_hosts`` is enforced, and with the list
    empty an engine bound to, or served on, loopback still answers only the
    loopback names (:func:`_host_is_allowed`). A request with no ``Host`` header passes: every
    browser sends one, so its absence is not a rebinding page.
    """
    dev_key = _no_usable_key()
    if dev_key and not _client_is_loopback(request):
        return status.HTTP_503_SERVICE_UNAVAILABLE, _NON_LOOPBACK_REFUSAL
    host = request.headers.get("host")
    server = request.scope.get("server")
    if host is not None and not _host_is_allowed(host, dev_key=dev_key, server=server):
        if dev_key:
            reason = (
                f"Refusing a request for host {host!r} while bearer auth is "
                "disabled. Open the engine at http://127.0.0.1 or "
                "http://localhost, list the name in api.allowed_hosts, or set "
                "PARTICLES_API_KEY."
            )
        elif not get_config().api.allowed_hosts:
            reason = (
                f"Refusing a request for host {host!r}: the engine is bound to loopback "
                "and api.allowed_hosts is empty, so it answers only http://127.0.0.1 and "
                "http://localhost. List the name in api.allowed_hosts."
            )
        else:
            reason = f"Refusing a request for host {host!r}: it is not in api.allowed_hosts."
        return status.HTTP_403_FORBIDDEN, reason
    return None


def _cross_site_refusal(reason: str) -> str:
    return (
        f"Refusing a cross-site browser request ({reason}). A web page on "
        "another site cannot drive this engine. Open the web UI from the "
        "engine itself (/app), or list a trusted browser origin in "
        "api.allowed_origins."
    )


def _refuse_cross_site(request: Request) -> None:
    """403 a cross-site browser request (security review 2026-10-23, critic's gap)."""
    reason = _cross_site_reason(request)
    if reason is None:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_cross_site_refusal(reason))


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
) -> Principal | None:
    """The bearer gate: the principal a keyed engine's request authenticates as.

    On a keyless engine the request is served only within the loopback and
    same-origin bounds and has no principal.
    """
    if engine_is_keyless():
        _bound_dev_key_request(request)
        return None
    refusal = request_bound_refusal(request)
    if refusal is not None:
        raise HTTPException(status_code=refusal[0], detail=refusal[1])
    return _resolve_principal(creds)


#: The refusal every write route gives on a keyless engine.
KEYLESS_WRITE_REFUSAL = (
    "This engine has no key (no PARTICLES_API_KEY and no per-client key), so it "
    "serves reads only. Restart it with PARTICLES_API_KEY set to a secret value, "
    "or mint a key on its host with `particles engine keys create`, and send the "
    "key as the bearer token (PARTICLES_ENGINE_TOKEN for the CLI or MCP server "
    "with engine.base_url set, the token field in the web app or the Obsidian "
    "plugin). The local CLI, hooks and MCP server write the store in-process "
    "and need no key."
)

#: The operator-scope routes' refusal under the dev-key skip. Since it
#: is the same text as every other write's: on a keyless engine the operator /
#: agent distinction no longer changes the answer.
OPERATOR_DEV_KEY_REFUSAL = KEYLESS_WRITE_REFUSAL


def _refuse_keyless_write() -> None:
    if engine_is_keyless():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=KEYLESS_WRITE_REFUSAL,
        )


def require_write_credential() -> None:
    """Refuse a write on a keyless engine, or from a key that cannot write.

    Any loopback process reaches a keyless engine (another local user, a
    sandboxed process, an agent with a shell), and a write names its own
    provenance, which the engine reads as recorded. So a keyless engine has no
    write route at all, not even from loopback, and no token changes that,
    because there is no key to check one against.

    503, not 401/403, for the reason ``_verify_key`` gives for its own dev-key
    refusal: no credential the caller could present would succeed. It also
    keeps the clients from misreading it, since they treat 401 as a bad token
    and 403 as a store with belief writes disabled. On a keyed engine the
    request's principal must hold ``AGENT_WRITE`` (403 otherwise). Callable
    inline in a gated route; :data:`AGENT_GATE` is the dependency form.
    """
    _refuse_keyless_write()
    principal = current_principal()
    if principal is None or not principal.can(Capability.AGENT_WRITE):
        raise _anonymous_or_role_refusal(principal, "write", "an agent or operator key")


def require_operator_credential() -> None:
    """Refuse an operator-scope act on a keyless engine, or from a non-operator key.

    The operator routes act with the operator's authority: an
    operator supersede or retract can retire any belief, and a review
    resolution decides between two claims and may write trust policy. Since
    this is where an operator key parts from every other key: the
    request's principal must hold ``OPERATOR_WRITE`` (403 otherwise). Callable
    inline for a route that writes on one branch only (``dry_run`` false,
    ``fix`` true); :data:`OPERATOR_GATE` is the dependency form.
    """
    _refuse_keyless_write()
    principal = current_principal()
    if principal is None or not principal.can(Capability.OPERATOR_WRITE):
        raise _anonymous_or_role_refusal(principal, "take this action", "an operator key")


def _anonymous_or_role_refusal(
    principal: Principal | None, action: str, needs: str
) -> HTTPException:
    if principal is None:
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return _role_refusal(principal, action, needs)


def _verify_write_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Gate for the agent writes: the request bounds, no write on a keyless engine, an agent key.

    Under the dev-key skip the bounds run first, so a non-loopback,
    rebinding or cross-site request keeps its own refusal, and every other
    request gets :data:`KEYLESS_WRITE_REFUSAL`. On a keyed engine the key must
    hold ``AGENT_WRITE``: an agent or operator key, never a reader's.
    """
    if engine_is_keyless():
        _bound_dev_key_request(request)
        _refuse_keyless_write()
    principal = _verify_key(request, creds)
    assert principal is not None  # a keyed engine always resolves one or raises
    if not principal.can(Capability.AGENT_WRITE):
        raise _role_refusal(
            principal, f"call {request.method} {request.url.path}", "an agent or operator key"
        )
    return principal


def _verify_operator_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Gate for operator-scope writes: never keyless, and only an operator key.

    On a keyless engine the bounds run first, as for the agent writes,
    so a non-loopback, rebinding or cross-site request keeps its own refusal.
    """
    if engine_is_keyless():
        _bound_dev_key_request(request)
        _refuse_keyless_write()
    principal = _verify_key(request, creds)
    assert principal is not None  # a keyed engine always resolves one or raises
    if not principal.can(Capability.OPERATOR_WRITE):
        raise _role_refusal(
            principal, f"call {request.method} {request.url.path}", "an operator key"
        )
    return principal


def _verify_read_key(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal | None:
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
    unconditional :data:`AUTH_GATE` instead, so they are gated by the bearer
    regardless of this flag. Every role may read. Without the
    flag, a bearer that does not resolve is ignored, as before, and the read is
    served unauthenticated.
    """
    if get_config().api.require_auth_for_reads:
        return _verify_key(request, creds)
    if engine_is_keyless():
        _bound_dev_key_request(request)
        return None
    refusal = request_bound_refusal(request)
    if refusal is not None:
        raise HTTPException(status_code=refusal[0], detail=refusal[1])
    if creds is not None:
        try:
            return _resolve_principal(creds)
        except HTTPException:
            pass
    # An unauthenticated read on a keyed engine: only the cross-site refusal applies.
    _refuse_cross_site(request)
    return None


async def _read_gate(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    _bind_principal(_verify_read_key(request, creds))


async def _auth_gate(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    _bind_principal(_verify_key(request, creds))


async def _agent_gate(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    _bind_principal(None)
    _bind_principal(_verify_write_key(request, creds))


async def _operator_gate(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    _bind_principal(None)
    _bind_principal(_verify_operator_key(request, creds))


#: The four route gates in decorator form (``dependencies=[AGENT_GATE]``).
#: FastAPI resolves a path operation's ``dependencies`` before its parameter
#: dependencies, so the gate refuses before any session or store lock is taken
#: (security review 2026-10-25, Q12). Each gate is async so the principal it
#: binds (:func:`current_principal`, and the event actor) is visible to the
#: handler and its inline checks: a sync dependency runs in a worker thread,
#: whose context changes never reach the handler. ``tests/test_write_route_gate.py``
#: pins that every route names one of them and that each write route names a
#: write gate.
READ_GATE = Depends(_read_gate)
AUTH_GATE = Depends(_auth_gate)
AGENT_GATE = Depends(_agent_gate)
OPERATOR_GATE = Depends(_operator_gate)

#: The gate functions, by which :func:`pre_body_refusal`'s caller finds a route's gate.
GATE_DEPENDENCIES: frozenset[object] = frozenset(
    {_read_gate, _auth_gate, _agent_gate, _operator_gate}
)

#: The capability each write gate asks for, and how its refusal names it.
_WRITE_GATE_NEEDS: dict[object, tuple[Capability, str]] = {
    _agent_gate: (Capability.AGENT_WRITE, "an agent or operator key"),
    _operator_gate: (Capability.OPERATOR_WRITE, "an operator key"),
}


def pre_body_refusal(request: Request, gate: object) -> tuple[int, str] | None:
    """The refusal ``gate`` gives ``request`` whatever its body says, else ``None``.

    FastAPI reads and parses a request's body, a multipart upload spooled to
    disk included, before it resolves the route's gate, so a request the gate
    refuses was parsed in full first (security review 2026-10-26, W8).
    :class:`particles.api._middleware.HostBoundMiddleware` asks this before the
    body is read. It refuses only what the gate refuses for every body: a
    cross-site browser request that holds no key, any write on a keyless
    engine, and, on a keyed engine, a bearer-gated request whose bearer does not
    resolve or whose key's role cannot use the route. The gate still runs and
    decides the rest.
    """
    if engine_is_keyless():
        reason = _cross_site_reason(request)
        if reason is not None:
            return status.HTTP_403_FORBIDDEN, _cross_site_refusal(reason)
        if gate in _WRITE_GATE_NEEDS:
            return status.HTTP_503_SERVICE_UNAVAILABLE, KEYLESS_WRITE_REFUSAL
        return None
    creds = _bearer_from_header(request)
    bearer_gated = gate is not _read_gate or get_config().api.require_auth_for_reads
    principal: Principal | None = None
    try:
        principal = _resolve_principal(creds)
    except HTTPException as exc:
        if bearer_gated:
            return exc.status_code, str(exc.detail)
    if creds is not None:
        request.scope[_SCOPE_PRINCIPAL] = principal
    if principal is None:
        # An open read with no key: the read gate's cross-site refusal.
        reason = _cross_site_reason(request)
        return None if reason is None else (status.HTTP_403_FORBIDDEN, _cross_site_refusal(reason))
    needs = _WRITE_GATE_NEEDS.get(gate)
    if needs is not None and not principal.can(needs[0]):
        refusal = _role_refusal(principal, f"call {request.method} {request.url.path}", needs[1])
        return refusal.status_code, str(refusal.detail)
    return None


def http_write_posture() -> str:
    """One line naming whether an engine started from this environment accepts writes.

    Read by ``particles config validate`` and the ``engine serve`` banner.
    It reads ``PARTICLES_API_KEY`` and the key file; it probes
    nothing.
    """
    if engine_is_keyless():
        return (
            "HTTP writes: refused (no PARTICLES_API_KEY and no per-client key; an "
            "engine started from this environment serves reads only)"
        )
    parts: list[str] = []
    try:
        live = _keys.live_keys()
    except _keys.KeyFileUnusable as exc:
        parts.append(f"per-client keys unusable ({exc})")
    else:
        if live:
            counts = {role: sum(1 for r in live if r.role is role) for role in Role}
            parts.append(
                "per-client keys ("
                + ", ".join(f"{counts[role]} {role.value}" for role in Role)
                + ")"
            )
    if get_particles_api_key() != _DEV_KEY:
        parts.append("the legacy PARTICLES_API_KEY (operator)")
    return "HTTP writes: a key required; accepted: " + " and ".join(parts)


def verify_request_bearer(request: Request) -> None:
    """Apply the bearer gate to a raw Starlette request.

    The same fail-closed posture as ``_verify_key`` (the ``Depends`` gate the
    route handlers use), but callable outside FastAPI's dependency-injection
    machinery — for the curation-PWA static-bundle mount, which is
    a mounted Starlette sub-application, not a path operation, so it cannot take
    a gate dependency. Parses the ``Authorization: Bearer`` header off the raw
    request and delegates to ``_verify_key`` so the dev-key loopback skip, the
    constant-time compare, and the 401/503 semantics live in one place. Raises
    ``HTTPException`` (401 / 503) exactly like ``_verify_key``.
    """
    _verify_key(request, _bearer_from_header(request))


def _bearer_from_header(request: Request) -> HTTPAuthorizationCredentials | None:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)
    return None


#: Where :func:`principal_for_request` caches its answer in the ASGI scope, so
#: the meter and the rate limiter resolve a request's key once.
_SCOPE_PRINCIPAL = "particles.principal"
_UNRESOLVED = object()


def principal_for_request(request: Request) -> Principal | None:
    """The principal ``request``'s bearer resolves to, or ``None``; never raises.

    Read before any gate runs, by the model-call meter and the rate limiter
    (:mod:`particles.api._rate_limit`), which key a request's bucket on its
    principal. A bearer that does not resolve, for any reason,
    reads as no principal here; the gate gives it its own refusal.
    """
    creds = _bearer_from_header(request)
    if creds is None:
        return None
    cached = request.scope.get(_SCOPE_PRINCIPAL, _UNRESOLVED)
    if cached is not _UNRESOLVED:
        return cached  # type: ignore[no-any-return]
    principal: Principal | None = None
    if not engine_is_keyless():
        try:
            principal = _resolve_principal(creds)
        except HTTPException:
            principal = None
    request.scope[_SCOPE_PRINCIPAL] = principal
    return principal


def request_holds_operator_credential(request: Request) -> bool:
    """True iff ``request``'s bearer resolves to an operator key.

    The one request whose model calls are not metered and whose probe verdicts
    are recorded (:class:`particles.api._rate_limit.ModelCallMeterMiddleware`;
    security review 2026-10-25-2, R2 and R5). Before every bearer
    holder was the operator; an agent or reader key is now metered like any
    other caller.
    """
    principal = principal_for_request(request)
    return principal is not None and principal.can(Capability.OPERATOR_WRITE)


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


def _startup_bind(cli_bind: str | None) -> tuple[str, str]:
    """The bind the startup checks judge, and where it was read from."""
    if cli_bind is None:
        return get_config().api.bind_host, "api.bind_host"
    return cli_bind, "the uvicorn command line"


def _startup_bind_is_loopback() -> bool:
    """Whether the bind the startup checks judge is a loopback address."""
    bind_host, _ = _startup_bind(_uvicorn_cli_bind())
    return _is_loopback_host(bind_host)


def startup_open_reads_warning() -> str | None:
    """:func:`open_reads_warning` for the bind this process was started with.

    ``engine serve`` prints the warning before it binds, but a raw ``uvicorn``
    launch never passed through it, so the lifespan logs this too, whatever
    launched the engine (security review 2026-10-24-2, N9). A UNIX-domain
    socket is reachable only from this host and is never warned about.
    """
    bind_host, _ = _startup_bind(_uvicorn_cli_bind())
    if bind_host == _UNIX_SOCKET_BIND:
        return None
    return open_reads_warning(bind_host)


#: The config a request on an address wider than the judged bind was last logged under.
_wide_bind_logged_for: object | None = None


def note_served_address(server: tuple[str, int | None] | None) -> None:
    """Log once per config when a request arrives on an address the startup checks did not judge.

    The startup checks read the bind from ``engine serve`` or the uvicorn
    command line, and fall back to ``api.bind_host`` for a launcher they cannot
    read, such as a script calling ``uvicorn.run`` or ``python -c`` (security
    review 2026-10-24, N9 and N11). Each request's ASGI scope names the local
    address it arrived on, so the first request on a non-loopback address under
    a loopback judgement says so: a keyless engine refuses every non-loopback
    client per request anyway, and a keyed one with open reads gets the
    open-reads warning it would have had at startup.
    """
    global _wide_bind_logged_for
    if server is None:
        return
    host = str(server[0])
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return
    config = get_config()
    if address.is_loopback or _wide_bind_logged_for is config:
        return
    _wide_bind_logged_for = config
    if not _startup_bind_is_loopback():
        return
    if engine_is_keyless():
        log.warning(
            "the engine received a request on %s, though the startup check read a "
            "loopback bind: this launcher binds wider than loopback. Every request "
            "from a non-loopback client is refused. Start the engine with `particles "
            "engine serve` or `uvicorn --host 127.0.0.1`.",
            host,
        )
        return
    warning = open_reads_warning(host)
    if warning is not None:
        log.warning("%s", warning)


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
    proves it is loopback, and so is a UNIX socket (``uvicorn --uds``), whose
    connections carry no peer, and a launch by another ASGI server
    (``_OPAQUE_LAUNCHERS``), whose command line this does not parse. The
    per-request bounds (:func:`request_bound_refusal`, applied on every path by
    the middleware) remain the backstop for a launcher this cannot see.

    Called from the FastAPI ``lifespan`` startup. The loopback dev case
    (auth disabled, bound to loopback) is allowed and only *warned* about
    (``warn_if_dev_auth_in_use``); this function returns silently for it.
    """
    if not engine_is_keyless():
        return
    cli_bind = _uvicorn_cli_bind()
    launcher = _launcher_name()
    if cli_bind is None and launcher in _OPAQUE_LAUNCHERS:
        raise RuntimeError(
            f"Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is unset, "
            f"empty or 'dev-key') and the engine was launched by {launcher!r}, whose bind "
            "this check cannot read, so nothing proves the API is reachable from "
            "loopback only. Start it with `particles engine serve` or `uvicorn`, or "
            "set PARTICLES_API_KEY to a secret value."
        )
    bind_host, source = _startup_bind(cli_bind)
    if bind_host == _UNIX_SOCKET_BIND:
        # uvicorn reports no peer for a UNIX-socket connection, so nothing tells
        # a local caller from one a proxy on the socket forwarded, and the
        # per-request bound refuses every peerless request (R1). Refuse here,
        # where the operator sees why, rather than serve nothing.
        raise RuntimeError(
            "Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is unset, "
            "empty or 'dev-key') and the engine was launched on a UNIX socket "
            "(uvicorn --uds), whose connections carry no peer address, so nothing "
            "proves a request came from this host rather than through a proxy on "
            "the socket. Set PARTICLES_API_KEY to a secret value, or bind a "
            "loopback interface (uvicorn --host 127.0.0.1)."
        )
    if _is_loopback_host(bind_host):
        return
    raise RuntimeError(
        "Refusing to start: bearer auth is disabled (PARTICLES_API_KEY is "
        f"unset, empty or 'dev-key') and the bind from {source}, {bind_host!r}, is not "
        "a loopback address, so the API would serve its read routes and raw "
        "corpus bytes unauthenticated beyond loopback (a keyless engine refuses "
        "every write). Set "
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
    if engine_is_keyless() or _is_loopback_host(bind_host):
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
    if not engine_is_keyless():
        return False
    log.warning(
        "PARTICLES_API_KEY is unset, empty or set to %r and no per-client key is "
        "minted: this engine serves reads only, and every write route answers 503. "
        "Set PARTICLES_API_KEY to a secret value, or mint a key with `particles "
        "engine keys create`, to accept writes, and before exposing the API on a "
        "non-loopback interface.",
        _DEV_KEY,
    )
    return True


def enforce_key_material_on_startup() -> None:
    """Refuse to start on key material the engine cannot use as intended.

    - A ``PARTICLES_API_KEY`` that begins with ``ptk_`` would still authenticate,
      since the legacy key is compared first, but a client reading its token's
      role from the prefix (``hook doctor``) would take it for a per-client key.
    - A key file that is group- or world-writable, owned by another user, or
      unparseable: serving would answer 503 to every per-client key, so the
      operator should see why before the engine starts.

    A key under a reserved name with no acknowledgement on record does not stop
    the start; it is named in the log, since it no longer authenticates.
    """
    if get_particles_api_key().startswith(_keys.KEY_PREFIX):
        raise RuntimeError(
            f"Refusing to start: PARTICLES_API_KEY begins with {_keys.KEY_PREFIX!r}, the "
            "prefix of a per-client key. Set it to any other secret value."
        )
    try:
        live = _keys.live_keys()
    except _keys.KeyFileUnusable as exc:
        raise RuntimeError(f"Refusing to start: the engine key file is unusable: {exc}") from exc
    for record in live:
        refusal = _keys.record_rule_refusal(record)
        if refusal is not None:
            # A hand-written record keys create would refuse, such as an
            # operator record under a principal that is not
            # curation.operator_identity (security review 2026-10-28-2, AA65).
            log.warning("engine key %s %s and is refused", record.key_id, refusal)
            continue
        if (
            _keys.is_harvester(record.role, record.principal)
            and record.memory_roots is None
            and record.archive_roots is None
        ):
            # A harvester key minted before its roots were recorded names no
            # file:// URI until it is minted again with them.
            log.warning(
                "engine key %s (%s) records no harvest roots, so its memory-file and "
                "archive deposits are refused; mint it again with `particles engine keys "
                "create --role agent --principal %s --reserved-principal --memory-root DIR "
                "--archive-root DIR` (on the client, with engine.base_url set, `particles "
                "init claude-code` prints the directories) and revoke this one",
                record.key_id,
                record.principal,
                record.principal,
            )
        if _keys.unacknowledged_reserved(record):
            # Minted before its name was reserved, or written by hand: it no
            # longer authenticates (security review 2026-10-27-1, G44).
            log.warning(
                "engine key %s (%s) holds a reserved name with no --reserved-principal "
                "on record and is refused; revoke it and mint it again with "
                "--reserved-principal if that name is intended",
                record.key_id,
                record.principal,
            )
