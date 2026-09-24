# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSRF protection (particles/url_safety.py).

Covers the scheme allowlist, IP-literal handling, and DNS-resolved hostname
handling (with a mocked socket.getaddrinfo to keep tests offline + fast).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from particles.url_safety import (
    UnsafeUrlError,
    ValidatingTransport,
    format_connect_pin,
    resolve_and_pin,
    validate_fetch_url,
)

# ---------------------------------------------------------------------------
# Scheme allowlist
# ---------------------------------------------------------------------------


class TestSchemeAllowlist:
    def test_https_allowed(self) -> None:
        with patch("socket.getaddrinfo", return_value=_mock_addrinfo(["8.8.8.8"])):
            validate_fetch_url("https://example.com/x")

    def test_http_allowed(self) -> None:
        with patch("socket.getaddrinfo", return_value=_mock_addrinfo(["8.8.8.8"])):
            validate_fetch_url("http://example.com/x")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(UnsafeUrlError, match="scheme"):
            validate_fetch_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self) -> None:
        with pytest.raises(UnsafeUrlError, match="scheme"):
            validate_fetch_url("ftp://example.com/x")

    def test_gopher_scheme_rejected(self) -> None:
        with pytest.raises(UnsafeUrlError, match="scheme"):
            validate_fetch_url("gopher://example.com/x")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(UnsafeUrlError):
            validate_fetch_url("")

    def test_no_hostname_rejected(self) -> None:
        with pytest.raises(UnsafeUrlError, match="hostname"):
            validate_fetch_url("http:///path")


# ---------------------------------------------------------------------------
# Literal-IP rejection (no DNS needed)
# ---------------------------------------------------------------------------


class TestLiteralIpRejection:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.1",  # RFC 1918
            "172.16.0.1",  # RFC 1918
            "192.168.1.1",  # RFC 1918
            "169.254.169.254",  # link-local — cloud metadata!
            "169.254.0.1",  # link-local
            "0.0.0.0",  # unspecified
            "224.0.0.1",  # multicast
        ],
    )
    def test_blocked_ipv4(self, ip: str) -> None:
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            validate_fetch_url(f"http://{ip}/x")

    @pytest.mark.parametrize(
        "ip",
        [
            "[::1]",  # loopback
            "[fe80::1]",  # link-local
            "[fc00::1]",  # private (ULA)
        ],
    )
    def test_blocked_ipv6(self, ip: str) -> None:
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            validate_fetch_url(f"http://{ip}/x")

    def test_public_ipv4_literal_allowed(self) -> None:
        # 8.8.8.8 is Google Public DNS — a real-world public IP
        validate_fetch_url("https://8.8.8.8/x")


# ---------------------------------------------------------------------------
# DNS-resolved hostnames
# ---------------------------------------------------------------------------


def _mock_addrinfo(ips: Iterable[str]) -> list[Any]:
    """Build a fake getaddrinfo response. Tuple shape:
    (family, type, proto, canonname, sockaddr) where sockaddr[0] is the IP."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0)) for ip in ips]


class TestDnsResolution:
    def test_hostname_resolving_to_public_ip_allowed(self) -> None:
        with patch("socket.getaddrinfo", return_value=_mock_addrinfo(["93.184.216.34"])):
            validate_fetch_url("https://example.com/x")

    def test_hostname_resolving_to_loopback_blocked(self) -> None:
        # E.g. localhost.localdomain or a malicious DNS record pointing to 127.x
        with (
            patch("socket.getaddrinfo", return_value=_mock_addrinfo(["127.0.0.1"])),
            pytest.raises(UnsafeUrlError, match="private/reserved"),
        ):
            validate_fetch_url("https://attacker-controls-dns.example/x")

    def test_hostname_resolving_to_cloud_metadata_blocked(self) -> None:
        with (
            patch("socket.getaddrinfo", return_value=_mock_addrinfo(["169.254.169.254"])),
            pytest.raises(UnsafeUrlError, match="private/reserved"),
        ):
            validate_fetch_url("https://my-internal.example/")

    def test_multiple_resolutions_blocked_if_any_is_private(self) -> None:
        """DNS rebinding defense: if any returned IP is private, reject."""
        with (
            patch("socket.getaddrinfo", return_value=_mock_addrinfo(["8.8.8.8", "10.0.0.1"])),
            pytest.raises(UnsafeUrlError, match="private/reserved"),
        ):
            validate_fetch_url("https://mixed.example/")

    def test_unresolvable_hostname_fails_closed(self) -> None:
        with (
            patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")),
            pytest.raises(UnsafeUrlError, match="resolve"),
        ):
            validate_fetch_url("https://this-does-not-exist.invalid/")


# ---------------------------------------------------------------------------
# deposit_url integration — the validation is at the entry point
# ---------------------------------------------------------------------------


class TestDepositUrlIntegration:
    @pytest.mark.asyncio
    async def test_deposit_url_rejects_loopback(self, db_session: Any) -> None:
        from particles.corpus.deposit import deposit_url

        with pytest.raises(UnsafeUrlError):
            await deposit_url(db_session, "http://127.0.0.1:8080/data")

    @pytest.mark.asyncio
    async def test_deposit_url_rejects_cloud_metadata(self, db_session: Any) -> None:
        from particles.corpus.deposit import deposit_url

        with pytest.raises(UnsafeUrlError):
            await deposit_url(db_session, "http://169.254.169.254/computeMetadata/v1/")

    @pytest.mark.asyncio
    async def test_deposit_url_rejects_file_scheme(self, db_session: Any) -> None:
        from particles.corpus.deposit import deposit_url

        with pytest.raises(UnsafeUrlError):
            await deposit_url(db_session, "file:///etc/passwd")


# ---------------------------------------------------------------------------
# CGNAT (RFC 6598) — not caught by ipaddress.is_private (F-4-ip)
# ---------------------------------------------------------------------------


class TestCgnatBlocking:
    @pytest.mark.parametrize(
        "ip",
        [
            "100.64.0.1",  # bottom of 100.64.0.0/10
            "100.96.0.1",  # middle of the range
            "100.127.255.254",  # top of 100.64.0.0/10
        ],
    )
    def test_cgnat_ipv4_literal_blocked(self, ip: str) -> None:
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            validate_fetch_url(f"http://{ip}/x")

    def test_cgnat_ipv4_mapped_ipv6_literal_blocked(self) -> None:
        # The IPv4-mapped form ::ffff:100.64.0.0/106 must also be blocked.
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            validate_fetch_url("http://[::ffff:100.64.0.1]/x")

    def test_cgnat_boundary_just_below_is_allowed(self) -> None:
        # 100.63.255.255 is one below the CGNAT range — a normal public IP.
        validate_fetch_url("http://100.63.255.255/x")

    def test_cgnat_boundary_just_above_is_allowed(self) -> None:
        # 100.128.0.0 is one above the CGNAT range — a normal public IP.
        validate_fetch_url("http://100.128.0.0/x")


# ---------------------------------------------------------------------------
# ValidatingTransport — the authoritative connect-time gate.
# Unit-tested against a stub resolver; no real network.
# ---------------------------------------------------------------------------


def _resolver(mapping: dict[str, list[str]]) -> Any:
    """Build a stub resolver: host -> list of IP strings."""

    def resolve(host: str) -> list[Any]:
        return [ipaddress.ip_address(ip) for ip in mapping[host]]

    return resolve


class TestValidatingTransport:
    @pytest.mark.asyncio
    async def test_dns_rebinding_blocked_at_connect(self) -> None:
        """F-1: the pre-flight sees a public IP, but the transport re-resolves
        to an internal one at connect time — the connection is refused before
        any socket is opened (the validated IP *is* the connected IP)."""
        # Pre-flight (independent resolution) sees a public address and passes.
        with patch("socket.getaddrinfo", return_value=_mock_addrinfo(["93.184.216.34"])):
            validate_fetch_url("https://rebind.example/")

        # The transport, resolving again, gets the flipped internal address.
        transport = ValidatingTransport(resolve=_resolver({"rebind.example": ["10.0.0.1"]}))
        request = httpx.Request("GET", "https://rebind.example/")
        with patch.object(
            httpx.AsyncHTTPTransport, "handle_async_request", new=AsyncMock()
        ) as parent:
            with pytest.raises(UnsafeUrlError, match="private/reserved"):
                await transport.handle_async_request(request)
            parent.assert_not_called()  # never reached the network

    @pytest.mark.asyncio
    async def test_resolves_to_cloud_metadata_blocked(self) -> None:
        transport = ValidatingTransport(resolve=_resolver({"meta.example": ["169.254.169.254"]}))
        request = httpx.Request("GET", "http://meta.example/latest/meta-data/")
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            await transport.handle_async_request(request)

    @pytest.mark.asyncio
    async def test_resolves_to_cgnat_blocked(self) -> None:
        transport = ValidatingTransport(resolve=_resolver({"nat.example": ["100.64.1.2"]}))
        request = httpx.Request("GET", "https://nat.example/")
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            await transport.handle_async_request(request)

    @pytest.mark.asyncio
    async def test_unresolvable_host_fails_closed(self) -> None:
        # A resolver returning no addresses is treated as failure-closed.
        transport = ValidatingTransport(resolve=_resolver({"nowhere.example": []}))
        request = httpx.Request("GET", "https://nowhere.example/")
        with pytest.raises(UnsafeUrlError, match="did not resolve"):
            await transport.handle_async_request(request)

    @pytest.mark.asyncio
    async def test_public_host_pins_ip_and_preserves_host_and_sni(self) -> None:
        """A public host connects to the vetted IP, with the Host header and
        TLS SNI kept bound to the original hostname (connect-by-IP)."""
        captured: dict[str, Any] = {}

        async def cap_parent(self: Any, request: httpx.Request) -> httpx.Response:
            captured["url_host"] = request.url.host
            captured["sni"] = request.extensions.get("sni_hostname")
            captured["host_header"] = request.headers.get("host")
            return httpx.Response(200, request=request, content=b"ok")

        transport = ValidatingTransport(resolve=_resolver({"example.com": ["93.184.216.34"]}))
        request = httpx.Request("GET", "https://example.com/x")
        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", cap_parent):
            resp = await transport.handle_async_request(request)

        assert resp.status_code == 200
        assert captured["url_host"] == "93.184.216.34"  # connected by IP
        assert captured["sni"] == "example.com"  # TLS verifies the real host
        assert captured["host_header"] == "example.com"  # Host header preserved

    @pytest.mark.asyncio
    async def test_redirect_to_internal_blocked_at_hop(self) -> None:
        """F-2: a genuinely-public URL that 302-redirects to the cloud-metadata
        endpoint is refused at the redirect hop — every hop is re-validated,
        not just the first."""
        resolve = _resolver(
            {
                "public.example": ["93.184.216.34"],
                "169.254.169.254": ["169.254.169.254"],
            }
        )

        async def redirecting_parent(self: Any, request: httpx.Request) -> httpx.Response:
            # The public hop "succeeds" with a 302 to the metadata endpoint;
            # the transport never calls the parent for the (blocked) hop 2.
            return httpx.Response(
                302,
                headers={"location": "http://169.254.169.254/latest/meta-data/"},
                request=request,
                content=b"",
            )

        transport = ValidatingTransport(resolve=resolve)
        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", redirecting_parent):
            async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
                with pytest.raises(UnsafeUrlError, match="private/reserved"):
                    await client.get("https://public.example/")

    @pytest.mark.asyncio
    async def test_public_redirect_to_public_is_allowed(self) -> None:
        """A public→public redirect is followed normally — the transport only
        blocks hops that resolve to internal addresses."""
        resolve = _resolver(
            {
                "a.example": ["93.184.216.34"],
                "b.example": ["93.184.216.35"],
            }
        )
        calls: list[str] = []

        async def parent(self: Any, request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if request.headers.get("host") == "a.example":
                return httpx.Response(
                    302, headers={"location": "https://b.example/"}, request=request, content=b""
                )
            return httpx.Response(200, request=request, content=b"final")

        transport = ValidatingTransport(resolve=resolve)
        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent):
            async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
                resp = await client.get("https://a.example/")

        assert resp.status_code == 200
        # Both hops connected by their vetted IPs (Host headers carried the names).
        assert calls == ["93.184.216.34", "93.184.216.35"]


# ---------------------------------------------------------------------------
# Multi-address failover across the vetted set (pin, backported
# to the httpx path). The parent transport is stubbed, so "connect failed" is a
# raised httpx.ConnectError and no socket is opened.
# ---------------------------------------------------------------------------


def _failing_parent(outcomes: dict[str, BaseException | None], seen: list[tuple[str, Any]]) -> Any:
    """A parent transport: per pinned IP, raise the mapped error or answer 200."""

    async def parent(self: Any, request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, (request.extensions.get("timeout") or {}).get("connect")))
        exc = outcomes.get(request.url.host)
        if exc is not None:
            raise exc
        return httpx.Response(200, request=request, content=b"ok")

    return parent


class TestValidatingTransportFailover:
    @pytest.mark.asyncio
    async def test_a_dead_first_address_fails_over_to_the_next_vetted_one(self) -> None:
        seen: list[tuple[str, Any]] = []
        parent = _failing_parent({"93.184.216.34": httpx.ConnectError("refused")}, seen)
        transport = ValidatingTransport(
            resolve=_resolver({"cdn.example": ["93.184.216.34", "93.184.216.35"]})
        )
        request = httpx.Request("GET", "https://cdn.example/x")
        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent):
            resp = await transport.handle_async_request(request)
        assert resp.status_code == 200
        assert [ip for ip, _ in seen] == ["93.184.216.34", "93.184.216.35"]
        assert request.extensions["sni_hostname"] == "cdn.example"
        assert request.headers["host"] == "cdn.example"

    @pytest.mark.asyncio
    async def test_a_healthy_first_address_is_the_only_one_tried(self) -> None:
        seen: list[tuple[str, Any]] = []
        parent = _failing_parent({}, seen)
        transport = ValidatingTransport(
            resolve=_resolver({"cdn.example": ["93.184.216.34", "93.184.216.35"]})
        )
        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent):
            await transport.handle_async_request(httpx.Request("GET", "https://cdn.example/"))
        assert [ip for ip, _ in seen] == ["93.184.216.34"]

    @pytest.mark.asyncio
    async def test_every_address_failing_raises_the_last_connect_error(self) -> None:
        seen: list[tuple[str, Any]] = []
        parent = _failing_parent(
            {
                "93.184.216.34": httpx.ConnectTimeout("first"),
                "93.184.216.35": httpx.ConnectError("second"),
            },
            seen,
        )
        transport = ValidatingTransport(
            resolve=_resolver({"cdn.example": ["93.184.216.34", "93.184.216.35"]})
        )
        with (
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent),
            pytest.raises(httpx.ConnectError, match="second"),
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://cdn.example/"))
        assert len(seen) == 2

    @pytest.mark.asyncio
    async def test_an_error_after_connecting_is_not_retried(self) -> None:
        """Only a failure to *connect* is safe to retry — once bytes may have been
        sent, a second attempt could repeat a non-idempotent request."""
        seen: list[tuple[str, Any]] = []
        parent = _failing_parent({"93.184.216.34": httpx.ReadTimeout("mid-response")}, seen)
        transport = ValidatingTransport(
            resolve=_resolver({"cdn.example": ["93.184.216.34", "93.184.216.35"]})
        )
        with (
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent),
            pytest.raises(httpx.ReadTimeout),
        ):
            await transport.handle_async_request(httpx.Request("POST", "https://cdn.example/"))
        assert [ip for ip, _ in seen] == ["93.184.216.34"]

    @pytest.mark.asyncio
    async def test_the_connect_budget_is_shared_not_multiplied(self) -> None:
        """Worst case, every attempt times out and burns its whole slice: each
        address but the last gets half of what remains, the last the rest, and
        the total is the one connect timeout the caller asked for."""
        clock = [100.0]
        seen: list[tuple[str, Any]] = []
        ips = ["93.184.216.34", "93.184.216.35", "93.184.216.36"]

        async def timing_out(self: Any, request: httpx.Request) -> httpx.Response:
            budget = request.extensions["timeout"]["connect"]
            seen.append((request.url.host, budget))
            clock[0] += budget  # the attempt waits out its slice
            raise httpx.ConnectTimeout("slow")

        transport = ValidatingTransport(resolve=_resolver({"cdn.example": ips}))
        request = httpx.Request(
            "GET",
            "https://cdn.example/",
            extensions={"timeout": {"connect": 8.0, "read": 30.0, "write": 30.0, "pool": 5.0}},
        )
        with (
            patch("particles.url_safety.time.monotonic", lambda: clock[0]),
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", timing_out),
            pytest.raises(httpx.ConnectTimeout),
        ):
            await transport.handle_async_request(request)
        assert [b for _, b in seen] == [4.0, 2.0, 2.0]
        assert clock[0] - 100.0 == 8.0  # a dead host still fails in one timeout
        assert request.extensions["timeout"]["read"] == 30.0  # the rest pass through

    @pytest.mark.asyncio
    async def test_a_fast_refusal_does_not_spend_the_budget(self) -> None:
        clock = [100.0]
        seen: list[tuple[str, Any]] = []
        ips = ["93.184.216.34", "93.184.216.35", "93.184.216.36"]
        parent = _failing_parent({ip: httpx.ConnectError("refused") for ip in ips[:2]}, seen)
        transport = ValidatingTransport(resolve=_resolver({"cdn.example": ips}))
        request = httpx.Request(
            "GET", "https://cdn.example/", extensions={"timeout": {"connect": 8.0}}
        )
        with (
            patch("particles.url_safety.time.monotonic", lambda: clock[0]),
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent),
        ):
            await transport.handle_async_request(request)
        assert [b for _, b in seen] == [4.0, 4.0, 8.0]

    @pytest.mark.asyncio
    async def test_no_connect_timeout_means_no_budget_to_split(self) -> None:
        seen: list[tuple[str, Any]] = []
        parent = _failing_parent({"93.184.216.34": httpx.ConnectError("down")}, seen)
        transport = ValidatingTransport(
            resolve=_resolver({"cdn.example": ["93.184.216.34", "93.184.216.35"]})
        )
        request = httpx.Request(
            "GET", "https://cdn.example/", extensions={"timeout": {"connect": None}}
        )
        with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", parent):
            await transport.handle_async_request(request)
        assert [b for _, b in seen] == [None, None]

    @pytest.mark.asyncio
    async def test_a_split_answer_is_still_refused_before_any_connect(self) -> None:
        """Failover widens the pin to every *vetted* address; it never relaxes
        the all-or-nothing vetting that precedes it."""
        seen: list[tuple[str, Any]] = []
        transport = ValidatingTransport(
            resolve=_resolver({"split.example": ["93.184.216.34", "10.0.0.1"]})
        )
        with (
            patch.object(
                httpx.AsyncHTTPTransport, "handle_async_request", _failing_parent({}, seen)
            ),
            pytest.raises(UnsafeUrlError, match="private/reserved"),
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://split.example/"))
        assert seen == []


# ---------------------------------------------------------------------------
# resolve_and_pin — the subprocess-egress counterpart.
# Same resolver, same blocklist, same fail-closed rule; only the connecting
# differs, so the guarantee reaches fetches that never enter httpx.
# ---------------------------------------------------------------------------


class TestResolveAndPin:
    def test_returns_every_vetted_address(self) -> None:
        """The whole set, not just the first — preserving libcurl's failover.

        Every candidate has already passed ``_is_blocked_ip``, so widening from
        one address to all cannot admit a blocked one.
        """
        addrs = resolve_and_pin(
            "cdn.example",
            resolve=_resolver({"cdn.example": ["93.184.216.34", "93.184.216.35"]}),
        )
        assert [str(a) for a in addrs] == ["93.184.216.34", "93.184.216.35"]

    def test_blocked_address_rejected(self) -> None:
        with pytest.raises(UnsafeUrlError, match="private/reserved"):
            resolve_and_pin(
                "metadata.example",
                resolve=_resolver({"metadata.example": ["169.254.169.254"]}),
            )

    def test_split_answer_is_all_or_nothing(self) -> None:
        """A hostile address cannot ride along with a clean one."""
        with pytest.raises(UnsafeUrlError, match="10.0.0.1"):
            resolve_and_pin(
                "split.example",
                resolve=_resolver({"split.example": ["93.184.216.34", "10.0.0.1"]}),
            )

    def test_empty_answer_fails_closed(self) -> None:
        with pytest.raises(UnsafeUrlError, match="did not resolve"):
            resolve_and_pin("void.example", resolve=_resolver({"void.example": []}))

    def test_shares_the_blocklist_with_validate_fetch_url(self) -> None:
        """CGNAT is blocked here too — one blocklist, three layers."""
        with pytest.raises(UnsafeUrlError):
            resolve_and_pin("cgnat.example", resolve=_resolver({"cgnat.example": ["100.64.0.1"]}))

    def test_real_resolver_is_the_default(self) -> None:
        """``resolve=None`` uses the same DNS path as ``validate_fetch_url``."""
        with patch("socket.getaddrinfo", return_value=_mock_addrinfo(["93.184.216.34"])):
            addrs = resolve_and_pin("example.com")
        assert [str(a) for a in addrs] == ["93.184.216.34"]


class TestFormatConnectPin:
    """The ``<host>:<port>:<addr>…`` wire format shared by curl and libcurl."""

    def test_single_ipv4(self) -> None:
        spec = format_connect_pin("example.com", 443, [ipaddress.ip_address("93.184.216.34")])
        assert spec == "example.com:443:93.184.216.34"

    def test_multiple_addresses_are_comma_joined(self) -> None:
        spec = format_connect_pin(
            "example.com",
            443,
            [ipaddress.ip_address("1.2.3.4"), ipaddress.ip_address("5.6.7.8")],
        )
        assert spec == "example.com:443:1.2.3.4,5.6.7.8"

    def test_ipv6_is_bracketed(self) -> None:
        """Otherwise an IPv6 address's own colons read as field separators."""
        spec = format_connect_pin(
            "example.com",
            443,
            [ipaddress.ip_address("2606:2800:220:1:248:1893:25c8:1946")],
        )
        assert spec == "example.com:443:[2606:2800:220:1:248:1893:25c8:1946]"

    def test_non_default_port_is_carried(self) -> None:
        spec = format_connect_pin("example.com", 8443, [ipaddress.ip_address("1.2.3.4")])
        assert spec == "example.com:8443:1.2.3.4"
