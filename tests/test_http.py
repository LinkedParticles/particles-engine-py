"""Tests for the shared HTTP helpers in ``particles/http.py``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from particles.http import (
    DEFAULT_RETRY_BACKOFFS,
    DEFAULT_RETRY_STATUSES,
    ResponseTooLarge,
    TransientHttpError,
    get_capped,
    get_with_retry,
)


async def _aiter_bytes(chunks: list[bytes]) -> Any:
    for chunk in chunks:
        yield chunk


def _capped_resp(status: int, *, body: bytes = b"") -> MagicMock:
    """A response shaped for ``get_capped`` (status + headers + a streamed body)."""
    r = MagicMock()
    r.status_code = status
    r.headers = {}
    r.aiter_bytes = lambda: _aiter_bytes([body])
    return r


def _stream_client(
    *, chunks: list[bytes], headers: dict[str, str] | None = None, status: int = 200
) -> MagicMock:
    """A mock httpx client whose .stream() yields a response streaming chunks."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.aiter_bytes = lambda: _aiter_bytes(chunks)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.stream = MagicMock(return_value=cm)
    return client


def _seq_stream_client(responses: list[MagicMock]) -> MagicMock:
    """A mock client whose .stream() returns each response in turn (for retry tests).

    ``get_with_retry`` now routes every attempt through ``get_capped``, which
    opens ``client.stream("GET", ...)`` — so a retry sequence is modelled as a
    list of stream context managers, one per attempt.
    """
    cms = []
    for resp in responses:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        cms.append(cm)
    client = MagicMock()
    client.stream = MagicMock(side_effect=cms)
    return client


class TestGetCapped:
    @pytest.mark.asyncio
    async def test_under_cap_returns_content(self) -> None:
        client = _stream_client(chunks=[b"hello ", b"world"])
        resp = await get_capped(client, "https://example.com/x", max_bytes=1024)
        # get_capped joins the streamed chunks into resp._content (httpx
        # surfaces that as resp.content on a real Response).
        assert resp._content == b"hello world"

    @pytest.mark.asyncio
    async def test_oversize_streamed_body_rejected(self) -> None:
        # Cap is 8 bytes; the streamed body crosses it mid-stream (gzip-bomb
        # shape: small Content-Length, large decompressed body).
        client = _stream_client(chunks=[b"aaaa", b"bbbb", b"cccc"])
        with pytest.raises(ResponseTooLarge, match="exceeded cap"):
            await get_capped(client, "https://example.com/x", max_bytes=8)

    @pytest.mark.asyncio
    async def test_oversize_content_length_rejected_up_front(self) -> None:
        client = _stream_client(chunks=[b"x"], headers={"content-length": "999999"})
        with pytest.raises(ResponseTooLarge, match="Content-Length"):
            await get_capped(client, "https://example.com/x", max_bytes=1024)

    @pytest.mark.asyncio
    async def test_malformed_content_length_falls_back_to_stream(self) -> None:
        # A bogus Content-Length is ignored; the streaming total still guards.
        client = _stream_client(chunks=[b"ok"], headers={"content-length": "not-a-number"})
        resp = await get_capped(client, "https://example.com/x", max_bytes=1024)
        assert resp._content == b"ok"

    @pytest.mark.asyncio
    async def test_cap_read_from_config_when_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Omitting max_bytes reads config.http.max_bytes at call time.
        cfg = MagicMock()
        cfg.http.max_bytes = 4
        monkeypatch.setattr("particles.config.get_config", lambda: cfg)

        client = _stream_client(chunks=[b"toolong"])
        with pytest.raises(ResponseTooLarge):
            await get_capped(client, "https://example.com/x")


class TestParticlesClient:
    @pytest.mark.asyncio
    async def test_routes_through_validating_transport(self) -> None:
        # Every particles_client routes through the SSRF connect-time gate
        #, so the validation covers redirects too.
        from particles.http import particles_client
        from particles.url_safety import ValidatingTransport

        async with particles_client() as client:
            assert isinstance(client._transport, ValidatingTransport)


class TestDefaults:
    def test_default_retry_statuses(self) -> None:
        assert frozenset({502, 503, 504}) == DEFAULT_RETRY_STATUSES

    def test_default_backoffs_imply_three_attempts(self) -> None:
        assert len(DEFAULT_RETRY_BACKOFFS) == 2  # 1 initial + 2 retries


class TestGetWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt_no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        client = _seq_stream_client([_capped_resp(200)])

        resp = await get_with_retry(client, "https://example.com/x", max_bytes=1024)
        assert resp.status_code == 200
        # Every attempt is a capped stream, never a raw client.get.
        assert client.stream.call_count == 1
        assert not client.get.called

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        client = _seq_stream_client([_capped_resp(502), _capped_resp(503), _capped_resp(200)])

        resp = await get_with_retry(client, "https://example.com/x", label="Test", max_bytes=1024)
        assert resp.status_code == 200
        assert client.stream.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        client = _seq_stream_client([_capped_resp(503), _capped_resp(503), _capped_resp(503)])

        with pytest.raises(TransientHttpError, match=r"Test unavailable \(503\)"):
            await get_with_retry(client, "https://example.com/x", label="Test", max_bytes=1024)
        # 1 + 2 backoffs = 3 attempts
        assert client.stream.call_count == 3

    @pytest.mark.asyncio
    async def test_non_transient_5xx_returned_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 500 is treated as permanent — no retries.
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        client = _seq_stream_client([_capped_resp(500)])

        resp = await get_with_retry(client, "https://example.com/x", max_bytes=1024)
        assert resp.status_code == 500
        assert client.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_4xx_returned_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        client = _seq_stream_client([_capped_resp(404)])

        resp = await get_with_retry(client, "https://example.com/x", max_bytes=1024)
        assert resp.status_code == 404
        assert client.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_retry_statuses(self) -> None:
        # Caller can extend the retry set, e.g. include 429.
        client = _seq_stream_client([_capped_resp(429), _capped_resp(200)])

        resp = await get_with_retry(
            client,
            "https://example.com/x",
            retry_statuses=frozenset({429}),
            backoffs=(0.0,),
            max_bytes=1024,
        )
        assert resp.status_code == 200
        assert client.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_oversize_body_rejected_on_retry_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A compression-bomb shaped 200 (small compressed, large decompressed)
        # is rejected by the cap rather than buffered, even through the retry
        # wrapper the GitHub helper uses.
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0, 0.0))

        client = _seq_stream_client([_capped_resp(200, body=b"x" * 64)])

        with pytest.raises(ResponseTooLarge):
            await get_with_retry(client, "https://example.com/x", max_bytes=8)

    @pytest.mark.asyncio
    async def test_label_appears_in_log_and_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        caplog.set_level(logging.WARNING, logger="particles.http")
        monkeypatch.setattr("particles.http.DEFAULT_RETRY_BACKOFFS", (0.0,))

        client = _seq_stream_client([_capped_resp(502), _capped_resp(502)])

        with pytest.raises(TransientHttpError, match="MyService"):
            await get_with_retry(client, "https://example.com/x", label="MyService", max_bytes=1024)
        assert any("MyService transient 502" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_stream(self) -> None:
        client = _seq_stream_client([_capped_resp(200)])

        await get_with_retry(
            client,
            "https://example.com/x",
            headers={"Accept": "application/json"},
            timeout=5,
            max_bytes=1024,
        )
        # get_capped opens client.stream("GET", url, **kwargs) — auth/headers
        # and per-request options ride through unchanged.
        args, kwargs = client.stream.call_args
        assert args[0] == "GET"
        assert kwargs["headers"] == {"Accept": "application/json"}
        assert kwargs["timeout"] == 5
