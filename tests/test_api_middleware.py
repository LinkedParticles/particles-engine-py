"""Tests for particles/api/_middleware.py (BodySizeLimitMiddleware).

The TestClient-driven tests in tests/test_app.py cover the common
``Content-Length`` rejection path. These drive the ASGI callable directly to
exercise the *streamed-body* path (no declared length, counted as chunks
arrive) and the scope/disabled short-circuits — cases TestClient cannot
construct because httpx always sends a Content-Length for in-memory bodies.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from particles.api._middleware import BodySizeLimitMiddleware
from particles.config import reset_config


def _ok_app() -> Any:
    """Minimal ASGI app: drains the body, then returns 200."""

    async def app(scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


async def _drive(
    mw: BodySizeLimitMiddleware, scope: dict[str, Any], chunks: list[tuple[bytes, bool]]
) -> list[dict[str, Any]]:
    """Feed ``chunks`` (body, more_body) into ``mw`` and collect sent messages."""
    pending = list(chunks)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if pending:
            body, more = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": more}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def test_streamed_body_over_limit_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "10")
    reset_config()
    mw = BodySizeLimitMiddleware(_ok_app())
    scope = {"type": "http", "headers": []}  # no content-length → counted path
    sent = asyncio.run(_drive(mw, scope, [(b"x" * 5, True), (b"y" * 20, False)]))
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    reset_config()


def test_streamed_body_within_limit_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "1000")
    reset_config()
    mw = BodySizeLimitMiddleware(_ok_app())
    scope = {"type": "http", "headers": []}
    sent = asyncio.run(_drive(mw, scope, [(b"x" * 5, True), (b"y" * 5, False)]))
    assert sent[0]["status"] == 200
    reset_config()


def test_zero_limit_disables_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "0")
    reset_config()
    mw = BodySizeLimitMiddleware(_ok_app())
    scope = {"type": "http", "headers": []}
    sent = asyncio.run(_drive(mw, scope, [(b"x" * 10_000, False)]))
    assert sent[0]["status"] == 200
    reset_config()


def test_non_http_scope_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_MAX_REQUEST_BODY_BYTES", "1")
    reset_config()

    seen: list[str] = []

    async def lifespan_app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    mw = BodySizeLimitMiddleware(lifespan_app)

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(message: dict[str, Any]) -> None:
        pass

    asyncio.run(mw({"type": "lifespan"}, receive, send))
    assert seen == ["lifespan"]  # middleware did not intercept
    reset_config()
