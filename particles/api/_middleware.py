"""ASGI middleware for the FastAPI app — request body-size limiting.

Defense-in-depth for the HTTP surface (security review F-7): uvicorn and
Starlette impose no default request-body limit, so an unbounded upload can
drive memory exhaustion before any handler runs. ``BodySizeLimitMiddleware``
rejects an oversized body with ``413 Request Entity Too Large``, reading the
cap from ``get_config().api.max_request_body_bytes`` at request time (so it
honours config reloads and the ``PARTICLES_MAX_REQUEST_BODY_BYTES`` override).

The limit is enforced two ways: a declared ``Content-Length`` over the cap is
rejected up front (no body read); a streamed/chunked body is counted as it
arrives and rejected the moment it crosses the cap. Setting the cap to ``0``
disables the in-app check — for deployments where a reverse proxy already
enforces a smaller limit.

This is a process-local backstop, not the deployment-time hardening gate
tracked separately — the proposed fail-closed-bind gate.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from particles.config import get_config


class _BodyTooLarge(Exception):
    """Internal signal: the streamed body crossed the configured cap."""


class BodySizeLimitMiddleware:
    """Reject HTTP requests whose body exceeds ``api.max_request_body_bytes``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = get_config().api.max_request_body_bytes
        if max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        # Up-front rejection when the client declares an oversized body.
        declared = _content_length(scope)
        if declared is not None and declared > max_bytes:
            await self._reject(send)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            # Only safe to synthesize a 413 if the handler had not already
            # begun its own response (FastAPI reads the body before responding,
            # so in practice the limit trips first).
            if response_started:
                raise
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps({"detail": "Request body too large"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
