# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the ``get_capped`` egress choke-point (security finding F14).

Every importer / extractor fetch routes through
:func:`particles.http.get_capped`, which opens ``client.stream("GET", url)``
(streaming + a decompressed-body size cap) rather than ``client.get(url)``.
Tests that previously wired ``mock_client.get = AsyncMock(...)`` therefore need
their mock client to expose a working ``.stream`` instead.

``set_capped_responses`` is the drop-in replacement: it mirrors the old
``return_value`` / ``side_effect`` semantics but wires ``client.stream`` so the
responses flow through ``get_capped`` unchanged. The response mocks keep their
existing ``.content`` / ``.json()`` / ``.status_code`` / ``.raise_for_status``
attributes — only the streaming seam (``.headers`` dict + ``.aiter_bytes``) is
added. A mock whose ``.json()`` was given a payload streams that payload's
JSON bytes, and every response's ``.json`` decodes the streamed bytes as
``httpx.Response`` does, so the byte cap and the real parser run on every
importer test (security review 2026-10-28-2, AA18; security review
2026-10-30, AB62).

Assertion mapping for converted tests:

* ``mock_client.get.call_count``           → ``mock_client.stream.call_count``
* ``call.args[0]`` (the URL on ``.get``)   → ``call.args[1]`` (``.stream("GET", url)``)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx


async def _aiter(chunks: list[bytes]) -> Any:
    for chunk in chunks:
        yield chunk


def _streamify(resp: MagicMock) -> AsyncMock:
    """Wrap a response mock in the async context manager ``get_capped`` drives.

    ``get_capped`` reads ``resp.headers.get("content-length")`` and iterates
    ``resp.aiter_bytes()``, so both must be real (a bare ``MagicMock`` header
    would make ``int(...)`` raise ``TypeError``). The streamed body is the mock's
    own ``.content`` when it is ``bytes``. Otherwise it is the payload the test
    gave ``.json``, serialized: its ``return_value``, or what a callable
    ``side_effect`` returns. A ``.json`` whose ``side_effect`` is an exception
    streams an empty body, which the real decoder refuses. Either way ``.json``
    then becomes a real decode of the bytes streamed, so the importer under
    test parses exactly what it was sent (security review 2026-10-30, AB62).

    Raises:
        TypeError: a ``.json`` payload ``json.dumps`` cannot serialize, or a
            ``side_effect`` sequence, which no single streamed body answers.
    """
    if not isinstance(getattr(resp, "headers", None), dict):
        resp.headers = {}
    body = getattr(resp, "content", b"")
    if not isinstance(body, bytes | bytearray):
        body = _payload_bytes(getattr(resp, "json", None))
    # A Mock still, so a test's call assertions keep working. It carries the
    # bytes it decodes, so a response wired into several calls keeps its body.
    resp.json = MagicMock(side_effect=httpx.Response(200, content=bytes(body)).json)
    resp.json.streamed_body = bytes(body)
    resp.aiter_bytes = lambda: _aiter([bytes(body)])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _payload_bytes(decoder: Any) -> bytes:
    """The JSON bytes of the payload a test gave a response's ``.json`` mock."""
    if not isinstance(decoder, Mock):
        return b""
    streamed = getattr(decoder, "streamed_body", None)
    if isinstance(streamed, bytes):
        return streamed
    effect = decoder.side_effect
    if effect is None:
        if isinstance(decoder.return_value, Mock):
            return b""
        payload = decoder.return_value
    elif isinstance(effect, BaseException) or (
        isinstance(effect, type) and issubclass(effect, BaseException)
    ):
        return b""
    elif callable(effect):
        payload = effect()
    else:
        raise TypeError(
            "a .json side_effect sequence cannot stream one body; give each response its own mock"
        )
    try:
        return json.dumps(payload).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"the .json payload is not JSON-serializable: {exc}") from exc


def set_capped_responses(
    client: MagicMock,
    *,
    return_value: MagicMock | None = None,
    side_effect: list[MagicMock] | None = None,
    router: Callable[[str], MagicMock] | None = None,
) -> None:
    """Wire ``client.stream`` so ``get_capped(client, url)`` yields the response(s).

    Mirrors ``client.get = AsyncMock(return_value=..., side_effect=...)``:

    * ``return_value`` — every ``get_capped`` call yields the same response.
    * ``side_effect`` — successive calls yield each response in turn (retry /
      multi-fetch sequences).
    * ``router`` — a *synchronous* ``url -> response`` function, for tests that
      route by URL across many endpoints (Firebase/Mastodon walkers). Replaces
      the old ``async def fake_get(url, ...)`` ``client.get`` side-effect.
    """
    if router is not None:

        def _stream(_method: str, url: str, *_a: Any, **_k: Any) -> AsyncMock:
            return _streamify(router(url))

        client.stream = MagicMock(side_effect=_stream)
    elif side_effect is not None:
        client.stream = MagicMock(side_effect=[_streamify(r) for r in side_effect])
    else:
        assert return_value is not None, "pass return_value, side_effect, or router"
        client.stream = MagicMock(return_value=_streamify(return_value))
