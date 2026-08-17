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
added.

Assertion mapping for converted tests:

* ``mock_client.get.call_count``           → ``mock_client.stream.call_count``
* ``call.args[0]`` (the URL on ``.get``)   → ``call.args[1]`` (``.stream("GET", url)``)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock


async def _aiter(chunks: list[bytes]) -> Any:
    for chunk in chunks:
        yield chunk


def _streamify(resp: MagicMock) -> AsyncMock:
    """Wrap a response mock in the async context manager ``get_capped`` drives.

    ``get_capped`` reads ``resp.headers.get("content-length")`` and iterates
    ``resp.aiter_bytes()``, so both must be real (a bare ``MagicMock`` header
    would make ``int(...)`` raise ``TypeError``). The streamed body is the mock's
    own ``.content`` when it is ``bytes``; otherwise an empty body (the test set
    ``.json`` directly and never populates ``.content``).
    """
    if not isinstance(getattr(resp, "headers", None), dict):
        resp.headers = {}
    body = getattr(resp, "content", b"")
    if not isinstance(body, bytes | bytearray):
        body = b""
    resp.aiter_bytes = lambda: _aiter([bytes(body)])
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


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
