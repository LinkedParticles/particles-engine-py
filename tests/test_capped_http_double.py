# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The capped-HTTP double answers ``.json`` only from the bytes it streamed.

The real ``httpx.Response.json`` decodes the bytes the stream delivered. A
double whose ``.json`` answered with something else let an importer test pass
on a body the real reader would refuse (security review 2026-10-30, AB62).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from tests._capped_http import set_capped_responses

_REFUSED = "<refused>"


def _decode(call: object) -> object:
    try:
        return call()  # type: ignore[operator]
    except ValueError:
        return _REFUSED


async def _streamed_and_decoded(resp: MagicMock) -> tuple[bytes, object]:
    from particles.http import get_capped

    client = MagicMock()
    set_capped_responses(client, return_value=resp)
    got = await get_capped(client, "https://example.test/x")
    body = b"".join([chunk async for chunk in got.aiter_bytes()])
    return body, _decode(got.json)


def _real_decode(body: bytes) -> object:
    return _decode(httpx.Response(200, content=body).json)


@pytest.mark.parametrize("shape", ["json_side_effect", "bytes_content_with_json", "exception"])
async def test_the_double_answers_json_only_from_the_bytes_it_streamed(shape: str) -> None:
    resp = MagicMock()
    resp.status_code = 200
    payload = {"id": "1", "kind": "status"}
    if shape == "json_side_effect":
        resp.json = MagicMock(side_effect=lambda: payload)
    elif shape == "bytes_content_with_json":
        resp.content = b"<html>not json</html>"
        resp.json = MagicMock(return_value=payload)
    else:
        resp.json = MagicMock(side_effect=ValueError("not JSON"))
    body, decoded = await _streamed_and_decoded(resp)
    assert decoded == _real_decode(body), (shape, body[:40], decoded)
    if shape == "json_side_effect":
        assert decoded == payload


def test_an_unserializable_payload_fails_the_helper_instead_of_streaming_nothing() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"id": "1", "tags": {"a", "b"}})
    with pytest.raises(TypeError):
        set_capped_responses(MagicMock(), return_value=resp)


async def test_a_plain_payload_streams_real_json() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"id": "1"})
    body, decoded = await _streamed_and_decoded(resp)
    assert json.loads(body) == decoded == {"id": "1"}
