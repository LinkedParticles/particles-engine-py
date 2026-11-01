# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 (the engine and keys_file fixtures are imported, then named as parameters)
"""The upload type sniff is bounded and never raises (security review 2026-10-30-2, AC2).

A ``.json`` upload's self-declared source type is sniffed from at most
``_SNIFF_MAX_BYTES``, parsed once, and a file the decoder cannot read (too
large, malformed, nested past its depth) is an ordinary file, never an error.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from particles.core.schema import SourceType
from particles.corpus.deposit import (
    _SNIFF_MAX_BYTES,
    _resolve_source_type,
    resolve_upload_source_type,
)
from tests.test_engine_keys import _bearer, _mint, engine, keys_file  # noqa: F401

_DEPTH = 200_000


def _deep() -> str:
    return "[" * _DEPTH + "]" * _DEPTH


def test_control_the_body_is_deep_enough_for_the_real_decoder() -> None:
    with pytest.raises(RecursionError):
        json.loads(_deep())


@pytest.mark.parametrize("suffix", [".json", ".jsonld"])
def test_the_upload_type_sniffers_read_a_deep_json_file_as_an_ordinary_file(
    tmp_path: Path, suffix: str
) -> None:
    """A deep ``.json`` upload resolves to a type rather than raising from the sniffer."""
    path = tmp_path / f"deep{suffix}"
    path.write_text(_deep())
    assert isinstance(resolve_upload_source_type(path), str)
    # The operator deposit resolves the same way from the whole content.
    assert isinstance(_resolve_source_type(path, path.read_bytes(), None), str)


def test_the_upload_type_sniffers_stay_within_a_small_multiple_of_the_upload(
    tmp_path: Path,
) -> None:
    """Sniffing a ``.json`` upload's type costs at most a few bytes of memory per input byte."""
    data = b"[" + b"{}," * (2 * 1024 * 1024 // 3) + b"{}]"
    path = tmp_path / "wide.json"
    path.write_bytes(data)
    tracemalloc.start()
    try:
        resolve_upload_source_type(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= 4 * len(data), (peak, len(data), peak / len(data))


def test_a_json_file_over_the_sniff_bound_is_an_ordinary_file(tmp_path: Path) -> None:
    """A policy shape padded past the bound is not sniffed as policy."""
    lens = {"kind": "TrustLensDefinition", "pad": "x" * _SNIFF_MAX_BYTES}
    path = tmp_path / "lens.json"
    path.write_text(json.dumps(lens))
    assert resolve_upload_source_type(path) not in {
        SourceType.TRUST_LENS_DEFINITION,
        SourceType.VOCABULARY_DOCUMENT,
        SourceType.TAXONOMY_DEFINITION,
    }


@pytest.mark.parametrize(
    ("doc", "suffix", "expected"),
    [
        ({"kind": "TrustLensDefinition"}, ".json", SourceType.TRUST_LENS_DEFINITION),
        ({"@type": "ppx:VocabularyDocument"}, ".jsonld", SourceType.VOCABULARY_DOCUMENT),
        ({"@type": ["ppx:VocabularyDocument"]}, ".json", SourceType.VOCABULARY_DOCUMENT),
        ({"name": "n", "version": "1", "tags": []}, ".json", SourceType.TAXONOMY_DEFINITION),
    ],
)
def test_a_policy_file_within_the_bound_is_still_sniffed(
    tmp_path: Path, doc: dict[str, object], suffix: str, expected: str
) -> None:
    path = tmp_path / f"policy{suffix}"
    path.write_text(json.dumps(doc))
    assert resolve_upload_source_type(path) == expected


def test_an_agent_keys_deep_json_upload_is_deposited_as_an_ordinary_file(
    engine: TestClient,
) -> None:
    """The confined upload route answers a deep ``.json`` file with a deposit, not a 500."""
    token = _mint("agent", "agent:ci-bot")
    resp = engine.post(
        "/corpus/deposit/file",
        files={"file": ("deep.json", _deep().encode(), "application/json")},
        headers=_bearer(token),
    )
    assert resp.status_code == 200, resp.text


def _record_sniff_threads(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per JSON type sniff: whether it ran on a thread with a running event loop."""
    import asyncio
    from typing import Any

    import particles.corpus.deposit as dep

    on_loop: list[bool] = []
    original = dep._sniffed_json

    def recording(content: bytes) -> Any:
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return original(content)

    monkeypatch.setattr(dep, "_sniffed_json", recording)
    return on_loop


def test_an_operator_keys_untyped_upload_is_sniffed_off_the_event_loop(
    engine: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every untyped upload's type sniff runs off the event loop, the operator's too
    (security review 2026-10-31-1, AD23)."""
    on_loop = _record_sniff_threads(monkeypatch)
    # The engine fixture's default bearer is the legacy operator key.
    resp = engine.post(
        "/corpus/deposit/file",
        files={"file": ("notes.json", b'{"a": 1}', "application/json")},
    )
    assert resp.status_code == 200, resp.text
    assert on_loop, "the sniff did not run"
    assert not any(on_loop), "an operator-key upload was sniffed on the event loop"
