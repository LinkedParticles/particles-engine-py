# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The no-egress setting end to end, and what each refused path degrades to.

The end-to-end test runs a representative workload with ``network.egress:
allowlist`` naming one in-process OpenAI-compatible stub **by name**, records
every outbound socket and name lookup the process attempts, and asserts the
set never leaves the allowlist. The rest pin the per-path degradation table:
a refusal is never cached, never counted as a miss, and never
mistaken for an outage.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles import egress
from particles import embeddings as ep
from particles.config import (
    EgressAllowEntry,
    NetworkConfig,
    OpenAICompatProviderConfig,
    ProviderSelection,
    get_config,
)

_EXTRACTION_REPLY = json.dumps(
    [
        {
            "content": "The Particles SDK stores beliefs with provenance.",
            "subjects": ["Particles SDK"],
            "confidence_value": 0.9,
            "uncertainty_nature": "EPISTEMIC",
        }
    ]
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    """An OpenAI-compatible ``chat/completions`` endpoint that always answers."""

    calls: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 — http.server's naming
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        type(self).calls.append(self.path)
        body = json.dumps(
            {
                "choices": [{"message": {"content": _EXTRACTION_REPLY}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return None


@pytest.fixture
def stub_server() -> Iterator[int]:
    _StubHandler.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class _Recorder:
    """Records outbound connects and name lookups; never refuses."""

    def __init__(self) -> None:
        self.active = False
        self.connects: list[tuple[str, int]] = []
        self.lookups: list[str] = []

    def hook(self, event: str, args: tuple[Any, ...]) -> None:
        if not self.active:
            return
        if event == "socket.connect":
            sock, address = args[0], args[1]
            if getattr(sock, "family", None) in (socket.AF_INET, socket.AF_INET6):
                self.connects.append((str(address[0]), int(address[1])))
        elif event == "socket.getaddrinfo":
            host = args[0]
            if isinstance(host, bytes):
                host = host.decode()
            if isinstance(host, str) and host and egress._ip_or_none(host) is None:
                self.lookups.append(host)


_RECORDER = _Recorder()
sys.addaudithook(_RECORDER.hook)


class _FakeEncoder:
    """A deterministic stand-in for the sentence encoder: no model, no download."""

    def encode(self, texts: list[str], **_kwargs: Any) -> Any:
        dim = get_config().embeddings.dim
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(dim)
            rows.append(vec / np.linalg.norm(vec))
        return np.array(rows)


@pytest.fixture
def posture(monkeypatch: pytest.MonkeyPatch, stub_server: int) -> Iterator[int]:
    """``allowlist`` naming the stub by name; every LLM purpose routed to it.

    ``models.test`` resolves to 127.0.0.1 through a resolver installed under the
    backstop's recording wrapper, so the name-entry path is the one exercised.
    The real backstop is armed for the test.
    """
    cfg = get_config()
    monkeypatch.setattr(
        cfg,
        "network",
        NetworkConfig(
            egress="allowlist",
            allow=[EgressAllowEntry(host="models.test", port=stub_server)],
        ),
    )
    cfg.llm.providers["stub"] = OpenAICompatProviderConfig(
        base_url=f"http://models.test:{stub_server}/v1", max_retries=0
    )
    monkeypatch.setattr(cfg.llm, "default", ProviderSelection(provider="stub", model="m"))
    for purpose in ("use_judge",):
        monkeypatch.setattr(cfg.llm, purpose, None)
    monkeypatch.setenv("NOTION_API_KEY", "secret-not-used")
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)

    egress.ensure_backstop()
    real = egress._real_getaddrinfo or socket.getaddrinfo

    def resolver(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        # anyio passes the IDNA-encoded name as bytes.
        if host in ("models.test", b"models.test"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 0))]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(egress, "_real_getaddrinfo", resolver)
    original = ep._embedding_model
    ep.set_embedding_model(_FakeEncoder())  # type: ignore[arg-type]
    try:
        yield stub_server
    finally:
        egress.disarm_backstop()
        ep.set_embedding_model(original)


@pytest.fixture
def live_wikidata_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse ``no_live_wikidata`` patch, so the egress gate is what stops it."""
    from particles import http
    from particles.ingest.authorities import wikidata

    monkeypatch.setattr(wikidata, "particles_client", http.particles_client)


# ---------------------------------------------------------------------------
# §6 item 2 and 3: no socket outside the allowlist
# ---------------------------------------------------------------------------


async def _workload(session: AsyncSession, tmp_path: Path) -> None:
    from particles.core.schema import FetchPolicy, Mutability, QueryRequest, WarcRecordType
    from particles.corpus.deposit import (
        deposit_file,
        deposit_text,
        deposit_url,
        save_blob,
        sha256,
        write_entry_and_snapshot,
    )
    from particles.exporters.notion import NotionExporter
    from particles.operations.corpus_refresh import refresh_entries
    from particles.operations.extract import extract_snapshot
    from particles.operations.lint.orchestrator import run_lint
    from particles.operations.query.main import query

    note = tmp_path / "note.md"
    note.write_text("The Particles SDK stores beliefs with provenance.\n")
    entry_id, snapshot_id = await deposit_file(session, note)
    await extract_snapshot(session, entry_id, snapshot_id)
    await session.commit()

    with pytest.raises(egress.EgressRefused):
        await deposit_url(session, "https://example.org/page")
    await session.rollback()

    wikidata_entity = {
        "id": "Q42",
        "labels": {"en": "Douglas Adams"},
        "statements": {
            "P31": [
                {
                    "rank": "normal",
                    "property": {"id": "P31", "data_type": "wikibase-item"},
                    "value": {"type": "value", "content": "Q5"},
                }
            ]
        },
    }
    wd_entry, wd_snapshot = await deposit_text(
        session, json.dumps(wikidata_entity), source_type="WIKIDATA_API"
    )
    await extract_snapshot(session, wd_entry, wd_snapshot)
    await session.commit()

    await query(session, QueryRequest(question="What does the Particles SDK store?"))
    await run_lint(session, semantic=True)

    with pytest.raises(egress.EgressRefused):
        await NotionExporter().export(session, None, database_id="db")

    page = b"<html>an earlier fetch</html>"
    digest = sha256(page)
    web_entry, _ = await write_entry_and_snapshot(
        session,
        "https://example.org/page",
        "WEB_PAGE",
        Mutability.MUTABLE,
        FetchPolicy.LAZY,
        page,
        digest,
        save_blob(page, digest),
        "operator",
        [],
        WarcRecordType.RESPONSE,
    )
    await session.commit()
    results = [r async for r in refresh_entries(session, [web_entry], force=True)]
    assert isinstance(results[0].error, egress.EgressRefused)


@pytest.mark.asyncio
async def test_workload_opens_no_socket_outside_the_allowlist(
    db_session: AsyncSession, posture: int, live_wikidata_path: None, tmp_path: Path
) -> None:
    from particles.store.wikidata_cache import WikidataLabelRow

    _RECORDER.connects.clear()
    _RECORDER.lookups.clear()
    _RECORDER.active = True
    try:
        await _workload(db_session, tmp_path)
    finally:
        _RECORDER.active = False

    assert set(_RECORDER.lookups) <= {"models.test"}, _RECORDER.lookups
    assert set(_RECORDER.connects) <= {("127.0.0.1", posture)}, _RECORDER.connects
    # The stub served the LLM calls: the name entry admitted its address.
    assert _StubHandler.calls
    assert all(path.endswith("/chat/completions") for path in _StubHandler.calls)
    # A refused label lookup is never persisted (finding 1).
    rows = (await db_session.execute(select(WikidataLabelRow))).scalars().all()
    assert {row.qid for row in rows}.isdisjoint({"P31", "Q5"})
    assert all(row.label != row.qid for row in rows)


@pytest.mark.asyncio
async def test_none_records_no_connect_and_no_lookup(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    live_wikidata_path: None,
    tmp_path: Path,
) -> None:
    from particles.corpus.deposit import deposit_file, deposit_url
    from particles.operations.extract import extract_snapshot

    monkeypatch.setattr(get_config(), "network", NetworkConfig(egress="none"))
    original = ep._embedding_model
    ep.set_embedding_model(_FakeEncoder())  # type: ignore[arg-type]
    egress.ensure_backstop()
    _RECORDER.connects.clear()
    _RECORDER.lookups.clear()
    _RECORDER.active = True
    try:
        note = tmp_path / "note.md"
        note.write_text("Local only.\n")
        entry_id, snapshot_id = await deposit_file(db_session, note)
        with pytest.raises(egress.EgressRefused):
            await extract_snapshot(db_session, entry_id, snapshot_id)
        await db_session.rollback()
        with pytest.raises(egress.EgressRefused):
            await deposit_url(db_session, "https://example.org/page")
    finally:
        _RECORDER.active = False
        egress.disarm_backstop()
        ep.set_embedding_model(original)

    assert _RECORDER.connects == []
    assert _RECORDER.lookups == []


# ---------------------------------------------------------------------------
# §4: what each refused path degrades to
# ---------------------------------------------------------------------------


@pytest.fixture
def refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``network.egress: none`` for one test, with no LLM key set."""
    monkeypatch.setattr(get_config(), "network", NetworkConfig(egress="none"))


class TestLlmPurposes:
    def test_get_provider_refuses_before_any_adapter(self, refusing: None) -> None:
        from particles.llm import CompletionError, PurposeRefused, get_provider

        with pytest.raises(PurposeRefused) as info:
            get_provider("extraction")
        # Every call site's CompletionError fallback applies, and a caller can
        # still tell a policy from an outage.
        assert isinstance(info.value, CompletionError)
        assert isinstance(info.value, egress.EgressRefused)
        assert info.value.host == "api.anthropic.com"
        assert info.value.llm_purpose == "extraction"

    def test_an_override_is_in_process_and_never_refused(self, refusing: None) -> None:
        from particles.llm import get_provider, override_providers, purpose_allowed

        sentinel: Any = object()
        with override_providers({"semantic_lint": sentinel}):
            assert purpose_allowed("semantic_lint")
            assert get_provider("semantic_lint") is sentinel

    @pytest.mark.asyncio
    async def test_breaker_neither_counts_nor_trips(self, refusing: None) -> None:
        from particles.llm import breaker

        before = breaker.llm_failure_count()
        assert await breaker._llm_call("prompt") is None
        assert await breaker._llm_call_many([]) == []
        assert breaker.llm_failure_count() == before
        assert not breaker.llm_circuit_open()

    def test_agent_side_purpose_is_unservable(self, refusing: None) -> None:
        """A refused semantic_lint holds agent-side near pairs."""
        from particles.extraction.agent import purpose_servable

        assert purpose_servable("semantic_lint") is False

    @pytest.mark.asyncio
    async def test_background_probe_fails_open_quietly(
        self, refusing: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        from particles.ingest.pipeline import _llm_confirms_contradiction

        assert await _llm_confirms_contradiction("a", "b") is None
        assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]

    def test_anthropic_client_keeps_the_sdk_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """§6 item 5: the hook adds a check and changes nothing else."""
        import anthropic

        from particles.llm import client as client_mod

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
        client_mod.set_client(None)
        try:
            ours = client_mod.get_client()
        finally:
            client_mod.set_client(None)
        plain = anthropic.Anthropic(api_key="sk-test-not-used")
        assert ours.max_retries == plain.max_retries
        assert ours.timeout == plain.timeout
        assert (
            ours._client._transport._pool._max_connections  # type: ignore[attr-defined]
            == plain._client._transport._pool._max_connections  # type: ignore[attr-defined]
        )
        assert len(ours._client.event_hooks["request"]) == 1

    def test_anthropic_hook_refuses(self, refusing: None) -> None:
        import httpx

        from particles.http import egress_request_hook

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        with pytest.raises(egress.EgressRefused):
            egress_request_hook("llm")(request)


class TestSubjectResolution:
    @pytest.mark.asyncio
    async def test_refused_authority_is_not_asked_and_not_negative_cached(
        self,
        db_session: AsyncSession,
        refusing: None,
        live_wikidata_path: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from particles.ingest.authorities import wikidata
        from particles.ingest.subject_resolver import resolve_subject
        from particles.store import subject_cache

        asked: list[str] = []

        async def _spy(name: str, *args: Any, **kwargs: Any) -> list[Any]:
            asked.append(name)
            return []

        monkeypatch.setattr(wikidata, "_wikidata_candidates", _spy)
        subject = await resolve_subject(db_session, "Ada Lovelace", source_type="WEB_PAGE")
        assert subject.canonical_name == "Ada Lovelace"
        assert subject.external_ids == []
        assert asked == []
        assert not subject_cache.negative_get("Ada Lovelace")

    @pytest.mark.asyncio
    async def test_split_by_qid_refuses_instead_of_naming_by_qid(
        self, db_session: AsyncSession, refusing: None
    ) -> None:
        from particles.ingest.subject_resolver import split_subject_resolving

        with pytest.raises(egress.EgressRefused):
            await split_subject_resolving(
                db_session,
                source_id="missing-subject",
                particle_ids=[],
                new_name=None,
                new_external_id="wikidata:Q30297735",
            )


class TestWikidataLabels:
    @pytest.mark.asyncio
    async def test_refused_label_is_returned_but_never_persisted(
        self, db_session: AsyncSession, refusing: None
    ) -> None:
        from particles.extraction import wikidata as wx
        from particles.store.wikidata_cache import get_label

        wx._item_label_cache.pop("Q42", None)
        assert await wx._fetch_item_label("Q42", session=db_session) == "Q42"
        assert await get_label(db_session, "Q42") is None
        assert "Q42" not in wx._item_label_cache


class TestDepositPaths:
    def test_api_deposit_is_a_403(self, refusing: None) -> None:
        from fastapi.testclient import TestClient

        from particles.api.app import app

        client = TestClient(app, client=("127.0.0.1", 50000))
        resp = client.post("/corpus/deposit/url", json={"url": "https://example.org/page"})
        assert resp.status_code == 403
        assert resp.json()["error"] == "egress_refused"
        assert resp.json()["host"] == "example.org"

    @pytest.mark.asyncio
    async def test_inbox_marks_refused_apart_from_failed(
        self, db_session: AsyncSession, refusing: None, tmp_path: Path
    ) -> None:
        from particles.operations.inbox import process_inbox

        inbox = tmp_path / "inbox.md"
        inbox.write_text("https://example.org/page\n")
        summary = await process_inbox(db_session, inbox)
        assert summary["refused"] == ["https://example.org/page"]
        assert summary["failed"] == []
        assert inbox.read_text().startswith("# Refused (network.egress) ")

    @pytest.mark.asyncio
    async def test_subprocess_pin_refuses_before_resolving(self, refusing: None) -> None:
        from particles.url_safety import resolve_and_pin

        def _no_resolve(host: str) -> list[Any]:
            raise AssertionError("resolved a refused host")

        with pytest.raises(egress.EgressRefused):
            resolve_and_pin("gist.github.com", resolve=_no_resolve)

    @pytest.mark.asyncio
    async def test_benchmark_dataset_points_at_dataset_file(
        self, refusing: None, tmp_path: Path
    ) -> None:
        from particles.benchmark.memory.loader import (
            EXPECTED_SHA256,
            MemoryDatasetLoadError,
            ensure_dataset,
        )

        variant = next(iter(EXPECTED_SHA256))
        with pytest.raises(MemoryDatasetLoadError, match="--dataset-file"):
            await ensure_dataset(variant, cache_dir=tmp_path, expected_sha256="0" * 64)
        assert not list(tmp_path.rglob("*.part"))


class TestExtractionAndMaintenance:
    @pytest.mark.asyncio
    async def test_refused_extraction_claims_nothing(
        self, db_session: AsyncSession, refusing: None, tmp_path: Path
    ) -> None:
        from particles.corpus.deposit import deposit_file
        from particles.corpus.store import SnapshotRow
        from particles.operations.extract import extract_snapshot

        note = tmp_path / "n.md"
        note.write_text("A note.\n")
        entry_id, snapshot_id = await deposit_file(db_session, note)
        await db_session.commit()
        with pytest.raises(egress.EgressRefused):
            await extract_snapshot(db_session, entry_id, snapshot_id)
        row = await db_session.get(SnapshotRow, snapshot_id)
        assert row is not None
        assert row.extraction_status == "PENDING"
        assert row.extraction_attempts == 0

    @pytest.mark.asyncio
    async def test_lint_reports_why_semantic_was_skipped(
        self, db_session: AsyncSession, refusing: None
    ) -> None:
        from particles.operations.lint.orchestrator import run_lint

        report = await run_lint(db_session, semantic=True)
        assert report.semantic_skipped is True
        assert report.semantic_skipped_reason == "egress"

    def test_consolidation_degrades_with_the_posture_named(self, refusing: None) -> None:
        from particles.operations.consolidation import _semantic_availability

        ok, reason = _semantic_availability(structural_only=False)
        assert ok is False
        assert reason is not None and reason.startswith("egress:")

    @pytest.mark.asyncio
    async def test_consolidation_extract_pass_is_skipped_not_failed(
        self, db_session: AsyncSession, refusing: None, tmp_path: Path
    ) -> None:
        from particles.corpus.deposit import deposit_file
        from particles.operations.consolidation import (
            ConsolidationReport,
            _EgressSkip,
            _pass_extract,
        )

        note = tmp_path / "n.md"
        note.write_text("A note.\n")
        await deposit_file(db_session, note)
        await db_session.commit()
        report = ConsolidationReport()
        with pytest.raises(_EgressSkip, match="stay PENDING"):
            await _pass_extract(db_session, report)
        assert report.pending_remaining == 1


class TestEncoderAndTelemetry:
    def test_encoder_loads_from_the_cache_only(
        self, refusing: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types

        seen: list[dict[str, Any]] = []

        class _ST:
            def __init__(self, model_id: str, **kwargs: Any) -> None:
                seen.append(kwargs)
                raise OSError("not cached")

        monkeypatch.setitem(
            sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_ST)
        )
        original = ep._embedding_model
        ep.set_embedding_model(None)
        try:
            assert ep.get_embedding_model() is None
            assert ep.get_embedding_model() is None  # remembered, not retried
        finally:
            ep.set_embedding_model(original)
        assert seen == [{"local_files_only": True}]

    def test_otlp_endpoint_refused(self, refusing: None) -> None:
        from particles.config import ObservabilityConfig
        from particles.observability.setup import _otlp_allowed

        assert _otlp_allowed(ObservabilityConfig(enabled=True, exporter="otlp")) is False


class TestReports:
    def test_posture_lists_every_destination(self, refusing: None) -> None:
        from particles.network_posture import posture

        report = posture()
        assert report.posture == "none"
        kinds = {item.kind for item in report.items}
        assert {"llm", "authority", "encoder"} <= kinds
        assert all(not i.allowed for i in report.items if i.kind in ("llm", "authority"))

    def test_config_validate_fails_on_a_refused_engine(
        self, refusing: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        monkeypatch.setattr(get_config().engine, "base_url", "http://engine.lan:8000")
        result = CliRunner().invoke(app, ["config", "validate"])
        assert result.exit_code == 1
        assert "network: egress=none" in result.output
        assert "engine.base_url" in result.output

    def test_config_validate_open_is_one_line(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["config", "validate"])
        assert result.exit_code == 0
        assert "network: egress=open (no check)" in result.output

    def test_doctor_line(self, refusing: None) -> None:
        from particles.api.cli.hook import _check_egress

        lines = _check_egress()
        assert lines[0].startswith("egress:         none")
        assert any("use_judge refused" in line for line in lines)
