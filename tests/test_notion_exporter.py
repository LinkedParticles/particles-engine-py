"""Notion exporter + API-target credential pattern.

The Notion HTTP API is mocked end-to-end — no live token, no network. The
fixtures seed a small store (subjects + ACTIVE particles linked via the join
table) and a stateful in-memory fake Notion workspace so the idempotent-upsert
behaviour (re-run updates, never duplicates) is assertable.

Coverage (per the ADR's § Testing and tests/AGENTS.md § What requires tests):

* missing-credential fails loud — at the exporter's first-statement getter
  (``export()``) AND at the generic CLI pre-flight (``required_secret``);
* dry-run makes ZERO API writes and reports planned totals;
* idempotent upsert — a second run updates the existing page, never duplicates;
* the ``min_particle_confidence`` filter is honoured;
* managed-range ownership (default) rewrites blocks on re-sync, while
  ``--no-update-blocks`` leaves an existing page's blocks untouched.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from particles.core.schema import (
    ApplicabilityClause,
    Confidence,
    ExtractorRecord,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.db import session_scope
from particles.exporters.notion import NotionExporter, _is_managed_heading, _subject_properties
from particles.exporters.registry import get_exporters, required_secret
from particles.store.extractor_store import invalidate_trust_cache, upsert_extractor_record
from particles.store.particle_store import insert_particle
from particles.store.subject_store import insert_subject

_DB_ID = "test-database-0000000000000000000000"


# ---------------------------------------------------------------------------
# Fake Notion workspace — a stateful dispatcher over the REST surface used by
# _NotionClient. Tracks created pages keyed by the upsert subject id so the
# idempotent-upsert assertion is meaningful.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeNotion:
    """In-memory Notion workspace. One client per export run via the patch."""

    def __init__(self, id_property: str) -> None:
        self.id_property = id_property
        # page_id -> {"subject_id": str, "properties": dict, "blocks": [ {id,...} ]}
        self.pages: dict[str, dict[str, Any]] = {}
        self._page_seq = 0
        self._block_seq = 0
        self.calls: list[tuple[str, str]] = []

    def _new_page_id(self) -> str:
        self._page_seq += 1
        return f"page-{self._page_seq}"

    def _new_block_id(self) -> str:
        self._block_seq += 1
        return f"block-{self._block_seq}"

    def _subject_id_of_props(self, properties: dict[str, Any]) -> str:
        prop = properties.get(self.id_property, {})
        runs = prop.get("rich_text", [])
        return "".join(r.get("text", {}).get("content", "") for r in runs)

    async def request(self, method: str, url: str, *, json: Any = None) -> _FakeResponse:
        # url is "https://api.notion.com/v1<path>"; strip the base + query.
        path = url.split("/v1", 1)[1]
        path_no_query = path.split("?", 1)[0]
        self.calls.append((method, path_no_query))

        if method == "POST" and path_no_query.endswith("/query"):
            wanted = (json or {}).get("filter", {}).get("rich_text", {}).get("equals")
            for pid, page in self.pages.items():
                if page["subject_id"] == wanted:
                    return _FakeResponse(200, {"results": [{"id": pid}]})
            return _FakeResponse(200, {"results": []})

        if method == "POST" and path_no_query == "/pages":
            props = (json or {}).get("properties", {})
            pid = self._new_page_id()
            self.pages[pid] = {
                "subject_id": self._subject_id_of_props(props),
                "properties": props,
                "blocks": [],
            }
            return _FakeResponse(200, {"id": pid})

        if method == "PATCH" and path_no_query.startswith("/pages/"):
            pid = path_no_query.split("/pages/", 1)[1]
            self.pages[pid]["properties"].update((json or {}).get("properties", {}))
            return _FakeResponse(200, {"id": pid})

        if method == "GET" and path_no_query.startswith("/blocks/"):
            pid = path_no_query.split("/blocks/", 1)[1].split("/children", 1)[0]
            return _FakeResponse(200, {"results": self.pages[pid]["blocks"], "has_more": False})

        if method == "PATCH" and path_no_query.startswith("/blocks/"):
            pid = path_no_query.split("/blocks/", 1)[1].split("/children", 1)[0]
            children = (json or {}).get("children", [])
            for child in children:
                stored = dict(child)
                stored["id"] = self._new_block_id()
                self.pages[pid]["blocks"].append(stored)
            return _FakeResponse(200, {"results": []})

        if method == "DELETE" and path_no_query.startswith("/blocks/"):
            bid = path_no_query.split("/blocks/", 1)[1]
            for page in self.pages.values():
                page["blocks"] = [b for b in page["blocks"] if b.get("id") != bid]
            return _FakeResponse(200, {"id": bid})

        raise AssertionError(f"unexpected Notion call: {method} {path_no_query}")


def _patch_notion(fake: _FakeNotion) -> Any:
    """Patch ``particles.http.particles_client`` to yield the fake client.

    ``_NotionClient._request`` imports ``particles_client`` inside the function
    body (tests/AGENTS.md § Mocking strategy), so the source-module patch reaches
    it. The async-context-manager yields a stand-in whose ``request`` is the
    fake's dispatcher.
    """
    mock_ctx = patch("particles.http.particles_client")
    started = mock_ctx.start()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=fake.request)
    started.return_value = mock_client
    return mock_ctx


# ---------------------------------------------------------------------------
# Store seeding
# ---------------------------------------------------------------------------


def _make_particle(
    content: str, *, confidence_value: float, extractor_id: str, subject_ids: list[str]
) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(
            value=confidence_value, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
        asserted_by=extractor_id,
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": extractor_id, "version": "0.1.0"},
        subject_ids=subject_ids,
    )


async def _seed(*, confidences: list[float]) -> Subject:
    """One subject with ``len(confidences)`` ACTIVE particles, all high-trust."""
    invalidate_trust_cache()
    subj = Subject(id=str(uuid.uuid4()), canonical_name="TestSubject", asserted_by="test")
    async with session_scope() as session:
        await upsert_extractor_record(
            session,
            ExtractorRecord(
                extractor_id="x",
                name="x",
                version="0.1.0",
                applicability=[
                    ApplicabilityClause(
                        keyword="MAY",
                        domain_uri="http://example.org/test",
                        domain_label="test",
                        source_types=["TEST"],
                    )
                ],
                trust_weight=1.0,
            ),
        )
        await insert_subject(session, subj)
        for i, c in enumerate(confidences):
            await insert_particle(
                session,
                _make_particle(
                    f"Claim {i} (raw {c})",
                    confidence_value=c,
                    extractor_id="x",
                    subject_ids=[subj.id],
                ),
            )
        await session.commit()
    return subj


@pytest.fixture(autouse=True)
def _notion_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "secret-test-token")


# ---------------------------------------------------------------------------
# Credential pattern (Part A)
# ---------------------------------------------------------------------------


class TestCredentialPattern:
    def test_exporter_declares_required_secret(self) -> None:
        exp = get_exporters()["notion"]
        assert required_secret(exp) == "NOTION_API_KEY"

    def test_filesystem_exporters_declare_no_secret(self) -> None:
        exporters = get_exporters()
        for fmt in ("obsidian", "anki", "wiki", "logseq", "jsonl"):
            assert required_secret(exporters[fmt]) is None

    @pytest.mark.asyncio
    async def test_missing_token_fails_loud_at_first_statement(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The getter is the authoritative check — it fires before any store read
        # or network call (no fake patched ⇒ a write would error if reached).
        monkeypatch.delenv("NOTION_API_KEY", raising=False)
        await _seed(confidences=[0.9])
        async with session_scope() as session:
            with pytest.raises(ValueError, match="NOTION_API_KEY"):
                await NotionExporter().export(session, None, database_id=_DB_ID, dry_run=True)


class TestCliPreflight:
    """The generic, REQUIRES_SECRET-driven CLI pre-flight."""

    def test_cli_aborts_when_secret_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        monkeypatch.delenv("NOTION_API_KEY", raising=False)
        result = CliRunner().invoke(app, ["export", "notion", "--dry-run"])
        assert result.exit_code == 2
        assert "NOTION_API_KEY is required" in result.output


# ---------------------------------------------------------------------------
# Mapping, dry-run, idempotency (Part B)
# ---------------------------------------------------------------------------


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_makes_zero_writes_and_reports_plan(self, db_session: object) -> None:
        await _seed(confidences=[0.9, 0.8])
        fake = _FakeNotion("Particle Subject ID")
        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                summary = await NotionExporter().export(
                    session, None, database_id=_DB_ID, dry_run=True
                )
        finally:
            ctx.stop()
        assert summary.dry_run is True
        assert summary.subjects_planned == 1
        assert summary.particles_synced == 2
        assert summary.pages_created is None  # no probe on dry-run
        assert summary.database_id == _DB_ID
        # The defining guarantee: not a single API call was issued.
        assert fake.calls == []


class TestUpsertIdempotency:
    @pytest.mark.asyncio
    async def test_first_run_creates_second_run_updates_never_duplicates(
        self, db_session: object
    ) -> None:
        await _seed(confidences=[0.9, 0.8])
        fake = _FakeNotion("Particle Subject ID")

        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                first = await NotionExporter().export(session, None, database_id=_DB_ID)
        finally:
            ctx.stop()
        assert first.pages_created == 1
        assert first.pages_updated == 0
        assert len(fake.pages) == 1

        # Re-run against the SAME fake workspace — must update, not duplicate.
        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                second = await NotionExporter().export(session, None, database_id=_DB_ID)
        finally:
            ctx.stop()
        assert second.pages_created == 0
        assert second.pages_updated == 1
        assert len(fake.pages) == 1  # still exactly one page

    @pytest.mark.asyncio
    async def test_managed_range_rewritten_on_resync(self, db_session: object) -> None:
        await _seed(confidences=[0.9, 0.8])
        fake = _FakeNotion("Particle Subject ID")
        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                await NotionExporter().export(session, None, database_id=_DB_ID)
        finally:
            ctx.stop()
        page_id = next(iter(fake.pages))
        # Heading + 2 particle blocks.
        assert len(fake.pages[page_id]["blocks"]) == 3

        # Re-sync (default = own the managed range): the old blocks are deleted
        # and re-appended, so the count stays 3 rather than doubling.
        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                await NotionExporter().export(session, None, database_id=_DB_ID)
        finally:
            ctx.stop()
        assert len(fake.pages[page_id]["blocks"]) == 3
        # The rewrite went through a delete of the old managed blocks.
        assert "DELETE" in {m for m, _ in fake.calls}

    @pytest.mark.asyncio
    async def test_no_update_blocks_leaves_existing_blocks_untouched(
        self, db_session: object
    ) -> None:
        await _seed(confidences=[0.9, 0.8])
        fake = _FakeNotion("Particle Subject ID")
        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                await NotionExporter().export(session, None, database_id=_DB_ID)
        finally:
            ctx.stop()
        page_id = next(iter(fake.pages))
        # Simulate a hand-edit inside the managed range.
        fake.pages[page_id]["blocks"].append({"id": "hand-edit", "type": "paragraph"})
        before = len(fake.pages[page_id]["blocks"])

        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                summary = await NotionExporter().export(
                    session, None, database_id=_DB_ID, no_update_blocks=True
                )
        finally:
            ctx.stop()
        assert summary.update_blocks is False
        # Create-only: no DELETE, no block append on the existing page.
        assert "DELETE" not in {m for m, _ in fake.calls}
        assert len(fake.pages[page_id]["blocks"]) == before


class TestQualityFilter:
    @pytest.mark.asyncio
    async def test_min_particle_confidence_drops_below_threshold(self, db_session: object) -> None:
        # 0.9 and 0.8 are kept at threshold 0.85? no — 0.8 < 0.85 drops.
        await _seed(confidences=[0.9, 0.8, 0.3])
        fake = _FakeNotion("Particle Subject ID")
        ctx = _patch_notion(fake)
        try:
            async with session_scope() as session:
                summary = await NotionExporter().export(
                    session,
                    None,
                    database_id=_DB_ID,
                    dry_run=True,
                    min_particle_confidence=0.85,
                )
        finally:
            ctx.stop()
        assert summary.particles_synced == 1  # only 0.9 survives
        assert summary.particles_dropped_below_threshold == 2


class TestMappingHelpers:
    def test_subject_properties_carry_title_and_upsert_key(self) -> None:
        subj = Subject(id="sub-123", canonical_name="Prometheus", asserted_by="t")
        props = _subject_properties(subj, "Prometheus (software)", 4, id_property="PID")
        assert props["Name"]["title"][0]["text"]["content"] == "Prometheus (software)"
        assert props["PID"]["rich_text"][0]["text"]["content"] == "sub-123"
        assert props["Particle Count"]["number"] == 4

    def test_is_managed_heading_matches_sentinel(self) -> None:
        block = {
            "type": "heading_2",
            "heading_2": {"rich_text": [{"plain_text": "Particles (managed)"}]},
        }
        assert _is_managed_heading(block, "Particles (managed)") is True
        assert _is_managed_heading(block, "Something else") is False
        assert _is_managed_heading({"type": "paragraph"}, "Particles (managed)") is False
