"""Tests for the SCHEMA_VERSION mismatch guard.

Covers both the pure exception shape (operator message contains the
exact remediation commands) and the integration: query refuses with the
exception when called against a store with mismatched particles, and the
CLI ``run()`` helper translates the exception to a clean exit + stderr
message.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import (
    SCHEMA_VERSION,
    AudienceHint,
    Confidence,
    Particle,
    QueryRequest,
    SchemaVersionMismatchError,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.operations.version_guard import assert_store_schema_current
from particles.store.particle_store import insert_particle


def _make_particle(*, schema_version: str = SCHEMA_VERSION) -> Particle:
    """Build an ACTIVE Particle with a specific schema_version override.

    Test-only: production code MUST NOT construct particles with a
    non-current schema_version; this fixture exists precisely to
    simulate the legacy-store condition the guard protects against.
    """
    return Particle(
        id=str(uuid.uuid4()),
        content="A test claim about a thing.",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        schema_version=schema_version,
    )


# ---------------------------------------------------------------------------
# SchemaVersionMismatchError shape
# ---------------------------------------------------------------------------


class TestExceptionShape:
    def test_message_names_the_remediation_commands(self) -> None:
        exc = SchemaVersionMismatchError(
            current_version="1.0.0",
            found_versions={"1.0.0": 42, "0.3.0": 7, "0.2.5": 2},
        )
        msg = str(exc)
        # The two mandated commands surface verbatim.
        assert "particles db init --force" in msg
        assert "particles extract --all-pending" in msg
        # The upgrade path is spelled out so the operator knows the policy.
        assert "upgrade path" in msg
        # The current and mismatched versions appear so the operator
        # can sanity-check the diagnosis.
        assert "v1.0.0" in msg
        assert "v0.2.5" in msg
        assert "v0.3.0" in msg
        # Mismatched counts are surfaced (7 + 2 = 9 mismatched).
        assert "7" in msg
        assert "2" in msg
        # The matching count (42) is NOT in the breakdown — only mismatches.
        assert "42" not in msg or "42 at v" not in msg

    def test_message_sorts_mismatched_versions(self) -> None:
        """Deterministic ordering for the operator's eye (and test stability)."""
        exc = SchemaVersionMismatchError(
            current_version="1.0.0",
            found_versions={"0.3.0": 1, "0.2.5": 1, "0.1.0": 1, "1.0.0": 1},
        )
        msg = str(exc)
        # 0.1.0 appears before 0.2.5 appears before 0.3.0
        i_010 = msg.index("v0.1.0")
        i_025 = msg.index("v0.2.5")
        i_030 = msg.index("v0.3.0")
        assert i_010 < i_025 < i_030

    def test_attributes_preserved_for_programmatic_callers(self) -> None:
        """The exception carries the data so API/MCP layers can re-render
        in their own structured format instead of parsing str()."""
        exc = SchemaVersionMismatchError(
            current_version="1.0.0",
            found_versions={"1.0.0": 10, "0.3.0": 3},
        )
        assert exc.current_version == "1.0.0"
        assert exc.found_versions == {"1.0.0": 10, "0.3.0": 3}


# ---------------------------------------------------------------------------
# assert_store_schema_current — integration with the store
# ---------------------------------------------------------------------------


class TestGuard:
    @pytest.mark.asyncio
    async def test_no_op_on_empty_store(self, db_session: AsyncSession) -> None:
        # Empty store has no mismatches; guard is silent.
        await assert_store_schema_current(db_session)

    @pytest.mark.asyncio
    async def test_no_op_when_all_particles_match(self, db_session: AsyncSession) -> None:
        await insert_particle(db_session, _make_particle())
        await insert_particle(db_session, _make_particle())
        await db_session.commit()
        # All particles at current SCHEMA_VERSION; guard is silent.
        await assert_store_schema_current(db_session)

    @pytest.mark.asyncio
    async def test_raises_when_any_active_particle_is_legacy(
        self, db_session: AsyncSession
    ) -> None:
        # One current + one legacy. The guard must raise.
        await insert_particle(db_session, _make_particle())
        await insert_particle(db_session, _make_particle(schema_version="0.3.0"))
        await db_session.commit()

        with pytest.raises(SchemaVersionMismatchError) as exc_info:
            await assert_store_schema_current(db_session)

        assert exc_info.value.current_version == SCHEMA_VERSION
        assert exc_info.value.found_versions.get("0.3.0") == 1
        assert exc_info.value.found_versions.get(SCHEMA_VERSION) == 1


# ---------------------------------------------------------------------------
# Query refuses on mismatch (integration: full operation entry point)
# ---------------------------------------------------------------------------


class TestQueryRefuses:
    @pytest.mark.asyncio
    async def test_query_raises_when_store_has_legacy_particle(
        self, db_session: AsyncSession
    ) -> None:
        from particles.operations.query import query

        await insert_particle(db_session, _make_particle(schema_version="0.3.0"))
        await db_session.commit()

        req = QueryRequest(question="anything", audience=AudienceHint.GENERAL)
        with pytest.raises(SchemaVersionMismatchError):
            await query(db_session, req)


# ---------------------------------------------------------------------------
# CLI run() translates the exception to a clean exit
# ---------------------------------------------------------------------------


class TestCliTranslation:
    def test_run_catches_and_translates_to_typer_exit(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI run() helper translates SchemaVersionMismatchError to
        a typer.Exit(1) + stderr message — no Python traceback bleeds
        through to the operator."""
        import typer

        from particles.api.cli import run

        async def _boom() -> None:
            raise SchemaVersionMismatchError(
                current_version="1.0.0",
                found_versions={"1.0.0": 5, "0.3.0": 2},
            )

        with pytest.raises(typer.Exit) as exit_info:
            run(_boom())

        assert exit_info.value.exit_code == 1
        # The canonical operator message reached stderr.
        captured = capsys.readouterr()
        assert "particles db init --force" in captured.err
        assert "upgrade path" in captured.err


# ---------------------------------------------------------------------------
# `extract --all-pending` fails fast on schema mismatch (no per-snapshot noise)
# ---------------------------------------------------------------------------


class TestExtractAllPendingFailFast:
    """Regression: when the particle store has mismatched-schema particles,
    ``extract --all-pending`` previously called the per-snapshot guard inside
    its loop and printed the (long) operator message once per pending snapshot.
    The upfront guard call lets the CLI's ``run()`` helper translate one
    raise into a single stderr line + exit 1.
    """

    def test_one_message_not_per_snapshot(
        self,
        cli_db: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio
        import uuid
        from datetime import UTC, datetime

        from typer.testing import CliRunner

        from particles.api.cli import app
        from particles.core.schema import (
            CorpusEntry,
            ExtractionStatus,
            Snapshot,
            WarcRecordType,
        )
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.db import session_scope

        # Seed THREE pending snapshots so the old per-snapshot behaviour
        # would have printed the message three times.
        async def _seed() -> None:
            async with session_scope() as session:
                for i in range(3):
                    entry = CorpusEntry(
                        entry_id=str(uuid.uuid4()),
                        source_type="WEB_PAGE",
                        uri_r=f"https://example.com/{i}",
                        deposited_by="test",
                    )
                    snap = Snapshot(
                        snapshot_id=str(uuid.uuid4()),
                        captured_at=datetime.now(UTC),
                        content_hash="a" * 64,
                        extraction_status=ExtractionStatus.PENDING,
                        warc_record_type=WarcRecordType.RESPONSE,
                    )
                    session.add(CorpusEntryRow.from_model(entry))
                    session.add(SnapshotRow.from_model(snap, entry.entry_id))
                # Seed one mismatched-schema ACTIVE particle so the guard fires.
                from particles.store.particle_store import insert_particle

                await insert_particle(session, _make_particle(schema_version="0.3.0"))
                await session.commit()

        asyncio.run(_seed())

        # Anthropic key is checked first by _extract_all_pending; stub it.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        from particles.llm import set_client

        set_client(object())  # any non-None value satisfies get_client()
        try:
            runner = CliRunner()
            result = runner.invoke(app, ["extract", "--all-pending"])
        finally:
            set_client(None)

        assert result.exit_code == 1
        # The operator message must appear exactly once, not three times —
        # the upfront guard short-circuits before the per-snapshot loop.
        output = (result.stdout or "") + (result.output or "")
        # Use a stable substring from the canonical error message.
        signature = "scrap-and-re-extract"
        assert output.count(signature) == 1, (
            f"expected 1 occurrence of {signature!r}, got {output.count(signature)}"
            f" — output was:\n{output}"
        )
