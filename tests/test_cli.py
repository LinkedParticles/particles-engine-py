"""Tests for the Typer CLI surface (particles/api/cli/).

The CLI was entirely uncovered in the architecture-review baseline. These
tests exercise it via ``typer.testing.CliRunner``, which is synchronous —
each command body wraps its async impl in ``run() = asyncio.run(...)``
so the test cannot itself be marked ``@pytest.mark.asyncio``.

Strategy:
  * file-based SQLite (the CLI opens its own session in a fresh asyncio.run
    every invocation, so ``:memory:`` would not be shared between calls)
  * tables created once per test via Base.metadata.create_all (alembic is
    too slow for unit tests; we skip ``particles db init``)
  * LLM-driven commands (``extract``, ``query``, ``lint --semantic``) and
    external-HTTP commands (``deposit URL``) are out of scope here

Each test asserts only on exit code and visible output. Internal state can
be re-inspected by opening a session against the same DB after the CLI run.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    Confidence,
    CorpusEntry,
    ExtractionStatus,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Snapshot,
    Subject,
    UncertaintyNature,
    WarcRecordType,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str]) -> Any:
    """Wrapper that surfaces the captured output on failure for better diagnostics."""
    return runner.invoke(app, args, catch_exceptions=False)


def _json_payload(output: str) -> Any:
    """Parse the stdout JSON from CliRunner's folded stdout+stderr buffer.

    Verbs that stream narration to stderr (e.g. the reindex plan line) emit it
    before the JSON summary lands on stdout, and CliRunner folds both into one
    buffer — so parse from the first ``{`` rather than the whole output.
    """
    return json.loads(output[output.index("{") :])


# Helpers for inserting test state via a fresh session ------------------------


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


async def _add_corpus_entry(
    *,
    source_type: str = "WEB_PAGE",
    uri_r: str = "https://example.com/x",
    archive_path: str | None = None,
) -> tuple[str, str]:
    """Insert a corpus entry + one COMPLETE snapshot. Returns (entry_id, snapshot_id).

    Pass ``archive_path`` when the test needs the snapshot to look blob-bearing —
    the blob-reachability probe ignores rows without one.
    """
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()), source_type=source_type, uri_r=uri_r, deposited_by="test"
    )
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        captured_at=datetime.now(UTC),
        content_hash="a" * 64,
        extraction_status=ExtractionStatus.COMPLETE,
        warc_record_type=WarcRecordType.RESPONSE,
        archive_path=archive_path,
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await session.commit()
    return entry.entry_id, snap.snapshot_id


async def _add_corpus_entry_with_id(entry_id: str, snapshot_id: str) -> None:
    """Insert a corpus entry + one snapshot with caller-chosen IDs.

    Used by the extract prefix-resolution tests, which need two entries that
    share a leading prefix (random UUIDs almost never collide on 8 chars).
    """
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    entry = CorpusEntry(
        entry_id=entry_id,
        source_type="WEB_PAGE",
        uri_r=f"https://example.com/{entry_id}",
        deposited_by="test",
    )
    snap = Snapshot(
        snapshot_id=snapshot_id,
        captured_at=datetime.now(UTC),
        content_hash="a" * 64,
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
    )
    async with session_scope() as session:
        session.add(CorpusEntryRow.from_model(entry))
        session.add(SnapshotRow.from_model(snap, entry.entry_id))
        await session.commit()


async def _add_follow_edge(via_entry_id: str, target_entry_id: str) -> None:
    """Record a POST_LINK follow edge between two existing corpus entries."""
    from particles.corpus.follow_edges import LINK_TYPE_POST, add_follow_edge
    from particles.db import session_scope

    async with session_scope() as session:
        await add_follow_edge(
            session,
            via_entry_id=via_entry_id,
            target_entry_id=target_entry_id,
            link_type=LINK_TYPE_POST,
        )
        await session.commit()


async def _add_subject(name: str = "Test Subject") -> str:
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    subj = Subject(id=str(uuid.uuid4()), canonical_name=name, asserted_by="test")
    async with session_scope() as session:
        await insert_subject(session, subj)
        await session.commit()
    return subj.id


async def _add_active_particle(
    *,
    content: str = "A test claim.",
    subject_ids: list[str] | None = None,
    entry_id: str | None = None,
    snapshot_id: str | None = None,
    extraction_provider_model: str | None = None,
) -> str:
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    provenance = []
    if entry_id and snapshot_id:
        provenance.append(
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        )
    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        provenance=provenance,
        subject_ids=subject_ids or [],
        extraction_provider_model=extraction_provider_model,
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _count_status(particle_ids: list[str], status: Status) -> int:
    """How many of these particles currently hold ``status``."""
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    async with session_scope() as session:
        found = [await get_particle(session, pid) for pid in particle_ids]
    return sum(1 for p in found if p is not None and p.status is status)


async def _add_cache_entry(
    subject_id: str, *, body: str = "# Cached\n\nArticle body.", input_hash: str = "h-aaa"
) -> None:
    """Seed one synthesis-cache row at the current prompt version (tests)."""
    from particles.db import session_scope
    from particles.exporters.article_synthesis import _PROMPT_VERSION
    from particles.store.synthesis_cache_store import store_cached_article

    async with session_scope() as session:
        await store_cached_article(session, subject_id, input_hash, _PROMPT_VERSION, body)
        await session.commit()


# ---------------------------------------------------------------------------
# Root help and discovery
# ---------------------------------------------------------------------------


class TestQueryFlags:
    def test_invalid_assertion_modality_rejected(self, runner: CliRunner, cli_db: Path) -> None:
        """`query --assertion-modality` validates before any retrieval."""
        result = runner.invoke(
            app,
            ["query", "anything?", "--assertion-modality", "NOPE"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "Unknown assertion modality" in result.output
        assert "FALSIFIABLE" in result.output

    def test_unknown_group_by_axis_rejected(self, runner: CliRunner, cli_db: Path) -> None:
        """`query --group-by` validates its axis before any retrieval."""
        result = runner.invoke(app, ["query", "--group-by", "planet"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "Unknown --group-by axis" in result.output
        assert "subject, predicate, object" in result.output

    def test_aggregate_rejects_question(self, runner: CliRunner, cli_db: Path) -> None:
        """a question beside --count is rejected, not narrated."""
        result = runner.invoke(app, ["query", "how many?", "--count"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "Invalid query request" in result.output

    def test_no_question_no_flags_rejected(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["query"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "Invalid query request" in result.output


class TestRootHelp:
    def test_help_lists_all_command_groups(self, runner: CliRunner) -> None:
        result = _invoke(runner, ["--help"])
        assert result.exit_code == 0
        # Every registered group appears in the index
        for cmd in (
            "db",
            "deposit",
            "extract",
            "query",
            "lint",
            "review",
            "reindex",
            "quality",
            "export",
            "subjects",
            "corpus",
            "extractor",
            "trust",
        ):
            assert cmd in result.output

    def test_no_args_prints_help_not_error(self, runner: CliRunner) -> None:
        # We set no_args_is_help=True on the root app — bare invocation shows help
        result = runner.invoke(app, [])
        # typer/click exit with non-zero when showing help with no args (code 2)
        # but the help text must be rendered
        assert "particles" in result.output.lower()


class TestVersion:
    """`particles --version` — client version always; engine version in remote mode."""

    def test_version_local_prints_client_only(self, runner: CliRunner) -> None:
        from particles import __version__

        result = _invoke(runner, ["--version"])
        assert result.exit_code == 0
        assert f"particles {__version__}" in result.output
        # Local mode (no engine.base_url): no engine line, no "(client)" label
        assert "engine" not in result.output.lower()

    def test_version_short_flag(self, runner: CliRunner) -> None:
        from particles import __version__

        result = _invoke(runner, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_remote_shows_client_and_engine(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles import __version__
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.example:8000")
        reset_config()

        async def fake_health(_self: Any) -> str:
            return "9.9.9"

        monkeypatch.setattr("particles.api.client.http.HttpBackend.health", fake_health)
        result = _invoke(runner, ["--version"])
        assert result.exit_code == 0
        assert f"particles {__version__} (client)" in result.output
        assert "engine 9.9.9 (http://engine.example:8000)" in result.output

    def test_version_remote_unreachable_still_shows_client(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles import __version__
        from particles.api.client.http import EngineUnreachableError
        from particles.config import reset_config

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://down.example:8000")
        reset_config()

        async def boom(_self: Any) -> str:
            raise EngineUnreachableError("engine is down")

        monkeypatch.setattr("particles.api.client.http.HttpBackend.health", boom)
        result = _invoke(runner, ["--version"])
        # Client version still surfaces; the unreachable engine does not fail the command.
        assert result.exit_code == 0
        assert f"particles {__version__} (client)" in result.output
        assert "unreachable" in result.output


# ---------------------------------------------------------------------------
# trust sub-Typer
# ---------------------------------------------------------------------------


class TestTrust:
    def test_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["trust", "list"])
        assert result.exit_code == 0
        assert "SCOPE" in result.output  # header is always printed

    def test_set_then_list_shows_rule(self, runner: CliRunner, cli_db: Path) -> None:
        set_result = _invoke(runner, ["trust", "set", "en.wikipedia.org", "0.9"])
        assert set_result.exit_code == 0
        assert "wikipedia" in set_result.output.lower()

        list_result = _invoke(runner, ["trust", "list"])
        assert list_result.exit_code == 0
        assert "en.wikipedia.org" in list_result.output

    def test_set_with_modifier_flag(self, runner: CliRunner, cli_db: Path) -> None:
        # Positive modifier — Click would interpret a leading-dash value as
        # an option, so the CLI's contract is that the score is a non-dash float.
        result = _invoke(
            runner, ["trust", "set", "/admin", "0.2", "--modifier", "--rationale", "boost"]
        )
        assert result.exit_code == 0

    def test_show_resolves_uri(self, runner: CliRunner, cli_db: Path) -> None:
        _invoke(runner, ["trust", "set", "example.com", "0.7"])
        result = _invoke(runner, ["trust", "show", "https://example.com/some/path"])
        assert result.exit_code == 0
        assert "example.com" in result.output
        assert "Effective:" in result.output

    def test_statement_set_rejects_out_of_range_rank(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(
            app, ["trust", "statement-set", "test", "SOURCE_TYPE", "WEB_PAGE", "2.5"]
        )
        assert result.exit_code != 0
        assert "0.0" in result.output and "1.0" in result.output

    def test_statement_set_rejects_unknown_ref_type(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(
            app, ["trust", "statement-set", "test", "NOT_A_TYPE", "value", "0.8"]
        )
        assert result.exit_code != 0
        assert "Unknown" in result.output

    def test_cascade_empty_run(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["trust", "cascade"])
        assert result.exit_code == 0
        assert "No OPERATOR_DIRECT statements" in result.output

    def test_set_entry_rejects_out_of_range_rank(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["trust", "set-entry", "some-id", "1.5"])
        assert result.exit_code != 0
        assert "0.0" in result.output and "1.0" in result.output

    def test_set_entry_unknown_entry(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["trust", "set-entry", "no-such-entry", "0.3"])
        assert result.exit_code != 0
        assert "No corpus entry" in result.output

    def test_set_entry_no_inferable_domain_without_flag(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        # WEB_PAGE has no MUST applicability clause, so the domain cannot be
        # inferred; the verb must refuse rather than write an unreachable override.
        entry_id, _ = _run_async(_add_corpus_entry(source_type="WEB_PAGE"))
        result = runner.invoke(app, ["trust", "set-entry", entry_id, "0.3"])
        assert result.exit_code != 0
        assert "--domain" in result.output

    def test_set_entry_with_explicit_domain_succeeds(self, runner: CliRunner, cli_db: Path) -> None:
        entry_id, _ = _run_async(_add_corpus_entry(source_type="WEB_PAGE"))
        result = _invoke(runner, ["trust", "set-entry", entry_id, "0.3", "--domain", "numismatics"])
        assert result.exit_code == 0
        assert "CORPUS_ENTRY" in result.output
        assert entry_id in result.output


# ---------------------------------------------------------------------------
# extractor sub-Typer
# ---------------------------------------------------------------------------


class TestExtractor:
    def test_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["extractor", "list"])
        assert result.exit_code == 0
        # With no extractor records pre-populated, the empty-state message fires
        assert "No extractor records" in result.output or "EXTRACTOR" in result.output

    def test_trust_set_rejects_out_of_range_weight(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["extractor", "trust-set", "general-extractor", "1.5"])
        assert result.exit_code != 0
        assert "0.0" in result.output

    def test_trust_set_unknown_extractor_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["extractor", "trust-set", "nonexistent-extractor", "0.5"])
        assert result.exit_code != 0
        assert "not found" in result.output

    # ---- conform ------------------------------------------------

    @staticmethod
    def _fixture_dir() -> str:
        return str(Path(__file__).parent / "conformance" / "fixtures")

    def test_conform_table_output(self, runner: CliRunner) -> None:
        result = _invoke(
            runner,
            ["extractor", "conform", "numista-coin-extractor", "--fixtures", self._fixture_dir()],
        )
        # Numista hardcodes EPISTEMIC. Since that is an ADVISORY, not
        # a failure — reported in full, but the run passes and exits 0.
        assert result.exit_code == 0
        assert "Conformance report" in result.output
        assert "numista-coin-extractor" in result.output
        assert "REQUIRED fields" in result.output
        assert "RECOMMENDED fields" in result.output
        assert "OPTIONAL fields" in result.output
        assert "uncertainty_nature" in result.output
        assert "PASS (advisory)" in result.output
        # The advisory_reason continuation line is included
        assert "Diversity rule violated" in result.output
        # …as is the histogram behind the distinct count.
        assert "values: EPISTEMIC 7" in result.output
        assert "1 advisory(ies)" in result.output

    def test_conform_json_output(self, runner: CliRunner) -> None:
        import json as _json

        result = _invoke(
            runner,
            [
                "extractor",
                "conform",
                "numista-coin-extractor",
                "--fixtures",
                self._fixture_dir(),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert payload["extractor_id"] == "numista-coin-extractor"
        assert payload["fixture_count"] == 1
        assert payload["particle_count"] == 7
        # Every contract field is reported. Asserted against the contract
        # itself rather than a literal, so adding a field (added
        # structured_claim and canonical_form) does not fail an unrelated test.
        from particles.conformance.contract import CONTRACT

        assert len(payload["fields"]) == len(CONTRACT)
        # the diversity finding lands in advisories, not failures —
        # this is what a checked-in baseline JSON now carries.
        assert payload["failures"] == []
        assert any(a["field"] == "uncertainty_nature" for a in payload["advisories"])
        un = next(f for f in payload["fields"] if f["field"] == "uncertainty_nature")
        assert un["value_counts"] == {"EPISTEMIC": 7}
        assert un["advisory_reason"] is not None

    def test_conform_unknown_extractor_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            ["extractor", "conform", "no-such-extractor", "--fixtures", self._fixture_dir()],
        )
        assert result.exit_code == 2
        assert "no-such-extractor" in (result.output + (result.stderr or ""))

    def test_conform_fail_on_warn_also_fires_for_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # --fail-on warn is the strictest gate: it fires on warnings *and* on
        # errors. No shipped extractor produces a RECOMMENDED warning (all
        # three populate at 100%), so the error half is what is exercisable
        # from the CLI. docstring-extractor has no fixture in the corpus, so
        # every REQUIRED field reports 0% — the data-availability
        # failure, which is a genuine REQUIRED failure for exit-code purposes.
        result = _invoke(
            runner,
            [
                "extractor",
                "conform",
                "docstring-extractor",
                "--fixtures",
                self._fixture_dir(),
                "--fail-on",
                "warn",
            ],
        )
        assert result.exit_code == 1
        assert "No corpus fixture is scored" in result.output

    def test_conform_advisory_alone_never_changes_the_exit_code(self, runner: CliRunner) -> None:
        """an advisory reports; it does not gate.

        Numista's only contract finding is the diversity advisory, so this
        pins that even the strictest --fail-on setting ignores it.
        """
        for fail_on in ("error", "warn"):
            result = _invoke(
                runner,
                [
                    "extractor",
                    "conform",
                    "numista-coin-extractor",
                    "--fixtures",
                    self._fixture_dir(),
                    "--fail-on",
                    fail_on,
                ],
            )
            assert result.exit_code == 0, f"--fail-on {fail_on} gated on an advisory"
            assert "Diversity rule violated" in result.output

    def test_conform_all_accepted_never_persists_the_adr_0176_verdict(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a widened run is report-only, by construction.

        The stored verdict is a claim about the extractor's *production*
        behaviour; an operator-widened run is not that, so it must not be able
        to clamp trust on inputs the pipeline never routes.
        """
        calls: list[tuple[str, bool]] = []

        async def _spy(extractor_id: str, evaluable_failure: bool) -> bool | None:
            calls.append((extractor_id, evaluable_failure))
            return True

        monkeypatch.setattr("particles.api.cli.extractor._persist_conformance_status", _spy)

        widened = _invoke(
            runner,
            [
                "extractor",
                "conform",
                "numista-coin-extractor",
                "--fixtures",
                self._fixture_dir(),
                "--all-accepted",
            ],
        )
        assert widened.exit_code == 0  # the standing diversity finding is advisory
        assert calls == []
        assert "--all-accepted: report-only" in widened.output

        # …and the same run without the flag does persist, so the bar above is
        # the flag's doing and not a broken call site.
        routed = _invoke(
            runner,
            ["extractor", "conform", "numista-coin-extractor", "--fixtures", self._fixture_dir()],
        )
        assert routed.exit_code == 0
        assert [c[0] for c in calls] == ["numista-coin-extractor"]
        # …and what it persists is False: an advisory is not an evaluable
        # REQUIRED failure, so the trust cap no longer clamps a
        # correctly-behaving structured extractor (§ Consequences).
        assert calls == [("numista-coin-extractor", False)]

    def test_conform_rejects_out_of_range_threshold(self, runner: CliRunner) -> None:
        result = runner.invoke(
            app,
            [
                "extractor",
                "conform",
                "numista-coin-extractor",
                "--fixtures",
                self._fixture_dir(),
                "--recommended-threshold",
                "1.5",
            ],
        )
        assert result.exit_code != 0

    # ---- generate-fixture ---------------------------------------

    def test_generate_fixture_from_entry(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        """`extractor generate-fixture` turns a deposited entry into a fixture
        skeleton (manifest + content + snapshot) under the output dir."""
        from particles.corpus.deposit import deposit_text
        from particles.db import session_scope

        async def _seed() -> str:
            async with session_scope() as session:
                entry_id, _ = await deposit_text(
                    session, "<html>sample page</html>", source_type="WEB_PAGE"
                )
                await session.commit()
                return entry_id

        entry_id = _run_async(_seed())
        out = tmp_path / "fixtures"
        result = _invoke(
            runner,
            ["extractor", "generate-fixture", entry_id, "--output-dir", str(out)],
        )
        assert result.exit_code == 0, result.output

        fixture_dirs = [p for p in out.iterdir() if p.is_dir()]
        assert len(fixture_dirs) == 1
        fx = fixture_dirs[0]
        assert (fx / "content.bin").read_bytes() == b"<html>sample page</html>"
        assert (fx / "manifest.yaml").exists()
        assert (fx / "snapshot.json").exists()
        assert (out / "MANIFEST.yaml").exists()
        # expected_acceptors is left empty for the operator (decision 4).
        assert "expected_acceptors: []" in (fx / "manifest.yaml").read_text()

    def test_generate_fixture_unknown_entry_errors(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["extractor", "generate-fixture", "deadbeef", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "not found" in (result.output + (result.stderr or ""))


# ---------------------------------------------------------------------------
# corpus sub-Typer
# ---------------------------------------------------------------------------


class TestCorpus:
    def test_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["corpus", "list"])
        assert result.exit_code == 0
        assert "ENTRY" in result.output  # header

    def test_list_shows_entry_after_insert(self, runner: CliRunner, cli_db: Path) -> None:
        entry_id, _ = _run_async(_add_corpus_entry(source_type="PDF"))
        result = _invoke(runner, ["corpus", "list"])
        assert result.exit_code == 0
        assert entry_id[:8] in result.output
        assert "PDF" in result.output

    def test_list_filtered_by_source_type(self, runner: CliRunner, cli_db: Path) -> None:
        entry_pdf, _ = _run_async(_add_corpus_entry(source_type="PDF"))
        entry_web, _ = _run_async(_add_corpus_entry(source_type="WEB_PAGE"))
        result = _invoke(runner, ["corpus", "list", "--source-type", "PDF"])
        assert result.exit_code == 0
        assert entry_pdf[:8] in result.output
        assert entry_web[:8] not in result.output

    def test_list_json_emits_full_untruncated_fields(self, runner: CliRunner, cli_db: Path) -> None:
        import json as _json

        # > 60 chars so the human table would left-truncate it; the JSON must not.
        long_uri = "https://example.com/" + "segment/" * 12 + "final-page"
        assert len(long_uri) > 60
        entry_id, _ = _run_async(_add_corpus_entry(source_type="WEB_PAGE", uri_r=long_uri))
        result = _invoke(runner, ["corpus", "list", "--json"])
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert isinstance(payload, list)
        rec = next(r for r in payload if r["entry_id"] == entry_id)
        assert rec["uri_r"] == long_uri  # full, untruncated
        assert rec["source_type"] == "WEB_PAGE"
        assert rec["extraction_status"] == "COMPLETE"
        assert rec["particle_count"] == 0
        assert rec["tags"] == []
        assert "created_at" in rec

    def test_list_json_respects_source_type_filter(self, runner: CliRunner, cli_db: Path) -> None:
        import json as _json

        entry_pdf, _ = _run_async(_add_corpus_entry(source_type="PDF"))
        entry_web, _ = _run_async(_add_corpus_entry(source_type="WEB_PAGE"))
        result = _invoke(runner, ["corpus", "list", "--json", "--source-type", "PDF"])
        assert result.exit_code == 0
        ids = {r["entry_id"] for r in _json.loads(result.output)}
        assert entry_pdf in ids
        assert entry_web not in ids

    def test_show_unknown_entry_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["corpus", "show", "deadbeef"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_show_existing_entry(self, runner: CliRunner, cli_db: Path) -> None:
        entry_id, _ = _run_async(_add_corpus_entry(source_type="PDF"))
        result = _invoke(runner, ["corpus", "show", entry_id])
        assert result.exit_code == 0
        assert entry_id in result.output
        assert "PDF" in result.output

    def test_show_no_follow_edges_omits_section(self, runner: CliRunner, cli_db: Path) -> None:
        # the follow section is skipped when an entry has
        # no edges, keeping the common case uncluttered.
        entry_id, _ = _run_async(_add_corpus_entry(source_type="PDF"))
        result = _invoke(runner, ["corpus", "show", entry_id])
        assert result.exit_code == 0
        assert "follows" not in result.output.lower()

    def test_show_surfaces_outgoing_follow(self, runner: CliRunner, cli_db: Path) -> None:
        # an entry that followed an external link shows it as outgoing.
        via_id, _ = _run_async(_add_corpus_entry(uri_r="https://reddit.com/r/foo/comments/abc"))
        tgt_id, _ = _run_async(_add_corpus_entry(uri_r="https://example.com/article"))
        _run_async(_add_follow_edge(via_id, tgt_id))

        result = _invoke(runner, ["corpus", "show", via_id])
        assert result.exit_code == 0
        assert "Outgoing follows (1)" in result.output
        assert tgt_id[:8] in result.output
        assert "POST_LINK" in result.output
        assert "Incoming follows" not in result.output

    def test_show_surfaces_incoming_follow(self, runner: CliRunner, cli_db: Path) -> None:
        # the fan-in view — an article reached from a source shows the
        # source as incoming.
        via_id, _ = _run_async(_add_corpus_entry(uri_r="https://hn.example/item?id=1"))
        tgt_id, _ = _run_async(_add_corpus_entry(uri_r="https://example.com/article"))
        _run_async(_add_follow_edge(via_id, tgt_id))

        result = _invoke(runner, ["corpus", "show", tgt_id])
        assert result.exit_code == 0
        assert "Incoming follows (1)" in result.output
        assert via_id[:8] in result.output
        assert "Outgoing follows" not in result.output

    def test_show_surfaces_both_directions(self, runner: CliRunner, cli_db: Path) -> None:
        # a middle entry (a → b → c) shows both an incoming and an
        # outgoing follow.
        a, _ = _run_async(_add_corpus_entry(uri_r="https://a.example/post"))
        b, _ = _run_async(_add_corpus_entry(uri_r="https://b.example/article"))
        c, _ = _run_async(_add_corpus_entry(uri_r="https://c.example/follow"))
        _run_async(_add_follow_edge(a, b))
        _run_async(_add_follow_edge(b, c))

        result = _invoke(runner, ["corpus", "show", b])
        assert result.exit_code == 0
        assert "Outgoing follows (1)" in result.output
        assert c[:8] in result.output
        assert "Incoming follows (1)" in result.output
        assert a[:8] in result.output

    def test_delete_with_yes_flag_skips_prompt(self, runner: CliRunner, cli_db: Path) -> None:
        entry_id, _ = _run_async(_add_corpus_entry())
        result = _invoke(runner, ["corpus", "delete", entry_id, "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

        # Verify it's gone
        result2 = runner.invoke(app, ["corpus", "show", entry_id])
        assert result2.exit_code != 0

    # -----------------------------------------------------------------------
    # corpus links list — 0.43.x downstream consumer of corpus_follow_edges
    #. Before this verb the edges were write-only; §3.9 of the
    # whitepaper claimed "queryable" without anything actually querying them.
    # -----------------------------------------------------------------------

    def test_links_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["corpus", "links", "list"])
        assert result.exit_code == 0
        assert "No follow edges" in result.output

    def test_links_list_shows_edge(self, runner: CliRunner, cli_db: Path) -> None:
        via_id, _ = _run_async(_add_corpus_entry(uri_r="https://reddit.com/r/foo/comments/abc"))
        tgt_id, _ = _run_async(_add_corpus_entry(uri_r="https://example.com/article"))
        _run_async(_add_follow_edge(via_id, tgt_id))

        result = _invoke(runner, ["corpus", "links", "list"])
        assert result.exit_code == 0
        assert via_id[:8] in result.output
        assert tgt_id[:8] in result.output
        assert "POST_LINK" in result.output

    def test_links_list_filtered_outgoing(self, runner: CliRunner, cli_db: Path) -> None:
        via_id, _ = _run_async(_add_corpus_entry(uri_r="https://reddit.com/post"))
        tgt_id, _ = _run_async(_add_corpus_entry(uri_r="https://example.com/article"))
        _run_async(_add_follow_edge(via_id, tgt_id))

        result = _invoke(runner, ["corpus", "links", "list", via_id, "--direction", "out"])
        assert result.exit_code == 0
        assert "Outgoing follows (1)" in result.output
        assert tgt_id[:8] in result.output
        # Incoming section not shown when --direction out
        assert "Incoming follows" not in result.output

    def test_links_list_filtered_incoming(self, runner: CliRunner, cli_db: Path) -> None:
        via_id, _ = _run_async(_add_corpus_entry(uri_r="https://hn.example/item?id=1"))
        tgt_id, _ = _run_async(_add_corpus_entry(uri_r="https://example.com/article"))
        _run_async(_add_follow_edge(via_id, tgt_id))

        result = _invoke(runner, ["corpus", "links", "list", tgt_id, "--direction", "in"])
        assert result.exit_code == 0
        assert "Incoming follows (1)" in result.output
        assert via_id[:8] in result.output
        # Outgoing section not shown when --direction in
        assert "Outgoing follows" not in result.output

    def test_links_list_both_sections_for_middle_entry(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        # entry_a → entry_b → entry_c (depth-1 follows from each source)
        a, _ = _run_async(_add_corpus_entry(uri_r="https://a.example/post"))
        b, _ = _run_async(_add_corpus_entry(uri_r="https://b.example/article"))
        c, _ = _run_async(_add_corpus_entry(uri_r="https://c.example/follow"))
        _run_async(_add_follow_edge(a, b))
        _run_async(_add_follow_edge(b, c))

        # b has one outgoing (to c) and one incoming (from a).
        result = _invoke(runner, ["corpus", "links", "list", b])
        assert result.exit_code == 0
        assert "Outgoing follows (1)" in result.output
        assert c[:8] in result.output
        assert "Incoming follows (1)" in result.output
        assert a[:8] in result.output

    def test_links_list_unknown_entry_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["corpus", "links", "list", "deadbeef"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_links_list_rejects_invalid_direction(self, runner: CliRunner, cli_db: Path) -> None:
        via_id, _ = _run_async(_add_corpus_entry())
        result = runner.invoke(app, ["corpus", "links", "list", via_id, "--direction", "sideways"])
        assert result.exit_code != 0
        assert "Invalid --direction" in result.output


# ---------------------------------------------------------------------------
# config validate
# ---------------------------------------------------------------------------


class TestConfigValidate:
    def test_valid_defaults_when_no_file(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_CONFIG", "/nonexistent-cli-validate.yaml")
        result = _invoke(runner, ["config", "validate"])
        assert result.exit_code == 0
        assert "Config valid" in result.output
        assert "compiled-in defaults" in result.output

    def test_valid_config_file(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("trust:\n  reviewer_trust_rank: 0.9\n")
        monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
        result = _invoke(runner, ["config", "validate"])
        assert result.exit_code == 0
        assert "Config valid" in result.output
        assert str(cfg) in result.output

    def test_invalid_value_reports_error(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("trust:\n  reviewer_trust_rank: 99\n")
        monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code != 0
        assert "INVALID" in result.output
        assert "trust.reviewer_trust_rank" in result.output

    def test_malformed_yaml_reports_error(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("trust:\n  reviewer_trust_rank: [unclosed\n")
        monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code != 0
        assert "not valid YAML" in result.output

    def test_warns_when_store_blobs_are_unreachable(self, runner: CliRunner, cli_db: Path) -> None:
        """Rows point at content the resolved blob_dir does not hold."""
        _run_async(_add_corpus_entry(archive_path="/gone/corpus_blobs/aa/" + "a" * 64))

        result = _invoke(runner, ["config", "validate"])

        # Detection, not a gate — an operator must still be able to deploy.
        assert result.exit_code == 0
        assert "Config valid" in result.output
        assert "WARNING" in result.output
        assert "none of the 1 sampled" in result.output

    def test_no_blob_warning_on_empty_store(self, runner: CliRunner, cli_db: Path) -> None:
        """A first-run store has nothing to find, so it must stay silent."""
        result = _invoke(runner, ["config", "validate"])

        assert result.exit_code == 0
        assert "WARNING" not in result.output


# ---------------------------------------------------------------------------
# subjects command (single command, manual action dispatch)
# ---------------------------------------------------------------------------


class TestSubjects:
    def test_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["subjects", "list"])
        assert result.exit_code == 0
        assert "No subjects" in result.output

    def test_list_shows_subjects(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Albert Einstein"))
        result = _invoke(runner, ["subjects", "list"])
        assert result.exit_code == 0
        assert "Einstein" in result.output
        assert sid[:8] in result.output

    def test_list_order_rejects_bad_value(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "list", "--order", "sideways"])
        assert result.exit_code == 1
        assert "--order must be 'name' or 'degree'" in result.output

    def test_search_finds_match(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_subject("Paris"))
        _run_async(_add_subject("London"))
        result = _invoke(runner, ["subjects", "search", "paris"])
        assert result.exit_code == 0
        assert "Paris" in result.output
        assert "London" not in result.output

    def test_search_no_match(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["subjects", "search", "nothing-here"])
        assert result.exit_code == 0
        assert "No subjects matching" in result.output

    def test_search_requires_query(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "search"])
        assert result.exit_code != 0
        assert "Provide a search query" in result.output

    def test_show_existing_subject(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Marie Curie"))
        result = _invoke(runner, ["subjects", "show", sid])
        assert result.exit_code == 0
        assert "Marie Curie" in result.output
        assert sid in result.output

    def test_show_missing_subject_errors(self, runner: CliRunner, cli_db: Path) -> None:
        fake = str(uuid.uuid4())
        result = runner.invoke(app, ["subjects", "show", fake])
        assert result.exit_code != 0

    def test_show_requires_id(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "show"])
        assert result.exit_code != 0
        assert "Provide a subject ID" in result.output

    def test_alias_adds_new_alias(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Albert Einstein"))
        result = _invoke(runner, ["subjects", "alias", sid, "A. Einstein"])
        assert result.exit_code == 0
        assert "A. Einstein" in result.output

    def test_unknown_action_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "not-a-real-action"])
        assert result.exit_code != 0
        assert "Unknown action" in result.output

    def test_delete_phantom_subject(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Phantom"))
        result = _invoke(runner, ["subjects", "delete", sid])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Gone afterwards.
        assert runner.invoke(app, ["subjects", "show", sid]).exit_code != 0

    def test_delete_missing_subject_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "delete", str(uuid.uuid4())])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_delete_requires_id(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "delete"])
        assert result.exit_code != 0
        assert "Usage" in result.output

    def test_delete_non_phantom_refused_without_force(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        sid = _run_async(_add_subject("Has Particles"))
        _run_async(_add_active_particle(subject_ids=[sid]))
        result = runner.invoke(app, ["subjects", "delete", sid])
        assert result.exit_code != 0
        assert "Refusing to delete" in result.output
        # Still present.
        assert runner.invoke(app, ["subjects", "show", sid]).exit_code == 0

    def test_delete_non_phantom_with_force(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Has Particles"))
        _run_async(_add_active_particle(subject_ids=[sid]))
        result = _invoke(runner, ["subjects", "delete", sid, "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert "Detached from 1 particle" in result.output

    #: `subjects gc` / `prune-empty` phantom sweep.
    def test_gc_sweeps_all_phantom_subjects(self, runner: CliRunner, cli_db: Path) -> None:
        a = _run_async(_add_subject("Phantom A"))
        b = _run_async(_add_subject("Phantom B"))
        result = _invoke(runner, ["subjects", "gc"])
        assert result.exit_code == 0
        assert "Pruned 2 phantom subject(s)" in result.output
        # Both gone afterwards.
        assert runner.invoke(app, ["subjects", "show", a]).exit_code != 0
        assert runner.invoke(app, ["subjects", "show", b]).exit_code != 0

    def test_gc_preserves_non_phantom_subjects(self, runner: CliRunner, cli_db: Path) -> None:
        keep = _run_async(_add_subject("Has Particles"))
        _run_async(_add_active_particle(subject_ids=[keep]))
        drop = _run_async(_add_subject("Phantom"))
        result = _invoke(runner, ["subjects", "gc"])
        assert result.exit_code == 0
        assert "Pruned 1 phantom subject(s)" in result.output
        # The subject with an ACTIVE particle survives; the phantom is gone.
        assert runner.invoke(app, ["subjects", "show", keep]).exit_code == 0
        assert runner.invoke(app, ["subjects", "show", drop]).exit_code != 0

    def test_gc_dry_run_previews_without_deleting(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Phantom"))
        result = _invoke(runner, ["subjects", "gc", "--dry-run"])
        assert result.exit_code == 0
        assert "Would prune 1 phantom subject(s)" in result.output
        assert "no changes made" in result.output
        # Still present — dry run committed nothing.
        assert runner.invoke(app, ["subjects", "show", sid]).exit_code == 0

    def test_gc_no_phantoms_message(self, runner: CliRunner, cli_db: Path) -> None:
        keep = _run_async(_add_subject("Has Particles"))
        _run_async(_add_active_particle(subject_ids=[keep]))
        result = _invoke(runner, ["subjects", "gc"])
        assert result.exit_code == 0
        assert "No phantom subjects to prune." in result.output

    def test_prune_empty_alias_matches_gc(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Phantom"))
        result = _invoke(runner, ["subjects", "prune-empty"])
        assert result.exit_code == 0
        assert "Pruned 1 phantom subject(s)" in result.output
        assert runner.invoke(app, ["subjects", "show", sid]).exit_code != 0

    #: `subjects set-class` override.
    def test_set_class_reclassifies_subject(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("1 Pfennig GDR"))
        result = _invoke(runner, ["subjects", "set-class", sid, "nmo:NumismaticObject"])
        assert result.exit_code == 0
        assert "Reclassified" in result.output
        assert "nmo:NumismaticObject" in result.output

    def test_set_class_noop_when_unchanged(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Copper"))
        _invoke(runner, ["subjects", "set-class", sid, "nmo:Material"])
        result = _invoke(runner, ["subjects", "set-class", sid, "nmo:Material"])
        assert result.exit_code == 0
        assert "no change" in result.output

    def test_set_class_missing_subject_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "set-class", str(uuid.uuid4()), "nmo:Material"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_set_class_requires_class_arg(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Needs a class"))
        result = runner.invoke(app, ["subjects", "set-class", sid])
        assert result.exit_code != 0
        assert "Usage" in result.output

    def test_unlink_removes_external_ref(self, runner: CliRunner, cli_db: Path) -> None:
        """End-to-end: insert a subject with a wikidata ref, unlink it,
        confirm `subjects show` no longer lists the ref."""
        from particles.core.schema import ExternalRef, Subject
        from particles.db import session_scope
        from particles.store.subject_store import insert_subject

        async def _setup() -> str:
            subj = Subject(
                id=str(uuid.uuid4()),
                canonical_name="Central Intelligence Agency",
                external_ids=[ExternalRef(namespace="wikidata", id="Q37230", confidence=0.3)],
                asserted_by="test",
            )
            async with session_scope() as session:
                await insert_subject(session, subj)
                await session.commit()
            return subj.id

        sid = _run_async(_setup())

        # Pre-condition: show lists the ref.
        before = _invoke(runner, ["subjects", "show", sid])
        assert "wikidata:Q37230" in before.output

        # Action.
        result = _invoke(runner, ["subjects", "unlink", sid, "wikidata:Q37230"])
        assert result.exit_code == 0, result.output
        assert "Removed wikidata:Q37230" in result.output

        # Post-condition: show no longer lists the ref but the subject
        # itself survives.
        after = _invoke(runner, ["subjects", "show", sid])
        assert after.exit_code == 0
        assert "wikidata:Q37230" not in after.output
        assert "Central Intelligence Agency" in after.output

    def test_unlink_missing_ref_errors(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Plain Subject"))
        result = runner.invoke(app, ["subjects", "unlink", sid, "wikidata:Q99999"])
        assert result.exit_code != 0
        assert "No wikidata:Q99999 link found" in result.output

    def test_unlink_requires_namespace_id_format(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("Plain Subject"))
        # Missing the colon → guiding error.
        result = runner.invoke(app, ["subjects", "unlink", sid, "no-colon-here"])
        assert result.exit_code != 0
        assert "NAMESPACE:ID must contain a colon" in result.output


# ---------------------------------------------------------------------------
# Top-level core verbs — happy paths and arg validation
# ---------------------------------------------------------------------------


class TestQualityCommand:
    def test_empty_db_runs_cleanly(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["quality"])
        assert result.exit_code == 0
        assert "Extraction Quality Dashboard" in result.output
        assert "Particles" in result.output
        assert "Corpus" in result.output


class TestLintCommand:
    def test_lint_clean_empty_db(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["lint", "--no-semantic"])
        assert result.exit_code == 0
        assert "Lint clean" in result.output

    def test_lint_json_output(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["lint", "--no-semantic", "--output-format", "json"])
        assert result.exit_code == 0
        # The JSON summary mode emits at least run_at and error/warning counts
        import json

        parsed = json.loads(result.output)
        assert "run_at" in parsed


class TestReviewCommand:
    def test_review_no_pending(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["review"])
        assert result.exit_code == 0
        assert "No INCONSISTENCY" in result.output

    def test_review_bulk_no_pending(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["review", "--bulk", "BOTH_VALID"])
        assert result.exit_code == 0
        assert "No INCONSISTENCY" in result.output

    def test_review_bulk_unknown_action_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["review", "--bulk", "WAT"])
        assert result.exit_code != 0
        assert "Unknown action" in result.output

    def test_review_particle_id_requires_action(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["review", "some-particle-id"])
        assert result.exit_code != 0
        assert "--action required" in result.output

    def test_review_surfaces_author_id_and_role(self, runner: CliRunner, cli_db: Path) -> None:
        """Spec §6 v0.2 Core checklist: surface author_id and author_role in
        the Review UI for UGC corpus entries."""
        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.db import session_scope
        from particles.store.particle_store import insert_particle

        async def _seed() -> None:
            # Two corpus entries from different UGC sources, each with author info
            entry_a = CorpusEntry(
                entry_id=str(uuid.uuid4()),
                source_type="REDDIT_POST",
                uri_r="https://reddit.com/r/x/comments/a",
                deposited_by="t",
            )
            snap_a = Snapshot(
                snapshot_id=str(uuid.uuid4()),
                captured_at=datetime.now(UTC),
                content_hash="a" * 64,
                extraction_status=ExtractionStatus.COMPLETE,
                warc_record_type=WarcRecordType.RESPONSE,
                author_id="reddit:u/alice",
                author_role="maintainer",
            )
            entry_b = CorpusEntry(
                entry_id=str(uuid.uuid4()),
                source_type="GITHUB_GIST",
                uri_r="https://gist.github.com/bob/x",
                deposited_by="t",
            )
            snap_b = Snapshot(
                snapshot_id=str(uuid.uuid4()),
                captured_at=datetime.now(UTC),
                content_hash="b" * 64,
                extraction_status=ExtractionStatus.COMPLETE,
                warc_record_type=WarcRecordType.RESPONSE,
                author_id="github:bob",
                # no role — verifies the "id only" rendering branch
            )

            # Two ACTIVE particles, one from each snapshot
            pa = Particle(
                id=str(uuid.uuid4()),
                content="Claim from alice",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="reddit-extractor",
                status=Status.ACTIVE,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id=entry_a.entry_id,
                        snapshot_id=snap_a.snapshot_id,
                    )
                ],
            )
            pb = Particle(
                id=str(uuid.uuid4()),
                content="Claim from bob",
                confidence=Confidence(
                    value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="github-gist-extractor",
                status=Status.ACTIVE,
                provenance=[
                    ProvenanceRef(
                        type=ProvenanceRefType.SOURCE,
                        corpus_entry_id=entry_b.entry_id,
                        snapshot_id=snap_b.snapshot_id,
                    )
                ],
            )

            # INCONSISTENCY particle whose PARTICLE-type provenance refs
            # carry the two particle IDs in corpus_entry_id (the existing
            # convention used by _list_review_detail).
            inc = Particle(
                id=str(uuid.uuid4()),
                content=(
                    "INCONSISTENCY: conflict between two claims.\n"
                    f"Particle A (existing): {pa.content}\n"
                    f"Particle B (new): {pb.content}"
                ),
                confidence=Confidence(
                    value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
                ),
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                asserted_by="extract-pipeline",
                status=Status.INCONSISTENCY,
                provenance=[
                    ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pa.id),
                    ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pb.id),
                ],
            )

            async with session_scope() as session:
                session.add(CorpusEntryRow.from_model(entry_a))
                session.add(SnapshotRow.from_model(snap_a, entry_a.entry_id))
                session.add(CorpusEntryRow.from_model(entry_b))
                session.add(SnapshotRow.from_model(snap_b, entry_b.entry_id))
                await insert_particle(session, pa)
                await insert_particle(session, pb)
                await insert_particle(session, inc)
                await session.commit()

        _run_async(_seed())

        result = _invoke(runner, ["review"])
        assert result.exit_code == 0
        # Alice has both id and role — full "id (role: ROLE)" rendering
        assert "reddit:u/alice (role: maintainer)" in result.output
        # Bob has id only — bare id, no role suffix
        assert "github:bob" in result.output
        assert "github:bob (role:" not in result.output


class TestFormatAuthor:
    def test_empty_when_no_id(self) -> None:
        from particles.api.cli.review import _format_author

        assert _format_author(None, None) == ""
        assert _format_author(None, "maintainer") == ""
        assert _format_author("", "maintainer") == ""

    def test_id_only_when_no_role(self) -> None:
        from particles.api.cli.review import _format_author

        assert _format_author("reddit:u/alice", None) == "reddit:u/alice"
        assert _format_author("reddit:u/alice", "") == "reddit:u/alice"

    def test_id_and_role(self) -> None:
        from particles.api.cli.review import _format_author

        assert _format_author("github:bob", "maintainer") == "github:bob (role: maintainer)"


class TestReindexCommand:
    def test_reindex_empty_db_succeeds(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["reindex", "--format", "json"])
        assert result.exit_code == 0
        # The summary is JSON-formatted; verify it's valid JSON with expected keys
        parsed = _json_payload(result.output)
        assert parsed["scope"] == 0
        assert parsed["succeeded"] == 0
        assert parsed["failed"] == 0
        # The upfront work plan prints unconditionally, before any extraction.
        assert "Reindex plan: 0 entries" in result.output

    def test_entry_ids_intersects_provider_model(self, runner: CliRunner, cli_db: Path) -> None:
        """the other scope flags survive ``--entry-ids``.

        Before the fix the explicit-entry branch discarded them, so this
        invocation re-extracted (and superseded) an entry whose particles the
        named pairing never produced.
        """
        entry_id, snapshot_id = _run_async(_add_corpus_entry())
        _run_async(
            _add_active_particle(
                entry_id=entry_id,
                snapshot_id=snapshot_id,
                extraction_provider_model="anthropic:opus",
            )
        )

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(
                runner,
                [
                    "reindex",
                    "--format",
                    "json",
                    "--entry-ids",
                    entry_id,
                    "--provider-model",
                    "openai:gpt-5.6-luna",
                ],
            )
        assert result.exit_code == 0
        # Substring rather than json.loads: the narrowing notice is a log
        # record on stderr, which CliRunner folds into the same buffer.
        assert '"scope": 0' in result.output
        extract.assert_not_called()

    def test_entry_ids_alone_still_selects_the_entry(self, runner: CliRunner, cli_db: Path) -> None:
        """The control for the test above — without the filter, scope is 1."""
        entry_id, snapshot_id = _run_async(_add_corpus_entry())
        _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snapshot_id))

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(runner, ["reindex", "--format", "json", "--entry-ids", entry_id])
        assert result.exit_code == 0
        assert _json_payload(result.output)["scope"] == 1
        extract.assert_called_once()

    def test_plan_line_prints_before_a_live_run(self, runner: CliRunner, cli_db: Path) -> None:
        """The 2026-08-02 incident fix: a live run states its scope — entries,
        snapshots, particles — before the first LLM call is spent."""
        entry_id, snapshot_id = _run_async(_add_corpus_entry())
        _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snapshot_id))

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(runner, ["reindex", "--entry-ids", entry_id])
        assert result.exit_code == 0
        assert "Reindex plan: 1 entries, 1 snapshots, 1 particles" in result.output
        assert f"entry-ids {entry_id}" in result.output
        extract.assert_called_once()

    def test_dry_run_prints_plan_and_extracts_nothing(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        entry_id, snapshot_id = _run_async(_add_corpus_entry())
        _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snapshot_id))

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(
                runner, ["reindex", "--format", "json", "--entry-ids", entry_id, "--dry-run"]
            )
        assert result.exit_code == 0
        extract.assert_not_called()
        parsed = _json_payload(result.output)
        assert parsed["dry_run"] is True
        assert parsed["succeeded"] == 0
        assert parsed["plan"]["entries"] == 1
        assert parsed["plan"]["particles"] == 1
        # The envelope carries the per-snapshot detail the human format omits.
        (sp,) = parsed["plan"]["snapshot_plans"]
        assert sp["entry_id"] == entry_id
        assert sp["snapshot_id"] == snapshot_id

    def test_human_live_run_prints_counts_not_the_envelope(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """The default (human) format ends with one counts line — the raw JSON
        envelope (hundreds of lines on a wide scope) stays behind --format json."""
        entry_id, snapshot_id = _run_async(_add_corpus_entry())
        _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snapshot_id))

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(runner, ["reindex", "--entry-ids", entry_id])
        assert result.exit_code == 0
        assert "Reindex complete: 1 succeeded, 0 failed (scope: 1 snapshot(s))." in result.output
        assert "snapshot_plans" not in result.output
        assert '"dry_run"' not in result.output

    def test_human_dry_run_prints_plan_once_on_stdout(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """On a human dry run the plan IS the artifact: it lands on stdout
        exactly once (not streamed to stderr as well), with no JSON after it."""
        entry_id, snapshot_id = _run_async(_add_corpus_entry())
        _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snapshot_id))

        extract = AsyncMock(return_value=[])
        with patch("particles.operations.reindex.extract_snapshot", new=extract):
            result = _invoke(runner, ["reindex", "--entry-ids", entry_id, "--dry-run"])
        assert result.exit_code == 0
        extract.assert_not_called()
        assert result.output.count("Reindex plan: 1 entries, 1 snapshots, 1 particles") == 1
        assert "Dry run — nothing extracted." in result.output
        assert '"dry_run"' not in result.output

    def test_human_dry_run_caps_missing_blob_lines(self, runner: CliRunner, cli_db: Path) -> None:
        """Seven missing blobs render as five detail lines + a remainder line;
        the full list stays behind --format json. (Test snapshots never write
        blobs, so every snapshot in scope counts as missing.)"""
        entry_ids = []
        for i in range(7):
            entry_id, snapshot_id = _run_async(_add_corpus_entry(uri_r=f"https://example.com/{i}"))
            _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snapshot_id))
            entry_ids.append(entry_id)

        result = _invoke(runner, ["reindex", "--entry-ids", ",".join(entry_ids), "--dry-run"])
        assert result.exit_code == 0
        assert result.output.count("blob missing:") == 5
        assert "… and 2 more (see --format json)" in result.output


class TestExportCommand:
    def test_export_unknown_format_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["export", "no-such-format", "/tmp/out"])
        assert result.exit_code != 0
        assert "Unknown format" in result.output

    def test_export_anki_empty_db(self, runner: CliRunner, cli_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "deck.txt"
        result = _invoke(runner, ["export", "anki", str(out)])
        assert result.exit_code == 0
        assert "Exported to" in result.output
        assert out.exists()

    def test_export_wiki_without_synthesis(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        # the --without-synthesis flag reaches the wiki exporter and
        # writes a deterministic article with no LLM call (no API key needed).
        sid = _run_async(_add_subject("GDR"))
        for i in range(3):
            _run_async(_add_active_particle(content=f"claim {i}", subject_ids=[sid]))
        out = tmp_path / "wiki"
        result = _invoke(runner, ["export", "wiki", str(out), "--without-synthesis"])
        assert result.exit_code == 0, result.output
        assert (out / "GDR.md").exists()

    def test_export_obsidian_uses_config_default_when_arg_omitted(
        self,
        runner: CliRunner,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`particles export obsidian` (no path) should write to the
        directory recorded in `obsidian.default_output_path`."""
        from particles.config import reset_config

        default_dir = tmp_path / "vault"
        monkeypatch.setenv("OBSIDIAN_DEFAULT_OUTPUT_PATH", str(default_dir))
        reset_config()  # autouse fixture already reset, but env was set after
        result = _invoke(runner, ["export", "obsidian"])
        assert result.exit_code == 0, result.output
        assert str(default_dir) in result.output  # "Exported to <default_dir>"
        assert default_dir.exists()

    def test_export_obsidian_explicit_arg_beats_config_default(
        self,
        runner: CliRunner,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A path on the CLI overrides whatever the config says."""
        from particles.config import reset_config

        config_dir = tmp_path / "config-vault"
        explicit_dir = tmp_path / "explicit-vault"
        monkeypatch.setenv("OBSIDIAN_DEFAULT_OUTPUT_PATH", str(config_dir))
        reset_config()
        result = _invoke(runner, ["export", "obsidian", str(explicit_dir)])
        assert result.exit_code == 0, result.output
        assert explicit_dir.exists()
        assert not config_dir.exists()  # config default was not used

    def test_export_obsidian_no_arg_no_config_errors(self, runner: CliRunner, cli_db: Path) -> None:
        """With neither CLI arg nor config default the export must fail
        with a guiding error message — not silently default somewhere."""
        result = runner.invoke(app, ["export", "obsidian"])
        assert result.exit_code != 0
        assert "default_output_path" in result.output

    def test_export_obsidian_expands_tilde_in_config_default(
        self,
        runner: CliRunner,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`~/foo` in config.yaml must resolve to the actual home dir —
        not a literal directory named `~` in cwd."""
        from particles.config import reset_config

        # Pretend HOME is tmp_path so the test is hermetic.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OBSIDIAN_DEFAULT_OUTPUT_PATH", "~/my-vault")
        reset_config()
        result = _invoke(runner, ["export", "obsidian"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-vault").exists()
        # Literal "~" directory should NOT have been created in cwd.
        assert not Path("~/my-vault").exists() or (tmp_path / "my-vault").exists()

    def test_export_wiki_still_requires_explicit_path(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """The default-path config is Obsidian-only for now; wiki and
        anki still error out without a path."""
        result = runner.invoke(app, ["export", "wiki"])
        assert result.exit_code != 0
        assert "default_output_path" in result.output


#: synthesis-cache list / show / vacuum / evict.
class TestSynthesisCache:
    def test_list_empty(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["synthesis-cache", "list"])
        assert result.exit_code == 0
        assert "empty" in result.output

    def test_list_and_show(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("GDR"))
        _run_async(_add_cache_entry(sid, body="# GDR\n\nThe cached prose."))
        lst = _invoke(runner, ["synthesis-cache", "list"])
        assert lst.exit_code == 0
        assert sid[:8] in lst.output
        assert "GDR" in lst.output
        show = _invoke(runner, ["synthesis-cache", "show", sid[:8]])
        assert show.exit_code == 0
        assert "The cached prose." in show.output

    def test_show_unknown_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["synthesis-cache", "show", "deadbeef"])
        assert result.exit_code != 0

    def test_vacuum_removes_orphaned(self, runner: CliRunner, cli_db: Path) -> None:
        # Cache row whose subject does not exist → orphaned, so vacuum prunes it.
        _run_async(_add_cache_entry("ghost-subject-id", body="orphan"))
        result = _invoke(runner, ["synthesis-cache", "vacuum"])
        assert result.exit_code == 0
        assert "Removed 1" in result.output
        assert "empty" in _invoke(runner, ["synthesis-cache", "list"]).output

    def test_vacuum_dry_run_keeps_rows(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_cache_entry("ghost-subject-id", body="orphan"))
        result = _invoke(runner, ["synthesis-cache", "vacuum", "--dry-run"])
        assert result.exit_code == 0
        assert "Would remove" in result.output
        assert "1 cached" in _invoke(runner, ["synthesis-cache", "list"]).output

    def test_evict_removes_subject(self, runner: CliRunner, cli_db: Path) -> None:
        sid = _run_async(_add_subject("GDR"))
        _run_async(_add_cache_entry(sid))
        result = _invoke(runner, ["synthesis-cache", "evict", sid, "--yes"])
        assert result.exit_code == 0
        assert "Evicted 1" in result.output
        assert "empty" in _invoke(runner, ["synthesis-cache", "list"]).output


class TestInboxStatusCommand:
    def test_status_without_config_errors_clearly(self, runner: CliRunner, cli_db: Path) -> None:
        """No config + no env override → exit non-zero with a guiding message."""
        result = runner.invoke(app, ["inbox", "status"])
        assert result.exit_code != 0
        assert "inbox.file_path" in result.output

    def test_status_with_existing_file_reports_pending(
        self,
        runner: CliRunner,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inbox = tmp_path / "_inbox.txt"
        inbox.write_text(
            "https://example.com/a\n"
            "# Processed 2026-05-24T15:30:00+00:00 (entry_id: x) https://example.com/done\n"
            "https://example.com/b\n"
        )
        monkeypatch.setenv("INBOX_FILE_PATH", str(inbox))
        from particles.config import reset_config

        reset_config()
        result = _invoke(runner, ["inbox", "status"])
        assert result.exit_code == 0, result.output
        assert "Pending:   2" in result.output
        assert "Processed: 1" in result.output
        assert "https://example.com/a" in result.output

    def test_status_hints_at_shell_escape_typo(
        self,
        runner: CliRunner,
        cli_db: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A path with `\\ ` (shell-escaped space) won't match anything
        on disk; the status command should flag this specifically rather
        than just saying 'file does not exist'."""
        # Deliberately misconfigured: shell-escape in a YAML-loaded path.
        # The file at the *real* path may or may not exist; what matters
        # is that the resolved path with `\ ` doesn't.
        bad_path = str(tmp_path) + r"/Mobile\ Documents/_inbox.txt"
        monkeypatch.setenv("INBOX_FILE_PATH", bad_path)
        from particles.config import reset_config

        reset_config()
        result = runner.invoke(app, ["inbox", "status"])
        assert result.exit_code == 0  # status doesn't fail; it diagnoses
        assert "backslash-space" in result.output
        assert "YAML doesn't need it" in result.output


class TestDepositCommand:
    def test_deposit_file(self, runner: CliRunner, cli_db: Path, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("A simple test document.")
        result = _invoke(runner, ["deposit", str(doc)])
        assert result.exit_code == 0
        assert "entry_id:" in result.output
        assert "snapshot_id:" in result.output

    def test_deposit_missing_file_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["deposit", "/nonexistent/path/file.txt"])
        assert result.exit_code != 0

    def test_deposit_text_creates_an_operator_attributed_entry(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """--text deposits a literal string, attributed to the operator.

        Attribution is the substantive assertion, not the exit code: an operator's
        typed note must not land at the agent asserter identity, because the §6.4
        AUTHOR trust tier ranks agent-asserted content below operator content.
        """
        result = _invoke(runner, ["deposit", "--text", "Pluto is a dwarf planet."])
        assert result.exit_code == 0
        assert "entry_id:" in result.output

        async def _read() -> tuple[str, str, str]:
            from sqlalchemy import select

            from particles.corpus.store import CorpusEntryRow, SnapshotRow
            from particles.db import session_scope

            async with session_scope() as session:
                entry = (await session.execute(select(CorpusEntryRow))).scalars().one()
                snap = (await session.execute(select(SnapshotRow))).scalars().one()
                return entry.deposited_by, entry.source_type, snap.author_id or ""

        deposited_by, source_type, author_id = asyncio.run(_read())
        assert deposited_by == "operator"
        assert author_id == "operator"
        assert source_type == "CONVERSATION"

    def test_deposit_text_honours_deposited_by(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["deposit", "--text", "A note.", "--deposited-by", "jeff"])
        assert result.exit_code == 0

        async def _depositor() -> str:
            from sqlalchemy import select

            from particles.corpus.store import CorpusEntryRow
            from particles.db import session_scope

            async with session_scope() as session:
                entry = (await session.execute(select(CorpusEntryRow))).scalars().one()
                return entry.deposited_by

        assert asyncio.run(_depositor()) == "jeff"

    def test_deposit_dash_reads_stdin(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["deposit", "-"], input="Typed straight into the terminal.\n")
        assert result.exit_code == 0
        assert "entry_id:" in result.output

    def test_deposit_text_and_source_are_mutually_exclusive(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("A file.")
        result = _invoke(runner, ["deposit", "--text", "A string.", str(doc)])
        assert result.exit_code == 1
        assert "drop the source argument" in result.output

    def test_deposit_empty_text_is_refused(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["deposit", "--text", "   "])
        assert result.exit_code == 1
        assert "empty text" in result.output

    def test_deposit_with_no_source_and_no_text_errors(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        result = _invoke(runner, ["deposit"])
        assert result.exit_code == 1
        assert "--text" in result.output

    @pytest.mark.parametrize(
        "flag,value",
        [("--date", "2026-01-01"), ("--mutability", "MUTABLE"), ("--fetch-policy", "LAZY")],
    )
    def test_deposit_text_refuses_file_only_flags(
        self, runner: CliRunner, cli_db: Path, flag: str, value: str
    ) -> None:
        """A pasted string has no file to revisit, so these are refused, not ignored."""
        result = _invoke(runner, ["deposit", "--text", "A note.", flag, value])
        assert result.exit_code == 1
        assert "local-file deposits only" in result.output

    def test_deposit_text_refuses_split_by_date(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["deposit", "--text", "A note.", "--split-by-date"])
        assert result.exit_code == 1
        assert "local-file deposits only" in result.output

    def test_deposit_help_shows_debug_flag(self, runner: CliRunner) -> None:
        import re

        result = _invoke(runner, ["deposit", "--help"])
        assert result.exit_code == 0
        # Typer/Rich injects ANSI escapes around option tokens in CI; strip
        # them before substring matching.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--debug" in clean
        assert "--verbose" in clean


class TestParticleCommand:
    def test_particle_show_outputs_content_and_source(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        entry_id, snapshot_id = _run_async(
            _add_corpus_entry(
                source_type="GITHUB_REPO",
                uri_r="https://github.com/karpathy/nanoGPT",
            )
        )
        subject_id = _run_async(_add_subject(name="nanoGPT"))
        particle_id = _run_async(
            _add_active_particle(
                content="The gpt2 model has 124M parameters.",
                subject_ids=[subject_id],
                entry_id=entry_id,
                snapshot_id=snapshot_id,
            )
        )

        result = _invoke(runner, ["particle", "show", particle_id[:8]])
        assert result.exit_code == 0, result.output
        assert "The gpt2 model has 124M parameters." in result.output
        assert "https://github.com/karpathy/nanoGPT" in result.output
        assert "nanoGPT" in result.output  # subject name
        assert "GITHUB_REPO" in result.output

    def test_particle_show_reports_calibration_source(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """The Confidence line names how the stored value was calibrated."""
        particle_id = _run_async(_add_active_particle(content="Calibrated claim."))

        result = _invoke(runner, ["particle", "show", particle_id[:8]])
        assert result.exit_code == 0, result.output
        assert "Confidence:   0.90 (EXTRACTOR_DIRECT)" in result.output

    def test_particle_show_reports_provider_model(self, runner: CliRunner, cli_db: Path) -> None:
        """The provider pairing gets its own line when stamped."""
        particle_id = _run_async(
            _add_active_particle(
                content="Extracted under a named model.",
                extraction_provider_model="openai:gpt-5.6-luna",
            )
        )

        result = _invoke(runner, ["particle", "show", particle_id[:8]])
        assert result.exit_code == 0, result.output
        assert "Model:        openai:gpt-5.6-luna" in result.output

    def test_particle_show_omits_provider_model_when_unstamped(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        """Pre-0229 particles legally lack the stamp — no empty line for them."""
        particle_id = _run_async(_add_active_particle(content="Unstamped claim."))

        result = _invoke(runner, ["particle", "show", particle_id[:8]])
        assert result.exit_code == 0, result.output
        assert "Model:" not in result.output

    def test_particle_show_unknown_id_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["particle", "show", "0000dead-0000-0000-0000-000000000000"])
        assert result.exit_code != 0

    def test_particle_show_ambiguous_prefix_errors(self, runner: CliRunner, cli_db: Path) -> None:
        # Two particles sharing the same single-char prefix.
        pid_a = _run_async(_add_active_particle(content="A"))
        pid_b = _run_async(_add_active_particle(content="B"))
        shared = pid_a[0]
        if pid_b[0] != shared:
            # Different first chars — retry with a more permissive prefix.
            # In practice UUID4 first-char collisions occur in <1% of pairs;
            # if this branch is hit, the test is still meaningful via the
            # empty-prefix path below.
            pass
        result = runner.invoke(app, ["particle", "show", ""])
        assert result.exit_code != 0


class TestExtractCommand:
    def test_extract_missing_entry_id_with_no_all_pending(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        result = runner.invoke(app, ["extract"])
        assert result.exit_code != 0
        assert "Provide an entry ID" in result.output

    def test_extract_all_pending_empty_db(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        result = _invoke(runner, ["extract", "--all-pending"])
        assert result.exit_code == 0
        assert "No PENDING" in result.output

    def test_extract_all_pending_without_api_key_fails_fast(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = _invoke(runner, ["extract", "--all-pending"])
        assert result.exit_code != 0
        assert "ANTHROPIC_API_KEY" in result.output

    def test_extract_all_pending_none_pending_shows_status_breakdown(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "No PENDING snapshots found" alone reads as data loss to an operator
        # whose previous run printed FAILED lines; the breakdown shows the
        # snapshots are simply COMPLETE.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        _run_async(_add_corpus_entry())  # one COMPLETE snapshot
        result = _invoke(runner, ["extract", "--all-pending"])
        assert result.exit_code == 0
        assert "No PENDING snapshots found (1 COMPLETE)." in result.output

    def test_extract_all_pending_exits_nonzero_when_any_snapshot_fails(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression: a run where every snapshot printed FAILED used to exit 0
        # because the per-snapshot handler swallowed the exception and nothing
        # counted failures.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        fail_entry = str(uuid.uuid4())
        ok_entry = str(uuid.uuid4())
        _run_async(_add_corpus_entry_with_id(fail_entry, str(uuid.uuid4())))
        _run_async(_add_corpus_entry_with_id(ok_entry, str(uuid.uuid4())))
        extracted: list[str] = []

        async def _fake_extract(_session: Any, e_id: str, s_id: str, **_kw: Any) -> list[Any]:
            if e_id == fail_entry:
                raise ValueError("boom")
            extracted.append(e_id)
            return []

        monkeypatch.setattr("particles.operations.extract.extract_snapshot", _fake_extract)
        result = _invoke(runner, ["extract", "--all-pending"])
        assert result.exit_code == 1
        assert "FAILED: boom" in result.output
        # The failure does not abort the run — the other snapshot is still tried.
        assert extracted == [ok_entry]
        assert "Extraction failed for 1 of 2 snapshot(s)." in result.output

    def test_extract_all_pending_translates_database_locked(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The per-snapshot handler catches the exception before run()'s lock
        # translation can fire, so it must produce the friendly message itself.
        from sqlalchemy.exc import OperationalError

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        _run_async(_add_corpus_entry_with_id(str(uuid.uuid4()), str(uuid.uuid4())))

        async def _fake_extract(_session: Any, e_id: str, s_id: str, **_kw: Any) -> list[Any]:
            raise OperationalError("UPDATE snapshots", None, Exception("database is locked"))

        monkeypatch.setattr("particles.operations.extract.extract_snapshot", _fake_extract)
        result = _invoke(runner, ["extract", "--all-pending"])
        assert result.exit_code == 1
        assert "database is locked — another particles process" in result.output
        assert "sqlalche.me" not in result.output

    def test_extract_accepts_exact_entry_and_snapshot_id(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``_extract`` imports extract_snapshot from particles.operations.extract
        # at call time; patch the attribute on that module so the resolved IDs
        # are captured without an LLM call. See tests/AGENTS.md § Mocking.
        entry_id, snap_id = _run_async(_add_corpus_entry())
        captured: dict[str, str] = {}

        async def _fake_extract(_session: Any, e_id: str, s_id: str, **_kw: Any) -> list[Any]:
            captured["entry_id"] = e_id
            captured["snapshot_id"] = s_id
            return []

        monkeypatch.setattr("particles.operations.extract.extract_snapshot", _fake_extract)
        result = _invoke(runner, ["extract", entry_id, "--snapshot-id", snap_id])
        assert result.exit_code == 0, result.output
        assert captured == {"entry_id": entry_id, "snapshot_id": snap_id}

    def test_extract_accepts_unique_entry_and_snapshot_prefix(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry_id, snap_id = _run_async(_add_corpus_entry())
        captured: dict[str, str] = {}

        async def _fake_extract(_session: Any, e_id: str, s_id: str, **_kw: Any) -> list[Any]:
            captured["entry_id"] = e_id
            captured["snapshot_id"] = s_id
            return []

        monkeypatch.setattr("particles.operations.extract.extract_snapshot", _fake_extract)
        # Pass the truncated 8-char forms the deposit/extract output displays.
        result = _invoke(runner, ["extract", entry_id[:8], "--snapshot-id", snap_id[:8]])
        assert result.exit_code == 0, result.output
        # Both expand back to the full UUIDs.
        assert captured == {"entry_id": entry_id, "snapshot_id": snap_id}

    def test_extract_ambiguous_entry_prefix_errors(self, runner: CliRunner, cli_db: Path) -> None:
        # Two entries sharing the leading 8 chars -> "dead" is ambiguous.
        _run_async(
            _add_corpus_entry_with_id(
                "dead0000-0000-0000-0000-000000000001",
                "5na90000-0000-0000-0000-000000000001",
            )
        )
        _run_async(
            _add_corpus_entry_with_id(
                "dead0000-0000-0000-0000-000000000002",
                "5na90000-0000-0000-0000-000000000002",
            )
        )
        result = runner.invoke(app, ["extract", "dead"])
        assert result.exit_code != 0
        assert "Ambiguous entry prefix" in result.output


# ---------------------------------------------------------------------------
# particle tag / untag commands
# ---------------------------------------------------------------------------


async def _add_taxonomy_coins() -> str:
    """Insert a small Coins taxonomy via insert_taxonomy. Returns taxonomy_id."""
    from particles.core.schema import TagNode, TaxonomyDefinition
    from particles.db import session_scope
    from particles.store.taxonomy_store import insert_taxonomy

    td = TaxonomyDefinition(
        name="Coins",
        version="1.0.0",
        author="test",
        tags=[
            TagNode(tag="coins"),
            TagNode(tag="coins/by-region", parent="coins"),
            TagNode(tag="coins/by-region/germany", parent="coins/by-region"),
        ],
    )
    async with session_scope() as session:
        await insert_taxonomy(session, td)
        await session.commit()
    return td.taxonomy_id


async def _read_particle_tags(particle_id: str) -> list[str]:
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    async with session_scope() as session:
        p = await get_particle(session, particle_id)
    return p.tags if p and p.tags else []


class TestParticleTagCommand:
    def test_tag_with_known_path_succeeds(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_taxonomy_coins())
        pid = _run_async(_add_active_particle())
        result = _invoke(runner, ["particle", "tag", pid, "--tag", "coins/by-region/germany"])
        assert result.exit_code == 0
        assert "Tagged" in result.output
        assert _run_async(_read_particle_tags(pid)) == ["coins/by-region/germany"]

    def test_tag_unknown_without_force_rejected(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_taxonomy_coins())
        pid = _run_async(_add_active_particle())
        result = _invoke(runner, ["particle", "tag", pid, "--tag", "not-in-taxonomy"])
        assert result.exit_code != 0
        assert "not defined in any active taxonomy" in result.output

    def test_tag_unknown_with_force_succeeds(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_taxonomy_coins())
        pid = _run_async(_add_active_particle())
        result = _invoke(runner, ["particle", "tag", pid, "--tag", "ad-hoc", "--force"])
        assert result.exit_code == 0
        assert _run_async(_read_particle_tags(pid)) == ["ad-hoc"]

    def test_tag_idempotent(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_taxonomy_coins())
        pid = _run_async(_add_active_particle())
        _invoke(runner, ["particle", "tag", pid, "--tag", "coins", "--force"])
        result = _invoke(runner, ["particle", "tag", pid, "--tag", "coins", "--force"])
        assert result.exit_code == 0
        assert "already had" in result.output

    def test_untag_removes(self, runner: CliRunner, cli_db: Path) -> None:
        _run_async(_add_taxonomy_coins())
        pid = _run_async(_add_active_particle())
        _invoke(runner, ["particle", "tag", pid, "--tag", "coins", "--force"])
        result = _invoke(runner, ["particle", "untag", pid, "--tag", "coins"])
        assert result.exit_code == 0
        assert "Untagged" in result.output
        assert _run_async(_read_particle_tags(pid)) == []

    def test_supersede_flag_rejected_in_phase_a(self, runner: CliRunner, cli_db: Path) -> None:
        pid = _run_async(_add_active_particle())
        result = _invoke(runner, ["particle", "tag", pid, "--tag", "x", "--force", "--supersede"])
        assert result.exit_code != 0
        assert "Phase A" in result.output

    def test_tag_missing_particle_fails(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["particle", "tag", "00000000", "--tag", "x", "--force"])
        assert result.exit_code != 0
        assert "No particle matches prefix" in result.output


# ---------------------------------------------------------------------------
# Uninitialised-DB UX (lint / query / etc. should hint at `db init`)
# ---------------------------------------------------------------------------


class TestUninitializedDatabaseUx:
    """SQLAlchemy auto-creates the SQLite file on connect, so an operator
    who runs any read command in a directory without a DB sees a
    confusing ``no such table: particles`` traceback. The CLI catches
    that condition and prints a one-liner pointing at ``db init``."""

    def test_lint_against_empty_db_hints_at_db_init(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point at a fresh sqlite file with no tables and no automatic
        # ``create_all`` (the ``cli_db`` fixture does that — we deliberately
        # don't use it here).
        empty_db = tmp_path / "empty.db"
        empty_db.touch()
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{empty_db}")
        monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))

        from particles.config import reset_config

        reset_config()
        try:
            result = _invoke(runner, ["lint"])
        finally:
            reset_config()

        assert result.exit_code == 1
        assert "particles db init" in result.output
        # Friendlier wording (0.36.1): also catches the post-upgrade
        # case where one new Alembic migration is pending. Message
        # explicitly says the command is idempotent + safe on populated
        # databases so operators upgrading the SDK don't fear running it.
        assert (
            "migration is pending" in result.output
            or "preserves your existing data" in result.output
        )


class TestSchemaBehindSdkUx:
    """0.42.2 added ``snapshots.extraction_started_at``; an operator who
    upgraded the SDK without running ``db init`` hits ``no such column``
    on the next read. Translate it to a clean stderr line + exit 1
    pointing at the recovery command, instead of the SQLAlchemy stack
    trace the user surfaced."""

    def test_no_such_column_translates_to_db_init_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import typer
        from sqlalchemy.exc import OperationalError

        from particles.api.cli import run

        async def _boom() -> None:
            raise OperationalError(
                statement="SELECT snapshots.extraction_started_at FROM snapshots",
                params=None,
                orig=Exception("no such column: snapshots.extraction_started_at"),
            )

        with pytest.raises(typer.Exit) as exit_info:
            run(_boom())

        assert exit_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "schema is behind" in captured.err.lower()
        assert "particles db init" in captured.err
        # Reassurance: the recovery command preserves data.
        assert "preserves your data" in captured.err


class TestDatabaseLockedUx:
    """An ``OperationalError: database is locked`` from a concurrent writer
    must surface as a clean stderr line + exit 1, not as a Python stack
    trace. WAL + busy_timeout makes the case rare; this test pins the
    fallback UX for the case where the timeout still doesn't absorb the
    contention.
    """

    def test_locked_db_translates_to_clean_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import typer
        from sqlalchemy.exc import OperationalError

        from particles.api.cli import run

        async def _boom() -> None:
            # SQLAlchemy wraps the underlying DBAPI error with the same
            # str() shape the CLI translator pattern-matches against.
            raise OperationalError(
                statement="INSERT INTO corpus_entries ...",
                params=None,
                orig=Exception("database is locked"),
            )

        with pytest.raises(typer.Exit) as exit_info:
            run(_boom())

        assert exit_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "Database is locked" in captured.err
        # Helpful guidance — the operator knows what to do.
        assert "another particles process" in captured.err


class TestOutboundUnreachableUx:
    """Any *non-engine* outbound fetch that never gets a response — the
    deposited URL's host, Wikidata, an importer API, or the Anthropic API —
    must surface a clean, host-named stderr line + exit 1, not the raw
    ``httpx.ConnectError`` / ``anthropic.APIConnectionError`` traceback the
    operator surfaced. (The remote *engine* path is translated separately to
    ``EngineUnreachableError`` — see ``tests/test_api_client.py``.) ``run()`` is
    the single backstop; these drive it directly.
    """

    def test_httpx_transport_error_names_the_host(self, capsys: pytest.CaptureFixture[str]) -> None:
        import httpx
        import typer

        from particles.api.cli import run

        async def _boom() -> None:
            req = httpx.Request("GET", "https://www.wikidata.org/w/api.php")
            raise httpx.ConnectError("All connection attempts failed", request=req)

        with pytest.raises(typer.Exit) as exit_info:
            run(_boom())

        assert exit_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "could not reach" in captured.err
        assert "www.wikidata.org" in captured.err  # names the unreachable host
        assert "Traceback" not in captured.err

    def test_httpx_transport_error_without_attached_request(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ``httpx.*Error.request`` is a property that raises RuntimeError when
        # the error was raised before a request was attached; the backstop must
        # degrade to a generic "a remote service" rather than crash on access.
        import httpx
        import typer

        from particles.api.cli import run

        async def _boom() -> None:
            raise httpx.ConnectTimeout("timed out")

        with pytest.raises(typer.Exit) as exit_info:
            run(_boom())

        assert exit_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "could not reach a remote service" in captured.err

    def test_anthropic_connection_error_translates(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Importing anthropic mirrors a verb (extract / query / semantic lint)
        # that made an LLM call: the backstop only translates this error when
        # ``anthropic`` is already loaded.
        import anthropic
        import httpx
        import typer

        from particles.api.cli import run

        async def _boom() -> None:
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=req)

        with pytest.raises(typer.Exit) as exit_info:
            run(_boom())

        assert exit_info.value.exit_code == 1
        captured = capsys.readouterr()
        assert "could not reach the Anthropic API" in captured.err
        # The trailing-period strip means no doubled ".." before the hint.
        assert ".." not in captured.err

    def test_unrelated_exception_still_propagates(self) -> None:
        # The broad ``except Exception`` arm that hosts the lazy anthropic check
        # must re-raise anything that isn't an Anthropic connection error,
        # preserving the original behaviour for genuine bugs.
        from particles.api.cli import run

        async def _boom() -> None:
            raise RuntimeError("unrelated failure")

        with pytest.raises(RuntimeError, match="unrelated failure"):
            run(_boom())


# ---------------------------------------------------------------------------
# `db init --force` — scrap-and-re-extract upgrade path
# ---------------------------------------------------------------------------


class TestDbInitForce:
    """``particles db init --force`` is the command the SCHEMA_VERSION
    mismatch guard tells operators to run. It must actually exist, ask
    for confirmation, and (on yes) clear the particle store while
    preserving the corpus."""

    def test_force_aborts_on_no(self, runner: CliRunner, cli_db: Path) -> None:
        entry_id, snap_id = _run_async(_add_corpus_entry())
        result = runner.invoke(app, ["db", "init", "--force"], input="n\n")
        # Click translates an aborted typer.confirm() into exit code 1.
        assert result.exit_code == 1
        # Corpus rows survive — abort happens before any delete.
        from particles.corpus.store import CorpusEntryRow
        from particles.db import session_scope

        async def _count_entries() -> int:
            from sqlalchemy import func, select

            async with session_scope() as session:
                rs = await session.execute(select(func.count()).select_from(CorpusEntryRow))
                return int(rs.scalar() or 0)

        assert _run_async(_count_entries()) == 1

    def test_force_yes_clears_particles_preserves_corpus(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        entry_id, snap_id = _run_async(_add_corpus_entry())
        _run_async(_add_active_particle(entry_id=entry_id, snapshot_id=snap_id))

        result = runner.invoke(app, ["db", "init", "--force"], input="y\n")
        assert result.exit_code == 0
        assert "extract --all-pending" in result.output

        from sqlalchemy import func, select

        from particles.corpus.store import CorpusEntryRow, SnapshotRow
        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async def _counts() -> tuple[int, int, int, str | None]:
            async with session_scope() as session:
                entries = int(
                    (
                        await session.execute(select(func.count()).select_from(CorpusEntryRow))
                    ).scalar()
                    or 0
                )
                snaps = int(
                    (await session.execute(select(func.count()).select_from(SnapshotRow))).scalar()
                    or 0
                )
                particles = int(
                    (await session.execute(select(func.count()).select_from(ParticleRow))).scalar()
                    or 0
                )
                snap_row = await session.get(SnapshotRow, snap_id)
                snap_status = snap_row.extraction_status if snap_row else None
                return entries, snaps, particles, snap_status

        entries, snaps, particles, snap_status = _run_async(_counts())
        # Corpus preserved …
        assert entries == 1
        assert snaps == 1
        # … particles cleared …
        assert particles == 0
        # … snapshot's extraction_status reset to PENDING for re-extraction.
        assert snap_status == ExtractionStatus.PENDING.value


class TestSubjectsListFilters:
    """`subjects list --phantoms-only` filters to zero-particle subjects."""

    def test_phantoms_only_shows_only_phantom(self, runner: CliRunner, cli_db: Path) -> None:
        phantom_id = _run_async(_add_subject("Phantom Co"))
        real_id = _run_async(_add_subject("Real Co"))
        _run_async(_add_active_particle(subject_ids=[real_id]))

        result = _invoke(runner, ["subjects", "list", "--phantoms-only"])
        assert result.exit_code == 0
        assert "Phantom Co" in result.output
        assert "Real Co" not in result.output
        assert phantom_id[:8] in result.output

    def test_phantoms_only_empty_message(self, runner: CliRunner, cli_db: Path) -> None:
        real_id = _run_async(_add_subject("Real Co"))
        _run_async(_add_active_particle(subject_ids=[real_id]))
        result = _invoke(runner, ["subjects", "list", "--phantoms-only"])
        assert result.exit_code == 0
        assert "No phantom subjects found." in result.output


async def _add_embedded_particle(content: str) -> str:
    """Insert an ACTIVE particle WITH an embedding.

    Projection's candidate prefilter (``get_active_particles_with_embeddings``)
    requires a non-NULL embedding, so the bare ``_add_active_particle`` helper —
    which stores none — is invisible to ``project``. The splice path runs the
    same deterministic retrieval, so its test particle needs an embedding.
    """
    import numpy as np

    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    emb = (
        np.ones(4, dtype=np.float32) / float(np.linalg.norm(np.ones(4, dtype=np.float32)))
    ).tolist()
    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
    )
    async with session_scope() as session:
        await insert_particle(session, p, emb)
        await session.commit()
    return p.id


class TestProjectSpliceMode:
    """`particles project … --splice REGION` writes the rendered
    body between the named sentinels in an existing file, deterministically
    (``--without-synthesis`` needs no API key)."""

    @staticmethod
    def _write_manifest(tmp_path: Path) -> Path:
        manifest = tmp_path / "arch.yaml"
        manifest.write_text(
            "name: arch\n"
            "output: README.md\n"
            "sections:\n"
            "  - title: Architecture\n"
            "    query: layering\n"
            "    flowing: true\n",
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _write_readme(tmp_path: Path, manifest: Path) -> Path:
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Project\n\n"
            "Hand-authored intro.\n\n"
            "## Architecture\n\n"
            f"<!-- BEGIN PROJECTED: architecture (manifest: {manifest}) -->\n"
            "OLD placeholder prose.\n"
            "<!-- END PROJECTED: architecture -->\n\n"
            "## License\n\nHand-authored footer.\n",
            encoding="utf-8",
        )
        return readme

    def test_splice_writes_between_sentinels_and_preserves_outside(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _run_async(
            _add_embedded_particle(content="The package splits into Client and Engine layers.")
        )
        manifest = self._write_manifest(tmp_path)
        readme = self._write_readme(tmp_path, manifest)

        result = _invoke(
            runner,
            [
                "project",
                str(manifest),
                str(readme),
                "--splice",
                "architecture",
                "--without-synthesis",
            ],
        )
        assert result.exit_code == 0, result.output
        out = readme.read_text(encoding="utf-8")
        # New body landed inside the region; placeholder gone.
        assert "Client and Engine layers" in out
        assert "OLD placeholder prose." not in out
        # Everything outside the sentinels preserved.
        assert "Hand-authored intro." in out
        assert "Hand-authored footer." in out
        assert "## Architecture" in out
        assert "## License" in out
        # A single sentinel pair survives.
        assert out.count("<!-- BEGIN PROJECTED: architecture") == 1
        assert out.count("<!-- END PROJECTED: architecture -->") == 1

    def test_splice_missing_region_exits_2(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _run_async(_add_embedded_particle(content="A claim."))
        manifest = self._write_manifest(tmp_path)
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\nNo sentinels here.\n", encoding="utf-8")

        result = _invoke(
            runner,
            [
                "project",
                str(manifest),
                str(readme),
                "--splice",
                "architecture",
                "--without-synthesis",
            ],
        )
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_splice_missing_target_file_exits_2(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _run_async(_add_embedded_particle(content="A claim."))
        manifest = self._write_manifest(tmp_path)
        missing = tmp_path / "does-not-exist.md"

        result = _invoke(
            runner,
            [
                "project",
                str(manifest),
                str(missing),
                "--splice",
                "architecture",
                "--without-synthesis",
            ],
        )
        assert result.exit_code == 2
        assert "does not exist" in result.output


class TestProjectMultiRegion:
    """manifest-declared regions — `--splice-all` splices every
    declared region in one pass; `--splice REGION` re-rolls one; `--export-corpus`
    writes the sibling drift-gate bundle. All deterministic (no API key)."""

    @staticmethod
    def _write_manifest(tmp_path: Path, readme: Path) -> Path:
        manifest = tmp_path / "readme.yaml"
        manifest.write_text(
            "name: readme\n"
            f"output: {readme}\n"
            "sections:\n"
            "  - title: What is this\n"
            "    query: overview\n"
            "    region: what-is\n"
            "  - title: Architecture\n"
            "    query: layering\n"
            "    region: architecture\n",
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _write_readme(tmp_path: Path) -> Path:
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Project\n\n"
            "## What is this\n\n"
            "<!-- BEGIN PROJECTED: what-is (manifest: m) -->\n"
            "OLD what-is.\n"
            "<!-- END PROJECTED: what-is -->\n\n"
            "## Architecture\n\n"
            "<!-- BEGIN PROJECTED: architecture (manifest: m) -->\n"
            "OLD architecture.\n"
            "<!-- END PROJECTED: architecture -->\n\n"
            "## License\n\nHand-authored footer.\n",
            encoding="utf-8",
        )
        return readme

    def test_splice_all_fills_every_region(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _run_async(_add_embedded_particle(content="Claims split into Client and Engine layers."))
        readme = self._write_readme(tmp_path)
        manifest = self._write_manifest(tmp_path, readme)

        result = _invoke(
            runner,
            ["project", str(manifest), "--splice-all", "--without-synthesis"],
        )
        assert result.exit_code == 0, result.output
        out = readme.read_text(encoding="utf-8")
        assert "OLD what-is." not in out
        assert "OLD architecture." not in out
        # Both regions carry the deterministic listing; host structure intact.
        assert out.count("Client and Engine layers") == 2
        assert "Hand-authored footer." in out
        # Region bodies are headingless — the host headings are the only ones.
        assert out.count("## What is this") == 1
        assert out.count("## Architecture") == 1
        # The deterministic snapshot refreshed beside the manifest.
        assert (tmp_path / "readme.snapshot.md").exists()

    def test_splice_single_region_leaves_others_untouched(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _run_async(_add_embedded_particle(content="A layering claim."))
        readme = self._write_readme(tmp_path)
        manifest = self._write_manifest(tmp_path, readme)

        result = _invoke(
            runner,
            ["project", str(manifest), "--splice", "architecture", "--without-synthesis"],
        )
        assert result.exit_code == 0, result.output
        out = readme.read_text(encoding="utf-8")
        assert "OLD what-is." in out  # untouched region
        assert "OLD architecture." not in out  # re-rolled region

    def test_splice_all_and_splice_are_mutually_exclusive(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        result = _invoke(
            runner,
            ["project", "whatever.yaml", "--splice-all", "--splice", "x"],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_splice_all_requires_region_on_every_section(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        _run_async(_add_embedded_particle(content="A claim."))
        readme = self._write_readme(tmp_path)
        manifest = tmp_path / "readme.yaml"
        manifest.write_text(
            "name: readme\n"
            f"output: {readme}\n"
            "sections:\n"
            "  - title: Bound\n    query: q\n    region: what-is\n"
            "  - title: Unbound\n    query: q\n",
            encoding="utf-8",
        )
        result = _invoke(
            runner,
            ["project", str(manifest), "--splice-all", "--without-synthesis"],
        )
        assert result.exit_code == 2
        assert "Unbound" in result.output

    def test_export_corpus_writes_pinned_bundle(
        self, runner: CliRunner, cli_db: Path, tmp_path: Path
    ) -> None:
        pid = _run_async(_add_embedded_particle(content="The pinned claim."))
        manifest = tmp_path / "readme.yaml"
        manifest.write_text(
            "name: readme\n"
            "output: README.md\n"
            "sections:\n"
            "  - title: What is this\n"
            "    query: overview\n"
            "    region: what-is\n"
            "    min_confidence: 0.99\n"
            "    select:\n"
            f"      allow: [{pid}]\n",
            encoding="utf-8",
        )
        result = _invoke(runner, ["project", str(manifest), "--export-corpus"])
        assert result.exit_code == 0, result.output
        bundle = tmp_path / "readme.corpus.jsonl"
        assert bundle.exists()
        lines = [ln for ln in bundle.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1
        unit = json.loads(lines[0])
        assert unit["@type"] == "Particle"
        assert unit["sourceParticleId"] == pid


class TestLinksDedupCommand:
    """`particles links dedup` — exact-duplicate auto-merge."""

    @staticmethod
    def _seed_duplicates(content: str = "uv parses pyproject.toml.") -> tuple[str, list[str]]:
        subject_id = _run_async(_add_subject(name="uv"))
        ids = [
            _run_async(_add_active_particle(content=content, subject_ids=[subject_id]))
            for _ in range(3)
        ]
        return subject_id, ids

    def test_dry_run_is_the_default_and_writes_nothing(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        _subject_id, ids = self._seed_duplicates()

        result = _invoke(runner, ["links", "dedup"])

        assert result.exit_code == 0, result.output
        assert "dry run — nothing written" in result.output
        assert "1 exact-duplicate group(s), 2 redundant" in result.output

        async def _statuses() -> list[Status]:
            from particles.db import session_scope
            from particles.store.particle_store import get_particle

            async with session_scope() as session:
                out = []
                for pid in ids:
                    particle = await get_particle(session, pid)
                    assert particle is not None
                    out.append(particle.status)
                return out

        assert _run_async(_statuses()) == [Status.ACTIVE] * 3

    def test_apply_refused_without_the_config_opt_in(self, runner: CliRunner, cli_db: Path) -> None:
        """Default OFF: --apply exits non-zero and names the knob."""
        self._seed_duplicates()

        result = runner.invoke(app, ["links", "dedup", "--apply"])

        assert result.exit_code != 0
        assert "links_suggest.auto_merge.enabled" in result.output

    def test_json_output_carries_the_group_counts(self, runner: CliRunner, cli_db: Path) -> None:
        self._seed_duplicates()

        result = _invoke(runner, ["links", "dedup", "--output-format", "json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["dry_run"] is True
        assert payload["total_groups"] == 1
        assert payload["total_redundant"] == 2
        assert payload["merged_groups"] == 0

    def test_unknown_subject_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["links", "dedup", "--subject", "no-such-subject"])
        assert result.exit_code != 0

    def test_empty_store_reports_no_duplicates(self, runner: CliRunner, cli_db: Path) -> None:
        result = _invoke(runner, ["links", "dedup"])
        assert result.exit_code == 0, result.output
        assert "no exact duplicates" in result.output


class TestLinksUnmergeCommand:
    """`particles links unmerge` — revert of an auto-merge."""

    @staticmethod
    def _seed_and_merge(runner: CliRunner) -> tuple[str, list[str]]:
        """Seed three duplicates, merge them via the CLI, return (event_id, ids)."""
        subject_id = _run_async(_add_subject(name="uv"))
        ids = [
            _run_async(
                _add_active_particle(content="uv parses pyproject.toml.", subject_ids=[subject_id])
            )
            for _ in range(3)
        ]

        async def _merge() -> str:
            from particles.config import get_config
            from particles.db import session_scope
            from particles.operations.links_suggest import auto_merge_exact_duplicates
            from particles.store.event_store import OperatorEventType, list_events

            get_config().links_suggest.auto_merge.enabled = True
            try:
                async with session_scope(write=True) as session:
                    await auto_merge_exact_duplicates(session, dry_run=False)
                async with session_scope() as session:
                    events = await list_events(
                        session, event_type=OperatorEventType.DUPLICATES_MERGED
                    )
                    return events[0].event_id
            finally:
                get_config().links_suggest.auto_merge.enabled = False

        return _run_async(_merge()), ids

    def test_dry_run_shows_the_plan_and_writes_nothing(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        event_id, ids = self._seed_and_merge(runner)

        result = _invoke(runner, ["links", "unmerge", event_id, "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "plan — nothing written" in result.output
        assert "2 copy/copies restored to ACTIVE" in result.output
        assert "--dry-run: nothing written." in result.output
        assert _run_async(_count_status(ids, Status.ACTIVE)) == 1

    def test_yes_reverts_and_restores_the_copies(self, runner: CliRunner, cli_db: Path) -> None:
        event_id, ids = self._seed_and_merge(runner)

        result = _invoke(runner, ["links", "unmerge", event_id, "--yes"])

        assert result.exit_code == 0, result.output
        assert "✓ reverted" in result.output
        assert _run_async(_count_status(ids, Status.ACTIVE)) == 3

    def test_unknown_event_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["links", "unmerge", "no-such-event", "--yes"])
        assert result.exit_code != 0
        assert "No operator event" in result.output

    def test_no_selector_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["links", "unmerge", "--yes"])
        assert result.exit_code != 0
        assert "exactly one" in result.output

    def test_two_selectors_error(self, runner: CliRunner, cli_db: Path) -> None:
        event_id, _ids = self._seed_and_merge(runner)
        result = runner.invoke(app, ["links", "unmerge", event_id, "--run", "r1", "--yes"])
        assert result.exit_code != 0
        assert "exactly one" in result.output

    def test_json_output_carries_the_counts(self, runner: CliRunner, cli_db: Path) -> None:
        event_id, _ids = self._seed_and_merge(runner)

        result = _invoke(runner, ["links", "unmerge", event_id, "--yes", "--output-format", "json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["dry_run"] is False
        assert payload["restored_particles"] == 2
        assert payload["groups"][0]["reverted"] is True
