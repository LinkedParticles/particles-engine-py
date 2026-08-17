"""Tests for `particles subjects split`.

Covers both the store helper (split_subject) and the CLI verb. The CLI's
resolver-driven path is exercised with a mocked subject resolver so the
test doesn't hit the live Wikidata API.
"""

from __future__ import annotations

import asyncio
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
    ExternalRef,
    Particle,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

# ---------------------------------------------------------------------------
# Fixtures + helpers (mirror tests/test_cli.py patterns)
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, args: list[str]) -> Any:
    return runner.invoke(app, args, catch_exceptions=False)


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


async def _insert_subject(name: str, *, external_ids: list[ExternalRef] | None = None) -> str:
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    subj = Subject(
        id=str(uuid.uuid4()),
        canonical_name=name,
        external_ids=external_ids or [],
        asserted_by="test",
    )
    async with session_scope() as session:
        await insert_subject(session, subj)
        await session.commit()
    return subj.id


async def _insert_particle_bound_to(*, content: str, subject_ids: list[str]) -> str:
    """Insert an ACTIVE CLAIM particle and bind it to ``subject_ids``.

    ``insert_particle`` already writes the join-table rows via
    ``link_particle_to_subjects`` (see particles/store/particle_store.py).
    The test only needs to call insert_particle and the bindings are in
    place.
    """
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle

    p = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        subject_ids=subject_ids,
    )
    async with session_scope() as session:
        await insert_particle(session, p)
        await session.commit()
    return p.id


async def _subject_id_for_particle(particle_id: str) -> list[str]:
    """Return the current particle_subjects bindings + denormalised json."""
    from sqlalchemy import select

    from particles.db import session_scope
    from particles.store.particle_store import ParticleRow
    from particles.store.subject_store import ParticleSubjectRow

    async with session_scope() as session:
        join_rows = await session.execute(
            select(ParticleSubjectRow.subject_id).where(
                ParticleSubjectRow.particle_id == particle_id
            )
        )
        join_ids = sorted(join_rows.scalars())
        p_row = await session.get(ParticleRow, particle_id)
        import json as _json

        denorm = sorted(_json.loads(p_row.subject_ids_json)) if p_row else []
    assert join_ids == denorm, (
        f"join table and denormalised subject_ids drifted: {join_ids=} {denorm=}"
    )
    return join_ids


# ---------------------------------------------------------------------------
# Store helper — split_subject
# ---------------------------------------------------------------------------


class TestSplitSubjectStoreHelper:
    """the store helper does the re-linking; it does NOT insert the
    new Subject. Caller wires the resolver and supplies the new Subject id."""

    @pytest.mark.asyncio
    async def test_basic_split_relinks_particle(self, db_session: object) -> None:
        from particles.db import session_scope
        from particles.store.subject_store import split_subject

        source_id = await _insert_subject("AAOI (misjoined)")
        new_id = await _insert_subject(
            "Applied Optoelectronics",
            external_ids=[ExternalRef(namespace="wikidata", id="Q30297735", confidence=1.0)],
        )
        pid = await _insert_particle_bound_to(
            content="The company has a Texas manufacturing facility.",
            subject_ids=[source_id],
        )

        async with session_scope() as s:
            relinked, not_bound = await split_subject(
                s,
                source_id=source_id,
                new_subject_id=new_id,
                particle_ids=[pid],
            )
            await s.commit()
        assert relinked == [pid]
        assert not_bound == []
        # Particle now bound to new subject only.
        assert await _subject_id_for_particle(pid) == [new_id]

    @pytest.mark.asyncio
    async def test_multi_subject_particle_partial_move(self, db_session: object) -> None:
        """A particle bound to [source, other] becomes [new, other] —
        only the source binding moves; the other is untouched."""
        from particles.db import session_scope
        from particles.store.subject_store import split_subject

        source_id = await _insert_subject("AAOI (misjoined)")
        other_id = await _insert_subject("United States")
        new_id = await _insert_subject("Applied Optoelectronics")

        pid = await _insert_particle_bound_to(
            content="Applied Optoelectronics has a Texas facility.",
            subject_ids=[source_id, other_id],
        )

        async with session_scope() as s:
            relinked, not_bound = await split_subject(
                s,
                source_id=source_id,
                new_subject_id=new_id,
                particle_ids=[pid],
            )
            await s.commit()
        assert relinked == [pid]
        bindings = await _subject_id_for_particle(pid)
        # new + other; source is gone.
        assert source_id not in bindings
        assert new_id in bindings
        assert other_id in bindings

    @pytest.mark.asyncio
    async def test_unbound_particle_is_no_op_warning(self, db_session: object) -> None:
        """A pid that isn't bound to the source is a no-op for the split
        but is returned in not_bound so the CLI can warn."""
        from particles.db import session_scope
        from particles.store.subject_store import split_subject

        source_id = await _insert_subject("Source")
        new_id = await _insert_subject("New")
        other_id = await _insert_subject("Other")
        # Bind the particle to other_id, not source_id.
        pid = await _insert_particle_bound_to(content="claim about other", subject_ids=[other_id])

        async with session_scope() as s:
            relinked, not_bound = await split_subject(
                s,
                source_id=source_id,
                new_subject_id=new_id,
                particle_ids=[pid],
            )
            await s.commit()
        assert relinked == []
        assert not_bound == [pid]
        # Particle's bindings unchanged.
        assert await _subject_id_for_particle(pid) == [other_id]

    @pytest.mark.asyncio
    async def test_empty_source_subject_survives(self, db_session: object) -> None:
        """If every particle was split off, the source Subject is preserved
        empty Subjects survive for audit-trail reasons."""
        from particles.db import session_scope
        from particles.store.subject_store import get_subject, split_subject

        source_id = await _insert_subject("Source (to be emptied)")
        new_id = await _insert_subject("New")
        pid = await _insert_particle_bound_to(content="the only claim", subject_ids=[source_id])

        async with session_scope() as s:
            await split_subject(
                s,
                source_id=source_id,
                new_subject_id=new_id,
                particle_ids=[pid],
            )
            await s.commit()

        async with session_scope() as s:
            source = await get_subject(s, source_id)
        assert source is not None  # not deleted
        assert source.id == source_id

    @pytest.mark.asyncio
    async def test_missing_source_raises(self, db_session: object) -> None:
        from particles.db import session_scope
        from particles.store.subject_store import split_subject

        new_id = await _insert_subject("New")
        async with session_scope() as s:
            with pytest.raises(ValueError, match="Source subject .* not found"):
                await split_subject(
                    s,
                    source_id="00000000-0000-0000-0000-000000000000",
                    new_subject_id=new_id,
                    particle_ids=[],
                )

    @pytest.mark.asyncio
    async def test_source_equals_new_raises(self, db_session: object) -> None:
        from particles.db import session_scope
        from particles.store.subject_store import split_subject

        source_id = await _insert_subject("Same")
        async with session_scope() as s:
            with pytest.raises(ValueError, match="must differ"):
                await split_subject(
                    s,
                    source_id=source_id,
                    new_subject_id=source_id,
                    particle_ids=[],
                )


# ---------------------------------------------------------------------------
# CLI verb — subjects split
# ---------------------------------------------------------------------------


def _sync_insert_subject(name: str) -> str:
    return _run_async(_insert_subject(name))


def _sync_insert_particle(content: str, subject_ids: list[str]) -> str:
    return _run_async(_insert_particle_bound_to(content=content, subject_ids=subject_ids))


class TestSubjectsSplitCli:
    def test_requires_source_id(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "split"])
        assert result.exit_code != 0
        assert "Usage:" in result.output

    def test_requires_at_least_one_particle(self, runner: CliRunner, cli_db: Path) -> None:
        source_id = _sync_insert_subject("Source")
        result = runner.invoke(
            app,
            [
                "subjects",
                "split",
                source_id,
                "--new-name",
                "Foo",
            ],
        )
        assert result.exit_code != 0
        assert "requires at least one --particle" in result.output

    def test_requires_name_or_external_id(self, runner: CliRunner, cli_db: Path) -> None:
        source_id = _sync_insert_subject("Source")
        pid = _sync_insert_particle("claim", [source_id])
        result = runner.invoke(
            app,
            ["subjects", "split", source_id, "--particle", pid],
        )
        assert result.exit_code != 0
        assert "requires --new-name or --new-external-id" in result.output

    def test_unknown_source_subject_errors(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(
            app,
            [
                "subjects",
                "split",
                "00000000",
                "--particle",
                "p1",
                "--new-name",
                "Foo",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_split_with_external_id_wikidata(self, runner: CliRunner, cli_db: Path) -> None:
        """--new-external-id wikidata:Q... fetches metadata via the
        Wikidata alias helper (which we mock to avoid network)."""
        source_id = _sync_insert_subject("AAOI (misjoined)")
        pid = _sync_insert_particle("Applied Optoelectronics is based in Texas.", [source_id])

        with patch(
            "particles.ingest.authorities.wikidata._wikidata_aliases",
            new=AsyncMock(return_value=["Applied Optoelectronics", "AAOI Inc."]),
        ):
            result = runner.invoke(
                app,
                [
                    "subjects",
                    "split",
                    source_id,
                    "--particle",
                    pid,
                    "--new-external-id",
                    "wikidata:Q30297735",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Split 1 particle(s)" in result.output
        assert "Applied Optoelectronics" in result.output
        # Particle re-bound to the new Wikidata-anchored Subject.
        bindings = _run_async(_subject_id_for_particle(pid))
        assert source_id not in bindings
        assert len(bindings) == 1

    def test_split_with_new_name_resolver_driven(self, runner: CliRunner, cli_db: Path) -> None:
        """--new-name "..." invokes the resolver. Mock the resolver to
        avoid the Wikidata API call but return a constructed Subject."""
        source_id = _sync_insert_subject("AAOI (misjoined)")
        pid = _sync_insert_particle("Some claim about the company.", [source_id])

        # Build the Subject the resolver will "return" (it would insert
        # internally; mock simulates that path).
        new_subject_obj = Subject(
            id=str(uuid.uuid4()),
            canonical_name="Applied Optoelectronics",
            external_ids=[ExternalRef(namespace="wikidata", id="Q30297735", confidence=0.95)],
            asserted_by="subjects-split",
        )

        async def fake_resolve(
            session: object, name: str, asserted_by: str = "general-extractor"
        ) -> Subject:
            from particles.store.subject_store import insert_subject

            await insert_subject(session, new_subject_obj)  # type: ignore[arg-type]
            return new_subject_obj

        with patch(
            "particles.ingest.subject_resolver.resolve_subject",
            new=fake_resolve,
        ):
            result = runner.invoke(
                app,
                [
                    "subjects",
                    "split",
                    source_id,
                    "--particle",
                    pid,
                    "--new-name",
                    "Applied Optoelectronics, Inc.",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Applied Optoelectronics" in result.output
        bindings = _run_async(_subject_id_for_particle(pid))
        assert source_id not in bindings
        assert new_subject_obj.id in bindings

    def test_split_dry_run_makes_no_changes(self, runner: CliRunner, cli_db: Path) -> None:
        source_id = _sync_insert_subject("Source")
        pid = _sync_insert_particle("claim", [source_id])

        with patch(
            "particles.ingest.authorities.wikidata._wikidata_aliases",
            new=AsyncMock(return_value=["Applied Optoelectronics"]),
        ):
            result = runner.invoke(
                app,
                [
                    "subjects",
                    "split",
                    source_id,
                    "--particle",
                    pid,
                    "--new-external-id",
                    "wikidata:Q30297735",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert "no changes made" in result.output
        # Particle bindings unchanged.
        bindings = _run_async(_subject_id_for_particle(pid))
        assert bindings == [source_id]

    def test_unbound_pid_is_warned_about(self, runner: CliRunner, cli_db: Path) -> None:
        source_id = _sync_insert_subject("Source")
        other_id = _sync_insert_subject("Other")
        # Particle bound to other, not source.
        pid_unbound = _sync_insert_particle("claim about other", [other_id])
        # And one bound to source so the split has something to do.
        pid_bound = _sync_insert_particle("claim about source", [source_id])

        with patch(
            "particles.ingest.authorities.wikidata._wikidata_aliases",
            new=AsyncMock(return_value=["Applied Optoelectronics"]),
        ):
            result = runner.invoke(
                app,
                [
                    "subjects",
                    "split",
                    source_id,
                    "--particle",
                    pid_bound,
                    "--particle",
                    pid_unbound,
                    "--new-external-id",
                    "wikidata:Q30297735",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Split 1 particle(s)" in result.output
        assert "Skipped 1 particle(s) not bound" in result.output
