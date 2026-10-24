# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single-pass N→1 `particles subjects merge`.

Covers the store helpers (plan_subject_merge, merge_subjects_many), the one
operator event a merge records, and the CLI verb's preview, confirmation and
refusals.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import Confidence, Particle, Subject, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


async def _insert_subject(name: str, *, aliases: list[str] | None = None) -> str:
    from particles.db import session_scope
    from particles.store.subject_store import insert_subject

    subj = Subject(
        id=str(uuid.uuid4()),
        canonical_name=name,
        aliases=aliases or [],
        asserted_by="test",
    )
    async with session_scope() as session:
        await insert_subject(session, subj)
        await session.commit()
    return subj.id


async def _insert_particle(*, content: str, subject_ids: list[str]) -> str:
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


async def _bindings(particle_id: str) -> tuple[list[str], list[str]]:
    """Return (join-table subject ids, denormalised subject_ids_json) for a particle."""
    from sqlalchemy import select

    from particles.db import session_scope
    from particles.store.particle_store import ParticleRow
    from particles.store.subject_store import ParticleSubjectRow

    async with session_scope() as session:
        rows = await session.execute(
            select(ParticleSubjectRow.subject_id).where(
                ParticleSubjectRow.particle_id == particle_id
            )
        )
        join_ids = sorted(rows.scalars())
        p_row = await session.get(ParticleRow, particle_id)
        denorm = json.loads(p_row.subject_ids_json) if p_row else []
    return join_ids, denorm


async def _get(subject_id: str) -> Subject | None:
    from particles.db import session_scope
    from particles.store.subject_store import get_subject

    async with session_scope() as session:
        return await get_subject(session, subject_id)


async def _merge(source_ids: list[str], target_id: str) -> Any:
    from particles.db import session_scope
    from particles.store.subject_store import merge_subjects_many

    async with session_scope() as session:
        result = await merge_subjects_many(session, source_ids, target_id)
        await session.commit()
    return result


async def _merge_events() -> list[Any]:
    from particles.db import session_scope
    from particles.store.event_store import OperatorEventType, list_events

    async with session_scope() as session:
        return await list_events(session, event_type=OperatorEventType.SUBJECTS_MERGED)


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


class TestMergeSubjectsMany:
    @pytest.mark.asyncio
    async def test_folds_every_source_into_target(self, db_session: object) -> None:
        a = await _insert_subject("Particles spec", aliases=["the spec"])
        b = await _insert_subject("specification")
        target = await _insert_subject("Particles specification")
        pa = await _insert_particle(content="Claim A.", subject_ids=[a])
        pb = await _insert_particle(content="Claim B.", subject_ids=[b])

        updated, plan = await _merge([a, b], target)

        assert updated.id == target
        assert set(plan.aliases_added) == {"Particles spec", "the spec", "specification"}
        assert set(updated.aliases) >= set(plan.aliases_added)
        assert plan.particles_relinked == 2
        assert await _get(a) is None
        assert await _get(b) is None
        assert await _bindings(pa) == ([target], [target])
        assert await _bindings(pb) == ([target], [target])

    @pytest.mark.asyncio
    async def test_particle_naming_two_sources_names_target_once(self, db_session: object) -> None:
        a = await _insert_subject("Jeff")
        b = await _insert_subject("Jeff Gage")
        other = await _insert_subject("Particles")
        target = await _insert_subject("the owner")
        pid = await _insert_particle(content="Jeff works on Particles.", subject_ids=[a, other, b])

        await _merge([a, b], target)

        join_ids, denorm = await _bindings(pid)
        assert join_ids == sorted([target, other])
        assert denorm == [target, other]

    @pytest.mark.asyncio
    async def test_particle_already_on_target_is_not_duplicated(self, db_session: object) -> None:
        a = await _insert_subject("GDR")
        target = await _insert_subject("East Germany")
        pid = await _insert_particle(content="The GDR existed.", subject_ids=[target, a])

        await _merge([a], target)

        assert await _bindings(pid) == ([target], [target])

    @pytest.mark.asyncio
    async def test_one_event_records_the_whole_merge(self, db_session: object) -> None:
        from particles.store.event_store import EventRefKind

        a = await _insert_subject("PRD v0.x")
        b = await _insert_subject("specification")
        target = await _insert_subject("Particles specification")
        pa = await _insert_particle(content="Claim A.", subject_ids=[a])

        await _merge([a, b], target)

        events = await _merge_events()
        assert len(events) == 1
        refs = {(r.ref_kind, r.ref_id) for r in events[0].refs}
        assert {
            (EventRefKind.SUBJECT, a),
            (EventRefKind.SUBJECT, b),
            (EventRefKind.SUBJECT, target),
            (EventRefKind.PARTICLE, pa),
        } <= refs
        payload = events[0].payload
        assert payload["sources"] == [a, b]
        assert payload["source_names"] == ["PRD v0.x", "specification"]
        assert payload["target"] == target
        assert payload["relinked"] == 1

    @pytest.mark.asyncio
    async def test_aliases_not_repeated_across_sources(self, db_session: object) -> None:
        a = await _insert_subject("Spec", aliases=["spec"])
        b = await _insert_subject("SPEC")
        target = await _insert_subject("Particles specification", aliases=["the spec"])

        _, plan = await _merge([a, b], target)

        assert plan.aliases_added == ("Spec",)

    @pytest.mark.asyncio
    async def test_source_equal_to_target_refuses_and_writes_nothing(
        self, db_session: object
    ) -> None:
        a = await _insert_subject("Spec")
        target = await _insert_subject("Specification")

        with pytest.raises(ValueError, match="is the merge target"):
            await _merge([a, target], target)

        assert await _get(a) is not None
        assert await _merge_events() == []

    @pytest.mark.asyncio
    async def test_source_listed_twice_refuses(self, db_session: object) -> None:
        a = await _insert_subject("Spec")
        target = await _insert_subject("Specification")

        with pytest.raises(ValueError, match="more than once"):
            await _merge([a, a[:8]], target)

    @pytest.mark.asyncio
    async def test_missing_source_refuses_whole_merge(self, db_session: object) -> None:
        from particles.store.subject_store import SubjectNotFoundError

        a = await _insert_subject("Spec")
        target = await _insert_subject("Specification")
        pa = await _insert_particle(content="Claim A.", subject_ids=[a])

        with pytest.raises(SubjectNotFoundError, match="not found"):
            await _merge([a, str(uuid.uuid4())], target)

        # The valid source was checked first but nothing moved.
        assert await _get(a) is not None
        assert await _bindings(pa) == ([a], [a])

    @pytest.mark.asyncio
    async def test_already_merged_source_is_named(self, db_session: object) -> None:
        from particles.store.subject_store import SubjectNotFoundError

        a = await _insert_subject("Spec")
        target = await _insert_subject("Specification")
        other = await _insert_subject("Other target")
        await _merge([a], target)

        with pytest.raises(SubjectNotFoundError, match=f"already merged into {target}"):
            await _merge([a], other)

    @pytest.mark.asyncio
    async def test_no_sources_refuses(self, db_session: object) -> None:
        target = await _insert_subject("Specification")
        with pytest.raises(ValueError, match="At least one source"):
            await _merge([], target)

    @pytest.mark.asyncio
    async def test_one_to_one_wrapper_keeps_its_return_shape(self, db_session: object) -> None:
        from particles.db import session_scope
        from particles.store.subject_store import merge_subjects

        a = await _insert_subject("BTC")
        target = await _insert_subject("Bitcoin")
        await _insert_particle(content="BTC is scarce.", subject_ids=[a])

        async with session_scope() as session:
            subject, added, relinked = await merge_subjects(session, a, target)
            await session.commit()

        assert subject.id == target
        assert added == ["BTC"]
        assert relinked == 1


# ---------------------------------------------------------------------------
# CLI verb
# ---------------------------------------------------------------------------


class TestSubjectsMergeCLI:
    def test_dry_run_lists_sources_and_writes_nothing(
        self, runner: CliRunner, cli_db: Path
    ) -> None:
        a = _run_async(_insert_subject("Particles spec"))
        b = _run_async(_insert_subject("specification"))
        target = _run_async(_insert_subject("Particles specification"))
        _run_async(_insert_particle(content="Claim A.", subject_ids=[a]))

        result = runner.invoke(
            app, ["subjects", "merge", a, b, target, "--dry-run"], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "Would merge 2 subject(s)" in result.output
        assert "Particles spec  (1 particle link(s))" in result.output
        assert "specification  (0 particle link(s))" in result.output
        assert "(no changes made)" in result.output
        assert _run_async(_get(a)) is not None
        assert _run_async(_merge_events()) == []

    def test_yes_merges_every_source(self, runner: CliRunner, cli_db: Path) -> None:
        a = _run_async(_insert_subject("Particles spec"))
        b = _run_async(_insert_subject("specification"))
        target = _run_async(_insert_subject("Particles specification"))

        result = runner.invoke(
            app, ["subjects", "merge", a[:8], b[:8], target[:8], "--yes"], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "Merged 2 subject(s)" in result.output
        assert _run_async(_get(a)) is None
        assert _run_async(_get(b)) is None
        assert len(_run_async(_merge_events())) == 1

    def test_declined_confirmation_writes_nothing(self, runner: CliRunner, cli_db: Path) -> None:
        a = _run_async(_insert_subject("Particles spec"))
        target = _run_async(_insert_subject("Particles specification"))

        result = runner.invoke(app, ["subjects", "merge", a, target], input="n\n")

        assert result.exit_code != 0
        assert "Merge 1 subject(s)?" in result.output
        assert _run_async(_get(a)) is not None
        assert _run_async(_merge_events()) == []

    def test_confirmation_accepted_merges(self, runner: CliRunner, cli_db: Path) -> None:
        a = _run_async(_insert_subject("Particles spec"))
        target = _run_async(_insert_subject("Particles specification"))

        result = runner.invoke(
            app, ["subjects", "merge", a, target], input="y\n", catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert _run_async(_get(a)) is None

    def test_source_equal_to_target_refused(self, runner: CliRunner, cli_db: Path) -> None:
        a = _run_async(_insert_subject("Particles spec"))
        target = _run_async(_insert_subject("Particles specification"))

        result = runner.invoke(app, ["subjects", "merge", a, target, target, "--yes"])

        assert result.exit_code == 1
        assert "is the merge target" in result.output
        assert _run_async(_get(a)) is not None

    def test_already_merged_source_refused(self, runner: CliRunner, cli_db: Path) -> None:
        a = _run_async(_insert_subject("Particles spec"))
        target = _run_async(_insert_subject("Particles specification"))
        _run_async(_merge([a], target))

        result = runner.invoke(app, ["subjects", "merge", a, target, "--yes"])

        assert result.exit_code == 1
        assert "already merged" in result.output

    def test_needs_a_source_and_a_target(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "merge", "only-one"])
        assert result.exit_code == 1
        assert "Usage: subjects merge SOURCE_ID [SOURCE_ID ...] TARGET_ID" in result.output
