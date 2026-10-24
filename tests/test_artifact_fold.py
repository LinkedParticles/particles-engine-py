# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The artifact fold over a store: carried refs, the root hook, apply and undo.

The pure planner is pinned in ``tests/test_artifact_fold_plan.py``. These tests
check what the planner cannot: that a merge carries a source's ``artifact``
ref so the next transcript does not re-split it, that a fold and its undo
leave the store as it was, and the CLI's refusals.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import claude_project_slug
from particles.core.schema import (
    Confidence,
    ExternalRef,
    Particle,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status
from particles.ingest.artifact_namespace import (
    register_artifact_namespace,
    register_artifact_relative_path,
)
from particles.ingest.authorities.artifact import ArtifactAuthority
from particles.ingest.authorities.registry import RecognizeContext
from particles.ingest.subject_resolver import resolve_qualified_subject
from particles.interchange.codec import SubjectRef
from particles.interchange.store import _resolve_subject_ref
from particles.operations.artifact_fold import build_fold_report, undo_fold
from particles.store.event_store import OperatorEventType, list_events
from particles.store.particle_store import ParticleRow, insert_particle
from particles.store.subject_store import (
    ParticleSubjectRow,
    get_subject,
    insert_subject,
    merge_subjects_many,
    plan_subject_merge,
)

NS = "proj"


@pytest.fixture(autouse=True)
def _default_hooks() -> Iterator[None]:
    register_artifact_namespace(None)
    register_artifact_relative_path(None)
    yield
    register_artifact_namespace(None)
    register_artifact_relative_path(None)


def _ref(path: str, ns: str = NS) -> ExternalRef:
    return ExternalRef(namespace="artifact", id=f"{ns}/{path}", confidence=1.0)


async def _file(session: Any, path: str, ns: str = NS) -> str:
    subject = Subject(
        canonical_name=path,
        external_ids=[_ref(path, ns)],
        subject_class="artifact:file",
        asserted_by="test",
    )
    await insert_subject(session, subject)
    return subject.id


async def _plain(session: Any, name: str, *, aliases: list[str] | None = None) -> str:
    subject = Subject(canonical_name=name, aliases=aliases or [], asserted_by="test")
    await insert_subject(session, subject)
    return subject.id


async def _claim(session: Any, content: str, subject_ids: list[str]) -> str:
    particle = Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        subject_ids=subject_ids,
    )
    await insert_particle(session, particle)
    return particle.id


async def _links(session: Any, particle_id: str) -> tuple[set[str], list[str]]:
    rows = await session.execute(
        select(ParticleSubjectRow.subject_id).where(ParticleSubjectRow.particle_id == particle_id)
    )
    p_row = await session.get(ParticleRow, particle_id)
    return set(rows.scalars().all()), json.loads(p_row.subject_ids_json)


def _context(ns: str = NS) -> RecognizeContext:
    return RecognizeContext(token_class="filename", namespace_key=ns, source_type="CONVERSATION")


class TestMergeCarriesArtifactRefs:
    async def test_a_re_mention_after_a_merge_binds_to_the_target(self, db_session: Any) -> None:
        bare = await _file(db_session, "codec.py")
        target = await _file(db_session, "particles/interchange/codec.py")
        await merge_subjects_many(db_session, [bare], target)

        again = await resolve_qualified_subject(db_session, "codec.py", _context())

        assert again is not None and again.id == target
        merged = await get_subject(db_session, target)
        assert merged is not None
        assert {r.id for r in merged.external_ids} == {
            f"{NS}/particles/interchange/codec.py",
            f"{NS}/codec.py",
        }

    async def test_another_projects_ref_is_withheld(self, db_session: Any) -> None:
        other = await _file(db_session, "codec.py", ns="elsewhere")
        target = await _file(db_session, "particles/interchange/codec.py")

        plan = await plan_subject_merge(db_session, [other], target)
        assert [r.id for r in plan.sources[0].refs_withheld] == ["elsewhere/codec.py"]
        await merge_subjects_many(db_session, [other], target)

        merged = await get_subject(db_session, target)
        assert merged is not None
        assert [r.id for r in merged.external_ids] == [f"{NS}/particles/interchange/codec.py"]

    async def test_a_ref_the_target_holds_is_not_duplicated(self, db_session: Any) -> None:
        target = await _file(db_session, "particles/config.py")
        twin = Subject(
            canonical_name="config twin",
            external_ids=[_ref("particles/config.py"), _ref("config.py")],
            subject_class="artifact:file",
            asserted_by="test",
        )
        await insert_subject(db_session, twin)

        await merge_subjects_many(db_session, [twin.id], target)

        merged = await get_subject(db_session, target)
        assert merged is not None
        assert sorted(r.id for r in merged.external_ids) == [
            f"{NS}/config.py",
            f"{NS}/particles/config.py",
        ]

    async def test_a_non_artifact_ref_is_not_carried(self, db_session: Any) -> None:
        target = await _file(db_session, "particles/config.py")
        wiki = Subject(
            canonical_name="Config",
            external_ids=[ExternalRef(namespace="wikidata", id="Q1", confidence=1.0)],
            asserted_by="test",
        )
        await insert_subject(db_session, wiki)

        await merge_subjects_many(db_session, [wiki.id], target)

        merged = await get_subject(db_session, target)
        assert merged is not None
        assert all(r.namespace == "artifact" for r in merged.external_ids)

    async def test_an_import_re_attaches_by_either_ref(self, db_session: Any) -> None:
        bare = await _file(db_session, "codec.py")
        target = await _file(db_session, "particles/interchange/codec.py")
        await merge_subjects_many(db_session, [bare], target)

        for path in ("codec.py", "particles/interchange/codec.py"):
            sid, created = await _resolve_subject_ref(
                db_session, SubjectRef(external_refs=[_ref(path)], canonical_name=path)
            )
            assert (sid, created) == (target, False)


class TestRootHook:
    def test_an_absolute_path_is_minted_root_relative(self) -> None:
        key = claude_project_slug("/Users/me/repo")
        register_artifact_relative_path(
            lambda ns, path: path.removeprefix("/Users/me/repo/") if ns == key else None
        )
        ref = ArtifactAuthority().recognize_in("/Users/me/repo/particles/config.py", _context(key))
        assert ref is not None and ref.id == f"{key}/particles/config.py"

    def test_with_nothing_registered_it_is_kept(self) -> None:
        ref = ArtifactAuthority().recognize_in("/Users/me/repo/particles/config.py", _context())
        assert ref is not None and ref.id == f"{NS}//Users/me/repo/particles/config.py"


class TestFoldAndUndo:
    async def _store(self, session: Any) -> dict[str, str]:
        ids = {
            "target": await _file(session, "particles/config.py"),
            "bare": await _file(session, "config_loader.py"),
            "module": await _plain(session, "particles.config", aliases=["the config module"]),
        }
        ids["p_bare"] = await _claim(session, "config_loader.py reads YAML.", [ids["bare"]])
        ids["p_module"] = await _claim(
            session, "particles.config holds the knobs.", [ids["module"]]
        )
        ids["p_both"] = await _claim(
            session, "particles.config is particles/config.py.", [ids["module"], ids["target"]]
        )
        await session.commit()
        return ids

    async def test_a_fold_and_its_undo_restore_the_store(self, db_session: Any) -> None:
        ids = await self._store(db_session)
        repo = ["particles/config.py", "particles/config_loader.py"]
        before_target = await get_subject(db_session, ids["target"])
        before_links = {
            p: await _links(db_session, ids[p]) for p in ("p_bare", "p_module", "p_both")
        }

        dry = await build_fold_report(db_session, NS, repo)
        assert {(f.source_name, f.rule.value) for f in dry.folds} == {
            ("particles.config", "module"),
        }
        assert {(r.name, r.reason.value) for r in dry.refusals} == {
            ("config_loader.py", "no Subject for the file's path yet"),
        }
        report = await build_fold_report(db_session, NS, repo, apply=True)
        await db_session.commit()

        assert len(report.applied_events) == 1
        assert await get_subject(db_session, ids["module"]) is None
        assert (await _links(db_session, ids["p_module"]))[0] == {ids["target"]}

        await undo_fold(db_session, report.applied_events[0])
        await db_session.commit()

        restored = await get_subject(db_session, ids["module"])
        assert restored is not None and restored.aliases == ["the config module"]
        assert await get_subject(db_session, ids["target"]) == before_target
        for p, (links, ids_json) in before_links.items():
            after_links, after_json = await _links(db_session, ids[p])
            assert after_links == links, p
            assert sorted(after_json) == sorted(ids_json), p

        splits = await list_events(db_session, event_type=OperatorEventType.SUBJECTS_SPLIT)
        assert splits[0].payload is not None
        assert splits[0].payload["reverses"] == report.applied_events[0]

    async def test_an_undo_twice_is_refused(self, db_session: Any) -> None:
        await self._store(db_session)
        report = await build_fold_report(db_session, NS, ["particles/config.py"], apply=True)
        await undo_fold(db_session, report.applied_events[0])
        with pytest.raises(ValueError, match="undone already"):
            await undo_fold(db_session, report.applied_events[0])

    async def test_an_operator_merge_is_not_a_fold(self, db_session: Any) -> None:
        a = await _plain(db_session, "A")
        b = await _plain(db_session, "B")
        await merge_subjects_many(db_session, [a], b)
        [event] = await list_events(db_session, event_type=OperatorEventType.SUBJECTS_MERGED)
        with pytest.raises(ValueError, match="not an artifact fold"):
            await undo_fold(db_session, event.event_id)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestFoldArtifactsCLI:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def _fake_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: list[str]
    ) -> tuple[Path, str]:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)

        def fake_run(argv: list[str], **_kw: Any) -> subprocess.CompletedProcess[bytes]:
            assert argv[:2] == ["git", "-C"] and argv[3:] == ["ls-files", "-z"]
            return subprocess.CompletedProcess(argv, 0, "\0".join(files).encode(), b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        return repo, claude_project_slug(repo.absolute())

    def test_dry_run_names_folds_and_refusals_and_writes_nothing(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo, key = self._fake_checkout(
            monkeypatch, tmp_path, ["particles/config.py", "particles/api/cli/config.py"]
        )

        async def seed() -> None:
            from particles.db import session_scope

            async with session_scope() as session:
                await _file(session, "particles/config.py", ns=key)
                await _file(session, "config.py", ns=key)
                await _plain(session, "particles.config")
                await session.commit()

        _run(seed())
        result = runner.invoke(app, ["subjects", "fold-artifacts", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "1 fold(s): 1 module" in result.output
        assert "config.py: more than one file in the checkout" in result.output
        assert "Nothing written." in result.output

    def test_a_checkout_of_another_project_is_refused(
        self, runner: CliRunner, cli_db: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo, _key = self._fake_checkout(monkeypatch, tmp_path, [])
        result = runner.invoke(
            app, ["subjects", "fold-artifacts", "--repo", str(repo), "--namespace", "other"]
        )
        assert result.exit_code == 1
        assert "refusing to fold one project against another's files" in result.output

    def test_no_checkout_and_no_namespace_is_refused(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "fold-artifacts"])
        assert result.exit_code == 1
        assert "Provide --repo DIR" in result.output

    def test_undo_of_an_unknown_event_is_refused(self, runner: CliRunner, cli_db: Path) -> None:
        result = runner.invoke(app, ["subjects", "fold-artifacts", "--undo", "nope"])
        assert result.exit_code == 1
        assert "not an artifact fold" in result.output
