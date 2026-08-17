"""Tests for particles/api/cli/_memory_projection.py — the MEMORY.md cycle.

Covers the ADR's test checklist: the §2 FIXED-POINT (render → splice → harvest
→ extract → render is byte-identical with zero new particles — the harvest-
extract leg simulated at the seam level per tests/AGENTS.md: the strip of a
pristine file is empty, and the hook deposits nothing for empty input), the
sentinel strip seam, drift / dirty-region routing, atomic write + one-deep
backup, the SpliceError refusal, fold-and-archive (+ opt-out), and the trailer freshness check (match / mismatch / parse failure).

CLI tests follow tests/test_claude_code_hook.py: the ``cli_db`` file-based
SQLite fixture plus ``HOME`` pointed at ``tmp_path`` so the state directory
(``~/.particles/claude-code``) lands in the sandbox.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli._claude_code import (
    MEMORY_REGION,
    filter_memory_file_for_deposit,
)
from particles.api.cli._memory_projection import (
    ARCHIVE_POINTER_PREFIX,
    archive_pointer_line,
    fold_authored_lines,
)
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_EMB = (np.ones(4, dtype=np.float32) / np.linalg.norm(np.ones(4, dtype=np.float32))).tolist()

#: A permissive single-bullets-section manifest (floor 0 so low-trust test
#: particles clear it; the default manifest's numbers are pinned elsewhere).
_MANIFEST_YAML = """\
name: memory-index
sections:
  - title: "Memory index"
    render: bullets
    top_k: 60
max_lines: 120
max_bytes: 16384
"""


def _particle(content: str, conf: float = 0.9) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=conf, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="entry-1", snapshot_id="snap-1"
            )
        ],
    )


# ---------------------------------------------------------------------------
# fixtures + helpers (CLI tier)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited git-context env vars so the git tests use the tmp repo.

    Under a pre-commit run the suite inherits GIT_DIR / GIT_INDEX_FILE pointing at
    the outer repo, which would hijack the tmp-repo ``git -C`` calls. Harmless for
    the non-git tests in this module.
    """
    from particles.api.cli._projection_git import _GIT_CONTEXT_VARS

    for var in _GIT_CONTEXT_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def projection_state(hook_home: Path) -> Path:
    """Enable the projection: write the manifest into the state directory."""
    state = hook_home / ".particles" / "claude-code"
    state.mkdir(parents=True, exist_ok=True)
    (state / "memory.yaml").write_text(_MANIFEST_YAML, encoding="utf-8")
    return state


def _seed(*beliefs: tuple[str, float]) -> list[str]:
    async def _run() -> list[str]:
        from particles.db import session_scope
        from particles.store.particle_store import insert_particle

        ids: list[str] = []
        async with session_scope() as session:
            for content, conf in beliefs:
                p = _particle(content, conf)
                await insert_particle(session, p, _EMB)
                ids.append(p.id)
            await session.commit()
        return ids

    return asyncio.run(_run())


def _project_dir(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """A Claude Code project dir with one transcript; returns (transcript, memory_dir)."""
    project = tmp_path / ".claude" / "projects" / name
    project.mkdir(parents=True, exist_ok=True)
    transcript = project / "sess.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hello"}}) + "\n"
    )
    return transcript, project / "memory"


def _payload(transcript: Path, session_id: str = "sess", source: str | None = None) -> str:
    data: dict[str, Any] = {
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": "/p",
        "reason": "prompt_input_exit",
    }
    if source is not None:
        data["source"] = source
    return json.dumps(data)


def _session_end(runner: CliRunner, transcript: Path) -> Any:
    return runner.invoke(
        app, ["hook", "session-end", "--store", "default"], input=_payload(transcript)
    )


def _session_start(runner: CliRunner, transcript: Path) -> Any:
    return runner.invoke(
        app,
        ["hook", "session-start", "--store", "default"],
        input=_payload(transcript, source="startup"),
    )


def _last_log(home: Path) -> dict[str, Any]:
    path = home / ".particles" / "claude-code" / "hooks.jsonl"
    return json.loads(path.read_text().splitlines()[-1])


def _corpus_state() -> list[tuple[str, int]]:
    """(uri, snapshot_count) per corpus entry — the fixed-point invariant data."""

    async def _run() -> list[tuple[str, int]]:
        from particles.corpus.store import list_entries, list_snapshots_for_entry
        from particles.db import session_scope

        async with session_scope() as session:
            entries = await list_entries(session, limit=200, source_type=None)
            return sorted(
                [
                    (e.uri_r, len(await list_snapshots_for_entry(session, e.entry_id)))
                    for e in entries
                ]
            )

    return asyncio.run(_run())


def _particle_count() -> int:
    async def _run() -> int:
        from sqlalchemy import func, select

        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async with session_scope() as session:
            return int(await session.scalar(select(func.count()).select_from(ParticleRow)) or 0)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# The §2 fixed point, at the seam level (pure render + strip, one DB session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fixed_point_render_splice_harvest_render(db_session: AsyncSession) -> None:
    """render → splice → harvest(strip) → extract → render is a fixed point:
    the strip of a pristine file is empty (nothing is deposited, so extraction
    of the stripped input produces nothing) and the re-render is byte-identical.
    """
    from particles.operations.projection import (
        DerivedSection,
        DocManifest,
        project_splice_body,
        splice_region,
    )
    from particles.render.markdown import (
        PROJECTED_BEGIN_TMPL,
        PROJECTED_END_TMPL,
        strip_projected_regions_for_deposit,
    )
    from particles.store.particle_store import insert_particle

    for content, conf in (("DCO is enforced.", 0.9), ("Prefer general mechanisms.", 0.7)):
        await insert_particle(db_session, _particle(content, conf), _EMB)
    await db_session.flush()

    manifest = DocManifest(
        name=MEMORY_REGION,
        max_lines=120,
        sections=[DerivedSection(title="Memory index", render="bullets", top_k=60)],
    )

    # render → splice into a MEMORY.md that carries the sentinel pair.
    body = (
        await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=False)
    ).document
    host = (
        PROJECTED_BEGIN_TMPL.format(region=MEMORY_REGION, manifest="memory.yaml")
        + "\n"
        + PROJECTED_END_TMPL.format(region=MEMORY_REGION)
        + "\n"
    )
    memory_md = splice_region(host, MEMORY_REGION, body, manifest="memory.yaml")
    snapshot = {MEMORY_REGION: body}

    # harvest leg: the pristine region strips to nothing ⇒ the hook deposits
    # nothing (empty input is skipped) ⇒ extract has no input ⇒ zero new
    # particles. Simulated at the seam per tests/AGENTS.md.
    stripped = strip_projected_regions_for_deposit(memory_md, snapshot)
    assert stripped.strip() == ""

    # re-render: byte-identical (no timestamp, no volatile banner).
    again = (
        await project_splice_body(db_session, manifest, base_dir=Path("."), synthesize=False)
    ).document
    assert again == body
    assert splice_region(memory_md, MEMORY_REGION, again, manifest="memory.yaml") == memory_md


# ---------------------------------------------------------------------------
# Fold-and-archive — pure half
# ---------------------------------------------------------------------------


class TestFoldAuthoredLines:
    REGION = (
        "<!-- BEGIN PROJECTED: memory-index (manifest: m.yaml) -->\n"
        "- a belief `p-aa` \n"
        "<!-- END PROJECTED: memory-index -->"
    )

    def test_moves_outside_lines_and_leaves_pointer(self) -> None:
        text = f"{self.REGION}\n\n- authored note one\n\n- authored note two\n"
        kept, folded = fold_authored_lines(text, "POINTER")
        assert folded == ["- authored note one", "- authored note two"]
        assert kept == f"{self.REGION}\n\nPOINTER\n"

    def test_nothing_to_fold_returns_text_unchanged(self) -> None:
        text = f"{self.REGION}\n"
        assert fold_authored_lines(text, "POINTER") == (text, [])

    def test_existing_pointer_line_is_not_refolded(self) -> None:
        pointer = archive_pointer_line()
        text = f"{self.REGION}\n\n{pointer}\n"
        kept, folded = fold_authored_lines(text, pointer)
        assert folded == []
        assert kept == text

    def test_no_region_means_no_fold(self) -> None:
        text = "# Memory\n- only authored content\n"
        assert fold_authored_lines(text, "POINTER") == (text, [])


# ---------------------------------------------------------------------------
# The harvest strip seam (stage-1 seam #2), with state-dir snapshot defaults
# ---------------------------------------------------------------------------


class TestFilterMemoryFileForDeposit:
    def test_reads_snapshots_from_state_dir_by_default(self, projection_state: Path) -> None:
        body = "- a belief `p-aa`\n\n<!-- sources: p-aa -->"
        (projection_state / "memory-index.snapshot.md").write_text(body + "\n")
        text = (
            "<!-- BEGIN PROJECTED: memory-index (manifest: m.yaml) -->\n"
            f"{body}\n"
            "<!-- END PROJECTED: memory-index -->\n\nauthored\n"
        )
        out = filter_memory_file_for_deposit(text)
        assert "p-aa" not in out
        assert "authored" in out

    def test_archive_pointer_line_is_dropped(self, projection_state: Path) -> None:
        text = f"authored line\n{archive_pointer_line()}\n"
        out = filter_memory_file_for_deposit(text)
        assert ARCHIVE_POINTER_PREFIX not in out
        assert "authored line" in out

    def test_dirty_region_body_is_kept(self, projection_state: Path) -> None:
        (projection_state / "memory-index.snapshot.md").write_text("- a belief `p-aa`\n")
        text = (
            "<!-- BEGIN PROJECTED: memory-index (manifest: m.yaml) -->\n"
            "- a belief `p-aa`\n- MY HAND-WRITTEN LINE\n"
            "<!-- END PROJECTED: memory-index -->\n"
        )
        out = filter_memory_file_for_deposit(text)
        assert "MY HAND-WRITTEN LINE" in out
        assert "BEGIN PROJECTED" not in out


# ---------------------------------------------------------------------------
# The render-splice cycle, end-to-end through `particles hook session-end`
# ---------------------------------------------------------------------------


class TestProjectionCycle:
    def test_renders_region_folds_and_backs_up(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        from particles.render.markdown import insert_projected_region_at_top

        _seed(("DCO is enforced.", 0.9), ("Prefer general mechanisms.", 0.7))
        transcript, memory_dir = _project_dir(tmp_path, "-p1")
        memory_dir.mkdir(parents=True)
        memory_md = memory_dir / "MEMORY.md"
        authored = "# Memory\n- an authored note about tabs\n"
        memory_md.write_text(insert_projected_region_at_top(authored, MEMORY_REGION, "memory.yaml"))
        pre_cycle = memory_md.read_text()

        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        record = _last_log(tmp_path)
        assert record["outcome"] == "ok"
        assert record["projection"]["outcome"] == "rendered"

        text = memory_md.read_text()
        # The region now carries the ranked bullets + trailer…
        assert "- DCO is enforced." in text
        assert "<!-- sources:" in text
        # …the authored lines were folded to the archive with a pointer left…
        assert "- an authored note about tabs" not in text
        assert ARCHIVE_POINTER_PREFIX in text
        archive = projection_state / "MEMORY.archive.md"
        assert "- an authored note about tabs" in archive.read_text()
        assert record["projection"]["folded"] == 2
        # …the pre-splice file was backed up one-deep, and the snapshot written.
        assert (projection_state / "MEMORY.md.pre-render").read_text() == pre_cycle
        snapshot = (projection_state / "memory-index.snapshot.md").read_text()
        assert "- DCO is enforced." in snapshot

    def test_cycle_is_a_fixed_point_no_new_corpus_entries(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        from particles.render.markdown import insert_projected_region_at_top

        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-p2")
        memory_dir.mkdir(parents=True)
        memory_md = memory_dir / "MEMORY.md"
        memory_md.write_text(
            insert_projected_region_at_top("- one authored note\n", MEMORY_REGION, "memory.yaml")
        )

        _session_end(runner, transcript)  # run 1: harvest + fold + first render
        _session_end(runner, transcript)  # run 2: archive deposited; region pristine
        state_after_2 = _corpus_state()
        text_after_2 = memory_md.read_text()
        particles_after_2 = _particle_count()

        result = _session_end(runner, transcript)  # run 3: the fixed point
        assert result.exit_code == 0
        assert memory_md.read_text() == text_after_2  # byte-identical
        assert _corpus_state() == state_after_2  # zero new deposits/snapshots
        assert _particle_count() == particles_after_2  # zero new particles
        # Belt 1: MEMORY.md's corpus entry has exactly the run-1 authored-note
        # snapshot — the pristine rendered region never re-entered the corpus.
        memory_snapshots = [n for uri, n in state_after_2 if uri.endswith("MEMORY.md")]
        assert memory_snapshots == [1]
        record = _last_log(tmp_path)
        assert record["projection"]["outcome"] == "rendered"
        assert record["projection"]["dirty_region"] is False

    def test_dirty_region_is_deposited_then_rerendered(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        from particles.render.markdown import insert_projected_region_at_top

        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-p3")
        memory_dir.mkdir(parents=True)
        memory_md = memory_dir / "MEMORY.md"
        memory_md.write_text(insert_projected_region_at_top("", MEMORY_REGION, "memory.yaml"))
        _session_end(runner, transcript)  # establish the rendered region + snapshot

        # A foreign write INSIDE the machine-owned region (§6).
        memory_md.write_text(
            memory_md.read_text().replace(
                "<!-- END PROJECTED", "- an edit inside the region\n<!-- END PROJECTED"
            )
        )
        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        record = _last_log(tmp_path)
        assert record["projection"]["outcome"] == "rendered"
        assert record["projection"]["dirty_region"] is True

        # The edit was routed through harvest (deposited as authored input)…
        blobs = _corpus_state()
        assert any(uri.endswith("MEMORY.md") for uri, _ in blobs)
        # …and the region was re-rendered from the store (edit no longer inline).
        assert "- an edit inside the region" not in memory_md.read_text()

    def test_damaged_sentinels_refuse_loudly_and_change_nothing(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-p4")
        memory_dir.mkdir(parents=True)
        memory_md = memory_dir / "MEMORY.md"
        damaged = (
            "<!-- BEGIN PROJECTED: memory-index (manifest: m.yaml) -->\n"
            "- content, but the END sentinel was deleted\n"
        )
        memory_md.write_text(damaged)

        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        record = _last_log(tmp_path)
        assert record["projection"]["skipped"] == "splice-error"
        assert memory_md.read_text() == damaged  # never "helpfully" regenerated

    def test_missing_memory_md_is_created_with_the_region(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        from particles.render.markdown import find_projected_regions

        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-p5")
        assert not memory_dir.exists()

        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        assert _last_log(tmp_path)["projection"]["outcome"] == "created"
        text = (memory_dir / "MEMORY.md").read_text()
        regions = find_projected_regions(text)
        assert len(regions) == 1 and regions[0].region == MEMORY_REGION
        assert "- DCO is enforced." in regions[0].body

    def test_git_history_commits_the_render_when_enabled(
        self,
        runner: CliRunner,
        cli_db: Path,
        projection_state: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """with git enabled and the memory dir in a repo, the render commits."""
        import subprocess

        from particles.config import reset_config

        config = tmp_path / "config.yaml"
        config.write_text(
            "agent_memory:\n  projection:\n    git:\n      enabled: true\n"
            f"storage:\n  database_url: sqlite+aiosqlite:///{cli_db}\n"
        )
        monkeypatch.setenv("PARTICLES_CONFIG", str(config))
        reset_config()

        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-pgit")
        memory_dir.mkdir(parents=True)
        # The operator keeps their memory directory under git (Letta-MemFS style).
        for args in (
            ["init", "-q"],
            ["config", "user.name", "Op"],
            ["config", "user.email", "op@example.test"],
        ):
            subprocess.run(["git", "-C", str(memory_dir), *args], check=True)

        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        projection = _last_log(tmp_path)["projection"]
        assert projection["outcome"] in ("created", "rendered")
        assert projection["git"].startswith("committed ")
        assert len(projection["run_id"]) == 8

        message = subprocess.run(
            ["git", "-C", str(memory_dir), "log", "-1", "--format=%B"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert message.splitlines()[0].startswith("memory-projection: ")
        assert f"run-id: {projection['run_id']}" in message
        assert "source of truth" in message  # the projected-view footer

    def test_git_history_degrades_silently_when_not_a_repo(
        self,
        runner: CliRunner,
        cli_db: Path,
        projection_state: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """git enabled but the memory dir is not a repo ⇒ inert, no failure."""
        from particles.config import reset_config

        config = tmp_path / "config.yaml"
        config.write_text(
            "agent_memory:\n  projection:\n    git:\n      enabled: true\n"
            f"storage:\n  database_url: sqlite+aiosqlite:///{cli_db}\n"
        )
        monkeypatch.setenv("PARTICLES_CONFIG", str(config))
        reset_config()

        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-pnorepo")
        memory_dir.mkdir(parents=True)

        result = _session_end(runner, transcript)
        assert result.exit_code == 0  # the projection itself still succeeds
        projection = _last_log(tmp_path)["projection"]
        assert projection["outcome"] in ("created", "rendered")
        assert projection["git"] == "skipped: not-a-git-repo"

    def test_fold_opt_out_keeps_authored_lines_in_place(
        self,
        runner: CliRunner,
        cli_db: Path,
        projection_state: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from particles.config import reset_config
        from particles.render.markdown import insert_projected_region_at_top

        config = tmp_path / "config.yaml"
        config.write_text(
            "agent_memory:\n  projection:\n    fold_authored_lines: false\n"
            f"storage:\n  database_url: sqlite+aiosqlite:///{cli_db}\n"
        )
        monkeypatch.setenv("PARTICLES_CONFIG", str(config))
        reset_config()

        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-p6")
        memory_dir.mkdir(parents=True)
        memory_md = memory_dir / "MEMORY.md"
        memory_md.write_text(
            insert_projected_region_at_top(
                "- an authored note to keep\n", MEMORY_REGION, "memory.yaml"
            )
        )

        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        assert _last_log(tmp_path)["projection"]["outcome"] == "rendered"
        text = memory_md.read_text()
        assert "- an authored note to keep" in text  # opt-out honored
        assert ARCHIVE_POINTER_PREFIX not in text
        assert not (projection_state / "MEMORY.archive.md").exists()

    def test_projection_disabled_leaves_memory_md_untouched(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        # No manifest in the state dir ⇒ projection not enabled ⇒ pure harvest.
        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, "-p7")
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("- authored only\n")

        result = _session_end(runner, transcript)
        assert result.exit_code == 0
        assert (memory_dir / "MEMORY.md").read_text() == "- authored only\n"
        assert "projection" not in _last_log(tmp_path)


# ---------------------------------------------------------------------------
# SessionStart trailer freshness (stage-1 seam #1)
# ---------------------------------------------------------------------------


class TestTrailerFreshness:
    def _rendered_memory(self, runner: CliRunner, tmp_path: Path, name: str) -> tuple[Path, Path]:
        """Seed one belief and run a full cycle so MEMORY.md holds the current view."""
        _seed(("DCO is enforced.", 0.9))
        transcript, memory_dir = _project_dir(tmp_path, name)
        _session_end(runner, transcript)
        return transcript, memory_dir / "MEMORY.md"

    def test_match_injects_nothing(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        transcript, _ = self._rendered_memory(runner, tmp_path, "-f1")
        result = _session_start(runner, transcript)
        assert result.exit_code == 0
        assert result.stdout == ""
        assert _last_log(tmp_path)["skipped"] == "projection-current"

    def test_mismatch_injects_only_the_difference(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        transcript, _ = self._rendered_memory(runner, tmp_path, "-f2")
        _seed(("A brand new belief learned elsewhere.", 0.8))  # the store moved

        result = _session_start(runner, transcript)
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        context = payload["hookSpecificOutput"]["additionalContext"]
        assert "A brand new belief learned elsewhere." in context
        assert "DCO is enforced." not in context  # already in the loaded region
        assert _last_log(tmp_path)["digest_mode"] == "projection-diff"

    def test_parse_failure_falls_back_to_full_digest(
        self, runner: CliRunner, cli_db: Path, projection_state: Path, tmp_path: Path
    ) -> None:
        transcript, memory_md = self._rendered_memory(runner, tmp_path, "-f3")
        # Hand-mangle the region so it carries no sources trailer.
        text = memory_md.read_text()
        mangled = "\n".join(
            line for line in text.splitlines() if not line.startswith("<!-- sources:")
        )
        memory_md.write_text(mangled + "\n")

        result = _session_start(runner, transcript)
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        context = payload["hookSpecificOutput"]["additionalContext"]
        assert "Memory digest" in context  # the full digest
        assert "skipped" not in _last_log(tmp_path)

    def test_projection_disabled_pushes_full_digest(
        self, runner: CliRunner, cli_db: Path, hook_home: Path, tmp_path: Path
    ) -> None:
        _seed(("DCO is enforced.", 0.9))
        transcript, _ = _project_dir(tmp_path, "-f4")
        result = _session_start(runner, transcript)
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "Memory digest" in payload["hookSpecificOutput"]["additionalContext"]
