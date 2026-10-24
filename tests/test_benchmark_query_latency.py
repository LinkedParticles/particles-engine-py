# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The query-latency benchmark: generator, stage timer, orchestration, report.

No assertion here is about speed. These pin what the published figures rest on:
the store is the shape it claims and the same for one seed, the stages sum to
the total and the counts match the store, the seams are restored, and a
re-launched run skips finished cells and records failures as results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from typer.testing import CliRunner

from particles.api.cli import app
from particles.api.cli.benchmark import _parse_sizes
from particles.benchmark.query_latency import runner as runner_mod
from particles.benchmark.query_latency.cell import run_cell, write_result
from particles.benchmark.query_latency.generator import (
    build_store,
    plan_counts,
    query_vectors,
    questions,
    store_file_name,
)
from particles.benchmark.query_latency.report import (
    _slope,
    export_by_size,
    render_markdown,
    size_label,
    summarise,
)
from particles.benchmark.query_latency.runner import RunPlan, plan_cells, run_plan
from particles.benchmark.query_latency.schema import (
    DIM,
    STAGES,
    CellResult,
    CellSpec,
    CellStatus,
    Mode,
    QuerySample,
    Variant,
)
from particles.benchmark.query_latency.timing import StageTimer, instrument
from particles.embeddings import get_embedding_model_id

SIZE = 400


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import asyncio

    path = tmp_path_factory.mktemp("qlat") / store_file_name(3, SIZE)
    asyncio.run(build_store(path, seed=3, size=SIZE))
    return path


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.with_suffix(".manifest.json").read_text())  # type: ignore[no-any-return]


def _spec(
    store: Path, tmp: Path, *, variant: Variant, mode: Mode = Mode.WARM, n: int = 3
) -> CellSpec:
    return CellSpec(
        cell_id="t",
        size=SIZE,
        seed=3,
        mode=mode,
        variant=variant,
        store_path=str(store),
        result_path=str(tmp / "t.json"),
        encoder=False,
        questions=n,
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def test_plan_counts_follow_the_measured_ratios() -> None:
    counts = plan_counts(56_000)
    assert counts["active"] == 56_000
    assert counts["inactive"] == round(56_000 * 0.204)
    assert counts["corpus_entries"] == round(56_000 / 82)
    assert counts["snapshots"] == round(56_000 / 32)
    assert counts["subjects"] == round(56_000 * 0.2584)
    assert plan_counts(10)["snapshots"] >= plan_counts(10)["corpus_entries"] >= 1


def test_store_holds_what_the_manifest_says(store: Path) -> None:
    manifest = _manifest(store)
    engine = create_engine(f"sqlite:///{store}")
    with engine.connect() as c:
        active = c.execute(
            text("SELECT count(*) FROM particles WHERE status = 'ACTIVE'")
        ).scalar_one()
        inactive = c.execute(
            text("SELECT count(*) FROM particles WHERE status != 'ACTIVE'")
        ).scalar_one()
        models = {
            r[0] for r in c.execute(text("SELECT DISTINCT embedding_model_id FROM particles"))
        }
        emb_bytes = c.execute(
            text("SELECT sum(length(embedding_json)) FROM particles WHERE status = 'ACTIVE'")
        ).scalar_one()
        links = c.execute(text("SELECT count(*) FROM particle_subjects")).scalar_one()
        sample = c.execute(text("SELECT embedding_json FROM particles LIMIT 1")).scalar_one()
        fanout = {
            sid: n
            for sid, n in c.execute(
                text(
                    "SELECT ps.subject_id, count(*) FROM particle_subjects ps JOIN particles p "
                    "ON p.id = ps.particle_id WHERE p.status = 'ACTIVE' GROUP BY ps.subject_id"
                )
            )
        }
    engine.dispose()
    assert active == SIZE == manifest["active"]
    assert inactive == manifest["inactive"]
    # The current model id, or the loader would skip every row.
    assert models == {get_embedding_model_id()} == {manifest["embedding_model_id"]}
    assert emb_bytes == manifest["embedding_json_bytes"]
    assert links == manifest["subject_links"]
    vec = np.array(json.loads(sample), dtype=np.float32)
    assert vec.shape == (DIM,)
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-5)
    subjects = manifest["query_subjects"]
    assert isinstance(subjects, list) and subjects
    for sid, n in subjects:
        assert fanout[sid] == n


async def test_one_seed_is_one_store(tmp_path: Path, store: Path) -> None:
    again = await build_store(tmp_path / "a.db", seed=3, size=SIZE)
    other = await build_store(tmp_path / "b.db", seed=4, size=SIZE)
    assert again.fingerprint == _manifest(store)["fingerprint"]
    assert other.fingerprint != again.fingerprint


async def test_a_built_store_is_reused(tmp_path: Path) -> None:
    path = tmp_path / store_file_name(5, 50)
    first = await build_store(path, seed=5, size=50)
    mtime = path.stat().st_mtime_ns
    second = await build_store(path, seed=5, size=50)
    assert second == first
    assert path.stat().st_mtime_ns == mtime
    assert not path.with_suffix(".building").exists()


def test_questions_and_vectors_are_seeded() -> None:
    assert questions(1, 4) == questions(1, 4)
    assert questions(1, 4) != questions(2, 4)
    v = query_vectors(1, 3)
    assert v.shape == (3, DIM)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v, query_vectors(1, 3))


# ---------------------------------------------------------------------------
# Stage timer and one cell
# ---------------------------------------------------------------------------


async def test_unfiltered_cell_counts_every_active_row(store: Path, tmp_path: Path) -> None:
    result = await run_cell(_spec(store, tmp_path, variant=Variant.UNFILTERED))
    assert result.status is CellStatus.OK
    assert len(result.samples) == 3
    assert result.warmup_seconds is not None
    for s in result.samples:
        assert set(s.stages) == set(STAGES)
        assert all(v >= 0 for v in s.stages.values())
        assert math.isclose(sum(s.stages.values()), s.total_seconds, rel_tol=1e-6)
        assert s.rows_read == SIZE
        # The warm-up query built the candidate cache, so a warm query
        # hydrates and decodes nothing.
        assert s.models_hydrated == s.embedding_json_bytes == 0
        assert 0 < s.candidates_scored <= SIZE
        assert s.snapshot_ids_looked_up > 0
        assert 0 < s.results <= 40
    assert result.cosine_reference is not None
    assert result.cosine_reference.vectors == SIZE
    assert result.peak_rss_bytes and result.peak_rss_bytes > 0


async def test_a_cold_unfiltered_query_decodes_every_active_row(
    store: Path, tmp_path: Path
) -> None:
    result = await run_cell(_spec(store, tmp_path, variant=Variant.UNFILTERED, mode=Mode.COLD, n=1))
    (sample,) = result.samples
    assert sample.rows_read == sample.models_hydrated == SIZE
    assert sample.embedding_json_bytes == _manifest(store)["embedding_json_bytes"]
    # The cache builds from a streamed result; its partition fetches are the
    # SQL fetch stage, not decode.
    assert sample.stages["sql_fetch"] > 0


async def test_subject_cell_reads_only_the_subjects_rows(store: Path, tmp_path: Path) -> None:
    result = await run_cell(_spec(store, tmp_path, variant=Variant.SUBJECT, mode=Mode.COLD, n=1))
    fanout = dict(map(tuple, _manifest(store)["query_subjects"]))  # type: ignore[call-overload]
    (sample,) = result.samples
    assert sample.subject_id is not None
    assert sample.rows_read == fanout[sample.subject_id]
    assert result.warmup_seconds is None


async def test_instrument_restores_every_seam(store: Path) -> None:
    import particles.corpus.store as corpus_store
    import particles.operations.query.main as main
    from particles.store.particle_store import ParticleRow

    before = (
        main._embed,
        main.load_active_candidates,
        main._gather_scored,
        main.load_source_rows,
        main.stale_support_discounts,
        main._collapse_co_evidential_top_k,
        ParticleRow.to_model,
        corpus_store.get_snapshot_source_meta,
        corpus_store.get_source_types_for_entries,
    )
    with pytest.raises(RuntimeError), instrument(StageTimer()):
        assert main._embed is not before[0]
        raise RuntimeError("boom")
    after = (
        main._embed,
        main.load_active_candidates,
        main._gather_scored,
        main.load_source_rows,
        main.stale_support_discounts,
        main._collapse_co_evidential_top_k,
        ParticleRow.to_model,
        corpus_store.get_snapshot_source_meta,
        corpus_store.get_source_types_for_entries,
    )
    assert after == before


def test_write_result_is_atomic(tmp_path: Path) -> None:
    out = tmp_path / "cells" / "x.json"
    write_result(_result(100, Mode.WARM, Variant.UNFILTERED, [0.1]), out)
    assert out.exists()
    assert not out.with_suffix(".json.tmp").exists()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_plan_cells_order_and_ids(tmp_path: Path) -> None:
    plan = RunPlan(sizes=(10, 20), cold_runs=2, warm_questions=7)
    cells = plan_cells(plan, tmp_path)
    assert [c.cell_id for c in cells[:6]] == [
        "n10-cold-unfiltered-r0",
        "n10-cold-unfiltered-r1",
        "n10-warm-unfiltered",
        "n10-cold-subject-r0",
        "n10-cold-subject-r1",
        "n10-warm-subject",
    ]
    assert len(cells) == 12
    cold = [c for c in cells if c.mode is Mode.COLD and c.size == 10]
    # Every cold repetition asks a different question, and the warm cell none of theirs.
    assert len({c.question_offset for c in cold if c.variant is Variant.UNFILTERED}) == 2
    warm = cells[2]
    assert warm.question_offset == 2 and warm.questions == 7
    assert warm.budget_seconds == plan.warm_budget_seconds
    assert all(c.budget_seconds == 0 for c in cold)


async def test_a_relaunch_skips_finished_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def in_process(spec: CellSpec, *_: object) -> CellResult | None:
        calls.append(spec.cell_id)
        write_result(await run_cell(spec), Path(spec.result_path))
        return None

    monkeypatch.setattr(runner_mod, "_run_child", in_process)
    plan = RunPlan(
        sizes=(60,), variants=(Variant.UNFILTERED,), cold_runs=1, warm_questions=2, encoder=False
    )
    first = await run_plan(plan, tmp_path)
    assert [r.status for r in first] == [CellStatus.OK, CellStatus.OK]
    assert len(calls) == 2
    second = await run_plan(plan, tmp_path)
    assert len(calls) == 2
    assert [r.spec.cell_id for r in second] == [r.spec.cell_id for r in first]


async def test_a_failed_cell_skips_its_size_and_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fail_unfiltered(spec: CellSpec, *_: object) -> CellResult | None:
        calls.append(spec.cell_id)
        if spec.variant is Variant.UNFILTERED:
            return CellResult(
                spec=spec,
                status=CellStatus.MEMORY_GUARD,
                started_at="t",
                finished_at="t",
                wall_seconds=1.0,
                error="killed when system swap grew",
            )
        write_result(await run_cell(spec), Path(spec.result_path))
        return None

    monkeypatch.setattr(runner_mod, "_run_child", fail_unfiltered)
    plan = RunPlan(sizes=(60,), cold_runs=2, warm_questions=1, encoder=False)
    results = {r.spec.cell_id: r.status for r in await run_plan(plan, tmp_path)}
    assert results["n60-cold-unfiltered-r0"] is CellStatus.MEMORY_GUARD
    assert results["n60-cold-unfiltered-r1"] is CellStatus.SKIPPED
    assert results["n60-warm-unfiltered"] is CellStatus.SKIPPED
    assert results["n60-warm-subject"] is CellStatus.OK
    assert "n60-cold-unfiltered-r1" not in calls


async def test_a_crashing_child_is_recorded_as_failed(tmp_path: Path) -> None:
    spec = CellSpec(
        cell_id="broken",
        size=10,
        seed=1,
        mode=Mode.COLD,
        variant=Variant.UNFILTERED,
        store_path=str(tmp_path / "missing.db"),
        result_path=str(tmp_path / "cells" / "broken.json"),
        encoder=False,
    )
    config = tmp_path / "config.yaml"
    config.write_text(runner_mod.PINNED_CONFIG)
    # The crash is what is under test, so both host guards are off: each reads
    # the runner's own disk and swap, and either fires MEMORY_GUARD if a poll
    # lands before the child exits. With the 4 GiB disk default a CI runner
    # with 3.7 GiB free did exactly that (seen twice on one runner, 2026-10-21).
    plan = RunPlan(min_free_disk_bytes=0, swap_guard_bytes=2**62)
    failure = await runner_mod._run_child(spec, plan, tmp_path, config)
    assert failure is not None
    assert failure.status is CellStatus.FAILED
    assert "child exited with" in (failure.error or "")
    assert not Path(spec.result_path).exists()


def test_swap_parsers() -> None:
    mac = "total = 2048.00M  used = 1116.38M  free = 931.62M  (encrypted)"
    assert runner_mod.parse_swapusage(mac) == int(1116.38 * 1024**2)
    assert runner_mod.parse_swapusage("") is None
    linux = (
        "MemTotal:       16000000 kB\nSwapTotal:       4194300 kB\nSwapFree:        4194200 kB\n"
    )
    assert runner_mod.parse_meminfo_swap(linux) == 100 * 1024
    assert runner_mod.parse_meminfo_swap("MemTotal: 1 kB\n") is None
    # The host's own reading must not raise, whatever the platform.
    used = runner_mod.swap_used_bytes()
    assert used is None or used >= 0


def test_child_env_pins_the_config_and_drops_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PARTICLES_QUERY_TOP_K", "3")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///x.db")
    monkeypatch.setenv("PARTICLES_BENCH_LOCK", "query-latency")
    env = runner_mod._child_env(tmp_path / "c.yaml")
    assert env["PARTICLES_CONFIG"] == str(tmp_path / "c.yaml")
    assert env["PARTICLES_BENCH_LOCK"] == "query-latency"
    assert "PARTICLES_QUERY_TOP_K" not in env
    assert "DATABASE_URL" not in env


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _result(
    size: int, mode: Mode, variant: Variant, totals: list[float], status: CellStatus = CellStatus.OK
) -> CellResult:
    spec = CellSpec(
        cell_id=f"n{size}-{mode.value}-{variant.value}",
        size=size,
        seed=1,
        mode=mode,
        variant=variant,
        store_path=f"/home/someone/bench/stores/{size}.db",
        result_path=f"/home/someone/bench/cells/{size}.json",
        encoder=False,
    )
    samples = [
        QuerySample(
            question_index=i,
            total_seconds=t,
            stages={**dict.fromkeys(STAGES, 0.0), "decode": t * 0.75, "hydrate": t * 0.25},
            rows_read=size,
            embedding_json_bytes=size * 8_400,
            models_hydrated=size,
            candidates_scored=size,
            snapshot_ids_looked_up=size // 32,
            entry_ids_looked_up=size // 82,
            results=40,
        )
        for i, t in enumerate(totals)
    ]
    return CellResult(
        spec=spec,
        status=status,
        started_at="t",
        finished_at="t",
        wall_seconds=1.0,
        samples=samples if status is CellStatus.OK else [],
    )


def test_slope_is_one_for_linear_growth() -> None:
    assert _slope([(1_000, 0.1), (10_000, 1.0), (100_000, 10.0)]) == pytest.approx(1.0)
    assert _slope([(1_000, 0.1)]) is None


def test_summary_pools_percentiles_shares_and_scaling() -> None:
    cells = [
        _result(1_000, Mode.WARM, Variant.UNFILTERED, [0.1, 0.1, 0.2]),
        _result(10_000, Mode.WARM, Variant.UNFILTERED, [1.0, 1.0, 2.0]),
        _result(100_000, Mode.WARM, Variant.UNFILTERED, [], status=CellStatus.MEMORY_GUARD),
    ]
    summary = summarise(cells)
    rows = {r.size: r for r in summary.rows}
    assert rows[1_000].n == 3 and rows[1_000].p50 == pytest.approx(0.1)
    assert rows[1_000].stage_share["decode"] == pytest.approx(0.75)
    assert rows[100_000].status is CellStatus.MEMORY_GUARD and rows[100_000].n == 0
    (scaling,) = summary.scaling
    assert scaling.sizes == [1_000, 10_000]
    assert scaling.slope == pytest.approx(1.0)
    assert scaling.microseconds_per_row_read[1_000] == pytest.approx(100.0)
    md = render_markdown(summary, raw_link=lambda n: f"raw-{size_label(n)}.json")
    assert "[json](raw-1k.json)" in md
    assert "memory_guard" in md


def test_export_cuts_host_paths(tmp_path: Path) -> None:
    (path,) = export_by_size([_result(2_000, Mode.COLD, Variant.SUBJECT, [0.5])], tmp_path, "q")
    assert path.name == "q-2k.json"
    body = path.read_text()
    assert "/home/someone" not in body
    assert json.loads(body)[0]["spec"]["store_path"] == "2000.db"


def test_size_labels() -> None:
    assert [size_label(n) for n in (400, 10_000, 56_000, 1_000_000, 2_500_000)] == [
        "400",
        "10k",
        "56k",
        "1M",
        "2.5M",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_sizes() -> None:
    assert _parse_sizes("10k, 56_000,1M,1.5k") == (10_000, 56_000, 1_000_000, 1_500)


@pytest.mark.parametrize("raw", ["ten", "5", "10k,,"])
def test_parse_sizes_refuses(raw: str) -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_sizes(raw)


def test_report_without_cells_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["benchmark", "query-latency", "report", "--work-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "no cell results" in result.output


def test_report_refuses_an_unknown_format(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["benchmark", "query-latency", "report", "--work-dir", str(tmp_path), "--format", "csv"],
    )
    assert result.exit_code == 2


def test_run_refuses_an_unknown_variant(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "query-latency",
            "run",
            "--work-dir",
            str(tmp_path),
            "--variant",
            "tag",
        ],
    )
    assert result.exit_code == 2


async def test_a_cached_store_is_brought_to_the_current_schema(tmp_path: Path) -> None:
    path = tmp_path / store_file_name(5, 50)
    await build_store(path, seed=5, size=50)

    def objects() -> dict[str, str | None]:
        engine = create_engine(f"sqlite:///{path}")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT name, sql FROM sqlite_master")).all()
        engine.dispose()
        return {name: sql for name, sql in rows}

    current = objects()
    # A store built before the change journal and the schema-version index.
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        for name in (
            "trg_particles_journal_insert",
            "trg_particles_journal_update",
            "trg_particles_journal_delete",
            "trg_particle_changes_prune",
        ):
            conn.execute(text(f"DROP TRIGGER {name}"))
        conn.execute(text("DROP TABLE particle_changes"))
        conn.execute(text("DROP INDEX ix_particles_status_schema_version"))
        ids_before = conn.execute(text("SELECT id FROM particles ORDER BY rowid")).scalars().all()
    engine.dispose()

    manifest = await build_store(path, seed=5, size=50)

    assert objects() == current
    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        assert (
            conn.execute(text("SELECT id FROM particles ORDER BY rowid")).scalars().all()
            == ids_before
        )
    engine.dispose()
    assert manifest.size == 50
