# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""One cell of the query-latency benchmark, run in its own process.

The orchestrator starts ``python -m particles.benchmark.query_latency.cell
<spec.json>`` once per cell, so a cold cell really is a fresh interpreter and
every cell's peak RSS is its own. The heavy imports are deferred into
:func:`run_cell` on purpose: importing the query path is a cost a cold cell
reports (``import_seconds``), so it must not happen at module load.

The result is written to ``spec.result_path`` through a temporary file and a
rename, so a result on disk is always a finished one.
"""

from __future__ import annotations

import asyncio
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from particles.benchmark.query_latency.schema import (
    CellResult,
    CellSpec,
    CosineReference,
    Environment,
    Mode,
    StoreManifest,
    Variant,
)


def peak_rss_bytes() -> int:
    """This process's peak resident set, in bytes (``ru_maxrss`` is KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _cmd(*args: str) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _cpu() -> str:
    if sys.platform == "darwin":
        return _cmd("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _memory_bytes() -> int | None:
    if sys.platform == "darwin":
        raw = _cmd("sysctl", "-n", "hw.memsize")
        return int(raw) if raw else None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def environment() -> Environment:
    """The reproduction tuple, read in the process that measures."""
    import sqlite3

    import numpy
    import pydantic
    import sqlalchemy

    from particles import __file__ as package_file

    try:
        from importlib.metadata import version

        package_version = version("linkedparticles")
    except Exception:  # noqa: BLE001 — an uninstalled tree still measures
        package_version = "unknown"
    return Environment(
        git_sha=_cmd("git", "-C", str(Path(package_file).parent), "rev-parse", "HEAD"),
        package_version=package_version,
        python=platform.python_version(),
        numpy=numpy.__version__,
        sqlite=sqlite3.sqlite_version,
        sqlalchemy=sqlalchemy.__version__,
        pydantic=pydantic.VERSION,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=_cpu(),
        cpu_count=os.cpu_count(),
        memory_bytes=_memory_bytes(),
        load_average_before=load_average(),
        bench_lock=os.environ.get("PARTICLES_BENCH_LOCK"),
        config=os.environ.get("PARTICLES_CONFIG", "(unset)"),
    )


async def run_cell(spec: CellSpec) -> CellResult:
    """Run one cell and return its result. Never writes the store."""
    started = datetime.now(UTC)
    wall0 = time.perf_counter()
    env = environment()

    t0 = time.perf_counter()
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from particles.benchmark.query_latency.generator import query_vectors, questions
    from particles.benchmark.query_latency.timing import (
        StageTimer,
        cosine_reference,
        instrument,
    )
    from particles.core.schema import QueryRequest
    from particles.db import _sqlite_set_pragmas
    from particles.operations.query.main import retrieve_ranked

    import_seconds = time.perf_counter() - t0

    result = CellResult(
        spec=spec,
        started_at=started.isoformat(),
        finished_at=started.isoformat(),
        wall_seconds=0.0,
        import_seconds=import_seconds,
        environment=env,
    )

    if spec.encoder:
        from particles.embeddings import get_embedding_model

        t0 = time.perf_counter()
        model = get_embedding_model()
        result.encoder_load_seconds = time.perf_counter() - t0
        if model is None:
            raise RuntimeError("no embedding model could be loaded; run with --no-encoder")

    store = Path(spec.store_path)
    manifest = StoreManifest.model_validate_json(store.with_suffix(".manifest.json").read_text())
    last = spec.question_offset + spec.questions + 1
    vectors = query_vectors(spec.seed, last)
    texts = questions(spec.seed, last)

    def subject_for(i: int) -> str | None:
        if spec.variant is Variant.UNFILTERED or not manifest.query_subjects:
            return None
        return manifest.query_subjects[i % len(manifest.query_subjects)][0]

    # The production engine's shape: default pool, the same WAL + busy_timeout
    # pragmas (particles.db.get_engine), one session per query as the query
    # verbs open one.
    engine = create_async_engine(f"sqlite+aiosqlite:///{store}")
    event.listen(engine.sync_engine, "connect", _sqlite_set_pragmas)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    timer = StageTimer()

    async def ask(i: int) -> int:
        timer.query_vector = None if spec.encoder else vectors[i]
        sid = subject_for(i)
        request = QueryRequest(question=texts[i], top_k=spec.top_k, subject_id=sid)
        async with factory() as session:
            timer.begin()
            ranked = await retrieve_ranked(session, request)
        return len(ranked)

    try:
        with instrument(timer):
            if spec.mode is Mode.WARM:
                # One untimed query to warm the process: the questions after it
                # are the warm ones.
                warm_index = spec.question_offset + spec.questions
                t0 = time.perf_counter()
                await ask(warm_index)
                result.warmup_seconds = time.perf_counter() - t0
            budget_from = time.perf_counter()
            first = spec.question_offset
            for n, i in enumerate(range(first, first + spec.questions)):
                # Only the final question keeps its candidate vectors for the
                # cosine reference: holding one query's vectors while the next
                # loads would inflate the peak RSS by a whole candidate set.
                spent = time.perf_counter() - budget_from
                mean = spent / n if n else 0.0
                budget_stop = bool(
                    spec.budget_seconds
                    and n + 1 >= spec.min_questions
                    and spent + mean > spec.budget_seconds
                )
                final = n == spec.questions - 1 or budget_stop
                timer.keep_candidates = final
                results = await ask(i)
                result.samples.append(
                    timer.finish(question_index=i, subject_id=subject_for(i), results=results)
                )
                if final:
                    if budget_stop and n < spec.questions - 1:
                        result.notes.append(
                            f"stopped after {n + 1} of {spec.questions} questions: the next "
                            f"would have passed the {spec.budget_seconds:.0f} s budget"
                        )
                    break
            # The cosine reference over the last candidates is taken after the
            # timed queries, never inside one.
            if timer.last_candidates and timer.last_query is not None:
                loop_s, stack_s, matmul_s = cosine_reference(
                    timer.last_candidates, timer.last_query
                )
                result.cosine_reference = CosineReference(
                    vectors=len(timer.last_candidates),
                    loop_seconds=loop_s,
                    stack_seconds=stack_s,
                    matmul_seconds=matmul_s,
                )
            timer.last_candidates = []
    finally:
        await engine.dispose()

    result.peak_rss_bytes = peak_rss_bytes()
    if result.environment is not None:
        result.environment.load_average_after = load_average()
    result.finished_at = datetime.now(UTC).isoformat()
    result.wall_seconds = time.perf_counter() - wall0
    return result


def write_result(result: CellResult, path: Path) -> None:
    """Write ``result`` to ``path`` atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(result.model_dump_json(indent=2))
    tmp.replace(path)


def main(argv: list[str]) -> int:
    """Entry point: ``python -m particles.benchmark.query_latency.cell <spec.json>``."""
    if len(argv) != 1:
        sys.stderr.write("usage: python -m particles.benchmark.query_latency.cell <spec.json>\n")
        return 2
    spec = CellSpec.model_validate_json(Path(argv[0]).read_text())
    result = asyncio.run(run_cell(spec))
    write_result(result, Path(spec.result_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
