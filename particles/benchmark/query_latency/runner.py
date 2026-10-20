# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The query-latency orchestrator: plan the cells, build the stores, run each cell.

Built to survive the machine it runs on: each cell is a child process that writes
its own result file when it finishes, every store is cached by ``(seed, size)``,
and a re-launch skips whatever is already on disk. A reboot mid-run therefore
loses only the cell in flight.

The orchestrator also watches each child: it samples the child's resident set,
kills it at ``cell_timeout``, and kills it when system swap grows past the swap
guard or free disk falls under the disk guard. A host that swaps rather than
running out of memory would otherwise grind for hours (and a swap file can fill
the disk the store lives on); the guard records that outcome as a result. After
a failed cell, the remaining cells of the same size and variant are recorded as
skipped rather than run into the same wall.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from particles.benchmark.query_latency.cell import environment
from particles.benchmark.query_latency.generator import build_store, store_file_name
from particles.benchmark.query_latency.schema import (
    DEFAULT_SEED,
    DEFAULT_SIZES,
    CellResult,
    CellSpec,
    CellStatus,
    Mode,
    Variant,
)

#: The child module the orchestrator starts per cell.
CELL_MODULE = "particles.benchmark.query_latency.cell"

#: The config every child loads: compiled defaults, nothing from the host.
PINNED_CONFIG = "# query-latency benchmark: compiled defaults only\n{}\n"

GIB = 1024**3


@dataclass(frozen=True)
class RunPlan:
    """What a run measures. The defaults are the published run's."""

    sizes: tuple[int, ...] = DEFAULT_SIZES
    seed: int = DEFAULT_SEED
    variants: tuple[Variant, ...] = (Variant.UNFILTERED, Variant.SUBJECT)
    cold_runs: int = 5
    warm_questions: int = 50
    warm_budget_seconds: float = 2400.0
    min_questions: int = 5
    encoder: bool = True
    top_k: int = 40
    cell_timeout_seconds: float = 7200.0
    swap_guard_bytes: int = 8 * GIB
    min_free_disk_bytes: int = 4 * GIB


def cell_id(size: int, mode: Mode, variant: Variant, rep: int | None = None) -> str:
    base = f"n{size}-{mode.value}-{variant.value}"
    return base if rep is None else f"{base}-r{rep}"


def _spec(  # noqa: PLR0913 — one cell's coordinates
    plan: RunPlan,
    work_dir: Path,
    size: int,
    variant: Variant,
    mode: Mode,
    rep: int | None,
    offset: int,
    questions: int,
) -> CellSpec:
    cid = cell_id(size, mode, variant, rep)
    return CellSpec(
        cell_id=cid,
        size=size,
        seed=plan.seed,
        mode=mode,
        variant=variant,
        store_path=str(work_dir / "stores" / store_file_name(plan.seed, size)),
        result_path=str(work_dir / "cells" / f"{cid}.json"),
        encoder=plan.encoder,
        question_offset=offset,
        questions=questions,
        budget_seconds=plan.warm_budget_seconds if mode is Mode.WARM else 0.0,
        min_questions=plan.min_questions,
        top_k=plan.top_k,
    )


def plan_cells(plan: RunPlan, work_dir: Path) -> list[CellSpec]:
    """Every cell of ``plan``, in run order. Pure.

    Per size and variant the cold repetitions run first (one question each, in
    fresh processes), then the warm cell. Sizes run smallest first, so the
    portable results are on disk before the largest store is attempted.
    """
    cells: list[CellSpec] = []
    for size in plan.sizes:
        for variant in plan.variants:
            cells.extend(
                _spec(plan, work_dir, size, variant, Mode.COLD, rep, rep, 1)
                for rep in range(plan.cold_runs)
            )
            cells.append(
                _spec(
                    plan,
                    work_dir,
                    size,
                    variant,
                    Mode.WARM,
                    None,
                    plan.cold_runs,
                    plan.warm_questions,
                )
            )
    return cells


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def _rss_bytes(pid: int) -> int | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps",
            "-o",
            "rss=",
            "-p",
            str(pid),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except OSError:
        return None
    text = out.decode().strip()
    return int(text) * 1024 if text.isdigit() else None


def parse_swapusage(text: str) -> int | None:
    """Bytes of swap in use from macOS ``sysctl -n vm.swapusage`` output. Pure."""
    m = re.search(r"used = ([\d.]+)([KMG])", text)
    if not m:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": GIB}[m.group(2)]
    return int(float(m.group(1)) * scale)


def parse_meminfo_swap(text: str) -> int | None:
    """Bytes of swap in use from Linux ``/proc/meminfo``. Pure."""
    info: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("SwapTotal", "SwapFree"):
            try:
                info[key] = int(rest.split()[0]) * 1024
            except (ValueError, IndexError):
                return None
    if len(info) != 2:
        return None
    return info["SwapTotal"] - info["SwapFree"]


def swap_used_bytes() -> int | None:
    """System swap in use, or ``None`` where it cannot be read."""
    if sys.platform == "darwin":
        import subprocess

        try:
            raw = subprocess.run(
                ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_swapusage(raw)
    try:
        return parse_meminfo_swap(Path("/proc/meminfo").read_text())
    except OSError:
        return None


def _child_env(config_path: Path) -> dict[str, str]:
    """The host environment minus anything that would reconfigure the child."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("PARTICLES_") and k != "PARTICLES_BENCH_LOCK") and k != "DATABASE_URL"
    }
    env["PARTICLES_CONFIG"] = str(config_path)
    return env


async def _run_child(
    spec: CellSpec,
    plan: RunPlan,
    work_dir: Path,
    config_path: Path,
) -> CellResult | None:
    """Run one cell's child; return a failure result, or ``None`` on success."""
    spec_path = work_dir / "specs" / f"{spec.cell_id}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec.model_dump_json(indent=2))
    log_path = work_dir / "logs" / f"{spec.cell_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    t0 = time.perf_counter()
    swap0 = swap_used_bytes()
    with log_path.open("w") as log:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            CELL_MODULE,
            str(spec_path),
            stdout=log,
            stderr=log,
            env=_child_env(config_path),
        )
        peak = 0
        status: CellStatus | None = None
        detail = ""
        while proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
                break
            except TimeoutError:
                pass
            rss = await _rss_bytes(proc.pid)
            peak = max(peak, rss or 0)
            elapsed = time.perf_counter() - t0
            swap = swap_used_bytes()
            free = shutil.disk_usage(work_dir).free
            if elapsed > plan.cell_timeout_seconds:
                status = CellStatus.TIMEOUT
                detail = f"killed at the {plan.cell_timeout_seconds:.0f} s cell timeout"
            elif swap is not None and swap0 is not None and swap - swap0 > plan.swap_guard_bytes:
                status = CellStatus.MEMORY_GUARD
                detail = (
                    f"killed when system swap grew by {(swap - swap0) / GIB:.1f} GiB "
                    f"(guard {plan.swap_guard_bytes / GIB:.0f} GiB)"
                )
            elif free < plan.min_free_disk_bytes:
                status = CellStatus.MEMORY_GUARD
                detail = f"killed when free disk fell to {free / GIB:.1f} GiB"
            if status is not None:
                proc.kill()
                await proc.wait()
                break
    if status is None and proc.returncode == 0 and Path(spec.result_path).exists():
        # Record the orchestrator's sampled peak beside the child's own.
        done = CellResult.model_validate_json(Path(spec.result_path).read_text())
        done.sampled_peak_rss_bytes = peak or None
        _write(done, Path(spec.result_path))
        return None
    if status is None:
        status = CellStatus.FAILED
        detail = f"child exited with {proc.returncode}"
    tail = log_path.read_text(errors="replace")[-4000:]
    return CellResult(
        spec=spec,
        status=status,
        started_at=started,
        finished_at=_now(),
        wall_seconds=time.perf_counter() - t0,
        sampled_peak_rss_bytes=peak or None,
        error=f"{detail}\n{tail}".strip(),
    )


def _write(result: CellResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(result.model_dump_json(indent=2))
    tmp.replace(path)


async def run_plan(
    plan: RunPlan,
    work_dir: Path,
    *,
    retry_failed: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[CellResult]:
    """Run every cell of ``plan`` not already finished under ``work_dir``.

    Returns every cell's result, read back from disk, including the ones a
    previous launch finished.
    """
    say = progress or (lambda _msg: None)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "config.yaml"
    config_path.write_text(PINNED_CONFIG)
    (work_dir / "run.json").write_text(environment().model_dump_json(indent=2))
    cells = plan_cells(plan, work_dir)
    failed: dict[tuple[int, Variant], str] = {}
    built: set[int] = set()
    for spec in cells:
        out = Path(spec.result_path)
        key = (spec.size, spec.variant)
        if out.exists():
            prior = CellResult.model_validate_json(out.read_text())
            if prior.status is CellStatus.OK or not retry_failed:
                if prior.status is not CellStatus.OK:
                    failed.setdefault(key, spec.cell_id)
                say(f"{spec.cell_id}: {prior.status.value} (on disk, skipped)")
                continue
        if key in failed:
            _write(
                CellResult(
                    spec=spec,
                    status=CellStatus.SKIPPED,
                    started_at=_now(),
                    finished_at=_now(),
                    wall_seconds=0.0,
                    notes=[f"not run: {failed[key]} failed at this size and variant"],
                ),
                out,
            )
            say(f"{spec.cell_id}: skipped after {failed[key]}")
            continue
        if spec.size not in built:
            say(f"building store n={spec.size} (seed {plan.seed}) …")
            manifest = await build_store(
                Path(spec.store_path), seed=plan.seed, size=spec.size, progress=say
            )
            say(
                f"store n={spec.size}: {manifest.db_bytes / GIB:.2f} GiB, "
                f"built in {manifest.build_seconds:.0f} s"
            )
            built.add(spec.size)
        say(f"{spec.cell_id}: running …")
        failure = await _run_child(spec, plan, work_dir, config_path)
        if failure is not None:
            _write(failure, out)
            failed[key] = spec.cell_id
            say(f"{spec.cell_id}: {failure.status.value}: {(failure.error or '').splitlines()[0]}")
        else:
            done = CellResult.model_validate_json(out.read_text())
            totals = sorted(s.total_seconds for s in done.samples)
            say(
                f"{spec.cell_id}: ok, {len(totals)} question(s), "
                f"median {totals[len(totals) // 2]:.3f} s"
            )
    return [
        CellResult.model_validate_json(Path(s.result_path).read_text())
        for s in cells
        if Path(s.result_path).exists()
    ]
