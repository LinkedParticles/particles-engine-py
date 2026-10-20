# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarise and render a query-latency run from its cell files. Pure over results.

Percentiles are over the samples a row holds and interpolate linearly; a row
says its ``n``, and a p95 over five samples is a description of five samples,
not a tail estimate. A stage's share is its time summed over the row's samples
divided by their summed totals. The scaling slope is the least-squares fit of
``log p50`` against ``log N`` over the sizes that finished: 1.0 is linear.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from particles.benchmark.query_latency.schema import (
    STAGES,
    CellResult,
    CellStatus,
    CosineReference,
    Environment,
    Mode,
    Variant,
)


class Row(BaseModel):
    """One (size, mode, variant): the cells it pools and their figures."""

    size: int
    mode: Mode
    variant: Variant
    status: CellStatus
    cell_ids: list[str]
    n: int = 0
    p50: float | None = None
    p95: float | None = None
    max: float | None = None
    stages_p50: dict[str, float] = Field(default_factory=dict)
    stage_share: dict[str, float] = Field(default_factory=dict)
    rows_read_p50: float | None = None
    embedding_json_bytes_p50: float | None = None
    models_hydrated_p50: float | None = None
    candidates_scored_p50: float | None = None
    snapshot_ids_p50: float | None = None
    peak_rss_bytes: int | None = None
    import_seconds_p50: float | None = None
    encoder_load_seconds_p50: float | None = None
    warmup_seconds: float | None = None
    cosine_reference: CosineReference | None = None
    notes: list[str] = Field(default_factory=list)


class Scaling(BaseModel):
    """How one (mode, variant) grows with N."""

    mode: Mode
    variant: Variant
    sizes: list[int]
    slope: float | None
    stage_slopes: dict[str, float]
    microseconds_per_row_read: dict[int, float]
    """p50 total divided by the p50 rows the loader read, per size: the per-row
    cost of the path, which a constant would make fall as N grows."""


class Summary(BaseModel):
    generated_at: str
    seed: int | None
    environment: Environment | None
    rows: list[Row]
    scaling: list[Scaling]


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values), q))


def _row(cells: list[CellResult]) -> Row:
    first = cells[0].spec
    statuses = [c.status for c in cells]
    worst = next((s for s in statuses if s is not CellStatus.OK), CellStatus.OK)
    row = Row(
        size=first.size,
        mode=first.mode,
        variant=first.variant,
        status=worst,
        cell_ids=[c.spec.cell_id for c in cells],
    )
    ok = [c for c in cells if c.status is CellStatus.OK]
    samples = [s for c in ok for s in c.samples]
    for c in cells:
        row.notes.extend(c.notes)
        if c.status is not CellStatus.OK and c.error:
            row.notes.append(f"{c.spec.cell_id}: {c.status.value}: {c.error.splitlines()[0]}")
    peaks = [p for c in cells for p in (c.peak_rss_bytes, c.sampled_peak_rss_bytes) if p]
    row.peak_rss_bytes = max(peaks) if peaks else None
    if not samples:
        return row
    totals = [s.total_seconds for s in samples]
    row.n = len(samples)
    row.p50 = _pct(totals, 50)
    row.p95 = _pct(totals, 95)
    row.max = max(totals)
    summed = sum(totals)
    for stage in STAGES:
        values = [s.stages.get(stage, 0.0) for s in samples]
        row.stages_p50[stage] = _pct(values, 50)
        row.stage_share[stage] = sum(values) / summed if summed else 0.0
    row.rows_read_p50 = _pct([float(s.rows_read) for s in samples], 50)
    row.embedding_json_bytes_p50 = _pct([float(s.embedding_json_bytes) for s in samples], 50)
    row.models_hydrated_p50 = _pct([float(s.models_hydrated) for s in samples], 50)
    row.candidates_scored_p50 = _pct([float(s.candidates_scored) for s in samples], 50)
    row.snapshot_ids_p50 = _pct([float(s.snapshot_ids_looked_up) for s in samples], 50)
    imports = [c.import_seconds for c in ok if c.import_seconds is not None]
    row.import_seconds_p50 = _pct(imports, 50) if imports else None
    loads = [c.encoder_load_seconds for c in ok if c.encoder_load_seconds is not None]
    row.encoder_load_seconds_p50 = _pct(loads, 50) if loads else None
    warm = [c.warmup_seconds for c in ok if c.warmup_seconds is not None]
    row.warmup_seconds = warm[0] if warm else None
    refs = [c.cosine_reference for c in ok if c.cosine_reference is not None]
    row.cosine_reference = refs[-1] if refs else None
    return row


def _slope(points: list[tuple[int, float]]) -> float | None:
    pts = [(math.log(n), math.log(v)) for n, v in points if v > 0]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return float(np.polyfit(xs, ys, 1)[0])


def summarise(cells: Iterable[CellResult]) -> Summary:
    """Pool cells into rows (one per size, mode, variant) and fit the scaling."""
    groups: dict[tuple[int, Mode, Variant], list[CellResult]] = {}
    for c in cells:
        groups.setdefault((c.spec.size, c.spec.mode, c.spec.variant), []).append(c)
    rows = [_row(groups[k]) for k in sorted(groups, key=lambda k: (k[1], k[2], k[0]))]
    scaling: list[Scaling] = []
    for mode in Mode:
        for variant in Variant:
            done = [r for r in rows if (r.mode, r.variant) == (mode, variant) and r.p50]
            if not done:
                continue
            scaling.append(
                Scaling(
                    mode=mode,
                    variant=variant,
                    sizes=[r.size for r in done],
                    slope=_slope([(r.size, r.p50 or 0.0) for r in done]),
                    stage_slopes={
                        stage: s
                        for stage in STAGES
                        if (s := _slope([(r.size, r.stages_p50.get(stage, 0.0)) for r in done]))
                        is not None
                    },
                    microseconds_per_row_read={
                        r.size: (r.p50 or 0.0) / r.rows_read_p50 * 1e6
                        for r in done
                        if r.rows_read_p50
                    },
                )
            )
    all_cells = [c for g in groups.values() for c in g]
    env = next((_portable(c).environment for c in all_cells if c.environment is not None), None)
    seed = all_cells[0].spec.seed if all_cells else None
    return Summary(
        generated_at=datetime.now(UTC).isoformat(),
        seed=seed,
        environment=env,
        rows=rows,
        scaling=scaling,
    )


def load_cells(cells_dir: Path) -> list[CellResult]:
    return [CellResult.model_validate_json(p.read_text()) for p in sorted(cells_dir.glob("*.json"))]


def size_label(n: int) -> str:
    """``10k``, ``56k``, ``1M``: how a store size is written in tables and file names."""
    if n < 1_000:
        return str(n)
    return f"{n / 1_000:g}k" if n < 1_000_000 else f"{n / 1_000_000:g}M"


def _secs(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 0.01:
        return f"{v * 1000:.1f} ms"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def _bytes(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1024**3:.2f} GiB" if v >= 1024**3 else f"{v / 1024**2:.0f} MiB"


_STAGE_LABEL = {
    "setup": "setup",
    "embed": "embed",
    "sql_fetch": "SQL fetch",
    "decode": "decode",
    "hydrate": "hydrate",
    "filters": "filters",
    "source_rows": "sources",
    "support_discounts": "discounts",
    "scoring": "scoring",
    "sort": "sort",
    "collapse": "collapse",
    "other": "other",
}


def render_markdown(
    summary: Summary,
    *,
    raw_link: Callable[[int], str] | None = None,
) -> str:
    """The tables the published page carries. ``raw_link(size)`` names each row's JSON."""

    def raw(size: int) -> str:
        return f"[json]({raw_link(size)})" if raw_link else ""

    out: list[str] = []
    for mode in Mode:
        for variant in Variant:
            rows = [r for r in summary.rows if (r.mode, r.variant) == (mode, variant)]
            if not rows:
                continue
            out.append(f"#### {mode.value}, {variant.value}\n")
            out.append(
                "| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) "
                "| peak RSS | status | raw |"
            )
            out.append("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                out.append(
                    f"| {size_label(r.size)} | {r.n} | {_secs(r.p50)} | {_secs(r.p95)} "
                    f"| {r.rows_read_p50 or 0:,.0f} | {_bytes(r.embedding_json_bytes_p50)} "
                    f"| {_bytes(r.peak_rss_bytes)} | {r.status.value} | {raw(r.size)} |"
                )
            out.append("")
            out.append("| Store | " + " | ".join(_STAGE_LABEL[s] for s in STAGES) + " |")
            out.append("|---|" + "---|" * len(STAGES))
            for r in rows:
                if not r.n:
                    continue
                cells = [
                    f"{_secs(r.stages_p50[s])} ({100 * r.stage_share[s]:.0f}%)" for s in STAGES
                ]
                out.append(f"| {size_label(r.size)} | " + " | ".join(cells) + " |")
            out.append("")
    if summary.scaling:
        out.append("#### Scaling\n")
        out.append(
            "| Mode, variant | sizes | slope of log p50 on log N "
            "| µs per row read (p50 / rows read) |"
        )
        out.append("|---|---|---|---|")
        for s in summary.scaling:
            per_row = ", ".join(
                f"{size_label(n)} {us:.1f}" for n, us in sorted(s.microseconds_per_row_read.items())
            )
            slope = "—" if s.slope is None else f"{s.slope:.2f}"
            out.append(
                f"| {s.mode.value}, {s.variant.value} | "
                f"{', '.join(size_label(n) for n in s.sizes)} | {slope} | {per_row} |"
            )
        out.append("")
    return "\n".join(out)


def _portable(cell: CellResult) -> CellResult:
    """``cell`` with host paths cut to file names, for publication."""
    out = cell.model_copy(deep=True)
    out.spec.store_path = Path(out.spec.store_path).name
    out.spec.result_path = Path(out.spec.result_path).name
    if out.environment is not None:
        out.environment.config = Path(out.environment.config).name
        if out.environment.bench_lock:
            out.environment.bench_lock = Path(out.environment.bench_lock).name
    return out


def export_by_size(cells: Iterable[CellResult], out_dir: Path, prefix: str) -> list[Path]:
    """Write one JSON per store size: every cell of that size, host paths cut to names."""
    by_size: dict[int, list[CellResult]] = {}
    for c in cells:
        by_size.setdefault(c.spec.size, []).append(_portable(c))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for size, group in sorted(by_size.items()):
        path = out_dir / f"{prefix}-{size_label(size)}.json"
        body = "[\n" + ",\n".join(c.model_dump_json(indent=2) for c in group) + "\n]\n"
        path.write_text(body)
        written.append(path)
    return written
