# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The parent side of the trip-versus-move case (§§ 2.2, 3 to 5).

Each run is one store in a child process (:mod:`._child`) with a private
environment, so no run can reach the reader's store, whatever path inside
``audit`` or ``memory consolidate`` opens a session on the default store (the
pooled extract pass does). The parent writes the child's config, starts it,
reads the finished store read-only, and scores it with :mod:`.checks`.

The parent never deletes a directory it did not create: a ``--store-dir`` that
exists and is not empty is refused, and without one the runs live in a temp
directory removed at exit.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from particles.benchmark.update_gate import _child
from particles.benchmark.update_gate.checks import CHECKS_VERSION, evaluate, read_store
from particles.benchmark.update_gate.estimate import CaseEstimate, default_ceiling
from particles.benchmark.update_gate.fixtures import MOVE_NOTES, case_note, sha256
from particles.benchmark.update_gate.schema import (
    CaseRun,
    FixtureFile,
    SavedEvent,
    SavedRow,
    Spend,
    TripVsMoveReport,
    Verdict,
)
from particles.config import ProviderSelection, env_override_names, get_config
from particles.llm.usage import LLMUsage

#: The record's protocol: the bar is stated only for these run counts.
RECORD_RUNS = {"move": 3, "trip-alone": 2}
#: A placeholder the scripted arm's child carries so the verbs' key gates pass.
#: Every purpose is overridden in process and egress is ``none``: nothing can
#: send it anywhere.
SCRIPTED_KEY_PLACEHOLDER = "scripted-arm-placeholder-not-a-key"


class CaseRefused(RuntimeError):
    """The case cannot start; nothing was spent."""


def selection_key(selection: ProviderSelection) -> str:
    """``provider:model``, the form reports and usage rows name a model by."""
    return f"{selection.provider}:{selection.model}"


def resolve_pins(
    *, provider: str | None, model: str | None, pins: Mapping[str, str]
) -> tuple[ProviderSelection, dict[str, ProviderSelection]]:
    """The one model, and every purpose's pinned selection.

    ``model`` and ``provider`` default to ``llm.default``'s. ``pins`` maps a
    purpose to a model on the same provider (``--pin``). Every purpose is
    pinned, read from the registry at run time, plus ``default``.

    Raises:
        CaseRefused: If a ``--pin`` names a purpose that does not exist.
    """
    default = get_config().llm.default
    base = ProviderSelection(
        provider=provider or default.provider,
        model=model or default.model,
    )
    purposes = _child.all_purposes()
    unknown = sorted(set(pins) - set(purposes))
    if unknown:
        raise CaseRefused(f"--pin names unknown purpose(s) {unknown}; known: {purposes}")
    out = {p: base for p in purposes}
    for purpose, pinned_model in pins.items():
        out[purpose] = ProviderSelection(provider=base.provider, model=pinned_model)
    return base, out


def child_config(
    run_dir: Path,
    pins: Mapping[str, ProviderSelection],
    *,
    same_slot_admits: bool,
    arm: str,
) -> dict[str, Any]:
    """The child's whole config: compiled defaults plus these keys, nothing else.

    Three sections come from the reader's own config: ``llm.providers`` (the
    endpoint wiring of a non-Anthropic model), ``llm.price_per_mtok`` (their
    prices) and ``network`` (their egress posture holds in the child). The
    scripted arm's egress is ``none``: it reaches nothing at all.
    """
    cfg = get_config()
    llm: dict[str, Any] = {p: {"provider": s.provider, "model": s.model} for p, s in pins.items()}
    llm["providers"] = {
        name: entry.model_dump(mode="json") for name, entry in cfg.llm.providers.items()
    }
    llm["price_per_mtok"] = {
        key: price.model_dump(mode="json") for key, price in cfg.llm.price_per_mtok.items()
    }
    network = cfg.network.model_dump(mode="json")
    if arm == "scripted":
        network = {"egress": "none"}
    return {
        "llm": llm,
        "network": network,
        "reconciliation": {"update_supersession": {"same_slot_admits": same_slot_admits}},
        "claude_code": {"state_dir": str(run_dir / "home" / ".particles" / "claude-code")},
        "agent_memory": {"projection": {"enabled": False}},
    }


def child_env(run_dir: Path, *, arm: str) -> dict[str, str]:
    """The child's environment: the parent's, scrubbed, with the run's own paths.

    Every ``PARTICLES_*`` variable and every config override name is removed,
    so no override of the reader's reaches the child. Secrets pass through:
    ``particles.secrets`` reads only the process environment, so a private
    ``HOME`` hides no key. ``HF_HOME`` points at the parent's model cache, so
    the embedding model is not downloaded again.
    """
    drop = env_override_names()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PARTICLES_") and k not in drop}
    home = run_dir / "home"
    (home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    env["HOME"] = str(home)
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{run_dir / 'particles.db'}"
    env["PARTICLES_BLOB_DIR"] = str(run_dir / "blobs")
    env["PARTICLES_CONFIG"] = str(run_dir / "config.yaml")
    if arm == "scripted":
        env["ANTHROPIC_API_KEY"] = SCRIPTED_KEY_PLACEHOLDER
    return env


@contextlib.contextmanager
def work_root(store_dir: Path | None) -> Iterator[Path]:
    """The directory the runs live in.

    A ``store_dir`` is kept, and must be absent or empty; a temp directory is
    removed at exit. A directory this function did not create is never deleted.

    Raises:
        CaseRefused: If ``store_dir`` exists and is not empty.
    """
    if store_dir is not None:
        if store_dir.exists() and any(store_dir.iterdir()):
            raise CaseRefused(f"--store-dir {store_dir} is not empty; nothing was run or deleted")
        store_dir.mkdir(parents=True, exist_ok=True)
        yield store_dir
        return
    root = Path(tempfile.mkdtemp(prefix="particles-trip-vs-move-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def model_leak(usage: LLMUsage | None, pins: Mapping[str, ProviderSelection]) -> str | None:
    """A usage row on a model its purpose was not pinned to, or ``None``."""
    if usage is None:
        return None
    pinned = {selection_key(s) for s in pins.values()}
    for row in usage.rows:
        got = f"{row.provider}:{row.model}"
        want = pins.get(row.purpose)
        if want is None:
            if got not in pinned:
                return f"model leak: {row.purpose} ran on {got}"
        elif got != selection_key(want):
            return f"model leak: {row.purpose} ran on {got}, pinned to {selection_key(want)}"
    return None


def budget_limited(db: Path) -> bool:
    """Whether the run's consolidation skipped LLM work for its budget."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        payloads = [
            json.loads(p or "{}")
            for (p,) in conn.execute(
                "select payload from operator_events where event_type = 'CONSOLIDATION_RUN'"
            )
        ]
    for payload in payloads:
        budget = payload.get("budget") or {}
        if budget.get("skipped_passes") or budget.get("skipped_requests"):
            return True
    return False


def run_one(
    run_dir: Path,
    *,
    variant: str,
    index: int,
    arm: str,
    pins: Mapping[str, ProviderSelection],
    same_slot_admits: bool,
    ceiling: float | None,
    timeout: float,
) -> CaseRun:
    """Run one store in a child and score it."""
    run_dir.mkdir(parents=True)
    config = child_config(run_dir, pins, same_slot_admits=same_slot_admits, arm=arm)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (run_dir / _child.PINS_FILE).write_text(
        json.dumps({p: selection_key(s) for p, s in pins.items()}, indent=1)
    )
    cmd = [
        sys.executable,
        "-m",
        _child.ENTRY_MODULE,
        "--run-dir",
        str(run_dir),
        "--variant",
        variant,
        "--arm",
        arm,
        "--parent-store",
        get_config().storage.database_url,
    ]
    if ceiling is not None:
        cmd += ["--ceiling", repr(ceiling)]
    proc = subprocess.run(  # noqa: S603 — our own interpreter and module, no shell
        cmd,
        cwd=run_dir,
        env=child_env(run_dir, arm=arm),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    result_file = run_dir / _child.RESULT_FILE
    if not result_file.exists():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return CaseRun(
            variant=variant,  # type: ignore[arg-type]
            index=index,
            passed=None,
            incomplete=f"the run's child exited {proc.returncode} without a result: {tail}",
        )
    result = json.loads(result_file.read_text())
    if "refused" in result:
        raise CaseRefused(f"a run refused to start: {result['refused']}")
    usage = LLMUsage.model_validate(result["usage"]) if result.get("usage") else None
    db = run_dir / "particles.db"
    rows, events = read_store(db)
    checks = evaluate(rows, events, trip_alone=variant == "trip-alone")
    incomplete = result.get("incomplete")
    if incomplete is None and budget_limited(db):
        incomplete = "budget"
    invalid = model_leak(usage, pins) if arm == "live" else None
    passed = None if (incomplete or invalid) else bool(checks["pass"])
    return CaseRun(
        variant=variant,  # type: ignore[arg-type]
        index=index,
        passed=passed,
        checks=checks,
        incomplete=incomplete,
        invalid=invalid,
        usage=usage,
        rows=[SavedRow.model_validate(r) for r in rows],
        events=[SavedEvent(event_type=k, payload=p) for k, p in events],
        refused_calls=result.get("refused_calls") or {},
    )


def _verdict(runs: list[CaseRun]) -> Verdict:
    move = [r for r in runs if r.variant == "move"]
    alone = [r for r in runs if r.variant == "trip-alone"]
    protocol = len(move) == RECORD_RUNS["move"] and len(alone) == RECORD_RUNS["trip-alone"]
    bar = None
    if protocol and all(r.passed is not None for r in runs):
        bar = all(r.passed for r in runs)
    return Verdict(
        move_passed=sum(bool(r.passed) for r in move),
        move_runs=len(move),
        trip_alone_passed=sum(bool(r.passed) for r in alone),
        trip_alone_runs=len(alone),
        bar_met=bar,
    )


def _spend(
    runs: list[CaseRun], estimate: CaseEstimate, ceilings: Mapping[str, float | None]
) -> Spend:
    total: float | None = 0.0
    priced = 0.0
    unpriced: list[str] = []
    for run in runs:
        if run.usage is None:
            continue
        priced += run.usage.priced_cost_usd
        if run.usage.cost_usd is None:
            total = None
        elif total is not None:
            total += run.usage.cost_usd
        unpriced += [u for u in run.usage.unpriced if u not in unpriced]
    ceiling_values = [c for c in ceilings.values() if c is not None]
    return Spend(
        estimate_source=estimate.source,
        estimate_usd=estimate.total_usd,
        estimate_calls=estimate.total_calls,
        cost_usd=total,
        priced_cost_usd=round(priced, 6),
        unpriced=unpriced,
        ceiling_usd=max(ceiling_values) if ceiling_values else None,
    )


def run_case(
    *,
    base: ProviderSelection,
    pins: Mapping[str, ProviderSelection],
    runs: Mapping[str, int],
    arm: str,
    same_slot_admits: bool,
    estimate: CaseEstimate,
    max_usd: float | None,
    store_dir: Path | None,
    progress: Callable[[str], None] | None = None,
) -> TripVsMoveReport:
    """Run every store of the case and build the report of record."""
    from particles import __version__

    cfg = get_config().benchmark_update_gate
    mixed = any(selection_key(s) != selection_key(base) for s in pins.values())
    ceilings = {
        v: (
            max_usd if max_usd is not None else default_ceiling(estimate, v, cfg.max_usd_multiplier)
        )
        for v in runs
    }
    if arm == "scripted":
        ceilings = dict.fromkeys(runs)
    done: list[CaseRun] = []
    with work_root(store_dir) as root:
        for variant, n in runs.items():
            for i in range(1, n + 1):
                run = run_one(
                    root / f"{variant}-{i}",
                    variant=variant,
                    index=i,
                    arm=arm,
                    pins=pins,
                    same_slot_admits=same_slot_admits,
                    ceiling=ceilings[variant],
                    timeout=cfg.run_timeout_seconds,
                )
                done.append(run)
                if progress is not None:
                    progress(_run_line(run))
    notes: list[str] = []
    if arm == "scripted":
        notes.append(
            "Scripted arm: every model answer came from the package's table. It proves the "
            "harness, the isolation and the checks, and nothing about any model's reading."
        )
    return TripVsMoveReport(
        checks_version=CHECKS_VERSION,
        sdk_version=__version__,
        run_at=datetime.now(UTC),
        model=selection_key(base),
        purposes={p: selection_key(s) for p, s in pins.items()},
        mixed=mixed,
        same_slot_admits=same_slot_admits,
        arm=arm,  # type: ignore[arg-type]
        fixture=[FixtureFile(name=n, sha256=sha256(case_note(n))) for n in MOVE_NOTES],
        runs=done,
        verdict=_verdict(done),
        spend=_spend(done, estimate, ceilings) if arm == "live" else Spend(estimate_source="none"),
        notes=notes,
    )


def _run_line(run: CaseRun) -> str:
    if run.invalid:
        state = f"INVALID ({run.invalid})"
    elif run.incomplete:
        state = f"INCOMPLETE ({run.incomplete})"
    else:
        state = "PASS" if run.passed else "FAIL"
    cost = ""
    if run.usage is not None and run.usage.rows and run.usage.cost_usd is not None:
        cost = f"  US${run.usage.cost_usd:.4f}"
    return f"{run.variant} run {run.index}: {state}{cost}"


def rescore(report: TripVsMoveReport, *, version: int = CHECKS_VERSION) -> TripVsMoveReport:
    """Re-run the checks over a saved report's rows and events; no store, no call.

    Raises:
        NotRescorable: If a run lacks a field ``version`` reads.
    """
    runs = []
    for run in report.runs:
        rows = [r.model_dump() for r in run.rows]
        events = [(e.event_type, e.payload) for e in run.events]
        checks = evaluate(rows, events, trip_alone=run.variant == "trip-alone", version=version)
        passed = None if (run.incomplete or run.invalid) else bool(checks["pass"])
        runs.append(run.model_copy(update={"checks": checks, "passed": passed}))
    return report.model_copy(
        update={
            "checks_version": version,
            "runs": runs,
            "verdict": _verdict(runs),
            "notes": [
                *report.notes,
                f"Re-scored under checks version {version} from the saved rows and events.",
            ],
        }
    )
