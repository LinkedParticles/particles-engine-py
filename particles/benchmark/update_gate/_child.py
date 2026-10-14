# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The child side of one trip-versus-move run.

One child process runs one store. The parent starts it with a private
environment: its own ``HOME``, ``DATABASE_URL``, ``PARTICLES_BLOB_DIR`` and
``PARTICLES_CONFIG``, everything else of the Particles kind removed. The child
drives the operator's own verbs, ``particles audit`` once per note and then
``particles memory consolidate --scope store``, as ``scripts/trip_vs_move_e2e.py``
did. It writes ``result.json`` in its run directory and nothing outside it.

The verbs are the Surface's, and this package is the Engine's, so the child is
split along the layer line: this module holds the run, and takes the verbs as
an injected ``invoke`` callable; ``particles.api.cli._trip_vs_move_child`` is
the process entry that binds them (:data:`ENTRY_MODULE`).

Before any call it refuses (exit 2, nothing spent) unless its resolved config
pins every purpose as the parent's pin map says, has the memory projection
off, and its store is not the parent's and lies inside its run directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Exit codes: a finished run, and a refusal before any call.
EXIT_DONE = 0
EXIT_REFUSED = 2

RESULT_FILE = "result.json"
PINS_FILE = "pins.json"
#: The module the parent runs with ``python -m``.
ENTRY_MODULE = "particles.api.cli._trip_vs_move_child"

#: Runs one CLI command (argv without the program name); returns exit code and output.
Invoke = Callable[[list[str]], tuple[int, str]]


def all_purposes() -> list[str]:
    """Every completion purpose, read from the registry at run time, plus ``default``."""
    from particles.llm import LLMPurpose

    return ["default", *typing.get_args(LLMPurpose)]


def store_file(dsn: str) -> Path | None:
    """The SQLite file a ``sqlite+aiosqlite:///`` DSN names, or ``None``."""
    prefix = "sqlite+aiosqlite:///"
    if not dsn.startswith(prefix):
        return None
    return Path(dsn[len(prefix) :]).resolve()


def preflight(run_dir: Path, parent_store: str, pins: dict[str, str]) -> str | None:
    """Why this child must not start, or ``None``. Reads config only."""
    from particles.config import get_config

    cfg = get_config()
    for purpose, want in pins.items():
        sel = cfg.llm.for_purpose(purpose) if purpose != "default" else cfg.llm.default
        got = f"{sel.provider}:{sel.model}"
        if got != want:
            return f"config pins {purpose} to {got}, not {want}"
    if cfg.agent_memory.projection.enabled:
        return "the memory projection is on"
    dsn = cfg.storage.database_url
    if dsn == parent_store:
        return "the child's store is the parent's store"
    path = store_file(dsn)
    if path is None or run_dir.resolve() not in path.parents:
        return f"the child's store {dsn} is not a SQLite file inside {run_dir}"
    return None


async def _init_store() -> None:
    from particles.db import create_tables

    await create_tables()


async def _dispose() -> None:
    from particles.db import get_engine

    await get_engine().dispose()


def run(
    run_dir: Path,
    *,
    variant: str,
    arm: str,
    parent_store: str,
    ceiling: float | None,
    invoke: Invoke,
) -> dict[str, Any]:
    """Deposit the notes, consolidate, and return what happened. Spends on the live arm."""
    # Deferred imports: the environment the parent set must be in place before
    # the SDK reads config.
    from particles.benchmark.update_gate.fixtures import (
        MOVE_NOTES,
        TRIP_ALONE_NOTES,
        case_note,
    )
    from particles.llm import override_providers, track_usage

    pins: dict[str, str] = json.loads((run_dir / PINS_FILE).read_text())
    refusal = preflight(run_dir, parent_store, pins)
    if refusal is not None:
        return {"refused": refusal}

    asyncio.run(_init_store())
    log = None
    overrides: dict[str, Any] = {}
    if arm == "scripted":
        from particles import embeddings
        from particles.benchmark.update_gate.scripted import (
            BagOfWordsEncoder,
            RequestLog,
            ScriptedTable,
            scripted_providers,
        )

        table = ScriptedTable.load()
        embeddings.set_embedding_model(BagOfWordsEncoder(table.embedding))  # type: ignore[arg-type]
        log = RequestLog()
        overrides = scripted_providers([p for p in all_purposes() if p != "default"], table, log)

    notes_dir = run_dir / "notes"
    notes_dir.mkdir()
    notes = TRIP_ALONE_NOTES if variant == "trip-alone" else MOVE_NOTES
    result: dict[str, Any] = {"exits": {}, "incomplete": None}
    # The ceiling is checked against measured spend between steps, never handed
    # to consolidation.budget_usd: that budget skips a pass whose cap-based
    # estimate would pass it, which changes the system under test (the first
    # reports of record lost the census's contradiction probe that way at
    # US$0.08 of US$0.45). Consolidation runs as the record's runs did,
    # bounded by its own probe caps.
    steps: list[tuple[str, list[str], str | None]] = [
        (f"audit {name}", ["audit", str(notes_dir), "--yes"], name) for name in notes
    ]
    steps.append(("memory consolidate", ["memory", "consolidate", "--scope", "store"], None))
    with override_providers(overrides), track_usage() as usage:  # type: ignore[arg-type]
        for label, argv, note in steps:
            if ceiling is not None and usage.snapshot().priced_cost_usd >= ceiling:
                result["incomplete"] = "budget"
                break
            if note is not None:
                shutil.copy(case_note(note), notes_dir / note)
            code, output = invoke(argv)
            result["exits"][label] = code
            if code != 0:
                result["incomplete"] = f"particles {label} exited {code}"
                result["output"] = output[-4000:]
                break
        result["usage"] = usage.snapshot().model_dump(mode="json")
    asyncio.run(_dispose())
    if log is not None:
        result["requests"] = log.entries
        result["refused_calls"] = log.refused_by_purpose()
    return result


def main(invoke: Invoke, argv: list[str] | None = None) -> int:
    """Parse the parent's arguments, run, and write ``result.json``."""
    parser = argparse.ArgumentParser(description="One trip-versus-move run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("move", "trip-alone"), required=True)
    parser.add_argument("--arm", choices=("live", "scripted"), required=True)
    parser.add_argument("--parent-store", required=True)
    parser.add_argument("--ceiling", type=float, default=None)
    args = parser.parse_args(argv)
    result = run(
        args.run_dir,
        variant=args.variant,
        arm=args.arm,
        parent_store=args.parent_store,
        ceiling=args.ceiling,
        invoke=invoke,
    )
    (args.run_dir / RESULT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    return EXIT_REFUSED if "refused" in result else EXIT_DONE
