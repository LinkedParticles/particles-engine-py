# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The Particles memory's child process for the door arm.

Particles enters the comparison the way every competitor does: as a memory in
its own process, written through its normal path and read through the door an
agent receives. The parent (:mod:`.particles_door`) starts this child with
isolation (a private ``HOME``, store, blob directory and
parent-written config, every other ``PARTICLES_*`` variable removed); the
process entry is the Surface module :data:`ENTRY_MODULE`, which binds the MCP
``query`` tool, so this Engine package never imports the MCP server.

Requests, one JSON object per line:

    import_session  text written_at uri session_id   a dated CONVERSATION deposit,
                                                     then extraction
    import_page     text written_at uri session_id   a WEB_PAGE deposit, then extraction
    import_memory   text written_at                  a CONVERSATION deposit (fallback)
    trust_rule      domain score                     a domain trust rule (a variant row)
    door_text       query                            the MCP query tool's result at its
                                                     defaults, rendered as FastMCP sends it
    usage                                            model spend so far, at list price
    census                                           particle counts by status

``--scripted`` runs the oracle perception instead (scripted
extraction and contradiction probe, every other purpose refused): a $0
plumbing run, never the product.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

#: The Surface module the parent runs (``python -m``).
ENTRY_MODULE = "particles.mcp._rot_door_child"
#: Exit code for a refusal before anything ran.
EXIT_REFUSED = 2

QueryFn = Callable[..., Awaitable[dict[str, Any]]]


def render_tool_result(result: dict[str, Any]) -> str:
    """A dict tool result as the MCP server sends it: indented JSON."""
    import pydantic_core

    return pydantic_core.to_json(result, fallback=str, indent=2).decode()


class _Child:
    """One store, one extractor, one usage scope: the state behind the ops."""

    def __init__(
        self, query_fn: QueryFn, *, scripted: bool, world_args: dict[str, Any], cache_dir: str
    ) -> None:
        from particles.benchmark.rot.generator import generate_world
        from particles.benchmark.rot.oracle import OracleExtractor
        from particles.benchmark.rot.schema import RotSession, RotWorld

        self.query_fn = query_fn
        self.scripted = scripted
        self.world: RotWorld | None = None
        self.sessions: dict[str, RotSession] = {}
        self.oracle: OracleExtractor | None = None
        self.cache: Any = None
        self.extractor: Any = None
        self.refused: dict[str, int] = {}
        if scripted:
            self.world = generate_world(
                int(world_args["seed"]), days=int(world_args["days"]), profile=world_args["profile"]
            )
            self.sessions = {s.session_id: s for s in self.world.sessions}
            self.oracle = OracleExtractor()
            self.extractor = self.oracle
        elif cache_dir:
            from particles.benchmark.rot.runner import ExtractionCache, _CachedExtractor
            from particles.core.schema import SourceType
            from particles.extraction.registry import select_extractor
            from particles.llm import get_provider

            inner = select_extractor(SourceType.CONVERSATION)
            self.cache = ExtractionCache(
                Path(cache_dir) / "rot-extraction.pickle",
                extractor=inner,
                model=get_provider("extraction").provider_model,
            )
            self.extractor = _CachedExtractor(inner, self.cache)

    async def deposit(
        self, *, text: str, written_at: str, uri: str, session_id: str, page: bool
    ) -> dict[str, Any]:
        from particles.benchmark.rot.oracle import oracle_claims
        from particles.core.schema import Mutability, SourceType
        from particles.corpus.deposit import deposit_text_versioned
        from particles.db import session_scope
        from particles.ingest.pipeline import extract_snapshot

        if self.oracle is not None and self.world is not None:
            self.oracle.register(text, oracle_claims(self.world, self.sessions[session_id]))
        async with session_scope() as db:
            entry_id, snapshot_id, unchanged = await deposit_text_versioned(
                db,
                text=text,
                uri_r=uri,
                source_type=SourceType.WEB_PAGE if page else SourceType.CONVERSATION,
                mutability=Mutability.STABLE,
                deposited_by="benchmark-rot-door",
                content_published_at=datetime.fromisoformat(written_at),
            )
            await db.commit()
            if not unchanged:
                await extract_snapshot(db, entry_id, snapshot_id, extractor=self.extractor)
                await db.commit()
        return {"entry_id": entry_id, "unchanged": unchanged}

    async def trust_rule(self, domain: str, score: float) -> None:
        from particles.db import session_scope
        from particles.store.trust_store import upsert_trust_rule

        async with session_scope() as db:
            await upsert_trust_rule(
                db,
                "domain",
                domain,
                score=score,
                modifier=None,
                rationale="rot door arm: the trust-rule variant row",
                asserted_by="benchmark-rot-door",
                actor="benchmark-rot-door",
            )
            await db.commit()

    async def census(self) -> dict[str, int]:
        from sqlalchemy import func, select

        from particles.db import session_scope
        from particles.store.particle_store import ParticleRow

        async with session_scope() as db:
            rows = await db.execute(
                select(ParticleRow.status, func.count()).group_by(ParticleRow.status)
            )
            return {str(status): int(n) for status, n in rows.all()}


async def _handle(child: _Child, req: dict[str, Any], acc: Any) -> Any:
    op = req.get("op")
    if op == "import_session":
        return await child.deposit(
            text=req["text"],
            written_at=req["written_at"],
            uri=req["uri"],
            session_id=req.get("session_id", ""),
            page=False,
        )
    if op == "import_page":
        return await child.deposit(
            text=req["text"],
            written_at=req["written_at"],
            uri=req["uri"],
            session_id=req.get("session_id", ""),
            page=True,
        )
    if op == "import_memory":
        return await child.deposit(
            text=req["text"],
            written_at=req["written_at"],
            uri=f"rotdoor://memory/{uuid.uuid4().hex}",
            session_id=req.get("session_id", ""),
            page=False,
        )
    if op == "trust_rule":
        await child.trust_rule(req["domain"], float(req["score"]))
        return None
    if op == "door_text":
        result = await child.query_fn(question=req["query"])
        return {"text": render_tool_result(result)}
    if op == "usage":
        usage = acc.snapshot()
        return {"priced_usd": usage.priced_cost_usd, "unpriced": list(usage.unpriced)}
    if op == "census":
        return await child.census()
    raise ValueError(f"unknown op {op!r}")


async def _serve(
    query_fn: QueryFn, args: argparse.Namespace, proto: TextIO, pins: dict[str, str]
) -> int:
    from particles.benchmark.update_gate._child import preflight
    from particles.db import create_tables, get_engine
    from particles.llm.usage import track_usage

    refusal = preflight(Path(args.run_dir), args.parent_store, pins)
    if refusal is not None:
        proto.write(json.dumps({"ok": False, "error": f"refused: {refusal}"}) + "\n")
        return EXIT_REFUSED
    await create_tables()
    world_args = {"seed": args.seed, "days": args.days, "profile": args.profile}
    child = _Child(
        query_fn, scripted=args.scripted, world_args=world_args, cache_dir=args.cache_dir or ""
    )
    loop = asyncio.get_running_loop()

    async def loop_requests(acc: Any) -> None:
        proto.write(json.dumps({"ok": True, "result": "ready"}) + "\n")
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return
            if not line.strip():
                continue
            try:
                reply = {"ok": True, "result": await _handle(child, json.loads(line), acc)}
            except Exception as exc:  # noqa: BLE001 — every failure goes back as data
                traceback.print_exc(file=sys.stderr)
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            proto.write(json.dumps(reply, default=str) + "\n")

    try:
        with track_usage() as acc:
            if args.scripted:
                from particles.benchmark.rot.runner import _arm_routing

                assert child.world is not None
                with _arm_routing("oracle", child.world, child.refused):
                    await loop_requests(acc)
            else:
                await loop_requests(acc)
    finally:
        if child.cache is not None:
            child.cache.flush()
        await get_engine().dispose()
    return 0


def main(query_fn: QueryFn, argv: list[str] | None = None) -> int:
    """Serve the Particles memory for one build until stdin closes."""
    parser = argparse.ArgumentParser(prog=ENTRY_MODULE)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-store", required=True)
    parser.add_argument("--pins", default="{}", help="JSON: purpose → provider:model")
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args(argv)
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    pins: dict[str, str] = json.loads(args.pins)
    return asyncio.run(_serve(query_fn, args, proto, pins))
