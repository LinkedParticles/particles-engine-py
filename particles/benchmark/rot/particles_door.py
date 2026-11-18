# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The Particles memory as the door arm drives it: a child process per build.

The child (:mod:`.door_child`, entered through the Surface module
:data:`~.door_child.ENTRY_MODULE`) gets isolation: a private
``HOME`` and ``claude_code.state_dir``, its own ``DATABASE_URL``, blob directory
and parent-written ``PARTICLES_CONFIG`` (compiled defaults plus the scratch
paths, and for the model-parity row the model pins), every other
``PARTICLES_*`` and config-override variable removed, and the same refusals to
open anything but its own store.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import IO, Any, get_args

import yaml

from particles.benchmark.rot.contract import DoorAdapterError, DoorMemory
from particles.benchmark.rot.door_child import ENTRY_MODULE
from particles.benchmark.update_gate.case import child_env
from particles.config import get_config


def child_config(
    run_dir: Path, *, parity: tuple[str, str] | None
) -> tuple[dict[str, Any], dict[str, str]]:
    """The child's whole config and its model pins (purpose → ``provider:model``).

    Compiled defaults plus the scratch paths. Three sections come from the
    operator's own config, as: ``llm.providers`` (endpoint
    wiring a parity model needs), ``llm.price_per_mtok`` (so spend is priced)
    and ``network`` (their egress posture holds in the child). ``parity`` is
    ``(provider, model)`` and pins every purpose to it.
    """
    from particles.llm import LLMPurpose

    cfg = get_config()
    llm: dict[str, Any] = {
        "providers": {
            name: entry.model_dump(mode="json") for name, entry in cfg.llm.providers.items()
        },
        "price_per_mtok": {
            key: price.model_dump(mode="json") for key, price in cfg.llm.price_per_mtok.items()
        },
    }
    pins: dict[str, str] = {}
    if parity is not None:
        provider, model = parity
        for purpose in ("default", *get_args(LLMPurpose)):
            llm[purpose] = {"provider": provider, "model": model}
            pins[purpose] = f"{provider}:{model}"
    network = cfg.network.model_dump(mode="json")
    config = {
        "llm": llm,
        "network": network,
        "claude_code": {"state_dir": str(run_dir / "home" / ".particles" / "claude-code")},
        "agent_memory": {"projection": {"enabled": False}},
    }
    return config, pins


class ParticlesDoorMemory(DoorMemory):
    """One build's Particles memory, served by a child process."""

    name = "particles"

    def __init__(
        self,
        run_dir: Path,
        *,
        world: tuple[int, int, str],
        parity: tuple[str, str] | None = None,
        scripted: bool = False,
        cache_dir: Path | None = None,
        trust_rule: tuple[str, float] | None = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        config, pins = child_config(run_dir, parity=parity)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
        seed, days, profile = world
        argv = [
            sys.executable,
            "-m",
            ENTRY_MODULE,
            "--run-dir",
            str(run_dir),
            "--parent-store",
            get_config().storage.database_url,
            "--pins",
            json.dumps(pins),
            "--seed",
            str(seed),
            "--days",
            str(days),
            "--profile",
            profile,
        ]
        if scripted:
            argv.append("--scripted")
        if cache_dir is not None:
            argv += ["--cache-dir", str(cache_dir)]
        self._log: IO[str] = (run_dir / "child.log").open("a", encoding="utf-8")
        self._proc = subprocess.Popen(  # noqa: S603 — our own interpreter and module
            argv,
            cwd=run_dir,
            env=child_env(run_dir, arm="live"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready = self._read()
        if not ready.get("ok"):
            self.close()
            raise DoorAdapterError(f"particles: {ready.get('error', 'child did not start')}")
        if trust_rule is not None:
            self._call("trust_rule", domain=trust_rule[0], score=trust_rule[1])

    def _read(self) -> dict[str, Any]:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise DoorAdapterError("particles: the child exited (see child.log in its run dir)")
        reply: dict[str, Any] = json.loads(line)
        return reply

    def _call(self, op: str, **args: Any) -> Any:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps({"op": op, **args}) + "\n")
        self._proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise DoorAdapterError(f"particles.{op}: {reply.get('error')}")
        return reply.get("result")

    def doors(self) -> list[str]:
        """The MCP ``query`` tool at its defaults: what an agent receives."""
        return ["mcp-query"]

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        """The tool's result as the MCP server renders it."""
        out = self._call("door_text", query=query)
        return str(out["text"]), None

    def import_memory(self, text: str, written_at: str) -> None:
        """A plain dated text deposit (unused: the arm requires ``import_session``)."""
        self._call("import_memory", text=text, written_at=written_at)

    def import_session(
        self, turns: list[dict[str, str]], written_at: str, *, text: str, session_id: str
    ) -> None:
        """A dated ``CONVERSATION`` deposit of the rendered session, then extraction."""
        self._call(
            "import_session",
            text=text,
            written_at=written_at,
            uri=f"rotbench://door/session/{session_id}",
            session_id=session_id,
        )

    def import_page(self, text: str, written_at: str, *, uri: str, session_id: str) -> None:
        """A ``WEB_PAGE`` deposit at the page's own URI, then extraction."""
        self._call("import_page", text=text, written_at=written_at, uri=uri, session_id=session_id)

    def supports(self, method: str) -> bool:
        """Particles implements both optional write methods and reports usage."""
        return method in ("import_session", "import_page", "usage")

    def usage_usd(self) -> float | None:
        """Priced model spend so far; ``None`` when a model in use is unpriced."""
        out = self._call("usage")
        return None if out["unpriced"] else float(out["priced_usd"])

    def census(self) -> dict[str, int]:
        """Particle counts by status."""
        return dict(self._call("census"))

    def close(self) -> None:
        """Close the child's stdin and wait; kill it if it lingers."""
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._log.close()
