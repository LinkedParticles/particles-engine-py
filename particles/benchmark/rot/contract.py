# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The memory contract the door arm drives, and the child process that serves one.

a competitor enters the rot world through **its adebench
adapter, unmodified**, from a clone pinned by commit. The door arm uses five
methods of that contract (``import_memory``, ``settle``, ``doors``,
``door_text``, ``health``) plus two optional ones (``import_session``, offered
upstream; ``import_page``, which only the Particles memory has). Each
competitor adapter runs in its own child process (:class:`ChildContractAdapter`)
under the clone's interpreter, started with ``-I`` and a whitelisted
environment, so a provider key one memory needs never reaches another memory's
process.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import IO, Any

from particles.url_safety import trusted_path

#: Variables every child inherits from the parent; everything else must be
#: named by the memory's own manifest entry. ``HOME`` is not
#: among them: :func:`child_env` points it at the build's scratch directory.
BASE_ENV: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
)

#: A child's door shim, shipped beside this module.
SHIM_PATH = Path(__file__).with_name("door_shim.py")


class DoorAdapterError(RuntimeError):
    """A memory's adapter failed to start, refused, or returned an error."""


class DoorMemory:
    """What the door arm drives: the contract subset, with no-op defaults.

    A subclass overrides what its memory supports and reports it through
    :meth:`supports`, so the driver can tell an absent method from one that
    ran and failed.
    """

    name = "memory"

    def health(self) -> bool:
        """True when the memory answers."""
        return True

    def doors(self) -> list[str]:
        """The memory's doors; the first is its declared default."""
        return ["default"]

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        """The text a client of ``door`` receives for ``query``, and the raw result."""
        raise NotImplementedError

    def import_memory(self, text: str, written_at: str) -> None:
        """Write one memory, dated ``written_at``, through the memory's import path."""
        raise NotImplementedError

    def import_session(
        self, turns: list[dict[str, str]], written_at: str, *, text: str, session_id: str
    ) -> None:
        """Write one session as speaker turns (optional)."""
        raise NotImplementedError

    def import_page(self, text: str, written_at: str, *, uri: str, session_id: str) -> None:
        """Write one fetched web page at its URI (optional)."""
        raise NotImplementedError

    def settle(self) -> None:
        """Let pending writes finish before the next read (optional)."""

    def supports(self, method: str) -> bool:
        """Whether an optional method is implemented."""
        return False

    def usage_usd(self) -> float | None:
        """Model spend so far, when the memory reports it; ``None`` otherwise."""
        return None

    def close(self) -> None:
        """Stop whatever the memory started."""


def child_env(
    parent: dict[str, str], declared: dict[str, str], *, store: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """A child's environment and the published record of it.

    ``declared`` comes from the memory's manifest entry: a literal value (with
    ``{store}`` replaced by the build's store directory) or ``{env:NAME}``,
    which passes the parent's ``NAME`` through. The record names every
    variable and shows literal values only; a passed-through value (a key) is
    recorded as ``<from env NAME>``.

    ``HOME`` (and ``USERPROFILE`` where set) is the build's own
    ``<store>/home``, never the operator's, so a competitor process has no
    reach into the operator's dotfiles or stores; an entry that needs another
    home declares ``HOME`` itself. The caller creates the directory.

    Raises:
        DoorAdapterError: When a ``{env:NAME}`` variable is not set.
    """
    env = {k: parent[k] for k in BASE_ENV if k in parent}
    env["HOME"] = str(store / "home")
    if "USERPROFILE" in parent:
        env["USERPROFILE"] = env["HOME"]
    record: dict[str, str] = {}
    for name, value in declared.items():
        if value.startswith("{env:") and value.endswith("}"):
            source = value[5:-1]
            if source not in parent:
                raise DoorAdapterError(f"{name} needs {source} in the environment; it is not set")
            env[name] = parent[source]
            record[name] = f"<from env {source}>"
        else:
            env[name] = value.replace("{store}", str(store))
            record[name] = value
    # Every program the build starts is looked up on the PATH entries no other
    # user may write, so a relative entry finds none in the build's directory.
    if "PATH" in env:
        env["PATH"] = trusted_path(env)
        if "PATH" in record and not record["PATH"].startswith("<from env"):
            record["PATH"] = env["PATH"]
    return env, record


class ChildContractAdapter(DoorMemory):
    """One adebench adapter in its own process, behind :data:`SHIM_PATH`."""

    def __init__(
        self,
        name: str,
        *,
        spec: str,
        clone: Path,
        python: Path,
        env: dict[str, str],
        log: Path,
    ) -> None:
        self.name = name
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log: IO[str] = log.open("a", encoding="utf-8")
        self._proc = subprocess.Popen(  # noqa: S603 — argv built from the operator's manifest
            [str(python), "-I", str(SHIM_PATH), "--adapter", spec, "--root", str(clone)],
            cwd=clone,
            env=env,
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
            raise DoorAdapterError(f"{name}: {ready.get('error', 'adapter did not start')}")
        self._methods = set(self._call("methods"))

    def _read(self) -> dict[str, Any]:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise DoorAdapterError(f"{self.name}: the adapter process exited (see its log)")
        reply: dict[str, Any] = json.loads(line)
        return reply

    def _call(self, op: str, **args: Any) -> Any:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps({"op": op, **args}) + "\n")
        self._proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise DoorAdapterError(f"{self.name}.{op}: {reply.get('error')}")
        return reply.get("result")

    def health(self) -> bool:
        """The adapter's own health check."""
        return bool(self._call("health"))

    def doors(self) -> list[str]:
        """The adapter's doors, its default first."""
        return list(self._call("doors"))

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        """The door text and the raw result, as the adapter returns them."""
        out = self._call("door_text", query=query, door=door)
        return str(out["text"]), out.get("raw")

    def import_memory(self, text: str, written_at: str) -> None:
        """The adapter's ``import_memory``."""
        self._call("import_memory", text=text, written_at=written_at)

    def import_session(
        self, turns: list[dict[str, str]], written_at: str, *, text: str, session_id: str
    ) -> None:
        """The adapter's ``import_session``, when upstream has added it."""
        self._call("import_session", turns=turns, written_at=written_at)

    def settle(self) -> None:
        """The adapter's ``settle``, when it has one."""
        if "settle" in self._methods:
            self._call("settle")

    def supports(self, method: str) -> bool:
        """Whether the adapter implements an optional contract method."""
        return method in self._methods

    def close(self) -> None:
        """Close the adapter's stdin and wait for it; kill it if it lingers."""
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._log.close()


class ManagedServer:
    """A per-build server a memory needs (a container, a local binary).

    Started before the adapter and stopped after the build, on the build's
    fresh store directory; readiness is the adapter's own ``health()``, polled
    by the driver, so no network code runs in the harness.
    """

    def __init__(self, argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log: IO[str] = log.open("a", encoding="utf-8")
        self._proc = subprocess.Popen(  # noqa: S603 — argv from the operator's manifest
            argv, cwd=cwd, env=env, stdout=self._log, stderr=subprocess.STDOUT
        )

    def alive(self) -> bool:
        """Whether the server process is still running."""
        return self._proc.poll() is None

    def stop(self) -> None:
        """Terminate the server, then kill it if it does not exit."""
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._log.close()


def wait_healthy(memory: DoorMemory, *, wait_s: float, server: ManagedServer | None) -> None:
    """Poll ``memory.health()`` until it answers or ``wait_s`` passes.

    Raises:
        DoorAdapterError: When the memory never becomes healthy, or its server dies.
    """
    deadline = time.monotonic() + wait_s
    last: Exception | None = None
    while True:
        try:
            if memory.health():
                return
        except DoorAdapterError as exc:
            last = exc
        if server is not None and not server.alive():
            raise DoorAdapterError(f"{memory.name}: its server exited before it became healthy")
        if time.monotonic() >= deadline:
            raise DoorAdapterError(f"{memory.name}: not healthy after {wait_s:.0f} s ({last})")
        time.sleep(min(5.0, max(0.5, wait_s / 60)))


def parent_environ() -> dict[str, str]:
    """The parent's environment, as a plain dict (a seam for the tests)."""
    return dict(os.environ)
