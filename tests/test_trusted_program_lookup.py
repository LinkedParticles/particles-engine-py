# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Every program Particles starts is found on a trusted ``PATH`` (security review 2026-11-17-1).

An empty or relative ``PATH`` entry makes a lookup search the working
directory, which may be a cloned repository. The Codex doctor once looked
``codex`` up on the inherited ``PATH`` and ran it with the inherited
environment (RA2, RA9): the egress chokepoint test judges network reach only,
so a local program was never classified. This invariant walks every spawn and
every ``shutil.which`` in ``particles/`` instead.

A ``shutil.which`` call passes a ``path=`` that is an absolute literal or is
built from :func:`particles.url_safety.pinned_subprocess_env` (directly, or
through ``trusted_path``). A spawn passes when its program is an absolute
literal, ``sys.executable``, or a name the same function bound from such a
lookup, or when it passes an ``env=`` built from the pinned environment.
Anything else needs a row in :data:`_ALLOWLIST` with its reason, and the row's
count is exact, so a second spawn in a listed function fails here too.
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

import particles.url_safety

PKG = Path(__file__).resolve().parent.parent / "particles"

_SUBPROCESS_SPAWNS = frozenset({"run", "Popen", "call", "check_call", "check_output"})
_ASYNCIO_SPAWNS = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
_OS_SPAWN_PREFIXES = ("exec", "spawn", "posix_spawn")
_OS_SPAWNS = frozenset({"system", "popen"})

#: The function every pinned environment is built from.
_PRIMITIVE = "pinned_subprocess_env"
#: The ``url_safety`` helpers built on it (checked below, not trusted by name alone).
_HELPERS = frozenset({"trusted_path", "trusted_which", "isolated_git_env"})
_PINNING = frozenset({_PRIMITIVE, *_HELPERS})

#: Spawns the structural rule cannot read, keyed by (file, enclosing function)
#: → (exact count, the call every such spawn's ``env=`` must make, or ``None``
#: when a separate test pins it, and why the program is never looked up in a
#: directory a clone or another user controls). Editing this table is the
#: moment a reviewer classifies a new spawn.
_ALLOWLIST: dict[tuple[str, str], tuple[int, frozenset[str] | None, str]] = {
    ("api/cli/_claude_launcher.py", "start_worker"): (
        1,
        frozenset({"absolute_path_env"}),
        "stdlib-only launcher (imports nothing from Particles): the installed worker "
        "command runs in the private spool directory with PATH cut to absolute entries",
    ),
    ("api/cli/_codex_launcher.py", "handoff"): (
        2,
        frozenset({"absolute_path_env", "child_env"}),
        "stdlib-only launcher: main() refuses a relative installed command, and both "
        "children run with PATH cut to absolute entries (child_env builds on it too)",
    ),
    ("benchmark/rot/door_runner.py", "factory"): (
        1,
        frozenset({"child_env"}),
        "the operator's own benchmark manifest; env is contract.child_env's, whose PATH "
        "is trusted_path (test_rot_child_env_pins_path)",
    ),
    ("benchmark/rot/contract.py", "__init__"): (
        2,
        None,
        "the adapter and server processes of the operator's benchmark manifest; env is "
        "the caller's contract.child_env (test_rot_child_env_pins_path)",
    ),
}

#: Spawn and lookup names that, imported bare or under an alias, would hide a
#: call from the dotted-name scan below.
_HIDDEN_IMPORTS = {
    "subprocess": _SUBPROCESS_SPAWNS,
    "shutil": frozenset({"which"}),
    "os": _OS_SPAWNS,
    "asyncio": _ASYNCIO_SPAWNS,
}
#: Event-loop spawns, whose program is not their first argument: always classified by hand.
_LOOP_SPAWNS = frozenset({"subprocess_exec", "subprocess_shell"})


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _calls_pinning(node: ast.AST | None) -> bool:
    """Whether ``node`` contains a call to the pinned environment or a helper on it."""
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _dotted(sub.func).rsplit(".", 1)[-1]
            if name in _PINNING:
                return True
    return False


def _is_which(call: ast.Call) -> bool:
    return _dotted(call.func) == "shutil.which"


def _which_path_is_trusted(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "path":
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return all(os.path.isabs(p) for p in value.value.split(os.pathsep))
            return _calls_pinning(value)
    return False


def _spawn_kind(call: ast.Call) -> str | None:
    name = _dotted(call.func)
    head, _, attr = name.rpartition(".")
    if head == "subprocess" and attr in _SUBPROCESS_SPAWNS:
        return "subprocess"
    if attr in _ASYNCIO_SPAWNS:
        return "asyncio"
    if head == "os" and (attr in _OS_SPAWNS or attr.startswith(_OS_SPAWN_PREFIXES)):
        return "os"
    if attr in _LOOP_SPAWNS:
        return "os"
    return None


class _Function:
    """The assignments of one function body, for following a name one step."""

    def __init__(self, func: ast.AST) -> None:
        self.assigned: dict[str, list[ast.expr]] = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._record(target, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._record(node.target, node.value)

    def _record(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.assigned.setdefault(target.id, []).append(value)
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "PATH"
        ):
            # ``env["PATH"] = <pinned>`` pins the name's PATH whatever it was built from.
            self.assigned.setdefault(f"{target.value.id}[PATH]", []).append(value)


def _head(call: ast.Call, kind: str, scope: _Function) -> ast.expr | None:
    if not call.args:
        return None
    first = call.args[0]
    if kind == "asyncio" or kind == "os":
        return first
    if isinstance(first, ast.Name) and len(scope.assigned.get(first.id, [])) == 1:
        first = scope.assigned[first.id][0]
    if isinstance(first, ast.List) and first.elts:
        return first.elts[0]
    return first


def _head_is_trusted(head: ast.expr | None, scope: _Function) -> bool:
    if isinstance(head, ast.Constant) and isinstance(head.value, str):
        return os.path.isabs(head.value)
    if head is not None and _dotted(head) == "sys.executable":
        return True
    if isinstance(head, ast.Name):
        bound = scope.assigned.get(head.id, [])
        return bool(bound) and all(
            isinstance(v, ast.Call)
            and (
                (_is_which(v) and _which_path_is_trusted(v))
                or _dotted(v.func).rsplit(".", 1)[-1] == "trusted_which"
            )
            for v in bound
        )
    return False


def _env_is_pinned(call: ast.Call, scope: _Function) -> bool:
    for kw in call.keywords:
        if kw.arg != "env":
            continue
        if _calls_pinning(kw.value):
            return True
        if isinstance(kw.value, ast.Name):
            name = kw.value.id
            if any(_calls_pinning(v) for v in scope.assigned.get(f"{name}[PATH]", [])):
                return True
            bound = scope.assigned.get(name, [])
            return bool(bound) and all(_calls_pinning(v) for v in bound)
    return False


def _env_calls(call: ast.Call, names: frozenset[str], func: ast.AST) -> bool:
    """Whether ``call``'s ``env=`` is built by one of ``names`` on every branch.

    A bare name counts when the enclosing function calls one of ``names``.
    """

    def calls(node: ast.AST) -> bool:
        return any(
            isinstance(sub, ast.Call) and _dotted(sub.func).rsplit(".", 1)[-1] in names
            for sub in ast.walk(node)
        )

    for kw in call.keywords:
        if kw.arg != "env":
            continue
        if isinstance(kw.value, ast.Name):
            return calls(func)
        if isinstance(kw.value, ast.IfExp):
            return calls(kw.value.body) and calls(kw.value.orelse)
        return calls(kw.value)
    return False


def _enclosing_functions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((node.name, node))
    return found


def _scan() -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
    """``(bad which calls, unclassified spawns, allowlisted spawn counts)``."""
    bad_which: list[str] = []
    unclassified: list[str] = []
    allowed: dict[tuple[str, str], int] = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Innermost function for each call: ast.walk is breadth-first, so a
        # nested def comes after the one enclosing it and overwrites it.
        owner: dict[int, tuple[str, ast.AST]] = {}
        for name, func in _enclosing_functions(tree):
            for sub in ast.walk(func):
                if isinstance(sub, ast.Call):
                    owner[id(sub)] = (name, func)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name, func = owner.get(id(node), ("<module>", tree))
            where = f"{rel}:{node.lineno} ({func_name})"
            if _is_which(node) and not _which_path_is_trusted(node):
                bad_which.append(where)
            kind = _spawn_kind(node)
            if kind is None:
                continue
            scope = _Function(func)
            if kind != "os" and (
                _head_is_trusted(_head(node, kind, scope), scope) or _env_is_pinned(node, scope)
            ):
                continue
            key = (rel, func_name)
            if key in _ALLOWLIST:
                allowed[key] = allowed.get(key, 0) + 1
                required = _ALLOWLIST[key][1]
                if required is not None and not _env_calls(node, required, func):
                    unclassified.append(f"{where}: its env= calls none of {sorted(required)}")
            else:
                unclassified.append(where)
    return bad_which, unclassified, allowed


def test_every_which_searches_a_trusted_path() -> None:
    bad_which, _, _ = _scan()
    assert not bad_which, (
        "shutil.which without a trusted path= (an absolute literal, or built from "
        "url_safety.pinned_subprocess_env; use url_safety.trusted_which): " + ", ".join(bad_which)
    )


def test_every_spawn_finds_its_program_on_a_trusted_path() -> None:
    _, unclassified, allowed = _scan()
    assert not unclassified, (
        "a spawn whose program is not an absolute literal and that passes no env= built "
        "from url_safety.pinned_subprocess_env; pin it, or classify it in _ALLOWLIST: "
        + ", ".join(unclassified)
    )
    expected = {key: count for key, (count, _, _) in _ALLOWLIST.items()}
    assert allowed == expected, "an _ALLOWLIST row's count no longer matches the code"


def test_no_spawn_or_lookup_is_imported_under_another_name() -> None:
    """The scan reads dotted names, so ``from subprocess import run`` would hide a spawn."""
    hidden: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module in _HIDDEN_IMPORTS:
                names = _HIDDEN_IMPORTS[node.module]
                hidden += [
                    f"{rel}:{node.lineno} from {node.module} import {a.name}"
                    for a in node.names
                    if a.name in names
                    or a.name == "*"
                    or (node.module == "os" and a.name.startswith(_OS_SPAWN_PREFIXES))
                ]
            elif isinstance(node, ast.Import):
                hidden += [
                    f"{rel}:{node.lineno} import {a.name} as {a.asname}"
                    for a in node.names
                    if a.name in _HIDDEN_IMPORTS and a.asname not in (None, a.name)
                ]
    assert not hidden, "import the module and call it dotted, so the scan sees it: " + ", ".join(
        hidden
    )


def test_the_pinning_helpers_are_built_on_the_pinned_environment() -> None:
    """A helper the scan trusts by name must itself call the primitive (or another helper)."""
    # Read the module where it is installed: in the split engine tree it ships
    # with the core distribution, not under this tree's `particles/`.
    source = Path(particles.url_safety.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
    }
    assert set(defs) == set(_HELPERS)
    for name, node in defs.items():
        assert _calls_pinning(node) and any(
            isinstance(sub, ast.Call)
            and _dotted(sub.func).rsplit(".", 1)[-1] in (_PINNING - {name})
            for sub in ast.walk(node)
        ), f"url_safety.{name} no longer builds on {_PRIMITIVE}"


def test_the_launchers_drop_relative_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from particles.api.cli import _claude_launcher, _codex_launcher

    monkeypatch.setenv("PATH", f".{os.pathsep}{os.pathsep}bin{os.pathsep}/usr/bin")
    assert _codex_launcher.child_env("/tmp/token")["PATH"] == "/usr/bin"
    assert _codex_launcher.absolute_path_env()["PATH"] == "/usr/bin"
    assert _claude_launcher.absolute_path_env()["PATH"] == "/usr/bin"


def test_rot_child_env_pins_path(tmp_path: Path) -> None:
    from particles.benchmark.rot.contract import child_env

    env, _ = child_env({"PATH": f".{os.pathsep}{os.pathsep}/usr/bin"}, {}, store=tmp_path)
    assert all(os.path.isabs(p) for p in env["PATH"].split(os.pathsep))
    env, record = child_env({"PATH": "/usr/bin"}, {"PATH": "rel"}, store=tmp_path)
    assert "rel" not in env["PATH"].split(os.pathsep)
    assert record["PATH"] == env["PATH"], "the published record names the PATH the child gets"


def test_the_scan_catches_the_codex_doctor_shape() -> None:
    """The shape RA2 shipped fails both rules."""
    source = (
        "import shutil, subprocess\n"
        "def doctor():\n"
        "    exe = shutil.which('codex')\n"
        "    subprocess.run([exe, '--version'])\n"
        "def runner(argv):\n"
        "    return subprocess.run(argv, capture_output=True)\n"
    )
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    which = next(c for c in calls if _is_which(c))
    assert not _which_path_is_trusted(which)
    for name, func in _enclosing_functions(tree):
        scope = _Function(func)
        for call in (n for n in ast.walk(func) if isinstance(n, ast.Call)):
            kind = _spawn_kind(call)
            if kind is not None:
                assert not _head_is_trusted(_head(call, kind, scope), scope), name
                assert not _env_is_pinned(call, scope), name


# ---------------------------------------------------------------------------
# The audit's failing proofs (RA2, RA9), ported
# ---------------------------------------------------------------------------


def _plant_codex(directory: Path, marker: Path) -> None:
    fake = directory / "codex"
    fake.write_text(f"#!/bin/sh\n: > '{marker}'\necho fake\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize("entry", [".", ""], ids=["relative-entry", "empty-entry"])
def test_codex_doctor_never_runs_a_program_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    import asyncio
    import contextlib

    from particles.api.cli.codex import codex_doctor
    from particles.config import reset_config

    clone = tmp_path / "clone"
    clone.mkdir()
    marker = tmp_path / "ran"
    _plant_codex(clone, marker)
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    monkeypatch.chdir(clone)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("PARTICLES_CODEX_STATE_DIR", str(tmp_path / "codex-state"))
    # A `PATH=.:$PATH` profile, or a stray leading colon.
    monkeypatch.setenv("PATH", entry + os.pathsep + "/usr/bin" + os.pathsep + "/bin")
    reset_config()
    try:
        with contextlib.suppress(Exception):
            asyncio.run(codex_doctor("default"))
    finally:
        reset_config()
    assert not marker.exists(), "codex_doctor ran the `codex` the working directory holds"


def test_run_command_runs_with_the_trusted_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A program run_command starts finds its own children on the trusted PATH only."""
    from particles.api.cli._codex_schedule import run_command

    marker = tmp_path / "ran"
    _plant_codex(tmp_path, marker)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f".{os.pathsep}/usr/bin{os.pathsep}/bin")
    done = run_command(["/bin/sh", "-c", 'codex; echo "$PATH"'])
    assert not marker.exists()
    assert "." not in done.stdout.strip().split(os.pathsep)
