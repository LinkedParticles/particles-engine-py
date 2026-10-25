# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Egress choke-point regression + enforcement (security finding F14).

Every importer / extractor that fetches external bytes into the corpus must go
through :func:`particles.http.get_capped` (streaming + a decompressed-body size
cap) rather than a raw ``client.get(url)`` (which httpx buffers, post-
decompression, with no cap — a compression-bomb / self-DoS).

Two layers of test:

1. **Behavioural** — a bomb-shaped response (small/absent ``Content-Length``,
   large decompressed body) on a real importer path raises
   :class:`~particles.http.ResponseTooLarge` instead of buffering unbounded;
   and Nomisma fetches its JSON-LD over ``https://`` (the one MITM-reachable
   leg F14 flagged).

2. **Structural / CI tripwire** — a source-grep that fails when a *new* raw
   ``client.<verb>(`` fetch site appears outside the audited allow-list, so the
   guarantee is enforced rather than relying on author discipline. This is the
   defense-in-depth answer to "no CI enforcement of the egress choke-point".

3. **Subprocess-egress inventory ratchet** — the same
   enforcement idea for the fetches that never enter ``httpx`` at all. An AST
   walk finds every ``create_subprocess_exec`` / ``subprocess.*`` call site
   whose argv head is a network-capable binary and fails unless it appears in
   an allow-list naming, per entry, which mechanism covers it. An import-linter
   contract cannot see a subprocess — nothing is imported — so the check has to
   read argv, which makes it a test.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import particles
from particles.http import ResponseTooLarge
from particles.ingest.importers.nomisma import NomismaImporter
from tests._capped_http import set_capped_responses

# A minimal valid Nomisma JSON-LD document (one concept node).
_NOMISMA_DOC: dict = {
    "@context": {"nm": "http://nomisma.org/id/", "skos": "http://www.w3.org/2004/02/skos/core#"},
    "@graph": [
        {
            "@id": "nm:al",
            "@type": ["skos:Concept"],
            "skos:prefLabel": [{"@language": "en", "@value": "Aluminum"}],
        }
    ],
}


def _client_returning(resp: MagicMock) -> MagicMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    set_capped_responses(client, return_value=resp)
    return client


# ---------------------------------------------------------------------------
# Behavioural regression
# ---------------------------------------------------------------------------


class TestCompressionBombRejected:
    @pytest.mark.asyncio
    async def test_bomb_on_importer_path_raises_response_too_large(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A small-compressed / large-decompressed body trips the get_capped cap.

        Models a compression bomb: no ``Content-Length`` over the cap (the
        up-front fast-path can't catch it), but the streamed decompressed body
        crosses ``config.http.max_bytes`` mid-stream. Without get_capped the
        importer would buffer the whole expansion into memory and a blob.
        """
        # Shrink the cap to 64 bytes (get_capped reads it at call time).
        cfg = MagicMock()
        cfg.http.max_bytes = 64
        monkeypatch.setattr("particles.config.get_config", lambda: cfg)

        bomb = MagicMock()
        bomb.status_code = 200
        bomb.content = b"x" * 4096  # decompressed body, far over the 64-byte cap
        bomb.raise_for_status = MagicMock()
        mock_client = _client_returning(bomb)

        with (
            patch("particles.http.particles_client", return_value=mock_client),
            pytest.raises(ResponseTooLarge),
        ):
            await NomismaImporter().deposit(
                AsyncMock(), "http://nomisma.org/id/al", "test-operator", []
            )

    @pytest.mark.asyncio
    async def test_oversize_content_length_rejected_before_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared Content-Length over the cap is rejected up front."""
        cfg = MagicMock()
        cfg.http.max_bytes = 64
        monkeypatch.setattr("particles.config.get_config", lambda: cfg)

        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"x" * 8  # body itself is tiny...
        resp.headers = {"content-length": "999999"}  # ...but the header lies big
        resp.raise_for_status = MagicMock()
        mock_client = _client_returning(resp)

        with (
            patch("particles.http.particles_client", return_value=mock_client),
            pytest.raises(ResponseTooLarge, match="Content-Length"),
        ):
            await NomismaImporter().deposit(
                AsyncMock(), "http://nomisma.org/id/al", "test-operator", []
            )


class TestNomismaHttps:
    @pytest.mark.asyncio
    async def test_jsonld_fetched_over_https(self) -> None:
        """Nomisma's JSON-LD document URI is fetched over https:// (F14)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.content = json.dumps(_NOMISMA_DOC).encode()
        resp.raise_for_status = MagicMock()
        mock_client = _client_returning(resp)

        with (
            patch("particles.http.particles_client", return_value=mock_client),
            patch(
                "particles.corpus.deposit.write_entry_and_snapshot",
                new_callable=AsyncMock,
                return_value=("entry-1", "snap-1"),
            ),
            patch("particles.corpus.deposit.sha256", return_value="hash123"),
            patch("particles.corpus.deposit.save_blob", return_value="/tmp/hash123"),
        ):
            await NomismaImporter().deposit(
                AsyncMock(), "http://nomisma.org/id/al", "test-operator", []
            )

        # get_capped opens client.stream("GET", url, ...) — assert the fetched
        # URL is the https JSON-LD document URI, not cleartext http://.
        assert mock_client.stream.call_count == 1
        fetched_url = mock_client.stream.call_args.args[1]
        assert fetched_url == "https://nomisma.org/id/al.jsonld"
        assert not fetched_url.startswith("http://")


# ---------------------------------------------------------------------------
# Structural enforcement — the CI tripwire
# ---------------------------------------------------------------------------

# Raw httpx client method calls. The choke point is the *call*, not the
# response read (``.json()`` / ``.content`` on a get_capped response is fine).
_RAW_CLIENT_CALL = re.compile(r"\bclient\.(get|post|put|patch|delete|stream|request|send)\(")

_PARTICLES_ROOT = Path(particles.__file__).resolve().parent

# Files permitted to issue a raw ``client.<verb>(`` — each audited:
#   http.py             — THE choke point itself (get_capped streams via
#                         client.stream; get_with_retry now delegates to it).
#   llm/adapters/openai_compat.py — completion POST to an operator-configured
#                         endpoint (loopback or hosted), not a
#                         corpus fetch; its short-lived client comes from
#                         particles.http.egress_client.
#   exporters/notion.py — outbound writes to the Notion API (a trusted export
#                         target), not untrusted-content ingestion.
#   api/client/http.py  — the Python SDK calling *our own* engine, not the web.
#   benchmark/memory/loader.py — streams the ~GB-scale pinned LongMemEval file
#                         to disk (too large for get_capped's
#                         buffered body cap). Not untrusted-content ingestion:
#                         the URL is revision-pinned, the payload is
#                         SHA-256-verified before anything parses or caches it,
#                         and the call still rides particles_client's
#                         SSRF-validating transport.
_ALLOWLISTED_RAW_CLIENT_FILES = {
    "http.py",
    "llm/adapters/openai_compat.py",
    "exporters/notion.py",
    "api/client/http.py",
    "benchmark/memory/loader.py",
}


_CLIENT_VERBS = frozenset({"get", "post", "put", "patch", "delete", "stream", "request", "send"})

# Callables that hand back an httpx client: the two choke-point factories and
# httpx's own constructors (which the construction ratchet below confines to
# particles/http.py).
_CLIENT_FACTORIES = frozenset({"particles_client", "egress_client", "AsyncClient", "Client"})


def _callee_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_client_annotation(node: ast.expr | None) -> bool:
    if node is None:
        return False
    text = ast.unparse(node)
    return "AsyncClient" in text or "httpx.Client" in text


def _client_names(tree: ast.Module) -> set[str]:
    """Names a module binds to an httpx client, whatever they are called (D44).

    The literal ``client.`` regex misses ``async with egress_client(…) as api:``
    followed by ``api.post(…)``, and a parameter annotated ``httpx.AsyncClient``
    under any name. Both are collected here.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                expr = item.context_expr
                if (
                    isinstance(expr, ast.Call)
                    and _callee_name(expr.func) in _CLIENT_FACTORIES
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    names.add(item.optional_vars.id)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _callee_name(node.value.func) in _CLIENT_FACTORIES:
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if _is_client_annotation(arg.annotation):
                    names.add(arg.arg)
    return names


def _aliased_client_call_lines(tree: ast.Module) -> list[int]:
    """Lines calling a client verb on a name bound to a client, or on ``self.*client``."""
    names = _client_names(tree)
    lines = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _CLIENT_VERBS:
            continue
        receiver = node.func.value
        if (isinstance(receiver, ast.Name) and receiver.id in names) or (
            isinstance(receiver, ast.Attribute) and receiver.attr.lower().endswith("client")
        ):
            lines.append(node.lineno)
    return lines


def _files_with_raw_client_calls(subdirs: list[str] | None = None) -> dict[str, list[int]]:
    roots = [_PARTICLES_ROOT / d for d in subdirs] if subdirs else [_PARTICLES_ROOT]
    hits: dict[str, list[int]] = {}
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(_PARTICLES_ROOT).as_posix()
            text = path.read_text()
            lines = {
                i
                for i, line in enumerate(text.splitlines(), start=1)
                if _RAW_CLIENT_CALL.search(line)
            }
            lines.update(_aliased_client_call_lines(ast.parse(text)))
            if lines:
                hits[rel] = sorted(lines)
    return hits


class TestEgressChokePointEnforced:
    def test_no_raw_client_calls_in_ingestion_surface(self) -> None:
        """Importers and extractors must fetch only through get_capped.

        The ingestion surface (``ingest/`` + ``extraction/``) has an EMPTY
        allow-list: a new domain importer/extractor that reaches for a raw
        ``client.get(url)`` fails here, steering it to the capped, SSRF-guarded
        choke point automatically.
        """
        offenders = _files_with_raw_client_calls(["ingest", "extraction"])
        assert offenders == {}, (
            "Raw httpx client calls found in the ingestion surface — route them "
            f"through particles.http.get_capped instead:\n{offenders}"
        )

    def test_raw_client_calls_confined_to_allowlist(self) -> None:
        """Any raw ``client.<verb>(`` anywhere in particles/ is audited.

        A NEW raw fetch site outside the audited allow-list fails CI: either
        route it through get_capped, or consciously add the file here with a
        one-line rationale (and confirm it is not untrusted-content ingestion).
        """
        offenders = {
            f: lines
            for f, lines in _files_with_raw_client_calls().items()
            if f not in _ALLOWLISTED_RAW_CLIENT_FILES
        }
        assert offenders == {}, (
            "Unaudited raw httpx client call(s) outside the egress choke point. "
            "Route through particles.http.get_capped, or add to "
            f"_ALLOWLISTED_RAW_CLIENT_FILES with a rationale:\n{offenders}"
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        """Every allow-listed file still actually issues a raw client call.

        Keeps the allow-list honest: if a refactor moves a call onto get_capped,
        the stale exemption must be removed rather than silently widening the
        permitted surface.
        """
        present = set(_files_with_raw_client_calls().keys())
        stale = _ALLOWLISTED_RAW_CLIENT_FILES - present
        assert stale == set(), f"Allow-list entries no longer issue raw client calls: {stale}"

    def test_detector_follows_a_client_under_any_name(self) -> None:
        """D44: the ratchet reads the binding, not the literal spelling ``client.``."""
        source = textwrap.dedent(
            """
            async def a():
                async with egress_client(purpose="x") as api:
                    await api.post("https://example.invalid")

            async def b(http: httpx.AsyncClient) -> None:
                await http.get("https://example.invalid")

            class C:
                async def c(self) -> None:
                    await self._http_client.send(request)

            async def d(resp):
                resp.json()  # a response read is not a call
            """
        )
        assert sorted(_aliased_client_call_lines(ast.parse(source))) == [4, 7, 11]


# ---------------------------------------------------------------------------
# Subprocess-egress inventory ratchet
# ---------------------------------------------------------------------------

# Binaries that can open a network connection. A subprocess spawning one of
# these is egress until proven otherwise, and must be classified below.
_NETWORK_BINARIES = frozenset({"curl", "git", "wget", "scp", "ssh", "nc", "rsync"})

# Spawn seams. ``asyncio.create_subprocess_*`` is matched on the attribute name
# alone (the module may be imported under any alias); ``subprocess.*`` requires
# the literal ``subprocess`` receiver so an unrelated ``foo.run(...)`` is not a
# false positive.
_ASYNCIO_SPAWNS = frozenset({"create_subprocess_exec", "create_subprocess_shell"})
_SUBPROCESS_SPAWNS = frozenset({"run", "Popen", "call", "check_call", "check_output"})

# argv head we could not read statically (``*cmd``, a variable, an f-string).
# Reported as its own binary so such a site still has to be classified — an
# unreadable argv is exactly where an unpinned fetcher would hide.
_DYNAMIC = "<dynamic>"

# Every network-capable subprocess in particles/, keyed by
# (file, enclosing function, argv head) → (call count, covering mechanism).
#
# The count is part of the key's value on purpose: adding a *second* git call
# to a function that already has one must fail here rather than inherit the
# existing entry's blessing.
#
# Editing this table is the moment a reviewer is forced to classify a new
# egress path — either it connects to an address this process resolved and
# vetted (``particles.url_safety.resolve_and_pin``), or it falls in one of
# 1's three exemption classes.
_SUBPROCESS_EGRESS_ALLOWLIST: dict[tuple[str, str, str], tuple[int, str]] = {
    ("extraction/reddit.py", "_fetch_with_curl", "curl"): (
        1,
        "resolve_and_pin → curl --resolve",
    ),
    ("extraction/reddit.py", "_resolve_reddit_redirect", "curl"): (
        1,
        "resolve_and_pin → curl --resolve, per hop (§2.3/§2.5)",
    ),
    ("ingest/importers/github.py", "_clone_gist_files", "git"): (
        2,
        "clone: resolve_and_pin → git -c http.curloptResolve; "
        "log: exemption class 3, local-only (no remote)",
    ),
    ("api/cli/_projection_git.py", "_git", _DYNAMIC): (
        1,
        "exemption class 3, local-only: rev-parse / add / diff / "
        "commit against a local working tree; adds no remote and pushes nothing",
    ),
    ("api/cli/subjects.py", "_checkout_files", "git"): (
        1,
        "exemption class 3, local-only: `git ls-files` over the "
        "checkout the operator names for `subjects fold-artifacts`; never fetches",
    ),
    ("api/cli/hook.py", "_git", "git"): (
        1,
        "exemption class 3, local-only: the doctor's rev-parse / "
        "describe / tag --list over the install's own checkout; never fetches",
    ),
    ("api/cli/init.py", "_claude_mcp", _DYNAMIC): (
        1,
        "exemption class 3, local-only: `claude mcp add` / `remove` "
        "edits the user's own Claude Code registration file; it "
        "starts no server and fetches nothing",
    ),
    ("benchmark/update_gate/case.py", "run_one", _DYNAMIC): (
        1,
        "exemption class 3, local-only: starts this interpreter "
        "(sys.executable -m the SDK's own child entry) for one isolated benchmark "
        "run; the child's own calls go through the SDK's checked "
        "transports under the operator's network.egress, copied into its config",
    ),
    ("benchmark/query_latency/runner.py", "_run_child", _DYNAMIC): (
        1,
        "exemption class 3, local-only: starts this interpreter "
        "(sys.executable -m the SDK's own cell module) for one benchmark cell; the "
        "cell reads a local SQLite file and loads the encoder through the SDK",
    ),
    ("benchmark/query_latency/cell.py", "_cmd", _DYNAMIC): (
        1,
        "exemption class 3, local-only: `git rev-parse HEAD` over "
        "the package's own checkout and `sysctl -n` hardware reads, for the "
        "result's reproduction tuple; never fetches",
    ),
}


# ``os`` / ``pty`` spawns, keyed to the position of the program argument
# (``os.system`` and ``os.popen`` take a shell string, whose first word is
# read; ``os.spawn*`` takes a mode first).
_OS_SPAWNS: dict[str, int] = {
    "system": 0,
    "popen": 0,
    **{name: 0 for name in ("execl", "execle", "execlp", "execlpe")},
    **{name: 0 for name in ("execv", "execve", "execvp", "execvpe")},
    **{name: 1 for name in ("spawnl", "spawnle", "spawnlp", "spawnlpe")},
    **{name: 1 for name in ("spawnv", "spawnve", "spawnvp", "spawnvpe")},
    "posix_spawn": 0,
    "posix_spawnp": 0,
}


class _SpawnImports:
    """How a module spells the spawn seams, so an alias cannot hide one (D44).

    ``import subprocess as sp`` / ``from subprocess import run`` /
    ``from asyncio import create_subprocess_exec`` / ``import os as _os`` /
    ``from os import system`` are all followed; a literal ``subprocess.``
    receiver is no longer required.
    """

    def __init__(self, tree: ast.Module) -> None:
        self.subprocess_modules = {"subprocess"}
        self.os_modules = {"os", "pty"}
        # local name -> (seam family, attribute)
        self.direct: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name == "subprocess":
                        self.subprocess_modules.add(local)
                    elif alias.name in ("os", "pty"):
                        self.os_modules.add(local)
            elif isinstance(node, ast.ImportFrom) and node.module in (
                "subprocess",
                "asyncio",
                "os",
                "pty",
            ):
                for alias in node.names:
                    self.direct[alias.asname or alias.name] = (node.module, alias.name)

    def seam(self, func: ast.expr) -> tuple[str, str] | None:
        """``(family, attribute)`` when ``func`` names a spawn seam, else None."""
        if isinstance(func, ast.Attribute):
            if func.attr in _ASYNCIO_SPAWNS:
                return ("asyncio", func.attr)
            if isinstance(func.value, ast.Name):
                if func.value.id in self.subprocess_modules and func.attr in _SUBPROCESS_SPAWNS:
                    return ("subprocess", func.attr)
                if func.value.id in self.os_modules and (
                    func.attr in _OS_SPAWNS or func.attr == "spawn"
                ):
                    return ("os", func.attr)
            return None
        if isinstance(func, ast.Name) and func.id in self.direct:
            family, attr = self.direct[func.id]
            if family == "asyncio" and attr in _ASYNCIO_SPAWNS:
                return ("asyncio", attr)
            if family == "subprocess" and attr in _SUBPROCESS_SPAWNS:
                return ("subprocess", attr)
            if family in ("os", "pty") and (attr in _OS_SPAWNS or attr == "spawn"):
                return ("os", attr)
        return None


def _literal_program(node: ast.expr, *, shell_string: bool) -> str:
    if isinstance(node, ast.List | ast.Tuple):
        node = node.elts[0] if node.elts else node
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value.split()[0] if shell_string and node.value.split() else node.value
        return Path(text).name
    return _DYNAMIC


def _spawn_binary(call: ast.Call, imports: _SpawnImports | None = None) -> str | None:
    """Return the argv head of a subprocess spawn, or None if not a spawn call."""
    seam = (imports or _SpawnImports(ast.Module(body=[], type_ignores=[]))).seam(call.func)
    if seam is None:
        return None
    family, attr = seam
    position = _OS_SPAWNS.get(attr, 0) if family == "os" else 0
    if len(call.args) <= position:
        return _DYNAMIC
    return _literal_program(
        call.args[position], shell_string=family == "os" and attr in ("system", "popen")
    )


class _SpawnVisitor(ast.NodeVisitor):
    """Collect (enclosing function, argv head) for every subprocess spawn."""

    def __init__(self, imports: _SpawnImports | None = None) -> None:
        self.imports = imports
        self.stack: list[str] = []
        self.found: list[tuple[str, str]] = []

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_func  # noqa: N815
    visit_AsyncFunctionDef = _visit_func  # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        binary = _spawn_binary(node, self.imports)
        if binary is not None:
            self.found.append((self.stack[-1] if self.stack else "<module>", binary))
        self.generic_visit(node)


def _spawn_sites_in(tree: ast.Module) -> list[tuple[str, str]]:
    visitor = _SpawnVisitor(_SpawnImports(tree))
    visitor.visit(tree)
    return visitor.found


def _network_subprocess_sites() -> dict[tuple[str, str, str], int]:
    """Every network-capable subprocess spawn in particles/, with its call count."""
    sites: dict[tuple[str, str, str], int] = {}
    for path in _PARTICLES_ROOT.rglob("*.py"):
        for func_name, binary in _spawn_sites_in(ast.parse(path.read_text())):
            if binary not in _NETWORK_BINARIES and binary != _DYNAMIC:
                continue
            key = (path.relative_to(_PARTICLES_ROOT).as_posix(), func_name, binary)
            sites[key] = sites.get(key, 0) + 1
    return sites


def _git_spawning_files() -> set[str]:
    """Files that spawn git: a literal ``git`` argv head, or a dynamic spawn beside a ``"git"``."""
    files: set[str] = set()
    for path in _PARTICLES_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        heads = {binary for _func, binary in _spawn_sites_in(tree)}
        names_git = any(
            isinstance(node, ast.Constant) and node.value == "git" for node in ast.walk(tree)
        )
        if "git" in heads or (_DYNAMIC in heads and names_git):
            files.add(path.relative_to(_PARTICLES_ROOT).as_posix())
    return files


class TestSubprocessEgressInventory:
    """a new unpinned subprocess fetcher must fail a gate."""

    def test_every_network_subprocess_is_classified(self) -> None:
        discovered = _network_subprocess_sites()
        expected = {k: v[0] for k, v in _SUBPROCESS_EGRESS_ALLOWLIST.items()}

        unclassified = {k: n for k, n in discovered.items() if k not in expected}
        assert unclassified == {}, (
            "Unclassified network-capable subprocess call site(s). Every outbound "
            "fetch whose target host derives from outside this process MUST connect "
            "to an address this process resolved and passed through _is_blocked_ip "
            ". Pin it with particles.url_safety.resolve_and_pin — "
            "`curl --resolve` or `git -c http.curloptResolve` — then add it to "
            "_SUBPROCESS_EGRESS_ALLOWLIST naming the mechanism. If it is genuinely "
            "exempt (operator-configured endpoint / compiled-in vendor endpoint / "
            f"local-only subprocess), record which class and why:\n{unclassified}"
        )

        drifted = {k: (n, expected[k]) for k, n in discovered.items() if expected[k] != n}
        assert drifted == {}, (
            "Subprocess call count changed at an already-classified site "
            "{found, allowlisted}. A new spawn in an existing function does not "
            "inherit that entry's classification — classify it, then update the "
            f"count:\n{drifted}"
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        """Every allow-listed site still exists — the exemption table stays honest."""
        stale = set(_SUBPROCESS_EGRESS_ALLOWLIST) - set(_network_subprocess_sites())
        assert stale == set(), f"Allow-list entries no longer match a subprocess call site: {stale}"

    def test_curl_and_git_egress_paths_are_pinned(self) -> None:
        """The two real egress paths name resolve_and_pin, not just an exemption.

        Guards the allow-list against being "maintained" by relabelling a pinned
        path as exempt — the classification must keep matching the code.
        """
        for key in (
            ("extraction/reddit.py", "_fetch_with_curl", "curl"),
            ("extraction/reddit.py", "_resolve_reddit_redirect", "curl"),
            ("ingest/importers/github.py", "_clone_gist_files", "git"),
        ):
            assert "resolve_and_pin" in _SUBPROCESS_EGRESS_ALLOWLIST[key][1]
            source = (_PARTICLES_ROOT / key[0]).read_text()
            assert "resolve_and_pin" in source, f"{key[0]} no longer pins its egress"

    def test_detector_follows_spawn_aliases(self) -> None:
        """D44: an aliased module, a from-import or an ``os`` spawn is still a spawn."""
        source = textwrap.dedent(
            """
            import os as _os
            import subprocess as sp
            from asyncio import create_subprocess_exec as spawn
            from subprocess import Popen

            def a():
                sp.run(["git", "status"])
                Popen(["curl", "https://example.invalid"])
                _os.system("wget https://example.invalid")
                _os.execvp("ssh", ["ssh", "host"])
                _os.spawnlp(_os.P_WAIT, "rsync", "rsync")

            async def b(cmd):
                await spawn(*cmd)
            """
        )
        assert _spawn_sites_in(ast.parse(source)) == [
            ("a", "git"),
            ("a", "curl"),
            ("a", "wget"),
            ("a", "ssh"),
            ("a", "rsync"),
            ("b", _DYNAMIC),
        ]

    def test_every_git_spawn_runs_no_program_a_config_names(self) -> None:
        """N13 / D22: a git child must not run a filter, fsmonitor or driver it is handed.

        The allow-list above classifies the git process; it cannot see the
        programs git itself starts from a repository's or the user's config. A
        local-only site passes ``url_safety.GIT_LOCAL_ARGS``; a site over a
        fetched repository runs under ``url_safety.isolated_git_env``.
        """
        files = _git_spawning_files()
        assert "ingest/importers/github.py" in files
        unhardened = sorted(
            rel
            for rel in files
            if not {"GIT_LOCAL_ARGS", "isolated_git_env"}
            & set(re.findall(r"\w+", (_PARTICLES_ROOT / rel).read_text()))
        )
        assert unhardened == [], (
            f"git subprocess without url_safety.GIT_LOCAL_ARGS or isolated_git_env: {unhardened}"
        )


# ---------------------------------------------------------------------------
# Client-construction ratchet (item 4)
# ---------------------------------------------------------------------------


def _client_constructions() -> list[tuple[str, int, str, ast.Call]]:
    """Every ``httpx.AsyncClient(`` / ``httpx.Client(`` / ``anthropic.Anthropic(`` call."""
    found: list[tuple[str, int, str, ast.Call]] = []
    for path in _PARTICLES_ROOT.rglob("*.py"):
        rel = path.relative_to(_PARTICLES_ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if not isinstance(receiver, ast.Name):
                continue
            name = f"{receiver.id}.{node.func.attr}"
            if name in {"httpx.AsyncClient", "httpx.Client", "anthropic.Anthropic"}:
                found.append((rel, node.lineno, name, node))
    return found


class TestClientConstructionRatchet:
    def test_httpx_clients_are_built_only_in_the_chokepoint(self) -> None:
        """Every outbound httpx client carries the no-egress check.

        ``particles_client`` (the SSRF transport) and ``egress_client`` (the
        egress-only transport) both live in ``particles/http.py``. A client built
        anywhere else would reach the network with neither.
        """
        offenders = [
            f"{rel}:{line} {name}("
            for rel, line, name, _ in _client_constructions()
            if name.startswith("httpx.") and rel != "http.py"
        ]
        assert offenders == [], (
            "httpx client constructed outside particles/http.py; use "
            f"particles_client() or egress_client() instead: {offenders}"
        )

    def test_anthropic_client_carries_the_egress_hook(self) -> None:
        """The SDK runs on httpx2, so the check rides its own client's request hook."""
        calls = [c for c in _client_constructions() if c[2] == "anthropic.Anthropic"]
        assert calls, "expected llm/client.py to build the Anthropic client"
        for rel, line, _name, node in calls:
            keywords = {kw.arg for kw in node.keywords}
            assert "http_client" in keywords, (
                f"{rel}:{line} builds anthropic.Anthropic without http_client=; pass "
                "anthropic.DefaultHttpxClient(event_hooks={'request': [egress_request_hook(...)]})"
            )


# ---------------------------------------------------------------------------
# The programs git starts by itself (N13, SECURITY_REVIEW_20261024-1; D22,
# SECURITY_REVIEW_20261023-1). Real git over local repositories only.
# ---------------------------------------------------------------------------


def _run_git(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, env=env)


def _marker_script(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """A script that touches a marker file (and passes stdin through, as a filter does)."""
    marker = tmp_path / f"{name}.ran"
    script = tmp_path / f"{name}.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\ncat\n", encoding="utf-8")
    script.chmod(0o755)
    return script, marker


@pytest.fixture
def planted_fsmonitor(tmp_path: Path) -> tuple[Path, Path]:
    """A checkout whose own ``.git/config`` names a program as ``core.fsmonitor``."""
    repo = tmp_path / "checkout"
    _run_git("init", "-q", str(repo))
    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    _run_git("-C", str(repo), "add", "tracked.txt")
    monitor, marker = _marker_script(tmp_path, "fsmonitor")
    _run_git("-C", str(repo), "config", "core.fsmonitor", str(monitor))
    # The plant is live: an unhardened listing runs it.
    _run_git("-C", str(repo), "ls-files")
    assert marker.exists()
    marker.unlink()
    return repo, marker


class TestLocalGitRunsNoRepositoryProgram:
    """D22: a "local-only" git query must not run the checkout's ``core.fsmonitor``."""

    def test_subjects_fold_listing(self, planted_fsmonitor: tuple[Path, Path]) -> None:
        from particles.api.cli.subjects import _checkout_files

        repo, marker = planted_fsmonitor
        _key, files = _checkout_files(repo)
        assert files == ["tracked.txt"]
        assert not marker.exists()

    def test_hook_doctor_query(self, planted_fsmonitor: tuple[Path, Path]) -> None:
        from particles.api.cli.hook import _git

        repo, marker = planted_fsmonitor
        assert _git(repo, "ls-files") == "tracked.txt"
        assert not marker.exists()

    def test_projection_commit(self, planted_fsmonitor: tuple[Path, Path]) -> None:
        import asyncio

        from particles.api.cli._projection_git import commit_projection

        repo, marker = planted_fsmonitor
        _run_git("-C", str(repo), "config", "user.name", "Test Operator")
        _run_git("-C", str(repo), "config", "user.email", "op@example.test")
        _run_git("-C", str(repo), "config", "commit.gpgsign", "false")
        memory = repo / "memory"
        memory.mkdir()
        (memory / "MEMORY.md").write_text("- a belief `p-aaaa11`\n", encoding="utf-8")
        status = asyncio.run(
            commit_projection(
                memory,
                store="default",
                outcome="created",
                body="- a belief `p-aaaa11`\n",
                snapshot=None,
                run_id="deadbeef",
            )
        )
        assert status.startswith("committed")
        assert not marker.exists()


class TestGistCloneRunsNoConfiguredFilter:
    """N13: a filter driver from the user's config never runs for a fetched tree."""

    def test_isolated_env_neutralises_a_global_filter_driver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.url_safety import isolated_git_env

        # The importing host's own config defines a smudge driver, as a stock
        # `git lfs install` defines filter.lfs.* globally.
        home = tmp_path / "home"
        home.mkdir()
        smudge, marker = _marker_script(tmp_path, "smudge")
        (home / ".gitconfig").write_text(
            f'[filter "planted"]\n\tsmudge = {smudge}\n', encoding="utf-8"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)

        # The fetched repository assigns that driver to its files.
        src = tmp_path / "gist"
        _run_git("init", "-q", str(src))
        (src / ".gitattributes").write_text("*.md filter=planted\n", encoding="utf-8")
        (src / "notes.md").write_text("hello\n", encoding="utf-8")
        _run_git("-C", str(src), "add", ".")
        _run_git(
            "-C",
            str(src),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "init",
        )

        # Under the user's config the driver runs: the plant is live.
        _run_git("clone", "-q", f"file://{src}", str(tmp_path / "plain"))
        assert marker.exists()
        marker.unlink()

        # Under the isolated env it does not, and the file still checks out.
        empty_home = tmp_path / "empty-home"
        empty_home.mkdir()
        env = isolated_git_env(os.environ, home=str(empty_home))
        env["GIT_ALLOW_PROTOCOL"] = "file"  # the test clones a local path
        _run_git("clone", "-q", f"file://{src}", str(tmp_path / "isolated"), env=env)
        assert not marker.exists()
        assert (tmp_path / "isolated" / "notes.md").read_text(encoding="utf-8") == "hello\n"

    def test_clone_spawns_git_under_the_isolated_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio
        import ipaddress

        from particles.ingest.importers import github

        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'filter.x.smudge=evil'")
        monkeypatch.setenv("GIT_SSL_NO_VERIFY", "1")
        seen: list[dict[str, str]] = []

        async def fake_exec(*_argv: str, env: dict[str, str], **_kw: Any) -> Any:
            seen.append(env)
            proc = MagicMock()
            proc.returncode = 128
            proc.communicate = AsyncMock(return_value=(b"", b"stopped by the test"))
            return proc

        monkeypatch.setattr(github.asyncio, "create_subprocess_exec", fake_exec)
        with pytest.raises(ValueError, match="stopped by the test"):
            asyncio.run(
                github._clone_gist_files(
                    "abc123", resolve=lambda _h: [ipaddress.ip_address("140.82.112.3")]
                )
            )
        (env,) = seen
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert env["GIT_ALLOW_PROTOCOL"] == "https"
        assert env["HOME"] != os.environ.get("HOME")
        assert "GIT_CONFIG_PARAMETERS" not in env
        assert "GIT_SSL_NO_VERIFY" not in env
