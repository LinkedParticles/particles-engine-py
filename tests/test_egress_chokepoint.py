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
import re
from pathlib import Path
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
#                         corpus fetch; uses its own short-lived httpx client.
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


def _files_with_raw_client_calls(subdirs: list[str] | None = None) -> dict[str, list[int]]:
    roots = [_PARTICLES_ROOT / d for d in subdirs] if subdirs else [_PARTICLES_ROOT]
    hits: dict[str, list[int]] = {}
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(_PARTICLES_ROOT).as_posix()
            lines = [
                i
                for i, line in enumerate(path.read_text().splitlines(), start=1)
                if _RAW_CLIENT_CALL.search(line)
            ]
            if lines:
                hits[rel] = lines
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
}


def _spawn_binary(call: ast.Call) -> str | None:
    """Return the argv head of a subprocess spawn, or None if not a spawn call."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    is_spawn = func.attr in _ASYNCIO_SPAWNS or (
        func.attr in _SUBPROCESS_SPAWNS
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )
    if not is_spawn:
        return None

    if not call.args:
        return _DYNAMIC
    head = call.args[0]
    # ``subprocess.run(["git", …])`` passes a list; the async form passes the
    # program as the first positional.
    if isinstance(head, ast.List | ast.Tuple):
        head = head.elts[0] if head.elts else head
    if isinstance(head, ast.Constant) and isinstance(head.value, str):
        return Path(head.value).name
    return _DYNAMIC


class _SpawnVisitor(ast.NodeVisitor):
    """Collect (enclosing function, argv head) for every subprocess spawn."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.found: list[tuple[str, str]] = []

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_func  # noqa: N815
    visit_AsyncFunctionDef = _visit_func  # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        binary = _spawn_binary(node)
        if binary is not None:
            self.found.append((self.stack[-1] if self.stack else "<module>", binary))
        self.generic_visit(node)


def _network_subprocess_sites() -> dict[tuple[str, str, str], int]:
    """Every network-capable subprocess spawn in particles/, with its call count."""
    sites: dict[tuple[str, str, str], int] = {}
    for path in _PARTICLES_ROOT.rglob("*.py"):
        visitor = _SpawnVisitor()
        visitor.visit(ast.parse(path.read_text()))
        for func_name, binary in visitor.found:
            if binary not in _NETWORK_BINARIES and binary != _DYNAMIC:
                continue
            key = (path.relative_to(_PARTICLES_ROOT).as_posix(), func_name, binary)
            sites[key] = sites.get(key, 0) + 1
    return sites


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
