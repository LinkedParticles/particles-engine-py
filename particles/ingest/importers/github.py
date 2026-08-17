"""GitHub importer (Engine layer; moved from particles.extraction.github.importer
).

``GitHubImporter`` is the single entry point for repo, gist, and Pages URLs
. It dispatches on URL pattern and writes blobs in the format
the matching extractor expects. The importer-only HTTP helpers
(``_fetch_last_commit_meta``, ``_fetch_gist_comments``, ``_clone_gist_files``,
``_parse_link_next``) move with the importer — none are referenced by the
Client-layer extractors. The shared URL/auth/HTTP helpers stay in
``particles.extraction.github._shared``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from particles.config import get_config
from particles.extraction.github._shared import (
    _GIST_RE,
    _PAGES_RE,
    _RAW_RE,
    _REPO_BLOB_RE,
    _REPO_ROOT_RE,
    GITHUB_API_BASE,
    SOURCE_TYPE_GIST,
    SOURCE_TYPE_PAGES,
    SOURCE_TYPE_REPO,
    _api_headers,
    _date_from_path,
    _github_get,
    _normalize_raw_url,
    _parse_gist_url,
    _parse_iso_utc,
    _parse_pages_url,
    _parse_repo_url,
    _raise_for_github_error,
)
from particles.secrets import get_github_api_key_optional
from particles.url_safety import Resolver, format_connect_pin, resolve_and_pin

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


class GitHubImporter:
    """Single importer for GitHub repos, gists, and Pages."""

    def accepts_url(self, url: str) -> bool:
        return bool(
            _PAGES_RE.match(url)
            or _GIST_RE.match(url)
            or _RAW_RE.match(url)
            or _REPO_BLOB_RE.match(url)
            or _REPO_ROOT_RE.match(url)
        )

    async def deposit(
        self,
        session: AsyncSession,
        url: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        log.debug("GitHubImporter: parsing %s", url)
        auth_present = get_github_api_key_optional() is not None
        log.debug("GitHubImporter: GITHUB_API_KEY %s", "set" if auth_present else "not set")
        # Pages first — it's the only pattern on a *.github.io host.
        pages = _parse_pages_url(url)
        if pages is not None:
            username, path = pages
            log.debug("GitHubImporter: classified as PAGES (user=%s)", username)
            return await self._deposit_pages(session, url, username, path, deposited_by, tags)
        gist = _parse_gist_url(url)
        if gist is not None:
            log.debug("GitHubImporter: classified as GIST (id=%s)", gist[1])
            return await self._deposit_gist(session, url, gist[1], deposited_by, tags)
        canonical = _normalize_raw_url(url)
        parsed = _parse_repo_url(canonical)
        if parsed is not None:
            owner, repo, branch, file_path = parsed
            log.debug(
                "GitHubImporter: classified as REPO (owner=%s, repo=%s, branch=%s, path=%s)",
                owner,
                repo,
                branch,
                file_path,
            )
            return await self._deposit_repo(
                session, canonical, owner, repo, branch, file_path, deposited_by, tags
            )
        raise ValueError(f"URL not recognised by GitHubImporter: {url}")

    # -- repo ---------------------------------------------------------------

    async def _deposit_repo(
        self,
        session: AsyncSession,
        canonical_url: str,
        owner: str,
        repo: str,
        branch: str | None,
        path: str | None,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot
        from particles.http import particles_client

        async with particles_client(extra_headers=_api_headers()) as client:
            if path is None:
                api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
            else:
                api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
                if branch:
                    api_url = f"{api_url}?ref={branch}"
            resp = await _github_get(client, api_url)
            _raise_for_github_error(resp, api_url)
            api_data: dict[str, Any] = resp.json()

            encoding = str(api_data.get("encoding") or "base64")
            content_b64 = str(api_data.get("content") or "")
            if encoding == "base64":
                content = base64.b64decode(content_b64) if content_b64 else b""
            else:
                content = content_b64.encode("utf-8", errors="replace")

            actual_path = str(api_data.get("path") or path or "README")

            author_id, content_published_at = await _fetch_last_commit_meta(
                client, owner, repo, actual_path, branch
            )

        if author_id is None:
            author_id = f"github:{owner}"

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info(
            "Deposited GitHub repo %s/%s/%s (%d bytes, author=%s)",
            owner,
            repo,
            actual_path,
            len(content),
            author_id,
        )
        return await write_entry_and_snapshot(
            session=session,
            uri_r=canonical_url,
            source_type=SOURCE_TYPE_REPO,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
            author_id=author_id,
            content_published_at=content_published_at,
        )

    # -- gist ---------------------------------------------------------------

    async def _deposit_gist(
        self,
        session: AsyncSession,
        url: str,
        gist_id: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot
        from particles.http import TransientHttpError, get_with_retry, particles_client

        api_url = f"{GITHUB_API_BASE}/gists/{gist_id}"
        # Try the REST API first. Large gists trip an internal serialization
        # limit on GitHub's side and return a stable cached 502 — when that
        # happens we clone the gist's git repo (anonymous, public-only) and
        # synthesize a gist-shaped JSON blob so the extractor is unchanged.
        try:
            async with particles_client(extra_headers=_api_headers()) as client:
                resp = await get_with_retry(client, api_url, label="GitHub API")
                _raise_for_github_error(resp, api_url)
                data: dict[str, Any] = resp.json()
                comments = await _fetch_gist_comments(client, gist_id)
        except TransientHttpError as exc:
            log.warning(
                "GitHub gist API exhausted retries for %s; falling back to git clone: %s",
                gist_id,
                exc,
            )
            return await self._deposit_gist_via_git(session, url, gist_id, deposited_by, tags)

        login = ((data.get("owner") or {}) if isinstance(data.get("owner"), dict) else {}).get(
            "login"
        )
        author_id = f"github:{login}" if login else None
        content_published_at = _parse_iso_utc(data.get("updated_at") or data.get("created_at"))

        # Embed comments in the stored envelope so extraction is reproducible
        # from the snapshot blob alone (no second API call at extract time).
        data["comments"] = comments
        content = json.dumps(data, sort_keys=True).encode()

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info(
            "Deposited GitHub gist %s (%d bytes, author=%s, %d comments)",
            gist_id,
            len(content),
            author_id,
            len(comments),
        )
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type=SOURCE_TYPE_GIST,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
            author_id=author_id,
            content_published_at=content_published_at,
        )

    async def _deposit_gist_via_git(
        self,
        session: AsyncSession,
        url: str,
        gist_id: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        """Fallback for gists too large for the REST API.

        Gists are real git repos hosted at ``https://gist.github.com/{id}.git``.
        Public gists clone anonymously over HTTPS without a token. We
        synthesize a JSON envelope shaped like the REST API response so
        ``GitHubGistExtractor`` needs no changes — losing only ``description``
        and precise ``created_at`` (we approximate ``updated_at`` with the
        last-commit committer date).

        Comments are still fetched via the REST API (a separate endpoint that
        works even when ``/gists/{id}`` returns a stable 5xx for large blobs).
        """
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot
        from particles.http import particles_client

        parsed = _parse_gist_url(url)
        if parsed is None:
            raise ValueError(f"Cannot parse gist URL for git fallback: {url}")
        owner_login = parsed[0]

        files, last_commit_iso = await _clone_gist_files(gist_id)

        async with particles_client(extra_headers=_api_headers()) as client:
            comments = await _fetch_gist_comments(client, gist_id)

        envelope = {
            "id": gist_id,
            "description": "",
            "owner": {"login": owner_login},
            "files": files,
            "comments": comments,
            "created_at": last_commit_iso,
            "updated_at": last_commit_iso,
            "_fallback": "git-clone",
        }
        content = json.dumps(envelope, sort_keys=True).encode()

        author_id = f"github:{owner_login}"
        content_published_at = _parse_iso_utc(last_commit_iso)

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info(
            "Deposited GitHub gist %s via git fallback "
            "(%d bytes, %d files, %d comments, author=%s)",
            gist_id,
            len(content),
            len(files),
            len(comments),
            author_id,
        )
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type=SOURCE_TYPE_GIST,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
            author_id=author_id,
            content_published_at=content_published_at,
        )

    # -- pages --------------------------------------------------------------

    async def _deposit_pages(
        self,
        session: AsyncSession,
        url: str,
        username: str,
        path: str,
        deposited_by: str,
        tags: list[str],
    ) -> tuple[str, str]:
        from email.utils import parsedate_to_datetime

        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot
        from particles.http import particles_client

        async with particles_client() as client:
            resp = await _github_get(client, url)
        _raise_for_github_error(resp, url)
        content = resp.content

        author_id = f"github:{username}"
        content_published_at = _date_from_path(path)
        if content_published_at is None:
            lm = resp.headers.get("Last-Modified")
            if lm:
                try:
                    parsed = parsedate_to_datetime(lm)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
                    content_published_at = parsed
                except (TypeError, ValueError):
                    content_published_at = None

        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)
        log.info(
            "Deposited GitHub Pages %s (%d bytes, author=%s, published=%s)",
            url,
            len(content),
            author_id,
            content_published_at,
        )
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type=SOURCE_TYPE_PAGES,
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
            author_id=author_id,
            content_published_at=content_published_at,
        )


# ---------------------------------------------------------------------------
# Importer-only HTTP fetchers
# ---------------------------------------------------------------------------


async def _fetch_last_commit_meta(
    client: Any,
    owner: str,
    repo: str,
    path: str,
    branch: str | None,
) -> tuple[str | None, datetime | None]:
    """Return (author_id, committer_date) from the latest commit touching path."""
    commits_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits?path={path}&per_page=1"
    if branch:
        commits_url = f"{commits_url}&sha={branch}"
    try:
        resp = await _github_get(client, commits_url)
    except Exception as exc:
        # Best-effort metadata; never block the deposit if this fails.
        log.warning("commits API call failed for %s/%s: %s", owner, repo, exc)
        return None, None
    if resp.status_code != 200:
        return None, None
    commits = resp.json()
    if not isinstance(commits, list) or not commits:
        return None, None
    first = commits[0]
    author_id: str | None = None
    author_block = first.get("author")
    if isinstance(author_block, dict):
        login = author_block.get("login")
        if login:
            author_id = f"github:{login}"
    commit_block = first.get("commit") or {}
    committer = commit_block.get("committer") if isinstance(commit_block, dict) else None
    date_str = committer.get("date") if isinstance(committer, dict) else None
    return author_id, _parse_iso_utc(date_str if isinstance(date_str, str) else None)


_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def _parse_link_next(link_header: str | None) -> str | None:
    """Return the URL marked ``rel="next"`` in a GitHub Link header, or None.

    GitHub paginated endpoints return a header like
    ``<...?page=2>; rel="next", <...?page=10>; rel="last"``; this picks
    out just the ``next`` URL for clean iteration. Returns None when
    there is no next page (i.e. we're on the last one).
    """
    if not link_header:
        return None
    m = _LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


async def _fetch_gist_comments(client: Any, gist_id: str) -> list[dict[str, Any]]:
    """Fetch a gist's comments via the GitHub REST API, paginating until done.

    Pagination follows the ``Link: …; rel="next"`` header until no
    next-link is present. A configurable ceiling
    ``github.gist_max_comments`` remains as an abuse-stop for
    mis-authenticated runs; set it to 0 to disable the cap.

    Best-effort: a failure on any page logs a warning and returns what we
    have so far. Comments are not load-bearing for the gist body, so a
    failure here must never block the deposit.

    Each item retains GitHub's shape: ``{"user": {"login": ...}, "body": ...,
    "created_at": ..., "updated_at": ...}``. We pass the API response through
    nearly unchanged so reindex can re-derive the rendered text.
    """
    cap = get_config().github.gist_max_comments
    out: list[dict[str, Any]] = []
    url: str | None = f"{GITHUB_API_BASE}/gists/{gist_id}/comments?per_page=100"
    while url is not None:
        try:
            resp = await _github_get(client, url)
        except Exception as exc:
            log.warning("gist comments fetch failed for %s at %s: %s", gist_id, url, exc)
            break
        if resp.status_code != 200:
            log.info(
                "gist comments fetch returned HTTP %d for %s at %s; stopping",
                resp.status_code,
                gist_id,
                url,
            )
            break
        raw = resp.json()
        if not isinstance(raw, list):
            break
        for c in raw:
            if not isinstance(c, dict):
                continue
            user = c.get("user") if isinstance(c.get("user"), dict) else {}
            login = (user or {}).get("login", "")
            body = c.get("body", "")
            if not isinstance(login, str) or not isinstance(body, str):
                continue
            out.append(
                {
                    "user": {"login": login},
                    "body": body,
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                }
            )
            if cap and len(out) >= cap:
                log.info(
                    "gist comments fetch hit cap %d for %s; raise "
                    "github.gist_max_comments to fetch more",
                    cap,
                    gist_id,
                )
                return out
        link_header = resp.headers.get("Link") if hasattr(resp, "headers") else None
        url = _parse_link_next(link_header)
    return out


_GIST_CLONE_HOST = "gist.github.com"


async def _clone_gist_files(
    gist_id: str,
    *,
    resolve: Resolver | None = None,
) -> tuple[dict[str, dict[str, str]], str | None]:
    """Shallow-clone a public gist and return its file contents + last-commit ISO date.

    Returns a mapping shaped like the GitHub gist API's ``files`` field
    (``{filename: {"content": "<utf-8 text>"}}``) plus the last commit's
    ISO-8601 committer date (or ``None`` if git log returned nothing).

    ``resolve`` is the injectable resolver seam; ``None`` uses real
    DNS. Unit tests stub it so no test touches the network.

    Raises:
        UnsafeUrlError: fail-closed, if the clone host does not resolve or
            resolves to a private/reserved address — before git is spawned.
    """
    clone_url = f"https://{_GIST_CLONE_HOST}/{gist_id}.git"

    # Restrict git's transport to https so neither the clone URL nor any
    # server-side redirect can downgrade to a non-https protocol (file://,
    # ext::, etc.). A no-op for the pinned gist.github.com host, but explicit
    # hardening before this code is public and forkable.
    git_env = os.environ.copy()
    git_env["GIT_ALLOW_PROTOCOL"] = "https"

    # Pin the clone's connection to addresses vetted in *this* process, so the
    # validated address is the connected address — the git-side
    # equivalent of `curl --resolve`, and of the validating transport for httpx
    #.
    # git verifies the certificate against the real hostname regardless, so the
    # pin adds no downgrade. Known asymmetry: git silently ignores unknown
    # `http.*` config, so on a libcurl without CURLOPT_RESOLVE the pin becomes a
    # no-op rather than an error (accepted — it degrades to exactly the host-pin
    # + https-only + no-redirect posture that preceded this, never below it;
    # runtime verification deferred).
    pin = format_connect_pin(
        _GIST_CLONE_HOST, 443, resolve_and_pin(_GIST_CLONE_HOST, resolve=resolve)
    )

    with tempfile.TemporaryDirectory(prefix="particles-gist-") as tmpdir:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-c",
                f"http.curloptResolve={pin}",
                "clone",
                "--depth=1",
                "--quiet",
                clone_url,
                tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=git_env,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "git binary not found; required for the gist fallback. "
                "Install git or use a smaller gist that the REST API can serve."
            ) from exc
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ValueError(
                f"git clone failed for gist {gist_id}: "
                f"{stderr.decode(errors='replace')[:200].strip()}"
            )

        # Last commit's committer date — best approximation of updated_at
        ts_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            tmpdir,
            "log",
            "-1",
            "--format=%cI",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=git_env,
        )
        ts_stdout, _ = await ts_proc.communicate()
        last_commit_iso: str | None = ts_stdout.decode().strip() or None

        files: dict[str, dict[str, str]] = {}
        tmpdir_path = Path(tmpdir)
        for entry in sorted(tmpdir_path.iterdir()):
            if entry.name == ".git" or entry.name.startswith(".git/"):
                continue
            if not entry.is_file():
                continue
            try:
                body = entry.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                body = entry.read_bytes().decode("utf-8", errors="replace")
            files[entry.name] = {"content": body}

    return files, last_commit_iso
