"""§9.1 Deposit operation — write a source to the corpus.

Deposit is intentionally trivial and carries no extraction cost.
Any agent or user can deposit at any time without schema knowledge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config, resolve_store_adjacent_path
from particles.core.schema import (
    CorpusEntry,
    ExtractionStatus,
    FetchPolicy,
    Mutability,
    Snapshot,
    SourceType,
    WarcRecordType,
)
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    get_entry_by_content_hash,
    get_entry_by_uri,
    list_snapshots_for_entry,
)
from particles.extraction.general import sniff_image_media_type
from particles.extraction.rdf import RDF_CONTENT_TYPES, RDF_SUFFIXES
from particles.http import get_capped, particles_client
from particles.observability import traced

log = logging.getLogger(__name__)


# A blob is content-addressed by its SHA-256 hex digest. Every internal caller
# passes an internally-computed digest, but ``content_hash`` is interpolated
# straight into the on-disk path (``hash[:2]/hash``); a future "fetch blob by
# hash" caller passing attacker-controlled input could otherwise traverse out of
# the blob store. Validate the charset up front (defence-in-depth; a real digest
# always passes, so this is non-breaking).
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def blob_path(content_hash: str) -> Path:
    """Resolved on-disk location for a content-addressed blob.

    ``<blob_dir>/<hh>/<hash>``, where ``blob_dir`` is resolved store-adjacent
    per policy. Public because the layout is the shared contract between the
    writer here and the readers that audit it — ``blob_health`` and
    ``blob_fsck`` must look exactly where extraction will look, not at a second
    guess at the same rule.

    Raises:
        ValueError: If ``content_hash`` is not a 64-char lowercase hex digest.
    """
    if not _SHA256_HEX_RE.match(content_hash):
        raise ValueError(
            f"content_hash must be a 64-char lowercase SHA-256 hex digest, got {content_hash!r}"
        )
    # Store-adjacent: a relative blob_dir anchors to the store, not to cwd
    # — otherwise an absolute DATABASE_URL plus the relative default
    # scatters blobs across whichever directory each process ran in.
    blob_dir = resolve_store_adjacent_path(get_config().storage.blob_dir)
    return blob_dir / content_hash[:2] / content_hash


def save_blob(content: bytes, content_hash: str) -> str:
    """Write bytes to the content-addressed blob store; return the file path."""
    path = blob_path(content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return str(path)


def sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def _detect_source_type(
    uri_r: str | None,
    content_type: str | None,
    path: Path | None,
) -> SourceType:
    if path is not None:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return SourceType.PDF
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            # standalone image → vision extraction.
            return SourceType.IMAGE
        if suffix == ".csv":
            return SourceType.CSV
        if suffix in (".md", ".markdown"):
            return SourceType.LOCAL_MARKDOWN
        if suffix == ".py":
            # a lone .py routes to the symbol-aware extractor too,
            # not just the bulk `import project` verb.
            return SourceType.PYTHON_SOURCE
        if suffix in RDF_SUFFIXES:
            # one source type for every RDF syntax. Note ``.json`` is
            # deliberately absent — a bare .json is contended (taxonomy /
            # trust-lens definitions) and this SDK's own interchange
            # bundles carry an @context, so an @context shape sniff would
            # hijack a store export into the RDF path. `.jsonld` or an explicit
            # --source-type is the contract.
            return SourceType.RDF_GRAPH
    if content_type:
        # Checked before the generic families below: `application/ld+json`
        # would otherwise never be reached by a JSON-shaped fallback.
        if any(ct in content_type for ct in RDF_CONTENT_TYPES):
            return SourceType.RDF_GRAPH
        if "pdf" in content_type:
            return SourceType.PDF
        if content_type.startswith("image/"):
            return SourceType.IMAGE
        if "csv" in content_type or "spreadsheet" in content_type:
            return SourceType.CSV
    if uri_r and any(uri_r.startswith(p) for p in ("http://", "https://")):
        return SourceType.WEB_PAGE
    return SourceType.LOCAL_FILE


def _is_taxonomy_definition(content: bytes, suffix: str) -> bool:
    """Return True if ``content`` parses as a TaxonomyDefinition JSON.

    Detection is shape-only: top-level object with ``name``, ``version``,
    and a ``tags`` list. The full Pydantic validation happens later in
    the TaxonomyExtractor — bad JSON that passes this check still produces
    an extraction failure, not a wrong source-type stamp.
    """
    if suffix != ".json":
        return False
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("name"), str)
        and isinstance(data.get("version"), str)
        and isinstance(data.get("tags"), list)
    )


def _is_trust_lens_definition(content: bytes, suffix: str) -> bool:
    """Return True if ``content`` declares itself a TrustLensDefinition.

    Detection keys on the explicit ``"kind"`` sentinel the artifact carries;
    full Pydantic validation happens later in the TrustLensExtractor.
    """
    if suffix != ".json":
        return False
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and data.get("kind") == "TrustLensDefinition"


# Leading Markdown heading / list / block-quote markers stripped before a line
# is tested as a standalone date (so `# 2026-03-15` and `- 2026-03-15` match).
_LEADING_MARKER_RE = re.compile(r"^[\s>#*\-+]+")
# An optional `Date:` / `Dated -` label prefix on the date line.
_DATE_LABEL_RE = re.compile(r"^dated?\s*[:\-]\s*", re.IGNORECASE)


def _parse_date_line(line: str) -> datetime | None:
    """A line that is *wholly* a date → its midnight-UTC datetime, else ``None``.

    Strips leading Markdown heading / list / block-quote markers and an optional
    ``Date:`` label, then attempts to parse the remainder **wholly** against
    ``deposit_date.formats``. A date sitting inside prose does not match —
    ``strptime`` rejects trailing text — which keeps false positives near zero.

    This is the boundary primitive shared leading-date capture
    (:func:`_detect_leading_date`, which applies it to the first few lines) and
    the date-line splitter (:func:`split_file_by_date`, which applies it
    to every line). Sharing one primitive keeps the two date behaviours from
    drifting. It is pure (no store) and ungated by ``detect_leading_date`` — that
    config flag governs auto-detection on the normal deposit path, not this test.
    """
    candidate = _LEADING_MARKER_RE.sub("", line.strip())
    candidate = _DATE_LABEL_RE.sub("", candidate).strip()
    if not candidate:
        return None
    for fmt in get_config().deposit_date.formats:
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=UTC)
    return None


def _detect_leading_date(content: bytes) -> datetime | None:
    """Return a date parsed from a standalone leading line of text content.

    Scans the first ``deposit_date.leading_date_scan_lines`` non-blank lines and
    returns the first :func:`_parse_date_line` hit as a midnight-UTC datetime, or
    ``None`` when leading-date detection is disabled, the content is not UTF-8
    text, or no line matches.
    """
    cfg = get_config().deposit_date
    if not cfg.detect_leading_date:
        return None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None

    scanned = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        scanned += 1
        if scanned > cfg.leading_date_scan_lines:
            break
        parsed = _parse_date_line(line)
        if parsed is not None:
            return parsed
    return None


def _segment_by_date_lines(text: str) -> list[tuple[datetime | None, str]]:
    """Segment ``text`` at standalone date-line boundaries.

    A **section** is a date line plus every following line up to (but excluding)
    the next date line; the date-line text is kept inside the section so it is
    self-describing. Any non-whitespace content *before* the first date line is
    returned as a leading **preamble** section with date ``None`` (a
    whitespace-only preamble is dropped, so leading blank lines do not become an
    empty entry). Only the first section can be dateless; every later section
    opens on a date line.

    Returns a list of ``(section_date, section_text)`` pairs in file order. A
    file with no date line yields a single ``(None, text)`` pair, which the
    caller treats as "not multi-entry."
    """
    sections: list[tuple[datetime | None, list[str]]] = []
    current_date: datetime | None = None
    current_lines: list[str] = []
    for raw in text.splitlines(keepends=True):
        parsed = _parse_date_line(raw.rstrip("\r\n"))
        if parsed is not None:
            if current_lines:
                sections.append((current_date, current_lines))
            current_date = parsed
            current_lines = [raw]
        else:
            current_lines.append(raw)
    if current_lines:
        sections.append((current_date, current_lines))

    joined = [(d, "".join(lines)) for d, lines in sections]
    # Drop a whitespace-only leading preamble (date None) — only the first
    # section can be dateless, so this checks index 0 alone.
    if joined and joined[0][0] is None and not joined[0][1].strip():
        joined = joined[1:]
    return joined


def _resolve_source_type(path: Path, content: bytes, override: str | None) -> str:
    """Resolve a local-file ``source_type`` (override › self-declared JSON › mime).

    Shared by :func:`deposit_file` and the splitter so every section of
    a split file inherits the same whole-file source-type detection.
    """
    if override is not None:
        return override
    # magic-byte image detection is authoritative — a PNG/JPEG is an
    # image regardless of file extension (or its absence).
    if sniff_image_media_type(content) is not None:
        return SourceType.IMAGE
    suffix = path.suffix.lower()
    if _is_trust_lens_definition(content, suffix):
        return SourceType.TRUST_LENS_DEFINITION
    if _is_taxonomy_definition(content, suffix):
        return SourceType.TAXONOMY_DEFINITION
    mime, _ = mimetypes.guess_type(str(path))
    return _detect_source_type(None, mime, path)


def _resolve_content_published_at(
    path: Path, content: bytes, override: datetime | None
) -> datetime | None:
    """Resolve ``content_published_at`` for a local-file deposit.

    Precedence, highest wins: an explicit operator ``override`` (the CLI
    ``--date``) › a leading date line in the content › the file mtime (when
    ``deposit_date.mtime_fallback`` is on). Returns ``None`` when none applies —
    today's behavior, so an undated deposit is unchanged.
    """
    if override is not None:
        return override
    leading = _detect_leading_date(content)
    if leading is not None:
        return leading
    if get_config().deposit_date.mtime_fallback:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(mtime, tz=UTC)
    return None


@traced("deposit")
async def deposit_file(
    session: AsyncSession,
    path: Path,
    deposited_by: str = "operator",
    mutability: Mutability | None = None,
    fetch_policy: FetchPolicy | None = None,
    source_type: str | None = None,
    tags: list[str] | None = None,
    content_date: datetime | None = None,
) -> tuple[str, str]:
    """Deposit a local file into the corpus.

    Source type is auto-detected from file extension. Deduplication is by
    SHA-256 content hash — re-depositing the same file returns the existing entry.

    ``content_published_at`` is captured at deposit time so an
    archival document is not stamped with the import date. Precedence (highest
    wins): an explicit ``content_date`` (the CLI ``--date``) › a leading date
    line in the content › the file mtime (see ``deposit_date`` config).
    ``deposit_vault`` routes through here, so vault files get the leading-date /
    mtime capture too (without the per-file ``--date`` override).

    Args:
        path: Absolute or relative path to the file.
        deposited_by: Agent ID recorded on the corpus entry.
        source_type: Override auto-detected source type.
        content_date: Explicit authorship date; overrides auto-detection.

    Returns:
        Tuple of (entry_id, snapshot_id).
    """
    content = path.read_bytes()
    content_hash = sha256(content)
    archive_path = save_blob(content, content_hash)
    content_published_at = _resolve_content_published_at(path, content, content_date)

    source_type = _resolve_source_type(path, content, source_type)

    if mutability is None:
        mutability = Mutability.STABLE
    if fetch_policy is None:
        fetch_policy = FetchPolicy.NEVER

    return await write_entry_and_snapshot(
        session=session,
        uri_r=path.resolve().as_uri(),  # file:///abs/path for display & dedup
        source_type=source_type,
        mutability=mutability,
        fetch_policy=fetch_policy,
        content=content,
        content_hash=content_hash,
        archive_path=archive_path,
        deposited_by=deposited_by,
        tags=tags or [],
        warc_record_type=WarcRecordType.RESPONSE,
        content_published_at=content_published_at,
    )


async def split_file_by_date(
    session: AsyncSession,
    path: Path,
    deposited_by: str = "operator",
    mutability: Mutability | None = None,
    fetch_policy: FetchPolicy | None = None,
    source_type: str | None = None,
    tags: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Deposit a multi-entry local file as N corpus entries, split at date lines.

    Segments ``path`` at standalone date-line boundaries
    (:func:`_segment_by_date_lines`) and writes **one corpus entry per dated
    section**, each carrying its own ``content_published_at`` — closing the silent
    date loss that one-file-one-entry deposit suffers on a journal / changelog /
    daily-log that concatenates many dated entries. The mechanism is genre-neutral
    (a date on its own line starts a new record); the personal journal is the
    motivating instance, not a genre silo.

    Each section flows through the same :func:`write_entry_and_snapshot` as an
    ordinary deposit, so downstream extraction (including the ``--journal``
    extractor), date extraction and over-length chunking, §6.6
    reconciliation, and the NARRATIVE post-pass all apply per-section unchanged.

    - **content_published_at** is the section's parsed date; the leading
      **preamble** (any non-whitespace content before the first date line) carries
      no date-line date and falls through remaining precedence (file
      mtime, then ``None``) — undated content stays undated.
    - **uri_r** is the source file URI plus a synthetic ``#entry-<n>`` fragment
      (1-based ordinal over the emitted sections), giving each section a
      human-readable identity tied to its source file.
    - **Dedup** is the existing two-tier ``write_entry_and_snapshot`` check
      (URI-R then content hash), so re-depositing an unchanged file is idempotent.
    - **source_type** is inherited from the whole-file detection / override, so a
      ``--journal`` split yields N JOURNAL entries.

    **No-op guard.** When the file yields fewer than two sections (no date line,
    or only a preamble), this falls back to the normal single-entry
    :func:`deposit_file` — so ``--split-by-date`` on a file that is not actually
    multi-entry is harmless.

    Args:
        session: Active SQLAlchemy async session.
        path: Local file to split; must be UTF-8-decodable (binary blobs cannot
            be line-split).
        deposited_by: Agent ID stamped on every section's corpus entry.
        source_type: Override the auto-detected whole-file source type.
        tags: Optional tag list applied to every section.

    Returns:
        List of ``(entry_id, snapshot_id)`` tuples, one per emitted section, in
        file order.

    Raises:
        ValueError: If ``path`` is not UTF-8-decodable.
    """
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"--split-by-date requires UTF-8 text content; {path} is not decodable "
            "(binary sources like PDFs cannot be line-split)."
        ) from exc

    sections = _segment_by_date_lines(text)
    if len(sections) < 2:
        # Not actually multi-entry → ordinary single-entry deposit (which still
        # captures the leading date). Harmless opt-in no-op.
        entry_id, snapshot_id = await deposit_file(
            session,
            path,
            deposited_by=deposited_by,
            mutability=mutability,
            fetch_policy=fetch_policy,
            source_type=source_type,
            tags=tags,
        )
        return [(entry_id, snapshot_id)]

    resolved_source_type = _resolve_source_type(path, content, source_type)
    if mutability is None:
        mutability = Mutability.STABLE
    if fetch_policy is None:
        fetch_policy = FetchPolicy.NEVER

    base_uri = path.resolve().as_uri()
    results: list[tuple[str, str]] = []
    for ordinal, (section_date, section_text) in enumerate(sections, start=1):
        section_bytes = section_text.encode("utf-8")
        section_hash = sha256(section_bytes)
        archive_path = save_blob(section_bytes, section_hash)
        if section_date is not None:
            content_published_at: datetime | None = section_date
        else:
            # Preamble: no date line → fallback (mtime, then None).
            content_published_at = _resolve_content_published_at(path, section_bytes, None)
        entry_id, snapshot_id = await write_entry_and_snapshot(
            session=session,
            uri_r=f"{base_uri}#entry-{ordinal}",
            source_type=resolved_source_type,
            mutability=mutability,
            fetch_policy=fetch_policy,
            content=section_bytes,
            content_hash=section_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=tags or [],
            warc_record_type=WarcRecordType.RESPONSE,
            content_published_at=content_published_at,
        )
        results.append((entry_id, snapshot_id))
    log.info("Split %s into %d dated corpus entries", path, len(results))
    return results


def _iter_tree(
    root: Path,
    suffixes: set[str],
    skip_part: Callable[[str], bool],
) -> list[Path]:
    """Walk ``root`` recursively; return files matching ``suffixes``.

    The shared recursive-deposit skeleton: recurse → ``is_file`` → suffix
    filter → deterministic sort, with the **ignore predicate injected** rather
    than hardcoded. ``skip_part`` is called on each path component *relative to
    ``root``*; a path is dropped if any of its components matches. Keying on the
    relative path means the user's home-directory components (``.config`` in
    ``~/.config/foo/vault``) never cause every file to be skipped.

    ``suffixes`` is matched case-insensitively against ``path.suffix.lower()``.
    Two ignore policies ride this one walker: ``deposit_vault``'s Obsidian rule
    (skip ``_``/``.`` components) and ``deposit_project``'s source-tree rule
    (skip dot-dirs + configured build/cache dirs, but **keep** underscore module
    files like ``__init__.py`` / ``_shared.py``).
    """
    results: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(root)
        if any(skip_part(part) for part in rel.parts):
            continue
        results.append(path)
    results.sort()
    return results


def _iter_vault_markdown(vault_dir: Path) -> list[Path]:
    """Walk ``vault_dir`` recursively and return ``.md`` / ``.markdown`` paths.

    Skips files and directories whose names start with ``_`` or ``.`` so
    Obsidian's ``.obsidian/`` settings dir, ``.trash/``, ``_attachments/``
    and similar hidden / scaffold paths do not pollute the corpus.

    Returns paths sorted deterministically so the progress callback and the
    returned ``(entry_id, snapshot_id)`` list are stable across runs.
    """
    return _iter_tree(vault_dir, {".md", ".markdown"}, lambda p: p.startswith(("_", ".")))


def iter_vault_files(vault_dir: Path) -> list[Path]:
    """The Markdown files ``import vault`` would deposit (ignore policy).

    The **shared walk seam**: :func:`deposit_vault` (the local path) and the
    remote ``import vault`` path (a client tree-walk that loops
    ``backend.deposit_file``) both call this, so the Obsidian ignore policy
    never drifts between the two.
    """
    return _iter_vault_markdown(vault_dir)


def iter_project_files(project_dir: Path, extensions: set[str] | None = None) -> list[Path]:
    """The source files ``import project`` would deposit (ignore policy).

    The **shared walk seam**: :func:`deposit_project` (the local path) and the
    remote ``import project`` path both call this, so the suffix set
    (``import_project.extensions``, or the ``extensions`` override) and the
    ignore-dir policy (``import_project.ignore_dirs``, plus dot-prefixed
    components) never drift between the two.
    """
    cfg = get_config().import_project
    if extensions is not None:
        suffixes = {(e if e.startswith(".") else f".{e}").lower() for e in extensions}
    else:
        suffixes = {e.lower() for e in cfg.extensions}
    ignore_dirs = set(cfg.ignore_dirs)
    return _iter_tree(project_dir, suffixes, lambda p: _skip_source_part(p, ignore_dirs))


async def deposit_vault(
    session: AsyncSession,
    vault_dir: Path,
    deposited_by: str = "operator",
    tags: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Deposit every Markdown file under ``vault_dir`` as ``LOCAL_MARKDOWN``.

    Recursively walks ``vault_dir`` for ``.md`` / ``.markdown`` files,
    skipping any path component starting with ``_`` or ``.`` (so Obsidian's
    ``.obsidian/`` settings, ``_attachments/`` scaffolds, and similar
    metadata-only paths are not deposited as content). Each file is routed
    through :func:`deposit_file`, which means existing ``content_hash``
    deduplication applies — re-running on the same vault is idempotent and
    will not create duplicate corpus entries.

    Honors the spec promise that operators with existing Obsidian
    vaults can onboard to Particles via one command, then `extract` + `lint`
    against the resulting corpus without rebuilding their notes.

    Args:
        session: Active SQLAlchemy async session.
        vault_dir: Vault root directory; walked recursively.
        deposited_by: Agent ID stamped on every corpus entry.
        tags: Optional tag list applied to every deposited entry.
        progress: Optional callback invoked once per file, before deposit,
            with a ``"[i/N] path/to/file.md"``-style message. Mirrors the
            progress shape used by ``particles reindex --verbose``.

    Returns:
        List of ``(entry_id, snapshot_id)`` tuples in deposit order; empty
        if ``vault_dir`` contains no Markdown files.
    """
    if not vault_dir.exists() or not vault_dir.is_dir():
        raise ValueError(f"Vault directory not found: {vault_dir}")

    files = iter_vault_files(vault_dir)
    if not files:
        if progress is not None:
            progress(f"No .md files found under {vault_dir}")
        return []

    results: list[tuple[str, str]] = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        if progress is not None:
            try:
                rel = path.relative_to(vault_dir)
            except ValueError:
                rel = path
            progress(f"[{i}/{total}] depositing {rel}")
        entry_id, snapshot_id = await deposit_file(
            session,
            path,
            deposited_by=deposited_by,
            source_type=SourceType.LOCAL_MARKDOWN,
            tags=tags or [],
        )
        results.append((entry_id, snapshot_id))
    return results


def _iter_clipper_markdown(captures_dir: Path) -> list[Path]:
    """Walk ``captures_dir`` for ``.md`` / ``.markdown`` captures.

    Uses the **vault ignore policy** (skip any path component starting with ``_``
    or ``.``) — a Clipper captures folder is an Obsidian vault folder, so
    ``.obsidian/`` settings and ``_attachments/`` scaffold must be skipped — over
    the shared :func:`_iter_tree` skeleton. Identical to
    :func:`_iter_vault_markdown`; named separately so the two intakes' ignore
    policies can diverge without entangling them.
    """
    return _iter_tree(captures_dir, {".md", ".markdown"}, lambda p: p.startswith(("_", ".")))


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a leading ``--- … ---`` YAML frontmatter block from the body.

    Returns ``(frontmatter_dict, body)``. When the text does not open with a
    frontmatter fence, the block does not close, or the YAML does not parse to a
    mapping, returns ``({}, text)`` — frontmatter is **trusted input that must
    degrade gracefully** (§ Consequences): a malformed header never
    raises, so one bad capture cannot abort a whole folder scan. The body keeps
    its original bytes minus the stripped header, so the extractor sees the
    article, not the YAML.
    """
    if not text.startswith("---"):
        return {}, text
    # The opening fence is the first line; it must be exactly `---` (allowing a
    # trailing CR). Anything else (e.g. a `***` horizontal rule) is not a fence.
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text
    # Find the closing fence.
    for idx in range(1, len(lines)):
        if lines[idx].rstrip("\r\n") == "---":
            header_text = "".join(lines[1:idx])
            body = "".join(lines[idx + 1 :])
            try:
                parsed = yaml.safe_load(header_text)
            except yaml.YAMLError:
                return {}, text
            if not isinstance(parsed, dict):
                return {}, text
            return parsed, body
    # No closing fence → not a frontmatter block.
    return {}, text


def _first_frontmatter_value(frontmatter: dict[str, Any], keys: list[str]) -> Any:
    """Return the first present, non-empty value among ``keys`` (in order)."""
    for key in keys:
        value = frontmatter.get(key)
        if value is not None and value != "":
            return value
    return None


def _frontmatter_tags(frontmatter: dict[str, Any], keys: list[str]) -> list[str]:
    """Collect tag values from ``keys`` — a list value extends, a scalar appends."""
    tags: list[str] = []
    for key in keys:
        value = frontmatter.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            tags.extend(str(t).strip() for t in value if str(t).strip())
        else:
            stripped = str(value).strip()
            if stripped:
                tags.append(stripped)
    return tags


def _strip_url_fragment(url: str) -> str:
    """Strip the ``#fragment`` from a URL (mirrors ``deposit_url``)."""
    from urllib.parse import urlsplit, urlunsplit

    split = urlsplit(url)
    if split.fragment:
        return urlunsplit(split._replace(fragment=""))
    return url


def _parse_frontmatter_date(value: Any) -> datetime | None:
    """Parse a frontmatter date value against ``deposit_date.formats``.

    A ``date`` / ``datetime`` (YAML parses ``published: 2026-05-12`` to a
    ``date``) is normalised to midnight UTC; a string is matched **wholly**
    against the configured formats. Returns ``None`` on no match — the caller
    then falls through remaining precedence.
    """
    from datetime import date as _date

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, _date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    for fmt in get_config().deposit_date.formats:
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=UTC)
    return None


async def deposit_web_clipper(
    session: AsyncSession,
    captures_dir: Path,
    deposited_by: str = "web-clipper",
    tags: list[str] | None = None,
    content_date: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Deposit a folder of frontmatter-Markdown captures as ``WEB_PAGE`` entries.

    The general frontmatter-Markdown intake, with the Obsidian Web Clipper as the
    first configured profile (``web_clipper`` config). Recursively
    walks ``captures_dir`` for ``.md`` / ``.markdown`` files (skipping ``_`` / ``.``
    components, the vault ignore policy), and for each capture:

    * parses the leading ``--- … ---`` YAML frontmatter (degrading gracefully to a
      plain ``LOCAL_MARKDOWN`` body deposit when the header is absent / malformed);
    * maps the configured ``url_keys`` → ``uri_r`` (fragment-stripped, **not**
      fetched — the local bytes are the record of what was seen), ``date_keys`` →
      ``content_published_at`` (below an explicit ``content_date``), and
      ``tag_keys`` → entry tags merged with the run-wide ``tags``;
    * deposits the **frontmatter-stripped body** as the entry's bytes, stamped
      ``WEB_PAGE`` (the configured ``source_type``), so the entry is trustable,
      decayable, and queryable as the web page it clipped — everything
      ``import vault`` discards.

    Reuses :func:`write_entry_and_snapshot`'s two-tier dedup wholesale: the body
    bytes are hashed (frontmatter stripped), so re-clips that differ only in
    clip-time metadata collapse to one entry, and a page clipped twice (same
    ``source:``) collapses on its ``uri_r`` rather than two ``file://`` paths.
    One-shot scan; the watching daemon is deferred.

    Args:
        session: Active SQLAlchemy async session.
        captures_dir: Captures root directory; walked recursively.
        deposited_by: Agent ID stamped on every corpus entry (default
            ``"web-clipper"``).
        tags: Optional run-wide tag list, merged with each capture's
            frontmatter tags (per-file ∪ run-wide).
        content_date: Explicit operator authorship date (the CLI ``--date``);
            overrides the frontmatter ``published`` date for every capture.
        progress: Optional callback invoked once per file, before deposit, with a
            ``"[i/N] depositing path"``-style message (mirrors ``deposit_vault``).

    Returns:
        List of ``(entry_id, snapshot_id)`` tuples in deposit order; empty if
        ``captures_dir`` contains no Markdown captures.

    Raises:
        ValueError: If ``captures_dir`` does not exist or is not a directory.
    """
    if not captures_dir.exists() or not captures_dir.is_dir():
        raise ValueError(f"Captures directory not found: {captures_dir}")

    files = _iter_clipper_markdown(captures_dir)
    if not files:
        if progress is not None:
            progress(f"No .md captures found under {captures_dir}")
        return []

    cfg = get_config().web_clipper
    run_tags = tags or []

    results: list[tuple[str, str]] = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        if progress is not None:
            try:
                rel: Path | str = path.relative_to(captures_dir)
            except ValueError:
                rel = path
            progress(f"[{i}/{total}] depositing {rel}")

        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            # A non-UTF-8 capture cannot carry frontmatter; deposit it verbatim
            # through the ordinary local-file path so the scan never aborts.
            entry_id, snapshot_id = await deposit_file(
                session, path, deposited_by=deposited_by, tags=run_tags, content_date=content_date
            )
            results.append((entry_id, snapshot_id))
            continue

        frontmatter, body = _split_frontmatter(text)

        uri_r_value = _first_frontmatter_value(frontmatter, cfg.url_keys)
        if not frontmatter or uri_r_value is None:
            # No parseable frontmatter, or no source URL → fall back to a plain
            # LOCAL_MARKDOWN body deposit (§ Consequences): the capture
            # still onboards, just without the restored provenance. file:// uri_r
            # via deposit_file; the leading-date / mtime path still runs.
            entry_id, snapshot_id = await deposit_file(
                session,
                path,
                deposited_by=deposited_by,
                source_type=SourceType.LOCAL_MARKDOWN,
                tags=run_tags,
                content_date=content_date,
            )
            results.append((entry_id, snapshot_id))
            continue

        uri_r = _strip_url_fragment(str(uri_r_value))

        # precedence: explicit --date wins; else the frontmatter date.
        if content_date is not None:
            content_published_at: datetime | None = content_date
        else:
            content_published_at = _parse_frontmatter_date(
                _first_frontmatter_value(frontmatter, cfg.date_keys)
            )

        # Per-file frontmatter tags ∪ run-wide --tags, order-stable + de-duped.
        merged_tags: list[str] = []
        for tag in (*_frontmatter_tags(frontmatter, cfg.tag_keys), *run_tags):
            if tag not in merged_tags:
                merged_tags.append(tag)

        content = body.encode("utf-8")
        content_hash = sha256(content)
        archive_path = save_blob(content, content_hash)

        entry_id, snapshot_id = await write_entry_and_snapshot(
            session=session,
            uri_r=uri_r,
            source_type=cfg.source_type,
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.NEVER,
            content=content,
            content_hash=content_hash,
            archive_path=archive_path,
            deposited_by=deposited_by,
            tags=merged_tags,
            warc_record_type=WarcRecordType.RESPONSE,
            content_published_at=content_published_at,
        )
        results.append((entry_id, snapshot_id))
    return results


def _skip_source_part(part: str, ignore_dirs: set[str]) -> bool:
    """Ignore predicate for a source tree.

    Skips a path component that starts with ``.`` (``.git``, ``.venv``,
    ``.mypy_cache``) or is in the configured ``ignore_dirs`` set
    (``__pycache__``, ``node_modules``, ``build``, ``dist``, …). Crucially it
    does **not** skip underscore-prefixed module files — a Python package's real
    source includes ``__init__.py`` / ``_shared.py`` / ``_logging.py``, which the
    vault walker's blanket ``_``-component skip would silently drop. This
    asymmetry is the substantive design point.
    """
    return part.startswith(".") or part in ignore_dirs


async def deposit_project(
    session: AsyncSession,
    project_dir: Path,
    deposited_by: str = "operator",
    tags: list[str] | None = None,
    extensions: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Deposit every source file under ``project_dir`` as ``PYTHON_SOURCE``.

    The recursive multi-file structured-source deposit: walks ``project_dir``
    for files matching the configured extension set (``import_project.extensions``,
    default ``[".py"]``), skipping dot-prefixed components and the configured
    build/cache directories (``import_project.ignore_dirs``) — but **keeping**
    underscore-prefixed module files (``__init__.py`` / ``_shared.py``). Each
    matched file is routed through :func:`deposit_file` with an explicit
    ``source_type=PYTHON_SOURCE`` (mirroring how :func:`deposit_vault` passes
    ``LOCAL_MARKDOWN``), so ``content_hash`` deduplication, content-date
    capture, and the per-file ``(entry_id, snapshot_id)`` return list all come
    for free. Re-running on the same tree is idempotent — editing one file and
    re-running appends a snapshot only to that file's entry.

    Python is the first registered glob instance (general
    multi-file structured-source deposit, not a Python silo); a future language
    is a new ``(extensions → source_type)`` pairing plus its own extractor.

    Args:
        session: Active SQLAlchemy async session.
        project_dir: Project root directory; walked recursively.
        deposited_by: Agent ID stamped on every corpus entry.
        tags: Optional tag list applied to every deposited entry.
        extensions: Per-invocation override of the deposited file suffixes
            (the ``--ext`` CLI flag). ``None`` uses ``import_project.extensions``.
            Compared case-insensitively; entries are normalised to a leading dot.
        progress: Optional callback invoked once per file, before deposit, with a
            ``"[i/N] depositing path"``-style message (mirrors ``deposit_vault``).

    Returns:
        List of ``(entry_id, snapshot_id)`` tuples in deposit order; empty if
        ``project_dir`` contains no matching source files.

    Raises:
        ValueError: If ``project_dir`` does not exist or is not a directory.
    """
    if not project_dir.exists() or not project_dir.is_dir():
        raise ValueError(f"Project directory not found: {project_dir}")

    files = iter_project_files(project_dir, extensions)
    if not files:
        if progress is not None:
            progress(f"No matching source files found under {project_dir}")
        return []

    results: list[tuple[str, str]] = []
    total = len(files)
    for i, path in enumerate(files, start=1):
        if progress is not None:
            try:
                rel = path.relative_to(project_dir)
            except ValueError:
                rel = path
            progress(f"[{i}/{total}] depositing {rel}")
        entry_id, snapshot_id = await deposit_file(
            session,
            path,
            deposited_by=deposited_by,
            source_type=SourceType.PYTHON_SOURCE,
            tags=tags or [],
        )
        results.append((entry_id, snapshot_id))
    return results


@traced("deposit")
async def deposit_url(
    session: AsyncSession,
    url: str,
    deposited_by: str = "operator",
    mutability: Mutability | None = None,
    fetch_policy: FetchPolicy | None = None,
    source_type: str | None = None,
    tags: list[str] | None = None,
    follow_post_links: bool | None = None,
    follow_comment_links: bool | None = None,
    out_follow_targets: list[tuple[str, str, str]] | None = None,
) -> tuple[str, str]:
    """Fetch a URL and deposit it into the corpus.

    Domain-specific importers (Wikidata, Numista, …) are tried first via the
    plugin registry. Falls back to a generic HTTP fetch for
    any URL not claimed by a registered importer.

    URL fragments (``#anchor``) are stripped before any importer lookup,
    fetch, or storage. HTTP servers never receive the fragment (it's
    client-side anchor navigation), so it carries no signal for
    extraction; keeping it in ``uri_r`` would just create spurious
    one-corpus-entry-per-anchor noise for operators who paste a
    section link.

    ``follow_post_links`` and ``follow_comment_links``: when
    the matched importer participates in deposit-time follow, the
    primary URL is fetched as a secondary corpus entry and a row is
    written to ``corpus_follow_edges`` recording the relationship.
    ``None`` (the default) consults the importer's
    ``DEFAULT_FOLLOW_POST_LINKS`` / ``DEFAULT_FOLLOW_COMMENT_LINKS``
    constants. Explicit True / False overrides the importer default.
    The recursive follow call hardcodes both flags to False — the
    depth-1 cap.

    ``out_follow_targets`` (optional): when supplied, each
    successful follow appends ``(entry_id, snapshot_id, uri_r)`` for
    the secondary corpus entry. Operator-facing surface (the CLI uses
    this to print the follow result so operators see both entries),
    not API-load-bearing.

    Returns (entry_id, snapshot_id) for the **primary** deposit. Follow
    targets surface via ``out_follow_targets``.
    """
    from urllib.parse import urlsplit, urlunsplit

    from particles.url_safety import validate_fetch_url

    split = urlsplit(url)
    if split.fragment:
        log.info("Stripped fragment '#%s' from deposit URL", split.fragment)
        url = urlunsplit(split._replace(fragment=""))

    validate_fetch_url(url)  # SSRF guard — rejects loopback, RFC1918, link-local, etc.

    # comment-link following is reserved-but-deferred.
    # Honour the flag's presence (don't silently drop it) but no-op the
    # behaviour with a one-time warning so operators see their intent
    # was acknowledged.
    if follow_comment_links:
        log.warning(
            "--follow-comment-links was passed but comment-link following is "
            "deferred (§ Deferred). Proceeding without comment-link "
            "follow."
        )

    if source_type is None:
        # defer: cycle break — `ingest` (the importer registry's package) imports
        # `corpus.deposit`/`corpus.store` at module top, so a top-level import here
        # would close a genuine corpus↔ingest cycle. The corpus→ingest edge exists
        # only because the corpus-writing ImporterPlugin role moved into
        # `ingest/importers/`. See root AGENTS.md § Code conventions → Deferred
        # imports (case 1: cycle break).
        from particles.ingest.importers.registry import get_importers

        importers = get_importers()
        log.debug("deposit_url: trying %d importers for %s", len(importers), url)
        for importer in importers:
            if importer.accepts_url(url):
                log.debug("deposit_url: matched %s", type(importer).__name__)
                entry_id, snapshot_id = await importer.deposit(
                    session, url, deposited_by, tags or []
                )
                # follow the primary URL of link-shaped posts.
                # Each importer that opts in defines a ``primary_url``
                # method; importers without the method don't participate.
                await _maybe_follow_primary_url(
                    session,
                    importer,
                    entry_id=entry_id,
                    snapshot_id=snapshot_id,
                    deposited_by=deposited_by,
                    tags=tags or [],
                    follow_post_links=follow_post_links,
                    out_follow_targets=out_follow_targets,
                )
                await _reconcile_mentions_for_deposit(session, url=url, entry_id=entry_id)
                return entry_id, snapshot_id
        log.debug("deposit_url: no importer matched; falling back to generic fetch")

    async with particles_client() as client:
        resp = await get_capped(client, url)
    if resp.status_code == 403:
        raise ValueError(
            f"403 Forbidden: the server refused the request for {url!r}. "
            "The site may require authentication or block automated access. "
            "Try downloading the page manually and using 'particles deposit <file>' instead."
        )
    resp.raise_for_status()

    content = resp.content
    content_hash = sha256(content)
    archive_path = save_blob(content, content_hash)
    content_type = resp.headers.get("content-type", "")

    if source_type is None:
        source_type = _detect_source_type(url, content_type, None)
    if mutability is None:
        mutability = Mutability.MUTABLE if "text/html" in content_type else Mutability.STABLE
    if fetch_policy is None:
        fetch_policy = FetchPolicy.LAZY if "text/html" in content_type else FetchPolicy.NEVER

    entry_id, snapshot_id = await write_entry_and_snapshot(
        session=session,
        uri_r=url,
        source_type=source_type,
        mutability=mutability,
        fetch_policy=fetch_policy,
        content=content,
        content_hash=content_hash,
        archive_path=archive_path,
        deposited_by=deposited_by,
        tags=tags or [],
        warc_record_type=WarcRecordType.RESPONSE,
        etag=resp.headers.get("etag"),
        last_modified=_parse_last_modified(resp.headers.get("last-modified")),
    )
    await _reconcile_mentions_for_deposit(session, url=url, entry_id=entry_id)
    return entry_id, snapshot_id


async def _maybe_follow_primary_url(
    session: AsyncSession,
    importer: Any,
    *,
    entry_id: str,
    snapshot_id: str,
    deposited_by: str,
    tags: list[str],
    follow_post_links: bool | None,
    out_follow_targets: list[tuple[str, str, str]] | None = None,
) -> None:
    """Depth-1 follow of a link-shaped post's primary URL.

    Called after the matched importer's ``deposit()`` returns. Looks up
    the importer's ``primary_url`` method (None if absent — the
    importer doesn't participate in following), resolves the follow
    policy from ``follow_post_links`` + the importer's
    ``DEFAULT_FOLLOW_POST_LINKS``, and recursively deposits the primary
    URL when the policy says so. The recursive call hardcodes both
    follow flags to False — the depth-1 cap.

    Records the relationship in ``corpus_follow_edges`` on success.
    Paywall / 403 / fetch failures on the follow log a warning but do
    not propagate — the primary deposit stays valid, the post's
    extraction still runs, and no edge is written.
    """
    primary_url_fn = getattr(importer, "primary_url", None)
    if primary_url_fn is None:
        return

    # Resolve the effective follow policy. Operator-explicit value
    # (True or False) wins; otherwise consult the importer's default.
    if follow_post_links is None:
        effective = bool(getattr(importer, "DEFAULT_FOLLOW_POST_LINKS", False))
    else:
        effective = follow_post_links
    if not effective:
        return

    # Load the deposited blob via its content_hash so the importer's
    # parser can read what it just wrote. Cheap on a single-user PKM;
    # avoids changing the ImporterPlugin protocol to return content.
    from particles.corpus.store import get_entry

    entry = await get_entry(session, entry_id)
    if entry is None:
        return
    snap = next((s for s in entry.snapshots if s.snapshot_id == snapshot_id), None)
    if snap is None:
        return
    try:
        content = load_blob(snap.content_hash)
    except FileNotFoundError:
        log.warning(
            "Follow skipped for %s — blob %s not found on disk",
            entry_id[:8],
            snap.content_hash[:12],
        )
        return

    # The importer's parser is deterministic — wrap broadly because
    # importer authors don't always exhaustively validate input shape
    # and we don't want a malformed blob to fail the primary deposit.
    try:
        target_url = primary_url_fn(content)
    except Exception as exc:
        log.warning("primary_url parser raised on %s: %s", entry_id[:8], exc)
        return
    if not target_url:
        log.info("No primary URL found for %s; follow skipped", entry_id[:8])
        return

    log.info("Following primary URL of %s: %s", entry_id[:8], target_url)
    try:
        # Hardcode depth-1: the recursive call cannot itself trigger
        # further follows.
        target_entry_id, target_snap_id = await deposit_url(
            session,
            target_url,
            deposited_by=deposited_by,
            tags=tags,
            follow_post_links=False,
            follow_comment_links=False,
        )
    except Exception as exc:
        log.warning(
            "Follow failed for %s (via %s): %s; primary deposit unaffected",
            target_url,
            entry_id[:8],
            exc,
        )
        return

    # Record the relationship. add_follow_edge is idempotent on
    # (via, target, link_type) so re-deposits don't create duplicates.
    from particles.corpus.follow_edges import LINK_TYPE_POST, add_follow_edge

    await add_follow_edge(
        session,
        via_entry_id=entry_id,
        target_entry_id=target_entry_id,
        link_type=LINK_TYPE_POST,
    )
    log.info("Recorded follow edge: %s → %s (POST_LINK)", entry_id[:8], target_entry_id[:8])

    if out_follow_targets is not None:
        out_follow_targets.append((target_entry_id, target_snap_id, target_url))


async def _reconcile_mentions_for_deposit(
    session: AsyncSession, *, url: str, entry_id: str
) -> None:
    """Bind prior undeposited mentions of a freshly-deposited URL.

    When a URL that other sources had only *cited* is finally deposited, its
    ``url_mentions`` rows reconcile to the new entry and each citing source
    gains a ``COMMENT_LINK`` follow edge — closing the
    suggestion → deposit loop and finally using the reserved sentinel. Best-
    effort: any failure is logged and swallowed so it never fails the deposit.
    """
    try:
        from particles.corpus.follow_edges import LINK_TYPE_COMMENT, add_follow_edge
        from particles.store.url_mention_store import reconcile_url_to_entry
        from particles.url_canonical import canonicalize_url

        canon = canonicalize_url(url)
        if canon is None:
            return
        vias = await reconcile_url_to_entry(session, canonical_url=canon, target_entry_id=entry_id)
        for via in vias:
            if via == entry_id:
                continue  # an entry that cited its own URL doesn't self-link
            await add_follow_edge(
                session,
                via_entry_id=via,
                target_entry_id=entry_id,
                link_type=LINK_TYPE_COMMENT,
            )
        if vias:
            log.info(
                "Reconciled %d prior mention(s) of %s → entry %s (COMMENT_LINK)",
                len(vias),
                canon,
                entry_id[:8],
            )
    except Exception as exc:
        log.warning("URL-mention reconciliation failed for entry %s: %s", entry_id[:8], exc)


@traced("deposit")
async def deposit_text(
    session: AsyncSession,
    text: str,
    deposited_by: str = "operator",
    source_type: str = SourceType.CONVERSATION,
    tags: list[str] | None = None,
    author_id: str | None = None,
) -> tuple[str, str]:
    """Deposit raw text (e.g. a conversation transcript).

    ``author_id`` records the principal the content is attributed to. For an
    agent assertion this is the server-bound asserter identity, which
    the §6.4 AUTHOR trust tier keys on (so agent-asserted beliefs rank below
    operator content). Returns (entry_id, snapshot_id).
    """
    content = text.encode("utf-8")
    content_hash = sha256(content)
    archive_path = save_blob(content, content_hash)

    return await write_entry_and_snapshot(
        session=session,
        uri_r=None,
        source_type=source_type,
        mutability=Mutability.STABLE,
        fetch_policy=FetchPolicy.NEVER,
        content=content,
        content_hash=content_hash,
        archive_path=archive_path,
        deposited_by=deposited_by,
        tags=tags or [],
        warc_record_type=WarcRecordType.RESPONSE,
        author_id=author_id,
    )


async def deposit_text_versioned(
    session: AsyncSession,
    *,
    text: str,
    uri_r: str,
    source_type: str,
    mutability: Mutability,
    tags: list[str] | None = None,
    deposited_by: str = "operator",
    content_published_at: datetime | None = None,
    fetch_policy: FetchPolicy | None = None,
) -> tuple[str, str, bool]:
    """Deposit text under a stable URI-R identity, as a no-op when unchanged.

    The Claude Code harvest hook re-deposits the same logical source repeatedly
    (a session transcript at every SessionEnd; a memory file whenever the agent
    edits it). Identity is the caller-supplied ``uri_r`` — every harvest of the
    same source lands on **one corpus entry** — and an unchanged re-deposit is
    detected up front: when the entry's latest snapshot already carries this
    content hash, nothing is written and the existing IDs are returned with
    ``unchanged=True``. A changed re-deposit appends a snapshot under the
    entry's mutability semantics (``APPEND_ONLY`` → the pipeline delta-extracts
    the tail, existing particles stay ACTIVE; ``MUTABLE`` → the prior
    snapshot's particles go PROVENANCE_STALE and the new snapshot is fully
    re-extracted).

    ``fetch_policy`` defaults to ``NEVER``, keeping every
    pre-0207 call site unchanged. An explicit value is **reconciled onto an
    existing entry even when the content is unchanged**: an entry's policy is
    not its content, and a rule file's normal state is "unchanged", so a
    reconciliation that only ran on the changed path would never enrol the
    common case. A prior promise was that the harvest paths would opt into
    ``LAZY``; this parameter is how they do it. Passing nothing never mutates a
    stored policy.

    Returns ``(entry_id, snapshot_id, unchanged)``; does not commit — the
    caller owns the transaction.
    """
    content = text.encode("utf-8")
    content_hash = sha256(content)

    existing = await get_entry_by_uri(session, uri_r)
    if existing is not None:
        await _reconcile_fetch_policy(session, existing, fetch_policy)
        snapshots = await list_snapshots_for_entry(session, existing.entry_id)
        if snapshots:
            latest = max(snapshots, key=lambda s: s.captured_at)
            if latest.content_hash == content_hash:
                return existing.entry_id, latest.snapshot_id, True

    archive_path = save_blob(content, content_hash)
    entry_id, snapshot_id = await write_entry_and_snapshot(
        session=session,
        uri_r=uri_r,
        source_type=source_type,
        mutability=mutability,
        fetch_policy=fetch_policy if fetch_policy is not None else FetchPolicy.NEVER,
        content=content,
        content_hash=content_hash,
        archive_path=archive_path,
        deposited_by=deposited_by,
        tags=tags or [],
        warc_record_type=WarcRecordType.RESPONSE,
        content_published_at=content_published_at,
    )
    return entry_id, snapshot_id, False


async def _reconcile_fetch_policy(
    session: AsyncSession,
    existing: CorpusEntry,
    requested: FetchPolicy | None,
) -> None:
    """Bring an existing entry's ``fetch_policy`` in line with an explicit request.

    ``write_entry_and_snapshot`` only sets the policy when it creates the entry
    row, so a re-deposit of a known URI-R would otherwise keep whatever policy
    the first deposit chose. The opposite is needed: an entry the
    harvest owns must be able to *enrol* in the refresh loop on a
    later run, including when the file has not changed at all.

    A ``None`` request is silence, not ``NEVER`` — it leaves the stored value
    alone.
    """
    if requested is None or existing.fetch_policy == requested:
        return
    row = await session.get(CorpusEntryRow, existing.entry_id)
    if row is None:
        return
    row.fetch_policy = requested.value
    await session.flush()
    log.info(
        "Updated fetch_policy for entry %s: %s → %s",
        existing.entry_id,
        existing.fetch_policy.value,
        requested.value,
    )


async def write_entry_and_snapshot(
    session: AsyncSession,
    uri_r: str | None,
    source_type: str,
    mutability: Mutability,
    fetch_policy: FetchPolicy,
    content: bytes,
    content_hash: str,
    archive_path: str,
    deposited_by: str,
    tags: list[str],
    warc_record_type: WarcRecordType,
    etag: str | None = None,
    last_modified: datetime | None = None,
    author_id: str | None = None,
    content_published_at: datetime | None = None,
) -> tuple[str, str]:
    """Core deposit logic: check for existing entry, create/update, return IDs."""
    # Check for existing entry: URI-R first, then content hash (deduplicates same
    # file deposited from different paths or re-deposited after a move).
    existing: CorpusEntry | None = None
    if uri_r:
        existing = await get_entry_by_uri(session, uri_r)
    if existing is None:
        existing = await get_entry_by_content_hash(session, content_hash)
        if existing is not None:
            log.info(
                "Deposit matched existing entry %s by content hash (skipping duplicate)",
                existing.entry_id,
            )

    now = datetime.now(UTC)

    snap = Snapshot(
        captured_at=now,
        content_hash=content_hash,
        etag=etag,
        last_modified=last_modified,
        warc_record_type=warc_record_type,
        archive_path=archive_path,
        extraction_status=ExtractionStatus.PENDING,
        author_id=author_id,
        content_published_at=content_published_at,
    )
    snap_row = SnapshotRow.from_model(
        snap,
        entry_id=existing.entry_id if existing else "TBD",
    )

    if existing:
        snap_row.entry_id = existing.entry_id
        session.add(snap_row)
        # If the new source_type is more specific than the existing one, update it.
        # This handles re-depositing a WEB_PAGE URL as WIKIDATA_API after the
        # domain-aware extractor is added.
        if existing.source_type != source_type:
            row = await session.get(CorpusEntryRow, existing.entry_id)
            if row is not None:
                row.source_type = source_type
                log.info(
                    "Updated source_type for entry %s: %s → %s",
                    existing.entry_id,
                    existing.source_type,
                    source_type,
                )
        await session.flush()
        log.info("Added snapshot %s to existing entry %s", snap.snapshot_id, existing.entry_id)
        resolved_entry_id = existing.entry_id
    else:
        entry = CorpusEntry(
            uri_r=uri_r,
            source_type=source_type,
            mutability=mutability,
            fetch_policy=fetch_policy,
            created_at=now,
            deposited_by=deposited_by,
            tags=tags,
        )
        entry_row = CorpusEntryRow.from_model(entry)
        snap_row.entry_id = entry.entry_id
        session.add(entry_row)
        session.add(snap_row)
        await session.flush()
        log.info("Deposited new entry %s (snapshot %s)", entry.entry_id, snap.snapshot_id)
        resolved_entry_id = entry.entry_id

    # cap. 2: capture this document's supersession relation via the
    # genre adapter (the ADR adapter reads `supersedes:` / `superseded_by:`
    # frontmatter). A no-op for every non-genre source — the §6.6 rung-1.5 prior
    # follows the stamped relation transitively at reconciliation time. Stamped
    # in the re-deposit branch too, so an edited `supersedes:` is picked up.
    if get_config().document_supersession.enabled:
        from particles.corpus.supersession import (
            document_relation_for_content,
            set_document_relation,
        )

        relation = document_relation_for_content(content)
        if relation is not None:
            await set_document_relation(session, resolved_entry_id, relation)

    return resolved_entry_id, snap.snapshot_id


def blob_exists(content_hash: str) -> bool:
    """Cheap presence probe for a content-addressed blob — no read, no raise.

    Lets planners (reindex ``--dry-run`` / the upfront work plan) flag a
    snapshot whose extraction would fail with ``FileNotFoundError`` before
    any LLM spend. A malformed hash reports absent rather than raising.
    """
    try:
        return blob_path(content_hash).exists()
    except ValueError:
        return False


def load_blob(content_hash: str) -> bytes:
    """Retrieve raw content from the blob store by SHA-256 hash.

    Raises:
        FileNotFoundError: If no blob exists for the given hash.
    """
    path = blob_path(content_hash)
    if not path.exists():
        raise FileNotFoundError(f"Blob not found for hash {content_hash}")
    return path.read_bytes()


def _parse_last_modified(value: str | None) -> datetime | None:
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None
