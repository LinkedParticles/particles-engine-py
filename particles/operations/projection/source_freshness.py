# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Advisory bundle-vs-repo source freshness for a gated projection.

The drift gate checks a projected document against its committed bundle, never
against the repository the bundle was extracted from. A pinned
claim whose source document has changed since extraction therefore stays in the
document until an operator re-renders. This module closes the reporting half of
that gap without touching the gate:

- :func:`bundle_source_records` (Engine, store-reading) runs at
  ``particles project --export-corpus`` and names, for each bundled particle,
  the repository file each of its ``SOURCE`` snapshots was deposited from and
  the SHA-256 of the bytes that snapshot holds. They are written beside the
  bundle as ``<name>.sources.json``.
- :func:`stale_bundle_sources` (no store) compares those digests with the
  files in the checkout now and returns what changed or went missing.

The check is advisory by construction: it returns findings and never raises on
bad data, and its callers print them without changing an exit code. A source
outside the repository (a URL, a file elsewhere on the exporting machine) is
not recorded, since a checkout cannot answer for it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from particles.config import get_config
from particles.core.schema import ProvenanceRefType
from particles.corpus.fetch import path_from_file_uri
from particles.corpus.local_read import read_local_file, within
from particles.corpus.store import get_entry_uri_map, get_snapshot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.core.schema import Particle
    from particles.operations.projection.manifest import DocManifest

#: The sidecar's format version, bumped on an incompatible change.
SOURCES_FORMAT = 1


def sources_path_for(manifest: DocManifest, *, base_dir: Path) -> Path:
    """The ``<name>.sources.json`` sidecar beside the manifest's bundle."""
    return base_dir / f"{manifest.name}.sources.json"


@dataclass(frozen=True)
class StaleSource:
    """One bundled particle whose recorded source no longer matches the checkout."""

    particle_id: str
    path: str
    reason: Literal["changed", "missing", "unreadable"]


def _repo_relative(path: Path, roots: list[Path]) -> str | None:
    """``path`` relative to the first root holding it, as a POSIX string, else ``None``."""
    for root in roots:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return None


async def bundle_source_records(
    session: AsyncSession, particles: list[Particle], *, repo_roots: list[Path]
) -> list[dict[str, str]]:
    """``{particle, path, sha256}`` for each repository source of each particle.

    ``repo_roots`` are the directories a recorded path is made relative to,
    tried in order (a worktree, then its main checkout, which is where the
    corpus was most likely deposited from). A source under none of them, or
    not a ``file://`` entry, is left out. Sorted for byte-stable output.
    """
    roots = [root.resolve() for root in repo_roots]
    refs = [
        (p.id, ref.corpus_entry_id, ref.snapshot_id)
        for p in particles
        for ref in p.provenance
        if ref.type is ProvenanceRefType.SOURCE and ref.corpus_entry_id and ref.snapshot_id
    ]
    uris = await get_entry_uri_map(session, {entry_id for _, entry_id, _ in refs})
    records: set[tuple[str, str, str]] = set()
    digests: dict[str, str | None] = {}
    for particle_id, entry_id, snapshot_id in refs:
        uri = uris.get(entry_id)
        if not uri or not uri.startswith("file:"):
            continue
        try:
            relative = _repo_relative(path_from_file_uri(uri), roots)
        except ValueError:
            continue
        if relative is None:
            continue
        if snapshot_id not in digests:
            snapshot = await get_snapshot(session, snapshot_id)
            digests[snapshot_id] = snapshot.content_hash if snapshot is not None else None
        digest = digests[snapshot_id]
        if digest:
            records.add((particle_id, relative, digest))
    return [
        {"particle": pid, "path": path, "sha256": digest} for pid, path, digest in sorted(records)
    ]


def encode_source_records(records: list[dict[str, str]]) -> str:
    """The sidecar's text: deterministic JSON, one record per line."""
    body = ",\n".join(f"    {json.dumps(r, sort_keys=True)}" for r in records)
    sources = f"[\n{body}\n  ]" if records else "[]"
    return f'{{\n  "format": {SOURCES_FORMAT},\n  "sources": {sources}\n}}\n'


def _safe_relative(path: object) -> PurePosixPath | None:
    """A recorded path that stays inside the checkout, else ``None``."""
    if not isinstance(path, str) or not path:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    return pure


def _file_digest(path: Path, root: Path) -> str | None:
    """SHA-256 of a regular file read contained to ``root``.

    ``None`` when no file is there; :data:`_UNREADABLE` when one is but the
    contained read refuses it (a link, a hard link, a file over
    ``local_files.max_bytes``), so the report does not call it missing.
    """
    try:
        content = read_local_file(
            path, max_bytes=get_config().local_files.max_bytes, allowed=within(root)
        ).content
    except FileNotFoundError:
        return None
    except OSError:
        return _UNREADABLE
    return hashlib.sha256(content).hexdigest()


#: The digest stand-in for a file that exists but could not be read.
_UNREADABLE = "unreadable"


def stale_bundle_sources(sources_text: str, repo_root: Path) -> list[StaleSource] | None:
    """The recorded sources that changed or vanished in the checkout at ``repo_root``.

    Returns ``None`` when ``sources_text`` is not a sidecar this version reads,
    so the caller can say the check did not run rather than report a clean
    result. Each distinct file is hashed once. A recorded path that would leave
    the checkout is skipped, never read.
    """
    try:
        data: Any = json.loads(sources_text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("format") != SOURCES_FORMAT:
        return None
    records = data.get("sources")
    if not isinstance(records, list):
        return None
    root = repo_root.resolve()
    digests: dict[PurePosixPath, str | None] = {}
    stale: list[StaleSource] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        relative = _safe_relative(record.get("path"))
        particle_id = record.get("particle")
        recorded = record.get("sha256")
        if relative is None or not isinstance(particle_id, str) or not isinstance(recorded, str):
            continue
        if relative not in digests:
            digests[relative] = _file_digest(root.joinpath(*relative.parts), root)
        current = digests[relative]
        if current is None:
            stale.append(StaleSource(particle_id, relative.as_posix(), "missing"))
        elif current == _UNREADABLE:
            stale.append(StaleSource(particle_id, relative.as_posix(), "unreadable"))
        elif current != recorded:
            stale.append(StaleSource(particle_id, relative.as_posix(), "changed"))
    return stale


def _record_count(sources_text: str) -> int:
    """How many records a sidecar ``stale_bundle_sources`` already accepted holds."""
    records = json.loads(sources_text).get("sources")
    return len(records) if isinstance(records, list) else 0


def advisory_lines(manifest: DocManifest, *, base_dir: Path, repo_root: Path) -> list[str]:
    """The advisory report for one manifest, ready to print (empty when fresh).

    Never raises: an unreadable or missing sidecar is itself one advisory line.
    A bundle with no sidecar yet says how to write one.
    """
    sidecar = sources_path_for(manifest, base_dir=base_dir)
    bundle = base_dir / f"{manifest.name}.corpus.jsonl"
    if not sidecar.is_file():
        if not bundle.is_file():
            return []
        return [
            f"advisory: {manifest.name}: no {sidecar.name} beside the bundle, so pinned "
            "claims are not checked against the current repository; it is written "
            "with the bundle the next time the bundle is exported from the store it "
            "was built in."
        ]
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        text = ""
    stale = stale_bundle_sources(text, repo_root)
    if stale is None:
        return [f"advisory: {manifest.name}: {sidecar.name} is not readable; source check skipped."]
    if _record_count(text) == 0:
        return [
            f"advisory: {manifest.name}: {sidecar.name} records no repository source, so no "
            "pinned claim is checked; the corpus was deposited from outside this checkout."
        ]
    lines = [
        f"advisory: {manifest.name}: pinned claim {s.particle_id[:8]} was extracted from "
        f"{s.path}, which has changed since; re-extract and re-render when it matters."
        for s in stale
        if s.reason == "changed"
    ]
    for path in sorted({s.path for s in stale if s.reason == "unreadable"}):
        lines.append(f"advisory: {manifest.name}: recorded source {path} could not be read.")
    missing = sorted({s.path for s in stale if s.reason == "missing"})
    if missing:
        lines.append(
            f"advisory: {manifest.name}: {len(missing)} recorded source file(s) are not in "
            f"this checkout ({', '.join(missing[:3])}{', …' if len(missing) > 3 else ''})."
        )
    return lines
