# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The agent-memory region render, wherever the store is.

The ``MEMORY.md`` render-splice cycle is split along the line: the
client owns the memory directory and the engine owns the store. This module is
the store half. It renders the region body from the client's own manifest and
answers the one store question the fold asks: for each pointer
target the client names, whether the region just rendered shows a belief that
file currently states. The local cycle calls it inside a session it opens, and
the engine's ``POST /memory/projection/{store}`` route calls it for a thin
client, so the two paths render the same bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from particles.operations.projection.manifest import DocManifest, MechanicalBlock
from particles.operations.projection.project import project_splice_body
from particles.render.markdown import parse_memory_bullet

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

#: Why a pointer target is not represented by the region.
RegionVerdict = Literal["not-extracted", "not-in-region"]


class ClientFileSectionError(ValueError):
    """A manifest section includes a file the renderer cannot read.

    A mechanical block names a path resolved against the manifest's own
    directory, which is on the client. The engine never reads it, so a remote
    render of such a manifest is refused.
    """


@dataclass(frozen=True)
class MemoryRegion:
    """One rendered region body and the fold verdicts asked of it."""

    body: str
    #: SHA-256 of ``body``'s UTF-8 bytes, so a client can tell it received the
    #: body the engine rendered.
    content_hash: str
    #: Per fold target (a file URI as the harvest deposited it): ``None`` when
    #: the region represents the file, otherwise why it does not.
    fold_verdicts: dict[str, RegionVerdict | None] = field(default_factory=dict)


def body_hash(body: str) -> str:
    """The content hash a :class:`MemoryRegion` carries."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def observer_trailer(observer_project: str, engaged: bool) -> str:
    """The region's last line under a project observer.

    It names the observer the region was rendered through, which is both the
    disclosure and what the SessionStart freshness check reads: a region
    rendered through a different observer, or through none, is never "current".
    """
    if engaged:
        return f"<!-- observer: project {observer_project} -->"
    return (
        f"<!-- observer: project {observer_project} NOT applied — this store has not been "
        "rescoped (`particles memory rescope`); the whole store is shown -->"
    )


def region_short_ids(body: str) -> set[str]:
    """The short ids the region's memory bullets cite."""
    ids: set[str] = set()
    for line in body.splitlines():
        parsed = parse_memory_bullet(line)
        if parsed is not None:
            ids.add(parsed[0])
    return ids


def has_client_file_section(manifest: DocManifest) -> bool:
    """Whether any section of ``manifest`` includes a file (a mechanical block)."""
    return any(isinstance(section, MechanicalBlock) for section in manifest.sections)


async def fold_verdicts(
    session: AsyncSession, target_uris: Collection[str], region_ids: Collection[str]
) -> dict[str, RegionVerdict | None]:
    """Per target file URI: ``None`` when represented, else why not.

    Represented means the file's newest live ``RESPONSE`` snapshot is
    ``COMPLETE`` and the region shows an ``ACTIVE`` belief anchored to that
    same snapshot. ``region_ids`` are the short ids the region's bullets cite.
    """
    from particles.core.schema import ExtractionStatus
    from particles.corpus.store import get_entry_by_uri, get_newest_response_snapshot
    from particles.store.particle_store import get_active_particle_ids_for_snapshot

    verdicts: dict[str, RegionVerdict | None] = {}
    for uri in target_uris:
        entry = await get_entry_by_uri(session, uri)
        newest = (
            None if entry is None else await get_newest_response_snapshot(session, entry.entry_id)
        )
        if entry is None or newest is None or newest[1] is not ExtractionStatus.COMPLETE:
            verdicts[uri] = "not-extracted"
            continue
        ids = await get_active_particle_ids_for_snapshot(session, entry.entry_id, newest[0])
        shown = any(pid.startswith(sid) for pid in ids for sid in region_ids)
        verdicts[uri] = None if shown else "not-in-region"
    return verdicts


async def render_memory_region(
    session: AsyncSession,
    manifest: DocManifest,
    *,
    base_dir: Path | None,
    observer_project: str | None,
    exclude_source_tags: Collection[str],
    fold_targets: Collection[str] = (),
) -> MemoryRegion:
    """Render the memory-index region body and answer its fold verdicts.

    Deterministic and budget-enforced: no synthesis, so no model call.
    ``base_dir`` is the manifest's directory, against which a mechanical
    block's file is read; ``None`` means the renderer cannot see the client's
    files, and a manifest with a block raises :class:`ClientFileSectionError`.
    ``fold_targets`` are read after the render, against the body just rendered,
    which is the order the local cycle has always used.
    """
    from particles.operations.query.observer_scope import lens_may_engage

    if base_dir is None and has_client_file_section(manifest):
        raise ClientFileSectionError(
            "the memory manifest includes a file section, which a remote engine cannot read"
        )
    result = await project_splice_body(
        session,
        manifest,
        base_dir=base_dir if base_dir is not None else Path(),
        synthesize=False,
        observer_project=observer_project,
        exclude_source_tags=exclude_source_tags,
    )
    body = result.document
    if observer_project is not None:
        engaged = await lens_may_engage(session)
        body = body.rstrip("\n") + "\n" + observer_trailer(observer_project, engaged) + "\n"
    verdicts = (
        await fold_verdicts(session, fold_targets, region_short_ids(body)) if fold_targets else {}
    )
    return MemoryRegion(body=body, content_hash=body_hash(body), fold_verdicts=verdicts)
