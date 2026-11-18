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
client, so the two paths render the same bytes. A thin client's manifest
blocks arrive as text the client read: the engine looks each one up
by the block's own string and never opens a path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from particles.operations.projection.manifest import DocManifest, MechanicalBlock
from particles.operations.projection.paths import read_block_text
from particles.operations.projection.project import project_splice_body
from particles.render.markdown import parse_memory_bullet

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

#: Why a pointer target is not represented by the region.
RegionVerdict = Literal["not-extracted", "not-in-region"]


#: The most UTF-8 bytes of block text one remote render may carry (
#: §4). A protocol bound, so a module constant both ends read rather than a
#: config field: a knob on the client would disagree with its engine. Four
#: times the default region budget and over twice the harness's 25 KB
#: session-start load window, so no block a ``MEMORY.md`` can usefully hold is
#: refused. The local render has no such cap.
MAX_REMOTE_BLOCK_BYTES = 64 * 1024


class ClientFileSectionError(ValueError):
    """A manifest section includes a file the renderer cannot read.

    A mechanical block names a path resolved against the manifest's own
    directory, which is on the client. The engine never reads it: the client
    sends each block's text with the render, and this is raised
    when it cannot or did not. The message names a block's position in the
    manifest and the reason, never the block's path or its text.
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


def reference_trailer(withheld: int) -> str:
    """The region's line saying how many reference beliefs it left out.

    A line of its own, before the observer trailer, so the freshness check's
    reading of the observer line is unchanged.
    """
    return (
        f"<!-- {withheld} reference belief(s) answer questions but are not listed "
        "(`particles query` finds them) -->"
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


def _block_positions(manifest: DocManifest) -> list[tuple[int, str]]:
    """``(1-based section position, block string)`` for each mechanical block."""
    return [
        (index, section.block)
        for index, section in enumerate(manifest.sections, start=1)
        if isinstance(section, MechanicalBlock)
    ]


def block_texts_size(block_texts: Mapping[str, str]) -> int:
    """The UTF-8 bytes the texts take on the wire, the quantity the cap bounds."""
    return sum(len(text.encode("utf-8")) for text in block_texts.values())


def read_block_texts(manifest: DocManifest, base_dir: Path) -> dict[str, str]:
    """Read each mechanical block's file on the client, for a remote render.

    Each file is read with the same call, and so the same containment rules,
    the local render uses (``paths.read_block_text`` with no outside paths): a
    relative path, no ``..`` or ``~``, inside the containment root, no
    symlinked final component, a regular file. A refused or missing file
    raises what the local render would raise. A total over
    :data:`MAX_REMOTE_BLOCK_BYTES` raises :class:`ClientFileSectionError`.
    """
    texts: dict[str, str] = {}
    for _position, block in _block_positions(manifest):
        if block not in texts:
            texts[block] = read_block_text(block, base_dir=base_dir, allow_outside=False)
    total = block_texts_size(texts)
    if total > MAX_REMOTE_BLOCK_BYTES:
        raise ClientFileSectionError(
            f"the memory manifest's file sections total {total} bytes, over the "
            f"{MAX_REMOTE_BLOCK_BYTES}-byte limit for a remote render"
        )
    return texts


def check_block_texts(manifest: DocManifest, block_texts: Mapping[str, str]) -> None:
    """Refuse texts that do not match the manifest's blocks exactly.

    Every block needs a text, every text needs a block, and the total stays
    under :data:`MAX_REMOTE_BLOCK_BYTES`. A key is compared, never opened. The
    message names positions and counts only, never a key or a text.
    """
    positions = _block_positions(manifest)
    named = {block for _position, block in positions}
    missing = [position for position, block in positions if block not in block_texts]
    if missing:
        listed = ", ".join(str(p) for p in missing)
        raise ClientFileSectionError(
            f"the request carries no text for the file section at manifest position {listed} "
        )
    extra = sum(1 for key in block_texts if key not in named)
    if extra:
        raise ClientFileSectionError(
            f"the request carries text for {extra} file section(s) the manifest does not name"
        )
    total = block_texts_size(block_texts)
    if total > MAX_REMOTE_BLOCK_BYTES:
        raise ClientFileSectionError(
            f"the request's file sections total {total} bytes, over the "
            f"{MAX_REMOTE_BLOCK_BYTES}-byte limit"
        )


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
    block_texts: Mapping[str, str] | None = None,
) -> MemoryRegion:
    """Render the memory-index region body and answer its fold verdicts.

    Deterministic and budget-enforced: no synthesis, so no model call.
    ``base_dir`` is the manifest's directory, against which a mechanical
    block's file is read; ``None`` means the renderer cannot see the client's
    files: each block's text then comes from ``block_texts``, which the client
    read and sent, and a manifest whose blocks the texts do not
    match exactly raises :class:`ClientFileSectionError`. With ``base_dir``
    given, ``block_texts`` is ignored and blocks are read from disk as before.
    ``fold_targets`` are read after the render, against the body just rendered,
    which is the order the local cycle has always used.
    """
    from particles.operations.query.observer_scope import lens_may_engage

    withheld: set[str] = set()
    texts: Mapping[str, str] | None = None
    if base_dir is None:
        texts = block_texts or {}
        if has_client_file_section(manifest) or texts:
            check_block_texts(manifest, texts)
    result = await project_splice_body(
        session,
        manifest,
        base_dir=base_dir if base_dir is not None else Path(),
        synthesize=False,
        observer_project=observer_project,
        exclude_source_tags=exclude_source_tags,
        reference_withheld_out=withheld,
        block_texts=texts,
    )
    body = result.document
    if withheld:
        body = body.rstrip("\n") + "\n" + reference_trailer(len(withheld)) + "\n"
    if observer_project is not None:
        engaged = await lens_may_engage(session)
        body = body.rstrip("\n") + "\n" + observer_trailer(observer_project, engaged) + "\n"
    verdicts = (
        await fold_verdicts(session, fold_targets, region_short_ids(body)) if fold_targets else {}
    )
    return MemoryRegion(body=body, content_hash=body_hash(body), fold_verdicts=verdicts)
