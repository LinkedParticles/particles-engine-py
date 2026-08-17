"""The MEMORY.md render-splice cycle + session-start freshness check.

Two consumers in ``cli/hook.py``:

- ``run_projection_cycle`` — the tail of every SessionEnd harvest cycle
  (harvest → extract → render → splice; the ordering is the §7 safety
  property): re-render the ``memory-index`` region from the store and splice
  it into ``MEMORY.md``, with fold-and-archive (§7, default-on), a one-deep
  backup, an atomic write, and a loud refusal on damaged sentinels.
- ``digest_decision`` — the double-occupancy freshness check for
  SessionStart: compare the projected region's sources trailer against the
  live selection and decide full push / skip / difference-only.

The module is Surface-tier (imports Engine freely); the sentinel parsing it
relies on lives in the Client-layer ``particles.render.markdown``, shared with
the harvest-side strip so splice and strip can never disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from particles.api.cli._claude_code import (
    ARCHIVE_POINTER_PREFIX,
    MEMORY_REGION,
    memory_archive_path,
    memory_backup_path,
    memory_manifest_path,
    memory_snapshot_path,
    projection_enabled,
)
from particles.config import get_config
from particles.render.markdown import (
    PROJECTED_BEGIN_TMPL,
    PROJECTED_END_TMPL,
    atomic_write_text,
    find_projected_regions,
    parse_sources_trailers,
)

log = logging.getLogger(__name__)


def archive_pointer_line() -> str:
    """The one line fold-and-archive leaves behind in MEMORY.md."""
    return (
        f"{ARCHIVE_POINTER_PREFIX} into the memory store; the moved originals "
        f"are archived at `{memory_archive_path()}`.*"
    )


# ---------------------------------------------------------------------------
# Fold-and-archive — pure text half
# ---------------------------------------------------------------------------


def fold_authored_lines(text: str, pointer_line: str) -> tuple[str, list[str]]:
    """Move authored lines outside every projected region out of ``text``.

    Returns ``(kept_text, folded_lines)``: ``kept_text`` is the projected
    region(s) verbatim followed by the archive ``pointer_line``;
    ``folded_lines`` is everything else non-blank (excluding any prior pointer
    line), in file order, destined for the append-only archive. When nothing
    is foldable, ``text`` is returned unchanged with an empty list — the
    pointer line is only introduced by an actual fold. Pure: the caller owns
    every write, in the §7 crash-safe order.
    """
    regions = find_projected_regions(text)
    if not regions:
        return text, []
    folded: list[str] = []
    cursor = 0
    outside_chunks: list[str] = []
    for region in regions:
        outside_chunks.append(text[cursor : region.start])
        cursor = region.end
    outside_chunks.append(text[cursor:])
    for chunk in outside_chunks:
        for line in chunk.splitlines():
            if line.strip() and not line.startswith(ARCHIVE_POINTER_PREFIX):
                folded.append(line)
    if not folded:
        return text, []
    kept = "\n\n".join(text[r.start : r.end] for r in regions)
    return kept + "\n\n" + pointer_line + "\n", folded


def _append_to_archive(lines: list[str], source_name: str) -> None:
    """Append folded lines to the state-dir archive (append-only, with a receipt)."""
    path = memory_archive_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n<!-- folded from {source_name} -->\n" + "\n".join(lines) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(block)


# ---------------------------------------------------------------------------
# The render-splice cycle (§6/§7)
# ---------------------------------------------------------------------------


async def _render_region_body(store: str) -> str:
    """Deterministic, budget-enforced render of the memory-index splice body."""
    from particles.db import session_scope
    from particles.operations.projection import load_manifest, project_splice_body

    manifest_path = memory_manifest_path()
    manifest = load_manifest(manifest_path)
    async with session_scope(store) as session:
        result = await project_splice_body(
            session, manifest, base_dir=manifest_path.parent, synthesize=False
        )
    return result.document


async def run_projection_cycle(
    store: str, memory_dir: Path, harvested_text: str | None
) -> dict[str, Any]:
    """Render → splice the memory-index region after a successful harvest.

    ``harvested_text`` is the raw ``MEMORY.md`` content the harvest step read
    (``None`` when the file did not exist then): the cycle refuses to touch a
    file that changed since it was harvested — the §7 invariant is *never
    corrupt or delete memory that hasn't been harvested*. All failures are
    contained: the returned dict is hook-log telemetry, and no partial write
    is ever visible (temp-file + ``os.replace``).

    Steps, in the §7 crash-safe order: render (from the store) → fold
    (compute, pure) → splice (compute; ``SpliceError`` refuses loudly) →
    backup (one-deep) → archive-append → atomic write → snapshot.
    """
    from particles.api.cli._projection_git import new_run_id
    from particles.operations.projection import SpliceError, splice_region

    run_id = new_run_id()
    try:
        body = await _render_region_body(store)
        manifest_ref = str(memory_manifest_path())
        memory_md = memory_dir / "MEMORY.md"
        current = memory_md.read_text(encoding="utf-8") if memory_md.is_file() else None

        if current is None and harvested_text is not None:
            return {"skipped": "file-vanished-since-harvest"}
        if current is not None and harvested_text is not None and current != harvested_text:
            # Someone wrote between harvest and render: that content has not
            # been harvested, so this cycle must not touch the file (§7).
            return {"skipped": "changed-since-harvest"}
        if current is not None and harvested_text is None:
            # The file appeared after the harvest step ran — same situation.
            return {"skipped": "changed-since-harvest"}

        if current is None:
            # No MEMORY.md at all: create it holding just the region — there
            # is no foreign content at risk (§7; init seeds existing files).
            begin = PROJECTED_BEGIN_TMPL.format(region=MEMORY_REGION, manifest=manifest_ref)
            end = PROJECTED_END_TMPL.format(region=MEMORY_REGION)
            stripped_body = body.strip("\n")
            created = f"{begin}\n{stripped_body}\n{end}\n"
            memory_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(memory_md, created)
            _write_snapshot(body)
            result: dict[str, Any] = {"outcome": "created", "run_id": run_id}
            git = await _maybe_commit(
                memory_dir, store=store, outcome="created", body=body, snapshot=None, run_id=run_id
            )
            if git is not None:
                result["git"] = git
            return result

        # Drift telemetry (§6): a dirty region's content was already deposited
        # as authored input by this cycle's harvest strip; the re-render below
        # routes it through the ladder instead of destroying it.
        snapshot = (
            memory_snapshot_path().read_text(encoding="utf-8")
            if memory_snapshot_path().is_file()
            else None
        )
        target = next(
            (r for r in find_projected_regions(current) if r.region == MEMORY_REGION), None
        )
        dirty = target is not None and (
            snapshot is None or target.body.strip("\n") != snapshot.strip("\n")
        )

        # Fold-and-archive (§7, default-on): compute first, write later.
        folded: list[str] = []
        text_to_splice = current
        if get_config().agent_memory.projection.fold_authored_lines:
            text_to_splice, folded = fold_authored_lines(current, archive_pointer_line())

        try:
            spliced = splice_region(text_to_splice, MEMORY_REGION, body, manifest=manifest_ref)
        except SpliceError as exc:
            # Missing / duplicated / inverted sentinels: report and skip —
            # never "helpfully" regenerate the whole file (§7).
            log.warning("memory projection splice refused: %s", exc)
            return {"skipped": "splice-error", "error": str(exc)}

        backup = memory_backup_path()
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(current, encoding="utf-8")
        if folded:
            _append_to_archive(folded, memory_md.name)
        atomic_write_text(memory_md, spliced)
        _write_snapshot(body)
        result = {
            "outcome": "rendered",
            "dirty_region": dirty,
            "folded": len(folded),
            "run_id": run_id,
        }
        git = await _maybe_commit(
            memory_dir, store=store, outcome="rendered", body=body, snapshot=snapshot, run_id=run_id
        )
        if git is not None:
            result["git"] = git
        return result
    except Exception as exc:  # noqa: BLE001 — hook telemetry, never a failure surface
        log.warning("memory projection cycle failed", exc_info=True)
        return {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"}


def _write_snapshot(body: str) -> None:
    """Persist the just-spliced region body — the §6 drift/pristine reference."""
    path = memory_snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip("\n") + "\n", encoding="utf-8")


async def _maybe_commit(
    memory_dir: Path,
    *,
    store: str,
    outcome: str,
    body: str,
    snapshot: str | None,
    run_id: str,
) -> str | None:
    """Optional git-versioned history of the just-written render.

    Returns ``None`` when the feature is off (config gate read at call time), else the best-effort commit's telemetry string. The commit is a
    bonus (never raises, never alters the projection outcome);
    ``snapshot`` is the *previous* region body, so the delta compares old vs new.
    """
    if not get_config().agent_memory.projection.git.enabled:
        return None
    from particles.api.cli._projection_git import commit_projection

    return await commit_projection(
        memory_dir, store=store, outcome=outcome, body=body, snapshot=snapshot, run_id=run_id
    )


# ---------------------------------------------------------------------------
# Session-start freshness check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DigestDecision:
    """What the SessionStart hook should inject."""

    action: Literal["full", "skip", "diff"]
    content: str | None = None


async def digest_decision(store: str, payload: dict[str, Any]) -> DigestDecision:
    """The double-occupancy freshness check (resolved).

    Reads the projected region's sources trailer from the ``MEMORY.md`` the
    harness will load, computes the live selection fingerprint (LLM-free,
    embedding-free), and decides:

    - **match** → ``skip``: the loaded file *is* the current view;
    - **mismatch** → ``diff``: inject only the difference — the fresh bullet
      lines (new / changed / newly-contested) absent from the loaded region;
    - **anything unparseable** (projection disabled, remote engine, no
      MEMORY.md, no region, no trailer) → ``full``: the plain digest push.
    """
    if not projection_enabled():
        return DigestDecision("full")
    from particles.api.client import get_backend

    if get_backend().remote:
        # The memory directory is machine-local; a remote store can't be
        # compared against it meaningfully — push the full digest.
        return DigestDecision("full")

    transcript_path = str(payload.get("transcript_path") or "")
    if not transcript_path:
        return DigestDecision("full")
    memory_md = Path(transcript_path).parent / "memory" / "MEMORY.md"
    if not memory_md.is_file():
        return DigestDecision("full")

    text = memory_md.read_text(encoding="utf-8", errors="replace")
    region = next((r for r in find_projected_regions(text) if r.region == MEMORY_REGION), None)
    if region is None:
        return DigestDecision("full")
    loaded_ids = parse_sources_trailers(region.body)
    if loaded_ids is None:
        return DigestDecision("full")

    fresh = await _render_region_body(store)
    fresh_ids = parse_sources_trailers(fresh)
    if fresh_ids is None:
        return DigestDecision("full")
    if fresh_ids == loaded_ids:
        return DigestDecision("skip")

    loaded_lines = set(region.body.splitlines())
    diff = [
        line for line in fresh.splitlines() if line.startswith("- ") and line not in loaded_lines
    ]
    if not diff:
        # Selection only shrank (or reordered): nothing new to top up — the
        # loaded region over-covers the current view.
        return DigestDecision("skip")
    header = (
        f"# Memory digest update — {store}\n\n"
        "_The loaded MEMORY.md projection is stale; beliefs new or changed since "
        "its last render:_\n\n"
    )
    return DigestDecision("diff", content=header + "\n".join(diff) + "\n")
