"""rules group — the rule-source set (operating docs as tracked sources).

``particles rules`` reports the resolved set and each file's registration state;
``particles rules sync`` deposits it as ``MUTABLE`` + ``LAZY`` so the loop keeps it fresh. The bare-verb-reports shape follows ``curate``.

The reporting verb exists because hand-written SQL was needed to
answer "is anything actually enrolled in the refresh loop?" — the answer on the
live store was *no*, and that should not require a database client to discover.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from particles.api.cli import app, run
from particles.api.cli._output import (
    DEBUG_OPTION,
    PROGRESS_OPTION,
    QUIET_OPTION,
    VERBOSE_OPTION,
    configure_output,
    narrate,
)
from particles.db import DEFAULT_STORE, session_scope

if TYPE_CHECKING:
    from particles.corpus.rule_sources import RuleSourceResolution

rules_app = typer.Typer(
    help="Operating-rule source documents tracked by this store.",
    no_args_is_help=False,
)
app.add_typer(rules_app, name="rules")

_STORE_OPTION = typer.Option(DEFAULT_STORE, "--store", help="Store handle to report on / write to.")


@rules_app.callback(invoke_without_command=True)
def rules_main(
    ctx: typer.Context,
    store: str = _STORE_OPTION,
    verbose: bool = VERBOSE_OPTION,
    debug: bool = DEBUG_OPTION,
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Show the resolved rule-source set and whether each file is tracked."""
    if ctx.invoked_subcommand is not None:
        return
    configure_output(verbose, debug, quiet, progress)
    run(_report(store))


@rules_app.command("sync")
def rules_sync_cmd(
    paths: list[Path] = typer.Argument(
        None,
        help=(
            "Files or directories to register, overriding rule_sources.paths for "
            "this run. Omit to use the configured (or discovered) set."
        ),
    ),
    store: str = _STORE_OPTION,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and print the set without depositing anything."
    ),
    restamp_only: bool = typer.Option(
        False,
        "--restamp-only",
        help=(
            "Skip the deposit half; only re-apply the scope exemption to "
            "particles already extracted from the tracked set."
        ),
    ),
    verbose: bool = VERBOSE_OPTION,
    debug: bool = DEBUG_OPTION,
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Deposit the rule-source set as MUTABLE + LAZY corpus entries."""
    configure_output(verbose, debug, quiet, progress)
    run(_sync([str(p) for p in paths] if paths else None, store, dry_run, restamp_only))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_resolution_notes(resolution: RuleSourceResolution) -> None:
    """Disclose discovery, truncation, and unresolvable roots — never silently."""
    if resolution.discovered:
        roots = ", ".join(str(r) for r in resolution.roots) or "(none found)"
        narrate(f"rule_sources.paths is empty — discovered roots: {roots}")
    for missing in resolution.missing:
        narrate(f"Warning: registered path does not exist: {missing}")
    if resolution.truncated:
        narrate(
            f"Warning: {resolution.truncated} more file(s) matched than "
            f"rule_sources.max_files allows and were dropped."
        )


async def _report(store: str) -> None:
    from particles.core.schema import FetchPolicy
    from particles.corpus.rule_sources import resolve_rule_sources
    from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry
    from particles.extraction.scope import is_scope_exempt_source

    if not _enabled_or_report():
        return

    resolution = resolve_rule_sources()
    _print_resolution_notes(resolution)
    if not resolution.files:
        typer.echo(
            "No rule sources resolved. Register paths in `rule_sources.paths`, or "
            "run `particles rules sync <path>` to track one now."
        )
        return

    tracked = enrolled = exempt = 0
    async with session_scope(store) as session:
        rows: list[tuple[str, str, str]] = []
        for path in resolution.files:
            entry = await get_entry_by_uri(session, path.as_uri())
            if entry is None:
                rows.append(("—", "not tracked", str(path)))
                continue
            tracked += 1
            exempt += int(is_scope_exempt_source(entry.tags))
            lazy = entry.fetch_policy is FetchPolicy.LAZY
            enrolled += int(lazy)
            snapshots = await list_snapshots_for_entry(session, entry.entry_id)
            checked = (
                max(snapshots, key=lambda s: s.captured_at).captured_at.strftime("%Y-%m-%d")
                if snapshots
                else "never"
            )
            state = f"{entry.fetch_policy.value.lower():<5} · {len(snapshots)} snap · {checked}"
            rows.append((entry.entry_id[:8], state, str(path)))

    for entry_id, state, shown in rows:
        typer.echo(f"  {entry_id:<8}  {state:<32}  {shown}")
    typer.echo(
        f"\n{len(resolution.files)} rule source(s): {tracked} tracked, "
        f"{enrolled} enrolled in the refresh loop (fetch_policy=LAZY), "
        f"{exempt} exempt from the document-meta exclusion."
    )
    if tracked and not exempt:
        # Legible rather than mysterious: an operator who emptied
        # `extraction_scope.exempt_source_tags` should see why their rules are
        # still missing from the projection.
        typer.echo(
            "No tracked entry is exempt — `extraction_scope.exempt_source_tags` does not "
            "match their tags, so a rule classified DOCUMENT_META stays off the default surface."
        )
    if tracked < len(resolution.files):
        typer.echo("Run `particles rules sync` to track the rest.")
    elif enrolled < tracked:
        # a file deposited with its projected regions stripped is
        # not byte-identical to its snapshot, so the byte-level tier
        # would see a change every night. `rules sync` refreshes it instead.
        typer.echo(
            f"{tracked - enrolled} file(s) carry a projected region, so their deposited "
            "body differs from their bytes; re-run `particles rules sync` to refresh those."
        )


def _enabled_or_report() -> bool:
    from particles.config import get_config

    if get_config().rule_sources.enabled:
        return True
    typer.echo("Rule-source tracking is off (`rule_sources.enabled: false`).")
    return False


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def projected_region_filter() -> Callable[[str], str]:
    """The deposit-body transform: strip pristine projected regions.

    The README was made to carry sentinel regions, and nothing stops an
    ``AGENTS.md`` from doing the same; depositing one unfiltered would feed the
    store its own rendered output and break belt 1 of the round-trip contract.
    The renderer's own strip is reused — not a second regex — so strip and
    splice can never disagree about where a region begins.

    It lives on this side of the seam because the region snapshots live in the
    CLI state directory: a direct reach from ``particles.corpus`` would close a
    ``store → corpus → render → store`` subpackage cycle.
    """
    from particles.api.cli._claude_code import load_projection_snapshots
    from particles.render.markdown import strip_projected_regions_for_deposit

    snapshots = load_projection_snapshots()

    def strip(text: str) -> str:
        return strip_projected_regions_for_deposit(text, snapshots)

    return strip


async def _restamp(session: Any, files: list[Path]) -> tuple[int, int]:
    """Re-apply the scope exemption to already-extracted rule sources.

    Returns ``(entries, particles)``. The stamp is a deterministic function of
    the entry's tags, so this is a policy re-application rather than a
    re-extraction — no LLM call, no new snapshot. It runs on every ``rules
    sync`` because a rule file's normal state is *unchanged*: gating it on a
    content change would mean the exemption never reached the files it exists
    for. Idempotent, so the steady state reports 0.

    Lives here rather than in ``particles.corpus.rule_sources`` because the
    corpus layer must not reach into the particle store — that back-edge was
    deliberately retired and re-adding it would fail the acyclic
    contract. The CLI is a Surface, so it may call both.
    """
    from particles.corpus.store import get_entry_by_uri
    from particles.extraction.scope import is_scope_exempt_source
    from particles.store.particle_store import stamp_scope_exemption_for_entry

    entries = particles = 0
    for path in files:
        entry = await get_entry_by_uri(session, path.as_uri())
        if entry is None or not is_scope_exempt_source(entry.tags):
            continue
        changed = await stamp_scope_exemption_for_entry(session, entry.entry_id)
        if changed:
            entries += 1
            particles += changed
    return entries, particles


async def _sync(paths: list[str] | None, store: str, dry_run: bool, restamp_only: bool) -> None:
    from particles.api.cli._remote import ensure_local
    from particles.corpus.rule_sources import RuleSyncReport, resolve_rule_sources
    from particles.corpus.rule_sources import sync_rule_sources as _sync_rule_sources

    if not _enabled_or_report():
        return
    ensure_local("rules sync")

    restamped = (0, 0)
    async with session_scope(store, write=True) as session:
        if restamp_only:
            report = RuleSyncReport(resolution=resolve_rule_sources(paths))
        else:
            report = await _sync_rule_sources(
                session,
                paths,
                dry_run=dry_run,
                filter_text=projected_region_filter(),
            )
        if not dry_run:
            # After the deposits, so an entry registered in this same run is
            # stamped too (its particles arrive at the next `extract`, which
            # stamps them itself — this covers the ones already there).
            restamped = await _restamp(session, report.resolution.files)
            await session.commit()

    _print_resolution_notes(report.resolution)
    if not report.resolution.files:
        typer.echo(
            "No rule sources resolved — nothing to sync. Register paths in "
            "`rule_sources.paths`, or pass them as arguments."
        )
        return

    verb = "would register" if dry_run else "registered"
    for path, _entry_id in report.deposited:
        typer.echo(f"  {verb}  {path}")
    for path in report.skipped_empty:
        narrate(f"  skipped (no authored content after the projected-region strip)  {path}")
    for path, error in report.failed:
        narrate(f"  error  {path}: {error}")

    if dry_run:
        typer.echo(f"\n--dry-run: {len(report.deposited)} file(s) would be registered.")
        return

    stamped_entries, stamped_particles = restamped
    if stamped_entries:
        typer.echo(
            f"  scope exemption applied to {stamped_particles} particle(s) across "
            f"{stamped_entries} entry(ies)"
        )

    if restamp_only:
        typer.echo(
            f"\n--restamp-only: {len(report.resolution.files)} rule source(s) checked, "
            f"{stamped_particles} particle(s) restamped."
        )
        return

    typer.echo(
        f"\n{len(report.resolution.files)} rule source(s): {report.changed} new/changed, "
        f"{len(report.unchanged)} unchanged, {len(report.failed)} failed."
    )
    if report.changed:
        typer.echo(
            "Run `particles extract --all-pending` to turn the new snapshots into beliefs "
            "(tonight's `particles memory consolidate` would do it too)."
        )
    else:
        typer.echo("All tracked and enrolled in the refresh loop.")
