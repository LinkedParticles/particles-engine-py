"""memory group — agent-memory maintenance verbs.

``rebuild-utility`` re-derives every utility channel from its own system of
record — the mined evidence from the harvested transcripts, the explicit
credits from the ``BELIEF_MARKED_USEFUL`` event log. The backfill that credits
history when the feature is turned on or the matchers change.

``useful`` is the explicit operator gesture: the second utility
channel, for the belief class action mining cannot reach at all. A prohibition
("never prepend ``export PATH``") or a design stance is complied with by *not*
acting, and the miner reads tool-call lines — so those beliefs are unobservable
by construction and stay out of the projection head no matter how the matchers
are tuned. Marking one useful credits it directly. Operator-only: there is
deliberately no MCP tool.

``consolidate`` is the dream cycle: the scheduled verb that runs
the cross-session maintenance passes in order — extract catch-up, reconcile,
capped+scoped census, curation-queue refresh, utility mining, projection
re-render — records each run as a ``CONSOLIDATION_RUN`` operator event, and
degrades to a disclosed structural-only pass without an LLM. Cadence comes
from launchd/cron (`--if-due` makes over-scheduling harmless); the CLI stays
thin over ``operations.consolidation.run_consolidation``.

``sweep-rank-lift`` is the calibration harness: a read-only sweep of
the usefulness rank-lift ``λ`` reporting where each operator-named target belief
lands, how many head slots hold distinct content, and the resulting admissible
band per rendered surface. Auto-fitting ``λ`` was declined on measured
grounds, so the harness ships instead of a fit — the operator supplies the one
input a fit cannot manufacture.

``serve`` runs the reference memory-server compatibility façade over stdio
 — the drop-in swap for ``@modelcontextprotocol/server-memory``,
backed by the store — and ``tools`` dumps its surface for the parity golden.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from particles.api.cli import app, run
from particles.api.cli._logging import configure_logging
from particles.api.cli._output import (
    DEBUG_OPTION,
    PROGRESS_OPTION,
    QUIET_OPTION,
    VERBOSE_OPTION,
    configure_output,
)
from particles.db import DEFAULT_STORE, session_scope
from particles.operations.utility_feedback import CLI_ACTOR
from particles.operations.utility_sweep import (
    DEFAULT_DISTINCT_RATIO,
    DEFAULT_GRID_MAX,
    DEFAULT_GRID_STEPS,
    DEFAULT_MAX_OWNER_SHARE,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.operations.consolidation import ProjectionRunner

memory_app = typer.Typer(
    help="Agent-memory maintenance.", no_args_is_help=True
)
app.add_typer(memory_app, name="memory")

_FORMATS = ("markdown", "json")
_SCOPES = ("delta", "store")


@memory_app.command("rebuild-utility")
def rebuild_utility_cmd(
    store: str = typer.Option(
        DEFAULT_STORE, "--store", help="Store handle to rebuild utility evidence for."
    ),
    verbose: bool = VERBOSE_OPTION,
    debug: bool = DEBUG_OPTION,
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Re-mine harvested session transcripts into fresh utility evidence."""
    configure_output(verbose, debug, quiet, progress)
    run(_rebuild_utility(store))


async def _rebuild_utility(store: str) -> None:
    from particles.operations.utility_mining import rebuild_store_utility

    result = await rebuild_store_utility(store)
    line = (
        f"Rebuilt utility evidence for store {store!r}: "
        f"{result.literal} literal + {result.behavioural} behavioural events "
        f"across {result.candidates} active beliefs "
        f"({result.behavioural_calls} LLM match call(s))."
    )
    if result.skipped_missing_blob:
        noun = "entry" if result.skipped_missing_blob == 1 else "entries"
        line += (
            f" Skipped {result.skipped_missing_blob} harvested {noun} with a missing"
            " corpus blob — those sessions contributed no evidence."
        )
    if result.explicit:
        noun = "credit" if result.explicit == 1 else "credits"
        line += f" Re-derived {result.explicit} explicit operator {noun} from the event log."
    typer.echo(line)


# ---------------------------------------------------------------------------
# useful — the explicit operator gesture
# ---------------------------------------------------------------------------


@memory_app.command("useful")
def useful_cmd(
    particle_id: str = typer.Argument(
        ...,
        metavar="PARTICLE_ID",
        help=(
            "The belief that earned its place. Full UUID, a unique id prefix, or the "
            "`p-xxxxxxxx` display form the digest and `particle show` print."
        ),
    ),
    reason: str | None = typer.Option(
        None, "--reason", help="Optional note recorded on the operator event."
    ),
    store: str = typer.Option(DEFAULT_STORE, "--store", help="Store handle."),
    verbose: bool = VERBOSE_OPTION,
    debug: bool = DEBUG_OPTION,
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Mark a belief useful — the explicit utility gesture.

    Use this for the beliefs the transcript miner cannot see: prohibitions
    ("never do X") and design stances, which you comply with by *not* acting and
    which therefore leave no tool-call trace. One press is worth
    `utility.explicit_weight` mined events, because the miner fires once per
    session while you fire once — and it is capped at one credit per belief per
    day, so pressing twice is recorded but not double-counted.

    This lifts the belief in the projection and digest **ranking only**. It never
    touches the stored confidence, never claims the belief is *true*, and can
    only promote — for "still true", the gesture is
    `particles curate apply affirm`.
    """
    configure_output(verbose, debug, quiet, progress)
    run(_useful(particle_id, reason=reason, store=store))


async def _useful(particle_id: str, *, reason: str | None, store: str) -> None:
    from particles.operations.utility_feedback import (
        BeliefNotCreditable,
        mark_belief_useful,
    )

    async with session_scope(store) as session:
        resolved = (
            await _resolve_target_ids(
                session,
                [particle_id],
                store=store,
                label="PARTICLE_ID",
                active_only_note=(
                    "Only ACTIVE beliefs are projected, so crediting a retracted or "
                    "superseded one could not surface anything."
                ),
            )
        )[0]
        try:
            result = await mark_belief_useful(session, resolved, actor=CLI_ACTOR, reason=reason)
        except BeliefNotCreditable as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from exc
        await session.commit()

    if result.counted:
        typer.echo(
            f"Marked {result.particle_id} useful. It now carries an explicit operator "
            "credit and will rank higher in the projection and digest."
        )
    else:
        typer.echo(
            f"Already marked {result.particle_id} useful today — recorded the gesture, "
            "but the credit is unchanged (one per belief per day). Mark it again on a "
            "later day to reinforce it further."
        )


# ---------------------------------------------------------------------------
# sweep-rank-lift — the calibration harness
# ---------------------------------------------------------------------------


@memory_app.command("sweep-rank-lift")
def sweep_rank_lift_cmd(
    store: str = typer.Option(DEFAULT_STORE, "--store", help="Store handle to sweep."),
    target: list[str] = typer.Option(
        [],
        "--target",
        help=(
            "Particle id of a belief you assert ought to reach the head; repeatable. "
            "Full UUID, a unique id prefix, or the `p-xxxxxxxx` digest display form; "
            "resolved against ACTIVE beliefs, and an id that matches none (or more "
            "than one) is an error rather than a silent rank-0. This is the judgment "
            "a fit cannot supply — without any, only head diversity "
            "constrains the band."
        ),
    ),
    head: list[int] = typer.Option(
        [],
        "--head",
        help=(
            "A rendered head size N to evaluate; repeatable. Defaults to the digest's "
            "mcp.recall.digest_max_beliefs. Pass every N you actually render — the "
            "band is a property of the surface, not the store."
        ),
    ),
    grid_max: float = typer.Option(
        DEFAULT_GRID_MAX, "--grid-max", help="Largest lambda to evaluate."
    ),
    grid_steps: int = typer.Option(
        DEFAULT_GRID_STEPS,
        "--grid-steps",
        help="Non-zero grid points; band edges resolve to one step.",
    ),
    distinct_ratio: float = typer.Option(
        DEFAULT_DISTINCT_RATIO,
        "--distinct-ratio",
        help=(
            "Fraction of head slots that must hold distinct content. Not 1.0 — that "
            "is unsatisfiable at large N on any store with over-extraction."
        ),
    ),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown (default) or json."
    ),
    verbose: bool = VERBOSE_OPTION,
    debug: bool = DEBUG_OPTION,
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Sweep the usefulness rank-lift and report its admissible band.

    Read-only: no writes, no LLM calls, no embeddings. `λ`
    (`utility.default.rank_lift`) is deliberately **not** auto-fitted
    measured every candidate closed form and found none defensible, because no
    label says which belief *should* occupy a head slot. This is the harness
    that makes setting it by hand a single command instead of a research
    project: name the beliefs that ought to reach the head with `--target`, and
    the sweep reports where they land, how many head slots hold distinct
    content, the resulting band per surface, and whether the configured value
    is inside it.
    """
    configure_output(verbose, debug, quiet, progress)
    if format_ not in _FORMATS:
        typer.echo(f"Error: --format must be one of: {', '.join(_FORMATS)}.", err=True)
        raise typer.Exit(2)
    if grid_max <= 0.0:
        typer.echo("Error: --grid-max must be greater than 0.", err=True)
        raise typer.Exit(2)
    if grid_steps < 1:
        typer.echo("Error: --grid-steps must be at least 1.", err=True)
        raise typer.Exit(2)
    if not 0.0 < distinct_ratio <= 1.0:
        typer.echo("Error: --distinct-ratio must be in (0, 1].", err=True)
        raise typer.Exit(2)
    if any(n < 1 for n in head):
        typer.echo("Error: --head must be at least 1.", err=True)
        raise typer.Exit(2)
    run(
        _sweep_rank_lift(
            store=store,
            targets=list(target),
            heads=list(head),
            grid_max=grid_max,
            grid_steps=grid_steps,
            distinct_ratio=distinct_ratio,
            fmt=format_,
        )
    )


async def _sweep_rank_lift(
    *,
    store: str,
    targets: list[str],
    heads: list[int],
    grid_max: float,
    grid_steps: int,
    distinct_ratio: float,
    fmt: str,
) -> None:
    import json

    from particles.config import get_config
    from particles.operations.utility_sweep import (
        render_rank_lift_sweep,
        sweep_store_rank_lift,
    )

    if not heads:
        # The one head size the store itself declares. A projection manifest's
        # per-section `top_k` is per-document, so it comes in via --head.
        digest_cap = get_config().mcp.recall.digest_max_beliefs
        heads = [digest_cap] if digest_cap > 0 else [60]

    async with session_scope(store) as session:
        # Resolve before scoring: an unresolvable target must fail fast and
        # loudly, not ride through the sweep as a rank-0 (see the helper).
        target_ids = await _resolve_target_ids(session, targets, store=store)
        sweep = await sweep_store_rank_lift(
            session,
            head_sizes=heads,
            target_ids=target_ids,
            grid_max=grid_max,
            grid_steps=grid_steps,
            distinct_ratio=distinct_ratio,
        )

    if fmt == "json":
        typer.echo(
            json.dumps(
                {
                    "scored": sweep.scored,
                    "configured_rank_lift": sweep.configured_rank_lift,
                    "configured_admissible": sweep.configured_admissible,
                    "intersection": {
                        "low": sweep.intersection.low,
                        "high": sweep.intersection.high,
                        "contiguous": sweep.intersection.contiguous,
                    },
                    "bands": [
                        {
                            "head_size": n,
                            "low": band.low,
                            "high": band.high,
                            "contiguous": band.contiguous,
                        }
                        for n, band in sweep.bands
                    ],
                    "points": [
                        {
                            "rank_lift": point.rank_lift,
                            "target_ranks": dict(point.heads[0].target_ranks),
                            "heads": [
                                {
                                    "head_size": h.head_size,
                                    "distinct_contents": h.distinct_contents,
                                    "largest_duplicate_cluster": h.largest_duplicate_cluster,
                                    "admissible": h.admissible,
                                }
                                for h in point.heads
                            ],
                        }
                        for point in sweep.points
                    ],
                },
                indent=2,
            )
        )
        return
    typer.echo(render_rank_lift_sweep(sweep))


async def _resolve_target_ids(
    session: AsyncSession,
    targets: list[str],
    *,
    store: str,
    label: str = "--target",
    active_only_note: str = (
        "The sweep ranks ACTIVE beliefs only, so a non-ACTIVE target can never reach the head."
    ),
) -> list[str]:
    """Resolve every ``--target`` to a full ACTIVE particle id, or exit 2.

    Shared by ``sweep-rank-lift`` and ``useful`` — ``label`` and
    ``active_only_note`` adapt the diagnostics to the calling verb, since the two
    have different reasons for insisting on a resolved ACTIVE id.

    Errors loudly on purpose. first acceptance criterion is "every
    named target ranks inside the head", and
    :class:`~particles.core.scoring.utility.HeadOutcome` scores a target it
    cannot find as rank ``0`` — which fails ``0 < rank <= head_size`` at every
    ``λ`` on the grid. A mistyped id therefore renders as an **empty admissible
    band on every surface**, indistinguishable from a store that genuinely
    cannot be calibrated. That is the worst failure mode a calibration tool
    has, so a target that resolves to nothing stops the sweep instead of
    quietly scoring zero.

    Resolution matches the sibling verbs (``particle show``, ``links add``):
    prefix-LIKE against ``ParticleRow.id``, so the truncated 8-char id an
    operator reads out of a digest works as well as the full UUID. It is
    additionally scoped to **ACTIVE** — the only population
    :func:`~particles.operations.utility_sweep.load_sweep_rows` ranks — so a
    retracted or superseded id is reported as such rather than resolving to a
    row the sweep will never score.

    Args:
        session: Active store session.
        targets: The raw ``--target`` values, display prefixes and all.
        store: Store handle, for the error messages.

    Returns:
        One full particle id per target, in the order given.
    """
    from sqlalchemy import select

    from particles.api.cli._id_norm import normalise_particle_id
    from particles.core.status import Status
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern
    from particles.store.particle_store import ParticleRow

    resolved: list[str] = []
    for raw in targets:
        norm = normalise_particle_id(raw)
        if not norm:
            typer.echo(f"Error: {label} must not be empty.", err=True)
            raise typer.Exit(2)
        rows = (
            await session.execute(
                select(ParticleRow.id, ParticleRow.status).where(
                    ParticleRow.id.like(f"{escape_like_pattern(norm)}%", escape=LIKE_ESCAPE)
                )
            )
        ).all()
        active = [pid for pid, status in rows if Status(status) is Status.ACTIVE]
        if len(active) == 1:
            resolved.append(str(active[0]))
            continue
        if len(active) > 1:
            typer.echo(
                f"Error: {label} {raw!r} is ambiguous — it matches {len(active)} ACTIVE beliefs:",
                err=True,
            )
            for pid in sorted(active)[:10]:
                typer.echo(f"  {pid}", err=True)
            typer.echo("Pass a longer prefix or the full id.", err=True)
            raise typer.Exit(2)
        if rows:
            seen = ", ".join(sorted({str(status) for _pid, status in rows}))
            typer.echo(
                f"Error: {label} {raw!r} matches no ACTIVE belief in store {store!r} "
                f"(matched {len(rows)} belief(s) with status: {seen}). {active_only_note}",
                err=True,
            )
            raise typer.Exit(2)
        typer.echo(
            f"Error: {label} {raw!r} matches no belief in store {store!r}. This "
            "accepts a full particle id, a unique id prefix, or the `p-xxxxxxxx` "
            "display form — check the id against `particles particle show`.",
            err=True,
        )
        raise typer.Exit(2)
    return resolved


# ---------------------------------------------------------------------------
# consolidate — the dream cycle
# ---------------------------------------------------------------------------


@memory_app.command("consolidate")
def consolidate_cmd(
    store: str = typer.Option(
        DEFAULT_STORE, "--store", help="Store handle to consolidate (default: the default store)."
    ),
    if_due: bool = typer.Option(
        False,
        "--if-due",
        help=(
            "Exit 0 without running unless the last successful run is older than "
            "consolidation.min_interval_hours — makes over-scheduling harmless."
        ),
    ),
    structural_only: bool = typer.Option(
        False,
        "--structural-only",
        help="Skip all LLM passes (disclosed in the report and the run record).",
    ),
    scope: str = typer.Option(
        "delta",
        "--scope",
        help=(
            "Semantic-pass scope: 'delta' (default — particles changed since the "
            "previous run's watermark) or 'store' (the whole store, still capped)."
        ),
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Also write the run report as Markdown to FILE."
    ),
    format_: str = typer.Option(
        "markdown", "--format", help="Terminal format: markdown (default) or json."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Run the scheduled consolidation cycle: the memory dream cycle.

    Exit codes (cron observability): 0 — success, including disclosed
    structural-only runs and --if-due / lock skips; 1 — one or more passes
    failed (run record written); 2 — the cycle could not start.
    """
    configure_logging(verbose, debug)
    if format_ not in _FORMATS:
        typer.echo(f"Error: --format must be one of: {', '.join(_FORMATS)}.", err=True)
        raise typer.Exit(2)
    if scope not in _SCOPES:
        typer.echo(f"Error: --scope must be one of: {', '.join(_SCOPES)}.", err=True)
        raise typer.Exit(2)
    run(
        _consolidate_impl(
            store=store,
            if_due=if_due,
            structural_only=structural_only,
            scope=scope,
            output=output,
            fmt=format_,
        )
    )


async def _consolidate_impl(
    *,
    store: str,
    if_due: bool,
    structural_only: bool,
    scope: str,
    output: Path | None,
    fmt: str,
) -> None:
    from particles.api.client import get_backend

    if get_backend().remote:
        typer.echo(
            "Error: `particles memory consolidate` consolidates one local store per "
            "invocation (§ Deferred); run it on the machine that holds the "
            "store.",
            err=True,
        )
        raise typer.Exit(2)

    # Deferred import: the operation pulls the reconcile/curation/lint stack —
    # and tests patch ``particles.operations.consolidation.run_consolidation``
    # at call time (tests/AGENTS.md § Mocking strategy).
    from particles.operations.consolidation import (
        render_consolidation_report,
        run_consolidation,
    )

    projection_runner, projection_skip = build_projection_runner(store)
    try:
        async with session_scope(store) as session:
            report = await run_consolidation(
                session,
                store=store,
                scope="store" if scope == "store" else "delta",
                structural_only=structural_only,
                if_due=if_due,
                projection_runner=projection_runner,
                projection_skip_reason=projection_skip,
            )
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 — §8 exit-code contract: 2 = could not start
        typer.echo(f"Error: consolidation could not start: {exc}", err=True)
        raise typer.Exit(2) from exc

    if report.outcome == "skipped":
        # Cron-friendly: contention / not-due is normal, not an alarm (§8).
        typer.echo(report.skip_reason or "consolidation skipped")
        return

    rendered = render_consolidation_report(report)
    if fmt == "json":
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(rendered)
    if output is not None:
        from particles.render.markdown import atomic_write_text

        atomic_write_text(output, rendered)
        typer.echo(f"Report written to {output}.")

    failed = report.failed_passes()
    if failed:
        typer.echo(f"consolidation: pass(es) failed: {', '.join(failed)}", err=True)
        raise typer.Exit(1)


def build_projection_runner(store: str) -> tuple[ProjectionRunner | None, str | None]:
    """The pass-6 projection callback — the SessionEnd harvest-then-render tail, reused.

    The Engine operation cannot import the CLI-side projection helpers (that
    would invert the Surface > Engine layer contract), so the Surface builds
    the callback and injects it. Returns ``(None, <disclosed reason>)`` when
    the projection is disabled or no Claude Code memory directory exists
    .

    Public because ``engine serve --daemon`` registers it as the daemon's
    projection-runner factory, so a resident daemon renders
    ``MEMORY.md`` exactly as the launchd recipe does.
    """
    from particles.api.cli._claude_code import projection_enabled

    if not projection_enabled():
        return None, "agent_memory.projection.enabled is false"
    root = Path.home() / ".claude" / "projects"
    memory_dirs = sorted(p for p in root.glob("*/memory") if p.is_dir()) if root.is_dir() else []
    if not memory_dirs:
        return None, f"no memory directories under {root}"

    async def _run() -> dict[str, Any]:
        # The same tail the SessionEnd hook runs (§6/§7): read +
        # sentinel-strip + deposit the memory files (dirty/authored content
        # routes through the ladder), THEN render-splice with the just-read
        # text — never a bare splice, so the §7 "never destroy unharvested
        # content" invariant is inherited rather than re-proven.
        from particles.api.cli._memory_projection import run_projection_cycle
        from particles.api.cli.hook import _harvest_memory_files

        telemetry: dict[str, Any] = {"dirs": len(memory_dirs), "harvested": 0, "rendered": 0}
        for memory_dir in memory_dirs:
            project = memory_dir.parent.name
            harvested, _unchanged, memory_md_text = await _harvest_memory_files(
                store, memory_dir, project
            )
            telemetry["harvested"] += len(harvested)
            outcome = await run_projection_cycle(store, memory_dir, memory_md_text)
            if outcome.get("outcome") in ("rendered", "created"):
                telemetry["rendered"] += 1
        return telemetry

    return _run, None


@memory_app.command("serve")
def memory_serve_cmd(
    store: str = typer.Option(
        None, "--store", help="Store handle to bind (default: the default store)."
    ),
) -> None:
    """Serve the reference memory-server compatibility façade over stdio.

    The drop-in swap for ``@modelcontextprotocol/server-memory``: same nine
    tools, same schemas, same responses, backed by a Particles store. In your
    MCP client config, replace the reference server's command with::

        "memory": {"command": "uv",
                   "args": ["run", "--project", "/path/to/particles",
                            "particles", "memory", "serve"]}

    Args:
        store: Store handle to bind. Defaults to the default store. Writes
            additionally require the store to be listed in
            ``mcp.write.enabled_stores`` (default-deny); the write
            tools stay visible either way and refuse with an actionable
            message when it is not.
    """
    from particles.mcp.memory_compat import main as serve_main

    serve_main(store)


@memory_app.command("tools")
def memory_tools_cmd(
    output_format: str = typer.Option(
        "json", "--format", help='Output format: "json" (default) or "text".'
    ),
) -> None:
    """Print the façade's tool surface — name, title, schemas, annotations.

    The debugging sibling of ``particles mcp tools``, and the generator for
    ``tests/mcp/memory-tool-schema.json``. That golden is what turns a parity
    regression into a failed build instead of a broken agent.

    Args:
        output_format: ``json`` (default) or ``text`` for a one-line summary.
    """
    import json

    from particles.mcp.memory_compat.server import tool_surface

    surface = tool_surface()

    if output_format == "text":
        for entry in surface:
            typer.echo(str(entry["name"]))
            description = str(entry["description"] or "")
            typer.echo(f"  {description.splitlines()[0] if description else ''}")
        return

    typer.echo(json.dumps(surface, indent=2, sort_keys=True))


@memory_app.command("sweep-owner-lift")
def sweep_owner_lift_cmd(
    store: str = typer.Option(DEFAULT_STORE, "--store", help="Store handle to sweep."),
    target: list[str] = typer.Option(
        [],
        "--target",
        help=(
            "Particle id of a belief that must STAY in the head; repeatable. Pass the "
            "beliefs your utility lift was calibrated to surface — the third criterion "
            " is that adding aboutness does not push them out. Same id "
            "forms as `sweep-rank-lift`."
        ),
    ),
    head: list[int] = typer.Option(
        [],
        "--head",
        help=(
            "A rendered head size N to evaluate; repeatable. Defaults to the digest's "
            "mcp.recall.digest_max_beliefs."
        ),
    ),
    grid_max: float = typer.Option(
        DEFAULT_GRID_MAX, "--grid-max", help="Largest omega to evaluate."
    ),
    grid_steps: int = typer.Option(
        DEFAULT_GRID_STEPS,
        "--grid-steps",
        help="Non-zero grid points; band edges resolve to one step.",
    ),
    min_owner_in_head: int = typer.Option(
        1,
        "--min-owner",
        help="Viewer beliefs the head must hold for an omega to pass (criterion 1).",
    ),
    max_owner_share: float = typer.Option(
        DEFAULT_MAX_OWNER_SHARE,
        "--max-owner-share",
        help=(
            "Largest fraction of the head the viewer cohort may occupy (criterion 2). "
            "This is the quantity to calibrate against: A(p) is a flat step, so omega "
            "behaves as a threshold over the whole cohort rather than a graded lift."
        ),
    ),
    format_: str = typer.Option(
        "markdown", "--format", help="Output format: markdown (default) or json."
    ),
    verbose: bool = VERBOSE_OPTION,
    debug: bool = DEBUG_OPTION,
    quiet: bool = QUIET_OPTION,
    progress: bool | None = PROGRESS_OPTION,
) -> None:
    """Sweep the owner-relevance rank-lift and report its band.

    Read-only: no writes, no LLM calls, no embeddings. `ω`
    (`owner_lens.rank_lift`) is store-specific and deliberately ships `0.0`
    (inert) — this is the harness for choosing it. Unlike the utility lift, `ω`
    multiplies a flat 0/1 indicator, so it acts as a *threshold* over the whole
    viewer cohort: below it nothing moves, above it every belief about the
    viewer arrives in the head at once. The report is therefore keyed on the
    cohort's **share of the head**, and the utility `λ` in force is held fixed
    so the non-regression criterion is measured against the head utility has
    already shaped.
    """
    configure_output(verbose, debug, quiet, progress)
    if format_ not in _FORMATS:
        typer.echo(f"Error: --format must be one of: {', '.join(_FORMATS)}.", err=True)
        raise typer.Exit(2)
    if grid_max <= 0.0:
        typer.echo("Error: --grid-max must be greater than 0.", err=True)
        raise typer.Exit(2)
    if grid_steps < 1:
        typer.echo("Error: --grid-steps must be at least 1.", err=True)
        raise typer.Exit(2)
    if not 0.0 < max_owner_share <= 1.0:
        typer.echo("Error: --max-owner-share must be in (0, 1].", err=True)
        raise typer.Exit(2)
    if min_owner_in_head < 1:
        typer.echo("Error: --min-owner must be at least 1.", err=True)
        raise typer.Exit(2)
    if any(n < 1 for n in head):
        typer.echo("Error: --head must be at least 1.", err=True)
        raise typer.Exit(2)
    run(
        _sweep_owner_lift(
            store=store,
            targets=list(target),
            heads=list(head),
            grid_max=grid_max,
            grid_steps=grid_steps,
            min_owner_in_head=min_owner_in_head,
            max_owner_share=max_owner_share,
            fmt=format_,
        )
    )


async def _sweep_owner_lift(
    *,
    store: str,
    targets: list[str],
    heads: list[int],
    grid_max: float,
    grid_steps: int,
    min_owner_in_head: int,
    max_owner_share: float,
    fmt: str,
) -> None:
    import json

    from particles.config import get_config
    from particles.operations.utility_sweep import (
        render_owner_rank_lift_sweep,
        sweep_store_owner_rank_lift,
    )

    if not heads:
        digest_cap = get_config().mcp.recall.digest_max_beliefs
        heads = [digest_cap] if digest_cap > 0 else [60]

    async with session_scope(store) as session:
        target_ids = await _resolve_target_ids(
            session,
            targets,
            store=store,
            active_only_note=(
                "The sweep ranks ACTIVE beliefs only, so a non-ACTIVE target can never "
                "stay in the head."
            ),
        )
        sweep = await sweep_store_owner_rank_lift(
            session,
            head_sizes=heads,
            target_ids=target_ids,
            grid_max=grid_max,
            grid_steps=grid_steps,
            min_owner_in_head=min_owner_in_head,
            max_owner_share=max_owner_share,
        )

    if fmt == "json":
        typer.echo(
            json.dumps(
                {
                    "scored": sweep.scored,
                    "owner_population": sweep.owner_population,
                    "configured_rank_lift": sweep.configured_rank_lift,
                    "configured_admissible": sweep.configured_admissible,
                    "intersection": {
                        "low": sweep.intersection.low,
                        "high": sweep.intersection.high,
                        "contiguous": sweep.intersection.contiguous,
                    },
                    "bands": [
                        {
                            "head_size": n,
                            "low": band.low,
                            "high": band.high,
                            "contiguous": band.contiguous,
                        }
                        for n, band in sweep.bands
                    ],
                    "points": [
                        {
                            "rank_lift": point.rank_lift,
                            "target_ranks": dict(point.heads[0].target_ranks),
                            "heads": [
                                {
                                    "head_size": h.head_size,
                                    "owner_in_head": h.owner_in_head,
                                    "owner_share": h.owner_share,
                                    "admissible": h.admissible,
                                }
                                for h in point.heads
                            ],
                        }
                        for point in sweep.points
                    ],
                },
                indent=2,
            )
        )
        return
    typer.echo(render_owner_rank_lift_sweep(sweep))
