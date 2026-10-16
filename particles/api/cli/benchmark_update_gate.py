# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles benchmark trip-vs-move`` and ``particles benchmark update-gate``.

Two groups beside the other system benchmarks, each with a free ``rescore``
over a saved report. Thin over :mod:`particles.benchmark.update_gate`.

Exit codes: 0 when every run passed (``trip-vs-move``) or every set met its
bar (``update-gate``), and after ``--estimate``; 1 when one did not; 2 when
nothing was run (a refusal or a declined confirmation).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from particles.api.cli import run
from particles.api.cli._progress import confirm
from particles.api.cli.benchmark import benchmark_app
from particles.config import ProviderSelection

if TYPE_CHECKING:
    from particles.benchmark.update_gate.schema import UpdateGateReport

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_STARTED = 2

trip_app = typer.Typer(
    help="The trip-versus-move case end to end, judge-free.",
    no_args_is_help=False,
)
benchmark_app.add_typer(trip_app, name="trip-vs-move")

gate_app = typer.Typer(
    help="The registered update-gate pair sets through the production probes.",
    no_args_is_help=False,
)
benchmark_app.add_typer(gate_app, name="update-gate")


class _Arm(StrEnum):
    live = "live"
    scripted = "scripted"


class _Case(StrEnum):
    reader = "reader"
    mid_trip = "mid-trip"


class _Format(StrEnum):
    table = "table"
    json = "json"


class _Set(StrEnum):
    gate = "gate"
    gate_heldout = "gate-heldout"
    past = "past"
    past_marker = "past-marker"


def _progress(line: str) -> None:
    typer.echo(f"[{datetime.now():%H:%M:%S}] {line}", err=True)


def _write(output: Path | None, as_json: str) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(as_json + "\n")
    typer.echo(f"Report written to {output}", err=True)


def _confirm_spend(yes: bool, what: str) -> None:
    """Every live run asks before its first call; ``--yes`` pre-confirms."""
    if yes:
        return
    if not sys.stdin.isatty():
        typer.echo(
            f"{what} makes live LLM calls and no --yes was given; nothing was run "
            "(non-interactive run).",
            err=True,
        )
        raise typer.Exit(EXIT_NOT_STARTED)
    if not confirm("Proceed with these live LLM calls?"):
        typer.echo("Aborted; nothing was run.", err=True)
        raise typer.Exit(EXIT_NOT_STARTED)


def _refuse_without_key(providers: set[str]) -> None:
    from particles.secrets import get_anthropic_api_key_optional

    if "anthropic" in providers and not get_anthropic_api_key_optional():
        typer.echo(
            "ANTHROPIC_API_KEY is not set, and the chosen model runs on Anthropic. "
            "Set it and re-run, or choose another --provider. Nothing was run.",
            err=True,
        )
        raise typer.Exit(EXIT_NOT_STARTED)


def _parse_pins(pins: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pins or []:
        purpose, sep, model = item.partition("=")
        if not sep or not purpose or not model:
            typer.echo(f"Error: --pin takes PURPOSE=MODEL, not {item!r}.", err=True)
            raise typer.Exit(EXIT_NOT_STARTED)
        out[purpose.strip()] = model.strip()
    return out


# ---------------------------------------------------------------------------
# particles benchmark trip-vs-move
# ---------------------------------------------------------------------------


@trip_app.callback(invoke_without_command=True)
def trip_vs_move_cmd(  # noqa: PLR0913 — the CLI option list is the API
    ctx: typer.Context,
    case_name: _Case = typer.Option(
        _Case.reader,
        "--case",
        help="reader (default): the reader's own case, a week in Mumbai and a move a month "
        "later. mid-trip: a harder, different case, where the move is announced the day "
        "after the trip note.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="The model every purpose runs on (default: llm.default's model). The page's "
        "headline run names claude-sonnet-5-5.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="The provider of --model: anthropic or an llm.providers entry "
        "(default: llm.default's provider).",
    ),
    pin: list[str] | None = typer.Option(
        None,
        "--pin",
        help="PURPOSE=MODEL: run one purpose on another model of the same provider. "
        "Repeatable. The report names the mixture.",
    ),
    move_runs: int | None = typer.Option(
        None, "--move-runs", min=0, help="Move-case runs (default: 3, the record's protocol)."
    ),
    trip_alone_runs: int | None = typer.Option(
        None,
        "--trip-alone-runs",
        min=0,
        help="Trip-alone runs (default: 2, the record's protocol).",
    ),
    arm: _Arm = typer.Option(
        _Arm.live,
        "--arm",
        help="live (the model's own answers, which costs money) or scripted (every "
        "answer from the package's table: free and key-free, a test of the harness only).",
    ),
    same_slot_admits: bool = typer.Option(
        True,
        "--same-slot-admits/--no-same-slot-admits",
        help="--no-same-slot-admits runs the gate before 1.187.0, which reproduces the "
        "failure the fix addressed.",
    ),
    max_usd: float | None = typer.Option(
        None,
        "--max-usd",
        min=0.0,
        help="Per-run ceiling in US$ at list price (default: four times a measured "
        "estimate), checked against measured spend before each step. A run that "
        "reaches it stops and is reported incomplete.",
    ),
    estimate: bool = typer.Option(
        False, "--estimate", help="Print the estimate and exit before any LLM call."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm the live spend in advance."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the JSON report of record here."
    ),
    output_format: _Format = typer.Option(
        _Format.table, "--format", help="table (default) or json (the report of record)."
    ),
    store_dir: Path | None = typer.Option(
        None,
        "--store-dir",
        help="Keep each run's scratch store here (must be absent or empty). Without it "
        "the runs live in a temporary directory that is removed.",
    ),
) -> None:
    """Run the trip-versus-move case end to end and check each store.

    A user who lives in Delhi says they are in Mumbai for a week, and a month
    later that they moved to Mumbai. Each run deposits the case's notes into a
    fresh scratch store in a child process with a private environment,
    consolidates it, and checks the store with judge-free checks. Nothing
    reaches your own store.
    """
    if ctx.invoked_subcommand is not None:
        return
    from particles.benchmark.update_gate.case import CaseRefused, resolve_pins, selection_key
    from particles.benchmark.update_gate.estimate import estimate_case, render_case_estimate
    from particles.benchmark.update_gate.render import render_case
    from particles.config import get_config

    cfg = get_config().benchmark_update_gate
    runs = {
        "move": cfg.move_runs if move_runs is None else move_runs,
        "trip-alone": cfg.trip_alone_runs if trip_alone_runs is None else trip_alone_runs,
    }
    try:
        base, pins = resolve_pins(provider=provider, model=model, pins=_parse_pins(pin))
    except CaseRefused as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_NOT_STARTED) from exc
    mixed = any(selection_key(s) != selection_key(base) for s in pins.values())
    est = estimate_case(pins, runs, mixed=mixed, case=case_name.value)
    if arm is _Arm.live:
        typer.echo(
            f"Case: {case_name.value}. " + render_case_estimate(est, selection_key(base)), err=True
        )
    else:
        typer.echo("Scripted arm: no LLM call leaves this process, and nothing is spent.", err=True)
    if estimate:
        typer.echo("--estimate: nothing was run.", err=True)
        return
    if arm is _Arm.live:
        _check_egress(pins.values())
        _refuse_without_key({s.provider for s in pins.values()})
        _confirm_spend(yes, "particles benchmark trip-vs-move")
    from particles.benchmark.update_gate.case import run_case

    try:
        report = run_case(
            case=case_name.value,
            base=base,
            pins=pins,
            runs=runs,
            arm=arm.value,
            same_slot_admits=same_slot_admits,
            estimate=est,
            max_usd=max_usd,
            store_dir=store_dir,
            progress=_progress,
        )
    except CaseRefused as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_NOT_STARTED) from exc
    as_json = report.model_dump_json(indent=1)
    typer.echo(as_json if output_format is _Format.json else render_case(report))
    _write(output, as_json)
    raise typer.Exit(EXIT_OK if all(r.passed for r in report.runs) else EXIT_FAILED)


def _check_egress(selections: Iterable[ProviderSelection]) -> None:
    from particles.benchmark.update_gate.pairs import ReplayRefused, check_egress

    for selection in selections:
        try:
            check_egress(selection)
        except ReplayRefused as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(EXIT_NOT_STARTED) from exc


@trip_app.command("rescore")
def trip_vs_move_rescore_cmd(
    report_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="A saved report."),
    checks_version: int | None = typer.Option(
        None, "--checks-version", help="Checks version to score under (default: the current one)."
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the re-scored report."),
    output_format: _Format = typer.Option(_Format.table, "--format", help="table or json."),
) -> None:
    """Re-run the checks over a saved report's rows and events, with no LLM call."""
    from particles.benchmark.update_gate.case import rescore
    from particles.benchmark.update_gate.checks import CHECKS_VERSION, NotRescorable
    from particles.benchmark.update_gate.render import render_case
    from particles.benchmark.update_gate.schema import TripVsMoveReport

    saved = TripVsMoveReport.model_validate_json(report_path.read_text())
    try:
        report = rescore(saved, version=checks_version or CHECKS_VERSION)
    except (NotRescorable, KeyError) as exc:
        typer.echo(f"Error: not re-scorable: {exc}", err=True)
        raise typer.Exit(EXIT_NOT_STARTED) from exc
    as_json = report.model_dump_json(indent=1)
    typer.echo(as_json if output_format is _Format.json else render_case(report))
    _write(output, as_json)
    raise typer.Exit(EXIT_OK if all(r.passed for r in report.runs) else EXIT_FAILED)


# ---------------------------------------------------------------------------
# particles benchmark update-gate
# ---------------------------------------------------------------------------


@gate_app.callback(invoke_without_command=True)
def update_gate_cmd(  # noqa: PLR0913 — the CLI option list is the API
    ctx: typer.Context,
    model: str | None = typer.Option(
        None, "--model", help="The model both probes run on (default: llm.default's model)."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="The provider of --model (default: llm.default's provider)."
    ),
    sets: list[_Set] | None = typer.Option(
        None, "--set", help="A pair set to replay; repeatable (default: all four)."
    ),
    samples: int | None = typer.Option(
        None, "--samples", min=1, help="Samples per pair (default: 3, as registered)."
    ),
    estimate: bool = typer.Option(
        False, "--estimate", help="Print the estimate and exit before any LLM call."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm the live spend in advance."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the JSON report of record here."
    ),
    output_format: _Format = typer.Option(
        _Format.table, "--format", help="table (default) or json (the report of record)."
    ),
) -> None:
    """Replay the registered update-gate pair sets through the production probes.

    Each pair is asked the contradiction probe and the update probe with its
    second reading, and three gates are scored from the same samples: the
    production gate, the update verdict alone, and the gate before 1.187.0.
    The registered bars apply to the production gate. No store is built.
    """
    if ctx.invoked_subcommand is not None:
        return
    from particles.benchmark.update_gate import pairs
    from particles.benchmark.update_gate.estimate import render_pairs_estimate
    from particles.benchmark.update_gate.fixtures import PAIR_SETS
    from particles.benchmark.update_gate.render import render_pairs
    from particles.config import get_config

    cfg = get_config()
    default = cfg.llm.default
    selection = ProviderSelection(
        provider=provider or default.provider, model=model or default.model
    )
    names = [s.value for s in sets] if sets else list(PAIR_SETS)
    n_samples = samples or cfg.benchmark_update_gate.samples
    calls = pairs.call_bound(names, n_samples)
    typer.echo(render_pairs_estimate(selection, names, calls), err=True)
    if estimate:
        typer.echo("--estimate: nothing was run.", err=True)
        return
    _check_egress([selection])
    _refuse_without_key({selection.provider})
    _confirm_spend(yes, "particles benchmark update-gate")
    report = run(_update_gate(selection, names, n_samples))
    as_json = report.model_dump_json(indent=1, by_alias=True)
    typer.echo(as_json if output_format is _Format.json else render_pairs(report))
    _write(output, as_json)
    raise typer.Exit(EXIT_OK if all(r.bars["bar"] for r in report.sets.values()) else EXIT_FAILED)


async def _update_gate(
    selection: ProviderSelection, names: list[str], samples: int
) -> UpdateGateReport:
    from particles.benchmark.update_gate import pairs
    from particles.benchmark.update_gate.case import selection_key
    from particles.benchmark.update_gate.schema import Spend
    from particles.llm import track_usage

    with pairs.pinned_probes(selection), track_usage() as usage:
        rows = await pairs.replay(names, samples=samples, progress=_progress)
        measured = usage.snapshot()
    return pairs.build_report(
        rows,
        model=selection_key(selection),
        samples=samples,
        spend=Spend(
            estimate_source="bound",
            cost_usd=measured.cost_usd,
            priced_cost_usd=round(measured.priced_cost_usd, 6),
            unpriced=measured.unpriced,
        ),
    )


@gate_app.command("rescore")
def update_gate_rescore_cmd(
    report_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="A saved report."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the re-scored report."),
    output_format: _Format = typer.Option(_Format.table, "--format", help="table or json."),
) -> None:
    """Re-score a saved report's samples under the current marker and bars, with no LLM call."""
    from particles.benchmark.update_gate.pairs import rescore
    from particles.benchmark.update_gate.render import render_pairs
    from particles.benchmark.update_gate.schema import UpdateGateReport

    report = rescore(UpdateGateReport.model_validate_json(report_path.read_text()))
    as_json = report.model_dump_json(indent=1, by_alias=True)
    typer.echo(as_json if output_format is _Format.json else render_pairs(report))
    _write(output, as_json)
    raise typer.Exit(EXIT_OK if all(r.bars["bar"] for r in report.sets.values()) else EXIT_FAILED)
