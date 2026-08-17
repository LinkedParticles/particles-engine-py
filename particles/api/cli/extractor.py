"""extractor sub-Typer — manage extractor registry (Extension A)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer

from particles.api.cli import app, run
from particles.conformance.types import ConformanceReport, FieldStat, FieldTier
from particles.db import session_scope
from particles.extraction.registry import get_extractors, selects

extractor_app = typer.Typer(help="Manage extractor registry (Extension A).", no_args_is_help=True)
app.add_typer(extractor_app, name="extractor")


# ---------------------------------------------------------------------------
# Conform — extractor conformance validator
# ---------------------------------------------------------------------------


class _ConformFormat(StrEnum):
    table = "table"
    json = "json"


class _ConformFailOn(StrEnum):
    error = "error"
    warn = "warn"


@extractor_app.command("conform")
def extractor_conform_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID of a registered extractor"),
    fixtures: Path | None = typer.Option(
        None,
        "--fixtures",
        help="Override fixture directory (default: tests/conformance/fixtures)",
    ),
    recommended_threshold: float = typer.Option(
        0.8,
        "--recommended-threshold",
        min=0.0,
        max=1.0,
        help="Minimum populate-rate for RECOMMENDED fields (0.0–1.0)",
    ),
    output_format: _ConformFormat = typer.Option(
        _ConformFormat.table, "--format", help="Output format"
    ),
    fail_on: _ConformFailOn = typer.Option(
        _ConformFailOn.error,
        "--fail-on",
        help="Exit non-zero on errors only (default) or on warnings as well",
    ),
    all_accepted: bool = typer.Option(
        False,
        "--all-accepted",
        help=(
            "Score every fixture the extractor accepts(), not just the ones the "
            "registry routes to it. Report-only: never stores the verdict"
        ),
    ),
) -> None:
    """Run an extractor against the conformance fixture corpus.

    Scores the fixtures the production registry routes to this extractor
    . ``--all-accepted`` widens the run to every fixture the
    extractor would take if handed it — for the fallback that is the
    whole corpus, so the result is a deliberate probe, not the extractor's
    conformance score, and it never updates the stored conformance verdict.

    Phase 1 (current): report-only. Exit code 0 unless --fail-on is given.
    Exit code 1 indicates the contract failed (a REQUIRED field missing, or a
    FAIL-severity diversity rule violated); --fail-on warn additionally treats
    RECOMMENDED warnings as fatal. An ADVISORY diversity finding (
    `uncertainty_nature` is the one shipped today) is reported and never
    affects the exit code.
    """
    run(
        _extractor_conform(
            extractor_id,
            fixtures,
            recommended_threshold,
            output_format,
            fail_on,
            all_accepted,
        )
    )


async def _extractor_conform(
    extractor_id: str,
    fixture_dir: Path | None,
    recommended_threshold: float,
    output_format: _ConformFormat,
    fail_on: _ConformFailOn,
    all_accepted: bool = False,
) -> None:
    from particles.conformance.validator import (
        ExtractorNotFoundError,
        has_evaluable_failure,
        validate_extractor,
    )

    try:
        report = await validate_extractor(
            extractor_id,
            fixture_dir=fixture_dir,
            recommended_threshold=recommended_threshold,
            all_accepted=all_accepted,
        )
    except ExtractorNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    if output_format is _ConformFormat.json:
        typer.echo(json.dumps(_report_to_dict(report), indent=2, default=str))
    else:
        _print_table(report)

    # best-effort persist of the evaluable-failure verdict so the
    # (opt-in) read-side trust cap can consult it. Conform stays a pure
    # validation command — a missing / uninitialised store (e.g. a CI run with
    # no DB) must not change its report, exit code, or JSON output, so any DB
    # error is swallowed. Conformance remains report-only as a gate; this writes
    # a read-side trust input, never a verdict that blocks anything.
    #
    # a --all-accepted run is barred from writing it. The stored
    # verdict is a claim about the extractor's *production* behaviour, and an
    # operator-widened run is by construction not that — for the fallback it
    # would clamp trust on inputs the pipeline never routes to it.
    if all_accepted:
        if output_format is not _ConformFormat.json:
            typer.echo(
                "(--all-accepted: report-only — the stored conformance verdict "
                "is left untouched; re-run without the flag to update it)"
            )
        _conform_exit(report, fail_on)
        return

    evaluable_failure = has_evaluable_failure(report)
    persisted = await _persist_conformance_status(report.extractor_id, evaluable_failure)
    if output_format is not _ConformFormat.json:
        if persisted is False:
            typer.echo(
                f"(conformance status not stored — {report.extractor_id} is not registered "
                "in this store; the trust cap applies only to registered extractors)"
            )
        elif persisted is True and evaluable_failure:
            typer.echo(
                "(stored: evaluable REQUIRED failure — the trust cap will clamp this "
                "extractor's effective weight when enabled)"
            )

    _conform_exit(report, fail_on)


def _conform_exit(report: ConformanceReport, fail_on: _ConformFailOn) -> None:
    """Apply the --fail-on exit policy. Shared by the routed and widened paths."""
    has_errors = bool(report.failures)
    has_warnings = bool(report.warnings)
    if fail_on is _ConformFailOn.warn and (has_errors or has_warnings):
        raise typer.Exit(1)
    if fail_on is _ConformFailOn.error and has_errors:
        raise typer.Exit(1)


async def _persist_conformance_status(extractor_id: str, evaluable_failure: bool) -> bool | None:
    """Best-effort write of the conformance verdict to the store.

    Returns ``True`` if persisted, ``False`` if the extractor has no registry
    row, or ``None`` if no store is reachable (no DB / un-migrated tables) — in
    which case conform simply behaves as the pure validation command it has
    always been. Never raises: a store-less ``extractor conform`` run is fully
    supported (CI, ad-hoc checks), so DB errors are caught and reported as
    ``None`` rather than propagating into the report path.
    """
    from sqlalchemy.exc import OperationalError

    from particles.db import session_scope
    from particles.store.extractor_store import set_conformance_status

    try:
        async with session_scope() as session:
            ok = await set_conformance_status(session, extractor_id, evaluable_failure)
            await session.commit()
        return ok
    except OperationalError:
        return None


# ---------------------------------------------------------------------------
# generate-fixture: corpus entry → conformance fixture skeleton
# ---------------------------------------------------------------------------


@extractor_app.command("generate-fixture")
def extractor_generate_fixture_cmd(
    entry_id: str = typer.Argument(..., help="Corpus entry ID (prefix OK)"),
    fixture_id: str | None = typer.Option(
        None, "--id", help="Fixture id (default: a slug of the entry URI + id prefix)"
    ),
    source_type: str | None = typer.Option(
        None, "--source-type", help="Override the entry's source type"
    ),
    output_dir: Path = typer.Option(
        Path("tests/conformance/fixtures"),
        "--output-dir",
        help="Fixture corpus directory (default: tests/conformance/fixtures)",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing fixture directory"),
) -> None:
    """Turn a deposited corpus entry into a conformance fixture skeleton.

    Reads the entry's latest stored snapshot + raw blob and writes
    ``manifest.yaml`` + ``content.bin`` + ``snapshot.json`` under
    ``<output-dir>/<fixture-id>``, registering it in ``MANIFEST.yaml``.
    ``expected_acceptors`` is left empty for you to fill after verifying which
    extractors the fixture should exercise.
    """
    run(_extractor_generate_fixture(entry_id, fixture_id, source_type, output_dir, force))


async def _extractor_generate_fixture(
    entry_id_prefix: str,
    fixture_id: str | None,
    source_type_override: str | None,
    output_dir: Path,
    force: bool,
) -> None:
    from sqlalchemy import select

    from particles.api.cli._remote import ensure_local
    from particles.conformance.fixtures import write_fixture
    from particles.core.schema import WarcRecordType
    from particles.corpus.deposit import load_blob
    from particles.corpus.store import CorpusEntryRow, get_entry
    from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern

    ensure_local("extractor generate-fixture")

    async with session_scope() as session:
        resolved_id = entry_id_prefix
        if len(entry_id_prefix) < 36:
            row = (
                await session.execute(
                    select(CorpusEntryRow).where(
                        CorpusEntryRow.entry_id.like(
                            f"{escape_like_pattern(entry_id_prefix)}%", escape=LIKE_ESCAPE
                        )
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                resolved_id = row.entry_id

        entry = await get_entry(session, resolved_id)
        if entry is None:
            typer.echo(f"Entry {entry_id_prefix!r} not found.", err=True)
            raise typer.Exit(1)

        usable = [
            s
            for s in entry.snapshots
            if s.warc_record_type == WarcRecordType.RESPONSE and s.content_hash
        ]
        if not usable:
            typer.echo(
                f"Entry {resolved_id[:8]} has no RESPONSE snapshot with stored content "
                "to build a fixture from.",
                err=True,
            )
            raise typer.Exit(1)
        snapshot = max(usable, key=lambda s: s.captured_at)

        try:
            content = load_blob(snapshot.content_hash)
        except FileNotFoundError as exc:
            typer.echo(
                f"Blob for snapshot {snapshot.snapshot_id[:8]} is not on disk "
                f"(content_hash {snapshot.content_hash[:8]}…).",
                err=True,
            )
            raise typer.Exit(1) from exc

        src_type = source_type_override or entry.source_type
        fid = fixture_id or _slug_fixture_id(entry.uri_r, src_type, entry.entry_id)
        notes = (
            f"Generated from corpus entry {resolved_id} "
            f"({entry.uri_r or 'no uri'}) via `extractor generate-fixture`. "
            "Fill expected_acceptors after verifying which extractors this fixture exercises."
        )

        try:
            fixture_dir = write_fixture(
                output_dir,
                fid,
                source_type=src_type,
                content=content,
                snapshot=snapshot,
                notes=notes,
                force=force,
            )
        except FileExistsError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc

    typer.echo(f"Wrote fixture '{fid}' → {fixture_dir}")
    typer.echo(f"  source_type: {src_type}  ({len(content)} bytes)")
    typer.echo("  expected_acceptors: []  ← fill in after verifying the fixture's coverage")
    typer.echo(
        "Note: adding a fixture changes the corpus hash and invalidates prior conformance "
        "reports — re-run `extractor conform` for affected extractors."
    )


def _slug_fixture_id(uri_r: str | None, source_type: str, entry_id: str) -> str:
    """Derive a filesystem-safe, unique fixture id from the entry URI + id prefix."""
    import re

    base = (uri_r or source_type).rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    if not slug:
        slug = source_type.lower()
    return f"{slug}-{entry_id[:8]}"


def _report_to_dict(report: ConformanceReport) -> dict[str, Any]:
    """Serialise a ConformanceReport to a JSON-friendly dict.

    ``asdict`` walks the nested FieldStat dataclasses; tier values are
    StrEnum so they serialise as strings out of the box. ``generated_at``
    is a datetime — let ``json.dumps(..., default=str)`` handle it.
    """
    return asdict(report)


def _print_table(report: ConformanceReport) -> None:
    """Render the table output documented."""
    short_hash = report.fixture_corpus_hash[:8] if report.fixture_corpus_hash else "—"
    typer.echo(f"Conformance report — {report.extractor_id} {report.extractor_version}")
    typer.echo(
        f"Fixtures: {report.fixture_count}  |  "
        f"Particles: {report.particle_count}  |  "
        f"Corpus hash: {short_hash}…"
    )
    # shown only when non-null. A deterministic extractor made no
    # completion call, and a permanently empty cell on six of seven reports
    # would read as a missing value rather than the honest one.
    if report.extraction_provider_model:
        typer.echo(f"Extraction pairing: {report.extraction_provider_model}")
    typer.echo("")

    grouped: dict[FieldTier, list[FieldStat]] = {
        FieldTier.REQUIRED: [],
        FieldTier.RECOMMENDED: [],
        FieldTier.OPTIONAL: [],
    }
    for stat in report.fields:
        grouped[stat.tier].append(stat)

    for tier in (FieldTier.REQUIRED, FieldTier.RECOMMENDED, FieldTier.OPTIONAL):
        rows = grouped[tier]
        if not rows:
            continue
        header = f"{tier.value} fields"
        if tier is FieldTier.OPTIONAL:
            typer.echo(f"{header:<42} {'rate':>5}  {'distinct':>8}")
        else:
            typer.echo(f"{header:<42} {'rate':>5}  {'distinct':>8}  status")
        typer.echo("-" * 72)
        for stat in rows:
            rate = f"{stat.rate:.0%}"
            distinct = str(stat.distinct_values) if stat.distinct_values else "—"
            if tier is FieldTier.OPTIONAL:
                line = f"  {stat.field:<40} {rate:>5}  {distinct:>8}"
            else:
                # an advisory never flips the verdict, so it renders
                # beside PASS rather than instead of it.
                status = "PASS" if stat.passes_threshold else "FAIL"
                if stat.advisory_reason:
                    status += " (advisory)"
                line = f"  {stat.field:<40} {rate:>5}  {distinct:>8}  {status}"
            # say what the rate was measured over whenever that is
            # not the whole run, so a 100 % is never read as wider than it is.
            if stat.excluded_count:
                line += (
                    f"\n    measured over {stat.total_count} of "
                    f"{stat.total_count + stat.excluded_count} particles; "
                    f"{stat.excluded_count} exempt (techspec §9)"
                )
            # Wrap long reasons onto indented continuation lines
            if stat.failure_reason:
                line += f"\n    {stat.failure_reason}"
            if stat.advisory_reason:
                line += f"\n    {stat.advisory_reason}"
            if stat.value_counts:
                # the distribution is the signal a bare distinct
                # count cannot carry — whether a passing diversity result held
                # by one particle or by half of them.
                spread = ", ".join(
                    f"{value} {count}" for value, count in sorted(stat.value_counts.items())
                )
                line += f"\n    values: {spread}"
            typer.echo(line)
        typer.echo("")

    overall = "PASS" if report.passed else "FAIL"
    typer.echo(
        f"Result: {overall}  ({report.failure_count} error(s), "
        f"{report.warning_count} warning(s), {report.advisory_count} advisory(ies))"
    )
    for note in report.quality_notes:
        typer.echo(f"note: {note}")


@extractor_app.command("list")
def extractor_list_cmd() -> None:
    """List all registered extractors with version, trust weight, and domain coverage."""
    run(_extractor_list())


async def _extractor_list() -> None:
    from particles.store.extractor_store import get_all_records

    async with session_scope() as session:
        records = await get_all_records(session)

    if not records:
        typer.echo("No extractor records found. Run: particles db init")
        return

    typer.echo(f"{'EXTRACTOR':<28}  {'VER':<7}  {'TRUST':>5}  DOMAINS")
    typer.echo("-" * 72)
    for r in records:
        domains = ", ".join(f"{c.domain_label} [{c.keyword}]" for c in r.applicability) or "—"
        typer.echo(f"{r.extractor_id:<28}  {r.version:<7}  {r.trust_weight:>5.2f}  {domains}")


@extractor_app.command("trust-set")
def extractor_trust_set_cmd(
    extractor_id: str = typer.Argument(..., help="Extractor ID"),
    weight: float = typer.Argument(..., help="New trust weight [0.0–1.0]"),
) -> None:
    """Override the trust weight for an extractor."""
    if not 0.0 <= weight <= 1.0:
        typer.echo("Weight must be in [0.0, 1.0]", err=True)
        raise typer.Exit(1)
    run(_extractor_trust_set(extractor_id, weight))


async def _extractor_trust_set(extractor_id: str, weight: float) -> None:
    from particles.store.extractor_store import invalidate_trust_cache, set_trust_weight

    async with session_scope() as session:
        ok = await set_trust_weight(session, extractor_id, weight)
        await session.commit()
    if not ok:
        typer.echo(f"Extractor {extractor_id!r} not found. Run: particles extractor list", err=True)
        raise typer.Exit(1)
    invalidate_trust_cache()
    typer.echo(f"Trust weight for {extractor_id!r} set to {weight:.2f}.")


# ---------------------------------------------------------------------------
# Benchmark — extraction-quality benchmark runner
# ---------------------------------------------------------------------------


class _BenchmarkFormat(StrEnum):
    table = "table"
    json = "json"


class _BenchmarkJudge(StrEnum):
    embedding = "embedding"
    llm = "llm"


class _BenchmarkFailOn(StrEnum):
    none = "none"
    precision = "precision"
    recall = "recall"
    calibration = "calibration"


@extractor_app.command("benchmark")
def extractor_benchmark_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID of a registered extractor"),
    suite: str | None = typer.Option(
        None,
        "--suite",
        help="Run only the suite with this suite_id (default: every suite the "
        "extractor is the routing choice for)",
    ),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override suite directory (default: tests/benchmark/suites)",
    ),
    fixtures_dir: Path | None = typer.Option(
        None,
        "--fixtures",
        help="Override fixture directory used to resolve `fixture:` "
        "references (default: tests/conformance/fixtures)",
    ),
    judge: _BenchmarkJudge = typer.Option(
        _BenchmarkJudge.embedding,
        "--judge",
        help="Equivalence judge: embedding cosine (default) or LLM-judge",
    ),
    threshold: float = typer.Option(
        0.80,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Cosine-similarity threshold for the embedding judge",
    ),
    output_format: _BenchmarkFormat = typer.Option(
        _BenchmarkFormat.table, "--format", help="Output format"
    ),
    fail_on: _BenchmarkFailOn = typer.Option(
        _BenchmarkFailOn.none,
        "--fail-on",
        help="Exit non-zero when the named metric falls below --fail-threshold",
    ),
    fail_threshold: float = typer.Option(
        0.9,
        "--fail-threshold",
        min=0.0,
        max=1.0,
        help="Threshold for --fail-on (precision/recall: minimum; calibration: maximum)",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Skip persisting the run report JSON under benchmark.runs_dir",
    ),
    runs: int = typer.Option(
        1,
        "--runs",
        min=1,
        help="Repeat each suite N times and report mean ± spread per metric "
        ". Costs N× the LLM calls; N=1 (default) is unchanged",
    ),
    estimate: bool = typer.Option(
        False,
        "--estimate",
        help="Print the projected LLM cost of the run and exit without running",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Pre-confirm the cost gate for a repeat run (--runs N)",
    ),
) -> None:
    """Run extraction-quality benchmarks against an extractor.

    Discovers every suite under --suites-dir that this extractor is the
    production routing choice for — the registry ladder, read back
    through ``select_extractor``, so the fallback extractor no
    longer inherits every domain suite. ``--suite`` runs a named suite
    regardless of routing. Emits one report per suite, and persists each
    report as a JSON file under ``benchmark.runs_dir`` (stamped with the
    resolved extraction provider:model pairing) unless --no-save is set.
    With --fail-on set, exits non-zero when any suite's named metric
    crosses the threshold.

    ``--runs N`` repeats each suite N times and reports each metric's mean,
    range and standard deviation instead of a single point estimate — the
    error bars a provider comparison needs. Every pass persists
    its own report file, so the series is still one run per JSON envelope.
    ``--fail-on`` is evaluated against the **mean** across runs.
    """
    run(
        _extractor_benchmark(
            extractor_id,
            suite,
            suites_dir,
            fixtures_dir,
            judge,
            threshold,
            output_format,
            fail_on,
            fail_threshold,
            no_save,
            runs,
            estimate,
            yes,
        )
    )


async def _extractor_benchmark(  # noqa: PLR0913 — CLI option list is the API
    extractor_id: str,
    suite_filter: str | None,
    suites_dir_override: Path | None,
    fixtures_dir_override: Path | None,
    judge: _BenchmarkJudge,
    threshold: float,
    output_format: _BenchmarkFormat,
    fail_on: _BenchmarkFailOn,
    fail_threshold: float,
    no_save: bool,
    runs: int,
    estimate_only: bool,
    yes: bool,
) -> None:
    from particles.benchmark.equivalence import EquivalenceJudge
    from particles.benchmark.loader import discover_suites
    from particles.benchmark.runner import run_benchmark, run_benchmark_repeated

    # Locate extractor
    extractor = next((e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID), None)
    if extractor is None:
        known = sorted(e.EXTRACTOR_ID for e in get_extractors())
        typer.echo(
            f"Unknown extractor_id {extractor_id!r}. Registered extractors: {known}",
            err=True,
        )
        raise typer.Exit(2)

    suites_dir = suites_dir_override or Path("tests/benchmark/suites")
    fixtures_dir = fixtures_dir_override or Path("tests/conformance/fixtures")
    judge_enum = EquivalenceJudge(judge.value)

    suites = list(discover_suites(suites_dir))
    if suite_filter is not None:
        suites = [s for s in suites if s.suite_id == suite_filter]
        if not suites:
            typer.echo(
                f"No suite with suite_id {suite_filter!r} found under {suites_dir}",
                err=True,
            )
            raise typer.Exit(2)
    else:
        # Auto-filter: only suites with overlapping source_types.
        # routing precedence, not accepts(). The fallback extractor
        # accepts every source type, so accepts() would hand it every domain
        # suite in the project and report the artifact as its score.
        suites = [s for s in suites if selects(extractor, s.source_types)]

    if not suites:
        typer.echo(
            f"No applicable benchmark suites found for {extractor_id!r}. Suites dir: {suites_dir}",
            err=True,
        )
        raise typer.Exit(2)

    # Cost gate — a repeat run costs N× the LLM calls, so it discloses the
    # projection before spending anything. A single run is
    # never gated: N=1 behaviour is byte-for-byte what it was before
    # --runs existed.
    if runs > 1 or estimate_only:
        _benchmark_cost_gate(suites, extractor, fixtures_dir, runs, estimate_only, yes)
        if estimate_only:
            return

    def _save(report: Any) -> None:
        if no_save:
            return
        saved = _persist_benchmark_run(report)
        if saved is not None:
            typer.echo(f"Run report saved to {saved}", err=True)

    any_failed = False
    for suite in suites:
        if runs == 1:
            report = await run_benchmark(
                suite,
                extractor,
                fixture_dir=fixtures_dir,
                judge=judge_enum,
                threshold=threshold,
            )
            if output_format is _BenchmarkFormat.json:
                typer.echo(json.dumps(_benchmark_report_to_dict(report), indent=2, default=str))
            else:
                _print_benchmark_table(report)
            _save(report)
            metrics = report.metrics
        else:

            def _on_report(index: int, report: Any, _suite_id: str = suite.suite_id) -> None:
                typer.echo(f"  {_suite_id}: run {index + 1}/{runs} done", err=True)
                _save(report)

            aggregate = await run_benchmark_repeated(
                suite,
                extractor,
                runs=runs,
                fixture_dir=fixtures_dir,
                judge=judge_enum,
                threshold=threshold,
                on_report=_on_report,
            )
            if output_format is _BenchmarkFormat.json:
                typer.echo(json.dumps(_aggregate_report_to_dict(aggregate), indent=2, default=str))
            else:
                _print_benchmark_aggregate_table(aggregate)
            metrics = aggregate.mean_metrics

        if fail_on is not _BenchmarkFailOn.none and _fail_on_metric(
            metrics, fail_on, fail_threshold
        ):
            any_failed = True

    if any_failed:
        raise typer.Exit(1)


def _benchmark_cost_gate(
    suites: list[Any],
    extractor: Any,
    fixtures_dir: Path,
    runs: int,
    estimate_only: bool,
    yes: bool,
) -> None:
    """Print the projected cost, then confirm above the config threshold.

    Mirrors the audit's gate and the memory benchmark's: the
    estimate always precedes the first LLM call, ``--estimate`` stops there,
    and a projection over ``benchmark.confirm_call_threshold`` needs ``--yes``
    or an interactive yes. The estimate goes to stderr so ``--format json``
    keeps a clean stdout.
    """
    from particles.benchmark.runner import estimate_benchmark_run, render_benchmark_estimate
    from particles.config import get_config

    cost = estimate_benchmark_run(suites, extractor, fixture_dir=fixtures_dir, runs=runs)
    typer.echo(render_benchmark_estimate(cost), err=True)
    if estimate_only:
        typer.echo("--estimate: nothing was run.", err=True)
        return

    threshold = get_config().benchmark.confirm_call_threshold
    if cost.estimated_extraction_calls > threshold and not yes:
        if not sys.stdin.isatty():
            typer.echo(
                f"Estimated extraction calls ({cost.estimated_extraction_calls}) exceed "
                f"benchmark.confirm_call_threshold ({threshold}) and no --yes was "
                f"given; aborting (non-interactive run).",
                err=True,
            )
            raise typer.Exit(1)
        if not typer.confirm(f"Proceed with ≥{cost.estimated_extraction_calls} extraction calls?"):
            typer.echo("Aborted.", err=True)
            raise typer.Exit(1)


def _fail_on_metric(metrics: dict[str, float], fail_on: _BenchmarkFailOn, threshold: float) -> bool:
    """Return True iff the named metric is on the wrong side of ``threshold``.

    Under ``--runs N`` the caller passes the per-metric **means**, not the
    worst run. Gating on the worst run would make the gate trip more often
    the more samples you take — the false-alarm rate would be a function of
    N rather than of the extractor — so the point estimate stays the mean
    and the spread is disclosed beside it.
    """
    if fail_on is _BenchmarkFailOn.precision:
        return float(metrics.get("precision", 0.0)) < threshold
    if fail_on is _BenchmarkFailOn.recall:
        return float(metrics.get("recall", 0.0)) < threshold
    if fail_on is _BenchmarkFailOn.calibration:
        # Lower calibration_error is better — fail if it's *above* threshold
        return float(metrics.get("calibration_error", 1.0)) > threshold
    return False


def _aggregate_report_to_dict(aggregate: Any) -> dict[str, Any]:
    return asdict(aggregate)


def _benchmark_report_to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)


#: Envelope-format stamp of a persisted benchmark-run file. Bump on any
#: envelope-shape change so downstream readers (drift diffs
#: time-series analysis) can dispatch on it.
_RUN_FILE_FORMAT = 1


def _persist_benchmark_run(report: Any) -> Path | None:
    """Write one run's report JSON under ``benchmark.runs_dir``.

    Persistence is a CLI concern (like --fail-on): the §13.3 harness stays
    report-only and its schema stays frozen — durable metadata the report
    model does not carry (the resolved extraction provider:model pairing)
    rides an envelope around the report dict instead. Returns the written
    path, or None when persistence failed — a full report already went to
    stdout, so a failure to archive it degrades to a warning, never an
    error exit.
    """
    from particles.config import get_config

    sel = get_config().llm.for_purpose("extraction")
    runs_dir = Path(get_config().benchmark.runs_dir).expanduser()
    stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}-benchmark-{report.extractor_id}-{report.suite_id}"
    envelope = {
        "format": _RUN_FILE_FORMAT,
        "extraction_provider_model": f"{sel.provider}:{sel.model}",
        "report": _benchmark_report_to_dict(report),
    }
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{base}.json"
        n = 2
        while path.exists():  # same suite twice within one second
            path = runs_dir / f"{base}-{n}.json"
            n += 1
        path.write_text(json.dumps(envelope, indent=2, default=str) + "\n")
    except OSError as exc:
        typer.echo(f"Warning: could not persist run report under {runs_dir}: {exc}", err=True)
        return None
    return path


# ---------------------------------------------------------------------------
# Benchmark-modality — journal modality-classification benchmark
# ---------------------------------------------------------------------------


@extractor_app.command("benchmark-modality")
def extractor_benchmark_modality_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID of a registered extractor"),
    suite: str | None = typer.Option(
        None,
        "--suite",
        help="Run only the modality suite with this suite_id (default: every "
        "suite the extractor is the routing choice for)",
    ),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override modality-suite directory (default: tests/benchmark/modality)",
    ),
    judge: _BenchmarkJudge = typer.Option(
        _BenchmarkJudge.embedding,
        "--judge",
        help="Claim-alignment judge: embedding cosine (default) or LLM-judge",
    ),
    threshold: float = typer.Option(
        0.65,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Cosine floor for aligning an emitted claim to a gold label "
        "(looser than the content harness's 0.80 — journal claims are reified "
        "paraphrases of their gold labels)",
    ),
    output_format: _BenchmarkFormat = typer.Option(
        _BenchmarkFormat.table, "--format", help="Output format"
    ),
) -> None:
    """Measure assertion_modality classification quality.

    Reports per-modality precision/recall, the dangerous **false-non-FALSIFIABLE
    rate** the journal extractor's inverted default raises, and the
    whole-entry **narrative-emission rate**. Discovers every modality
    suite under --suites-dir the extractor is the production routing choice
    for (or runs only --suite). Report-only and **integration-tier** — it drives the
    extractor's LLM call, so it needs ANTHROPIC_API_KEY.
    """
    run(
        _extractor_benchmark_modality(
            extractor_id, suite, suites_dir, judge, threshold, output_format
        )
    )


async def _extractor_benchmark_modality(  # noqa: PLR0913 — CLI option list is the API
    extractor_id: str,
    suite_filter: str | None,
    suites_dir_override: Path | None,
    judge: _BenchmarkJudge,
    threshold: float,
    output_format: _BenchmarkFormat,
) -> None:
    from particles.benchmark.equivalence import EquivalenceJudge
    from particles.benchmark.modality import (
        discover_modality_suites,
        run_modality_benchmark,
    )

    extractor = next((e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID), None)
    if extractor is None:
        known = sorted(e.EXTRACTOR_ID for e in get_extractors())
        typer.echo(
            f"Unknown extractor_id {extractor_id!r}. Registered extractors: {known}",
            err=True,
        )
        raise typer.Exit(2)

    suites_dir = suites_dir_override or Path("tests/benchmark/modality")
    judge_enum = EquivalenceJudge(judge.value)

    suites = list(discover_modality_suites(suites_dir))
    if suite_filter is not None:
        suites = [s for s in suites if s.suite_id == suite_filter]
        if not suites:
            typer.echo(
                f"No modality suite with suite_id {suite_filter!r} found under {suites_dir}",
                err=True,
            )
            raise typer.Exit(2)
    else:
        # routing precedence, not accepts() — see the note in
        # _extractor_benchmark.
        suites = [s for s in suites if selects(extractor, [s.source_type])]

    if not suites:
        typer.echo(
            f"No applicable modality suites found for {extractor_id!r}. Suites dir: {suites_dir}",
            err=True,
        )
        raise typer.Exit(2)

    for suite_obj in suites:
        report = await run_modality_benchmark(
            suite_obj, extractor, judge=judge_enum, threshold=threshold
        )
        if output_format is _BenchmarkFormat.json:
            typer.echo(json.dumps(_modality_report_to_dict(report), indent=2, default=str))
        else:
            _print_modality_table(report)


def _modality_report_to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)


def _print_modality_table(report: Any) -> None:
    """Render the modality-benchmark table."""
    typer.echo(
        f"Modality benchmark — {report.extractor_id} {report.extractor_version}  "
        f"against  {report.suite_id} {report.suite_version}"
    )
    typer.echo(
        f"Cases: {report.cases_run}/{report.cases_total}     "
        f"Claims aligned: {report.claims_aligned}     "
        f"unaligned: {report.claims_unaligned}"
    )
    typer.echo(f"Judge: {report.judge} @ ≥{report.alignment_threshold:.2f}")
    typer.echo("")
    typer.echo(
        f"  false-non-FALSIFIABLE rate : {report.false_non_falsifiable_rate:.2f}   "
        "(real facts wrongly demoted out of the truth engine — lower is better)"
    )
    typer.echo(
        f"  narrative-emission rate    : {report.narrative_emission_rate:.2f}   "
        f"({report.narrative_cases_emitted}/{report.narrative_cases_expected} entries)"
    )
    typer.echo("")

    expected_support: dict[str, int] = {}
    emitted_support: dict[str, int] = {}
    for cell in report.confusion:
        expected_support[cell.expected] = expected_support.get(cell.expected, 0) + cell.count
        emitted_support[cell.emitted] = emitted_support.get(cell.emitted, 0) + cell.count

    typer.echo(f"{'MODALITY':<16}  {'precision':>9}  {'recall':>7}  {'exp':>4}  {'emit':>4}")
    typer.echo("-" * 52)
    for modality, prec in report.precision.items():
        typer.echo(
            f"  {modality:<14}  {prec:>9.2f}  {report.recall[modality]:>7.2f}  "
            f"{expected_support.get(modality, 0):>4}  {emitted_support.get(modality, 0):>4}"
        )
    typer.echo("")

    if report.confusion:
        typer.echo("Confusion (expected → emitted : count)")
        for cell in report.confusion:
            mark = "  " if cell.expected == cell.emitted else "✗ "
            typer.echo(f"  {mark}{cell.expected:<14} → {cell.emitted:<14} : {cell.count}")
        typer.echo("")

    typer.echo("Per-case detail")
    for c in report.per_case:
        nar = "✓" if c.narrative_emitted else ("✗" if c.narrative_expected else "—")
        total = c.claims_aligned + c.claims_unaligned
        typer.echo(
            f"  {c.case_id}    emitted: {c.claims_emitted}    "
            f"aligned: {c.claims_aligned}/{total}    narrative: {nar}"
        )
    if report.quality_notes:
        typer.echo("")
        typer.echo("Notes:")
        for n in report.quality_notes:
            typer.echo(f"  {n}")


# ---------------------------------------------------------------------------
# Benchmark-polarity — claim-polarity classification benchmark
# ---------------------------------------------------------------------------


@extractor_app.command("benchmark-polarity")
def extractor_benchmark_polarity_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID of a registered extractor"),
    suite: str | None = typer.Option(
        None,
        "--suite",
        help="Run only the polarity suite with this suite_id (default: every "
        "suite the extractor is the routing choice for)",
    ),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override polarity-suite directory (default: tests/benchmark/polarity)",
    ),
    judge: _BenchmarkJudge = typer.Option(
        _BenchmarkJudge.embedding,
        "--judge",
        help="Claim-alignment judge: embedding cosine (default) or LLM-judge",
    ),
    threshold: float = typer.Option(
        0.65,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Cosine floor for aligning an emitted claim to a gold label "
        "(looser than the content harness's 0.80 — the general extractor emits "
        "near-paraphrases of its gold labels)",
    ),
    output_format: _BenchmarkFormat = typer.Option(
        _BenchmarkFormat.table, "--format", help="Output format"
    ),
) -> None:
    """Measure claim-polarity classification quality.

    Reports the dangerous **wrong-`DECLINED` rate** — a real current decision
    (ASSERTED) wrongly classified DECLINED and thereby silently hidden from the
    default surface (the headline, the README-projection-trust risk;
    cap. 1) — plus its superset the wrong-hidden rate and per-polarity
    precision/recall. Discovers every polarity suite under --suites-dir the
    extractor is the production routing choice for (or runs only
    --suite). Report-only and
    **integration-tier** — it drives the extractor's LLM call, so it needs
    ANTHROPIC_API_KEY.
    """
    run(
        _extractor_benchmark_polarity(
            extractor_id, suite, suites_dir, judge, threshold, output_format
        )
    )


async def _extractor_benchmark_polarity(  # noqa: PLR0913 — CLI option list is the API
    extractor_id: str,
    suite_filter: str | None,
    suites_dir_override: Path | None,
    judge: _BenchmarkJudge,
    threshold: float,
    output_format: _BenchmarkFormat,
) -> None:
    from particles.benchmark.equivalence import EquivalenceJudge
    from particles.benchmark.polarity import (
        discover_polarity_suites,
        run_polarity_benchmark,
    )

    extractor = next((e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID), None)
    if extractor is None:
        known = sorted(e.EXTRACTOR_ID for e in get_extractors())
        typer.echo(
            f"Unknown extractor_id {extractor_id!r}. Registered extractors: {known}",
            err=True,
        )
        raise typer.Exit(2)

    suites_dir = suites_dir_override or Path("tests/benchmark/polarity")
    judge_enum = EquivalenceJudge(judge.value)

    suites = list(discover_polarity_suites(suites_dir))
    if suite_filter is not None:
        suites = [s for s in suites if s.suite_id == suite_filter]
        if not suites:
            typer.echo(
                f"No polarity suite with suite_id {suite_filter!r} found under {suites_dir}",
                err=True,
            )
            raise typer.Exit(2)
    else:
        # routing precedence, not accepts() — see the note in
        # _extractor_benchmark.
        suites = [s for s in suites if selects(extractor, [s.source_type])]

    if not suites:
        typer.echo(
            f"No applicable polarity suites found for {extractor_id!r}. Suites dir: {suites_dir}",
            err=True,
        )
        raise typer.Exit(2)

    for suite_obj in suites:
        report = await run_polarity_benchmark(
            suite_obj, extractor, judge=judge_enum, threshold=threshold
        )
        if output_format is _BenchmarkFormat.json:
            typer.echo(json.dumps(_polarity_report_to_dict(report), indent=2, default=str))
        else:
            _print_polarity_table(report)


def _polarity_report_to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)


def _print_polarity_table(report: Any) -> None:
    """Render the polarity-benchmark table."""
    typer.echo(
        f"Polarity benchmark — {report.extractor_id} {report.extractor_version}  "
        f"against  {report.suite_id} {report.suite_version}"
    )
    typer.echo(
        f"Cases: {report.cases_run}/{report.cases_total}     "
        f"Claims aligned: {report.claims_aligned}     "
        f"unaligned: {report.claims_unaligned}"
    )
    typer.echo(f"Judge: {report.judge} @ ≥{report.alignment_threshold:.2f}")
    if not report.polarity_classifier_enabled:
        typer.echo("")
        typer.echo(
            "  ⚠ extraction_polarity.enabled is False — classifier OFF; rates are "
            "vacuously 0.0 (not real)."
        )
    typer.echo("")
    typer.echo(
        f"  wrong-DECLINED rate : {report.wrong_declined_rate:.2f}   "
        "(real decisions wrongly hidden as rejected — the headline danger, lower is better)"
    )
    typer.echo(
        f"  wrong-hidden  rate : {report.wrong_hidden_rate:.2f}   "
        "(real decisions hidden as DECLINED or HYPOTHETICAL — the superset)"
    )
    typer.echo("")

    expected_support: dict[str, int] = {}
    emitted_support: dict[str, int] = {}
    for cell in report.confusion:
        expected_support[cell.expected] = expected_support.get(cell.expected, 0) + cell.count
        emitted_support[cell.emitted] = emitted_support.get(cell.emitted, 0) + cell.count

    typer.echo(f"{'POLARITY':<16}  {'precision':>9}  {'recall':>7}  {'exp':>4}  {'emit':>4}")
    typer.echo("-" * 52)
    for polarity, prec in report.precision.items():
        typer.echo(
            f"  {polarity:<14}  {prec:>9.2f}  {report.recall[polarity]:>7.2f}  "
            f"{expected_support.get(polarity, 0):>4}  {emitted_support.get(polarity, 0):>4}"
        )
    typer.echo("")

    if report.confusion:
        typer.echo("Confusion (expected → emitted : count)")
        for cell in report.confusion:
            mark = "  " if cell.expected == cell.emitted else "✗ "
            typer.echo(f"  {mark}{cell.expected:<14} → {cell.emitted:<14} : {cell.count}")
        typer.echo("")

    typer.echo("Per-case detail")
    for c in report.per_case:
        total = c.claims_aligned + c.claims_unaligned
        typer.echo(
            f"  {c.case_id}    emitted: {c.claims_emitted}    aligned: {c.claims_aligned}/{total}"
        )
    if report.quality_notes:
        typer.echo("")
        typer.echo("Notes:")
        for n in report.quality_notes:
            typer.echo(f"  {n}")


# ---------------------------------------------------------------------------
# Benchmark-validity — event-anchored-validity benchmark
# ---------------------------------------------------------------------------


@extractor_app.command("benchmark-validity")
def extractor_benchmark_validity_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID of a registered extractor"),
    suite: str | None = typer.Option(
        None,
        "--suite",
        help="Run only the validity suite with this suite_id (default: every "
        "suite the extractor is the routing choice for)",
    ),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override validity-suite directory (default: tests/benchmark/validity)",
    ),
    judge: _BenchmarkJudge = typer.Option(
        _BenchmarkJudge.embedding,
        "--judge",
        help="Claim-alignment judge: embedding cosine (default) or LLM-judge",
    ),
    threshold: float = typer.Option(
        0.65,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Cosine floor for aligning an emitted claim to a gold label "
        "(looser than the content harness's 0.80 — the general extractor emits "
        "near-paraphrases of its gold labels)",
    ),
    output_format: _BenchmarkFormat = typer.Option(
        _BenchmarkFormat.table, "--format", help="Output format"
    ),
) -> None:
    """Measure event-anchored-validity quality.

    Reports the dangerous **wrong-expiry rate** — of the aligned claims whose
    gold is durable (no boundary), the fraction the extractor wrongly assigned a
    ``valid_until`` and thereby set up for silent retirement by the §9.3
    staleness lint (the headline, the over-eager-expiry risk) — plus existence
    precision/recall of correct date-bounded extraction and date accuracy.
    Discovers every validity suite under --suites-dir the extractor is the
    production routing choice for (or runs only --suite). Report-only
    and **integration-tier** — it drives the extractor's LLM call, so it needs
    ANTHROPIC_API_KEY.
    """
    run(
        _extractor_benchmark_validity(
            extractor_id, suite, suites_dir, judge, threshold, output_format
        )
    )


async def _extractor_benchmark_validity(  # noqa: PLR0913 — CLI option list is the API
    extractor_id: str,
    suite_filter: str | None,
    suites_dir_override: Path | None,
    judge: _BenchmarkJudge,
    threshold: float,
    output_format: _BenchmarkFormat,
) -> None:
    from particles.benchmark.equivalence import EquivalenceJudge
    from particles.benchmark.validity import (
        discover_validity_suites,
        run_validity_benchmark,
    )

    extractor = next((e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID), None)
    if extractor is None:
        known = sorted(e.EXTRACTOR_ID for e in get_extractors())
        typer.echo(
            f"Unknown extractor_id {extractor_id!r}. Registered extractors: {known}",
            err=True,
        )
        raise typer.Exit(2)

    suites_dir = suites_dir_override or Path("tests/benchmark/validity")
    judge_enum = EquivalenceJudge(judge.value)

    suites = list(discover_validity_suites(suites_dir))
    if suite_filter is not None:
        suites = [s for s in suites if s.suite_id == suite_filter]
        if not suites:
            typer.echo(
                f"No validity suite with suite_id {suite_filter!r} found under {suites_dir}",
                err=True,
            )
            raise typer.Exit(2)
    else:
        # routing precedence, not accepts() — see the note in
        # _extractor_benchmark.
        suites = [s for s in suites if selects(extractor, [s.source_type])]

    if not suites:
        typer.echo(
            f"No applicable validity suites found for {extractor_id!r}. Suites dir: {suites_dir}",
            err=True,
        )
        raise typer.Exit(2)

    for suite_obj in suites:
        report = await run_validity_benchmark(
            suite_obj, extractor, judge=judge_enum, threshold=threshold
        )
        if output_format is _BenchmarkFormat.json:
            typer.echo(json.dumps(_validity_report_to_dict(report), indent=2, default=str))
        else:
            _print_validity_table(report)


def _validity_report_to_dict(report: Any) -> dict[str, Any]:
    return asdict(report)


def _print_validity_table(report: Any) -> None:
    """Render the validity-benchmark table."""
    typer.echo(
        f"Validity benchmark — {report.extractor_id} {report.extractor_version}  "
        f"against  {report.suite_id} {report.suite_version}"
    )
    typer.echo(
        f"Cases: {report.cases_run}/{report.cases_total}     "
        f"Claims aligned: {report.claims_aligned}     "
        f"unaligned: {report.claims_unaligned}"
    )
    typer.echo(f"Judge: {report.judge} @ ≥{report.alignment_threshold:.2f}")
    if not report.validity_extractor_enabled:
        typer.echo("")
        typer.echo(
            "  ⚠ extraction_validity.enabled is False — extractor OFF; rates are "
            "vacuously safe (not real)."
        )
    typer.echo("")
    typer.echo(
        f"  wrong-expiry rate : {report.wrong_expiry_rate:.2f}   "
        "(durable facts wrongly given a valid_until → silently retired — the headline "
        "danger, lower is better)"
    )
    s = report.support
    typer.echo(
        f"  expiry precision  : {report.expiry_precision:.2f}   "
        f"recall : {report.expiry_recall:.2f}   "
        f"date-acc (±{report.date_tolerance_days}d) : {report.date_accuracy:.2f}"
    )
    typer.echo(
        f"  support           : durable={s.get('durable', 0)}  bounded={s.get('bounded', 0)}  "
        f"emitted={s.get('emitted', 0)}  both={s.get('both', 0)}"
    )
    typer.echo("")

    typer.echo("Per-case detail (expected → emitted boundary)")
    for c in report.per_case:
        total = c.claims_aligned + c.claims_unaligned
        typer.echo(
            f"  {c.case_id}    emitted: {c.claims_emitted}    aligned: {c.claims_aligned}/{total}"
        )
        for exp, emi in c.pairs:
            mark = "  " if exp == emi else "✗ "
            typer.echo(f"    {mark}{exp:<12} → {emi}")
    if report.quality_notes:
        typer.echo("")
        typer.echo("Notes:")
        for n in report.quality_notes:
            typer.echo(f"  {n}")


# ---------------------------------------------------------------------------
# Benchmark-compare — multi-extractor comparison
# ---------------------------------------------------------------------------


@extractor_app.command("benchmark-compare")
def extractor_benchmark_compare_cmd(
    extractor_ids: list[str] = typer.Option(
        ...,
        "--extractor-id",
        help="EXTRACTOR_ID to include in the comparison (repeat ≥2 times)",
    ),
    suite: str | None = typer.Option(
        None,
        "--suite",
        help="Restrict to a single suite_id (default: every suite whose "
        "source_types intersect ANY supplied extractor's accepts())",
    ),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override suite directory (default: tests/benchmark/suites)",
    ),
    fixtures_dir: Path | None = typer.Option(
        None,
        "--fixtures",
        help="Override fixture directory (default: tests/conformance/fixtures)",
    ),
    judge: _BenchmarkJudge = typer.Option(
        _BenchmarkJudge.embedding,
        "--judge",
        help="Equivalence judge: embedding cosine (default) or LLM-judge",
    ),
    threshold: float = typer.Option(
        0.80, "--threshold", min=0.0, max=1.0, help="Embedding-judge cosine threshold"
    ),
    output_format: _BenchmarkFormat = typer.Option(
        _BenchmarkFormat.table, "--format", help="Output format"
    ),
) -> None:
    """Compare two or more extractors against the same benchmark corpus.

    \b
        particles extractor benchmark-compare \\
            --extractor-id numista-coin-extractor \\
            --extractor-id numista-coin-extractor-v3

    Cells where an extractor declined a suite's source_type render as
    `—` in the table view and `null` in JSON output.
    """
    if len(extractor_ids) < 2:
        typer.echo(
            "benchmark-compare requires at least two --extractor-id flags. "
            "For a single extractor, use `particles extractor benchmark <id>`.",
            err=True,
        )
        raise typer.Exit(2)
    run(
        _extractor_benchmark_compare(
            extractor_ids,
            suite,
            suites_dir,
            fixtures_dir,
            judge,
            threshold,
            output_format,
        )
    )


async def _extractor_benchmark_compare(  # noqa: PLR0913 — CLI option list is the API
    extractor_ids: list[str],
    suite_filter: str | None,
    suites_dir_override: Path | None,
    fixtures_dir_override: Path | None,
    judge: _BenchmarkJudge,
    threshold: float,
    output_format: _BenchmarkFormat,
) -> None:
    from particles.benchmark.compare import BenchmarkComparison, compare_benchmarks
    from particles.benchmark.equivalence import EquivalenceJudge
    from particles.benchmark.loader import discover_suites

    known = {e.EXTRACTOR_ID: e for e in get_extractors()}
    missing = [eid for eid in extractor_ids if eid not in known]
    if missing:
        typer.echo(
            f"Unknown extractor_id(s) {missing!r}. Registered: {sorted(known)}",
            err=True,
        )
        raise typer.Exit(2)
    extractors = [known[eid] for eid in extractor_ids]

    suites_dir = suites_dir_override or Path("tests/benchmark/suites")
    fixtures_dir = fixtures_dir_override or Path("tests/conformance/fixtures")
    judge_enum = EquivalenceJudge(judge.value)

    suites = list(discover_suites(suites_dir))
    if suite_filter is not None:
        suites = [s for s in suites if s.suite_id == suite_filter]
        if not suites:
            typer.echo(
                f"No suite with suite_id {suite_filter!r} found under {suites_dir}",
                err=True,
            )
            raise typer.Exit(2)
    else:
        # Deliberately accepts(), not the routing-precedence filter the
        # single-extractor verbs use: a named cross-extractor bake-off is this
        # verb's whole job, and routing precedence would leave at most one
        # extractor per suite and empty the matrix. §3 of that ADR.
        suites = [
            s for s in suites if any(ext.accepts(st) for ext in extractors for st in s.source_types)
        ]

    if not suites:
        typer.echo(
            f"No applicable benchmark suites found for {extractor_ids!r}. Suites dir: {suites_dir}",
            err=True,
        )
        raise typer.Exit(2)

    per_suite_comparisons = []
    for suite_obj in suites:
        comparison = await compare_benchmarks(
            suite_obj,
            extractors,
            fixture_dir=fixtures_dir,
            judge=judge_enum,
            threshold=threshold,
        )
        per_suite_comparisons.extend(comparison.suites)

    aggregated = BenchmarkComparison(
        judge=judge_enum.value,
        threshold=threshold,
        extractor_ids=extractor_ids,
        suites=per_suite_comparisons,
        generated_at=per_suite_comparisons[0].reports[extractor_ids[0]].generated_at
        if per_suite_comparisons
        else _now_utc(),
    )

    if output_format is _BenchmarkFormat.json:
        typer.echo(aggregated.model_dump_json(indent=2))
    else:
        _print_benchmark_compare_table(aggregated)


def _now_utc() -> datetime:
    from datetime import UTC

    return datetime.now(UTC)


def _print_benchmark_compare_table(comparison: Any) -> None:
    """Render the suite × extractor matrix table (§Decision)."""
    typer.echo("Benchmark comparison")
    typer.echo(f"Judge: {comparison.judge} @ ≥{comparison.threshold:.2f}")
    typer.echo(f"Suites: {len(comparison.suites)}     Extractors: {len(comparison.extractor_ids)}")
    # Column width: enough for the longest extractor_id + 2 padding.
    col_w = max(24, max(len(eid) for eid in comparison.extractor_ids) + 2)
    suite_col_w = max(20, max((len(s.suite_id) for s in comparison.suites), default=20))
    for metric in ("recall", "precision", "calibration_error"):
        typer.echo("")
        typer.echo(metric)
        header = f"{'SUITE':<{suite_col_w}}  " + "  ".join(
            f"{eid:>{col_w}}" for eid in comparison.extractor_ids
        )
        typer.echo(header)
        typer.echo("-" * len(header))
        for suite in comparison.suites:
            cells = []
            for eid in comparison.extractor_ids:
                v = suite.metrics.get(metric, {}).get(eid)
                cell = "—" if v is None else f"{v:.2f}"
                cells.append(f"{cell:>{col_w}}")
            typer.echo(f"{suite.suite_id:<{suite_col_w}}  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# Calibrate — temperature-scaling calibration
# ---------------------------------------------------------------------------

#: Where `extractor calibrate` looks by default. A *sibling* of
#: `tests/benchmark/suites/`, not a subdirectory of it, and holding ordinary
#: `BenchmarkSuite` YAML read by the same frozen schema / loader / runner — the
#: separation is the directory and nothing else.
#:
#: The split exists because the two purposes pull opposite ways on one
#: artifact's coverage fraction. A §13.3 suite is under standing pressure
#: toward *total* gold coverage (precision and calibration_error are computed
#: over every emitted particle, so a sparse gold set reports a well-behaved
#: extractor as imprecise), and total coverage is exactly the all-True label
#: set a temperature fit cannot use. `extractor benchmark` never discovers this
#: directory, so a deliberately-partial gold set never scores itself down.
DEFAULT_CALIBRATION_SUITES_DIR = Path("tests/benchmark/calibration")


def _selected_suite_ids(extractor: Any, suites_dir: Path) -> set[str] | None:
    """Suite ids the auto-filter selects for ``extractor``, or None.

    ``None`` means the question could not be asked — the suites directory is
    absent or unreadable, which is the normal state of an *installed* SDK:
    `tests/` ships in neither the wheel nor the sdist, so the staleness predicate is repo-only by construction. Callers degrade to an
    unannotated listing rather than failing; a missing suites tree is not an
    error, it is the absence of an input.
    """
    from particles.benchmark.loader import discover_suites

    if not suites_dir.is_dir():
        return None
    return {s.suite_id for s in discover_suites(suites_dir) if selects(extractor, s.source_types)}


def _unapplied_note(calibration: Any) -> str | None:
    """One-line note for a stored record this SDK will not apply, or None."""
    from particles.extraction.calibration import scaler_for_record

    if scaler_for_record(calibration) is not None:
        return None
    return (
        "NOT APPLIED: fitted before — its temperature parameterises the "
        "retired clamp(raw/T, 0, 1) form and was fitted against all-False labels. "
        "This pairing mints EXTRACTOR_DIRECT particles until it is re-fitted."
    )


def _staleness_note(calibration: Any, selected: set[str]) -> str | None:
    """One-line staleness annotation for a stored calibration, or None if clean."""
    from particles.extraction.calibration import fitted_suite_ids, is_suite_stale

    if not is_suite_stale(calibration.benchmark_suite_id, selected):
        return None
    fitted = "+".join(sorted(fitted_suite_ids(calibration.benchmark_suite_id))) or "(none)"
    now = "+".join(sorted(selected)) or "(none — this extractor auto-matches no suite)"
    return f"STALE: fitted over {fitted}; this extractor's suites are now {now}"


@extractor_app.command("calibrate")
def extractor_calibrate_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID of a registered extractor"),
    suite: str | None = typer.Option(
        None,
        "--suite",
        help="Restrict calibration to a single suite_id (default: every suite the "
        "extractor is the routing choice for)",
    ),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override suite directory (default: tests/benchmark/calibration)",
    ),
    fixtures_dir: Path | None = typer.Option(
        None,
        "--fixtures",
        help="Override fixture directory used to resolve `fixture:` "
        "references (default: tests/conformance/fixtures)",
    ),
    judge: _BenchmarkJudge = typer.Option(
        _BenchmarkJudge.llm,
        "--judge",
        help="Equivalence judge for the calibration label: LLM-judge (default) or embedding cosine",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Fit and print but do not persist the calibration record",
    ),
    regenerate: bool = typer.Option(
        False,
        "--regenerate/--no-regenerate",
        help="Overwrite an existing calibration; without it, an extractor "
        "that already has one exits 1",
    ),
) -> None:
    """Fit a temperature-scaling calibration for an extractor.

    Runs every applicable **calibration** suite — `tests/benchmark/calibration/`,
    a sibling of the §13.3 `suites/` directory whose gold sets are deliberately
    partial — collects (raw_confidence, correct) pairs from
    emitted-vs-matched, fits a single T via NLL minimisation, and persists the
    result on the extractor record. Subsequent particles produced by this
    extractor carry calibration_source=CALIBRATED_BENCHMARK and a
    temperature-scaled confidence value. Pre-existing particles are
    unaffected — operators who want retroactive application should run
    `particles reindex --extractor-id <id>`.

    The fit is refused rather than persisted when it cannot mean anything
    : degenerate labels, a temperature on an
    optimizer bound, fewer than two distinct movable confidences, or a
    calibration that does not reduce calibration error.

    Unlike `extractor benchmark`, this verb defaults to the **LLM** equivalence
    judge — see `_extractor_calibrate` for why the calibration label
    cannot afford the embedding judge's paraphrase misses.
    """
    run(
        _extractor_calibrate(
            extractor_id,
            suite,
            suites_dir,
            fixtures_dir,
            judge,
            dry_run,
            regenerate,
        )
    )


async def _extractor_calibrate(  # noqa: PLR0913 — CLI option list is the API
    extractor_id: str,
    suite_filter: str | None,
    suites_dir_override: Path | None,
    fixtures_dir_override: Path | None,
    judge: _BenchmarkJudge,
    dry_run: bool,
    regenerate: bool,
) -> None:
    from datetime import UTC, datetime

    from particles.benchmark.equivalence import EquivalenceJudge
    from particles.benchmark.loader import discover_suites
    from particles.benchmark.runner import graded_pairs, run_benchmark
    from particles.config import get_config
    from particles.core.schema import ExtractorCalibration
    from particles.extraction.calibration import (
        TRANSFORM_LOGIT,
        TemperatureScaler,
        calibration_error,
    )
    from particles.store.extractor_store import (
        LEGACY_PROVIDER_MODEL,
        get_calibration,
        get_calibrations,
        get_extractor_record,
        upsert_calibration,
    )

    # Locate extractor
    extractor = next((e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID), None)
    if extractor is None:
        known = sorted(e.EXTRACTOR_ID for e in get_extractors())
        typer.echo(
            f"Unknown extractor_id {extractor_id!r}. Registered extractors: {known}",
            err=True,
        )
        raise typer.Exit(2)

    # calibration reads its own suite family, whose gold sets are
    # deliberately partial. `extractor benchmark` never discovers this dir.
    suites_dir = suites_dir_override or DEFAULT_CALIBRATION_SUITES_DIR
    fixtures_dir = fixtures_dir_override or Path("tests/conformance/fixtures")

    suites = list(discover_suites(suites_dir))
    if suite_filter is not None:
        suites = [s for s in suites if s.suite_id == suite_filter]
        if not suites:
            typer.echo(
                f"No suite with suite_id {suite_filter!r} found under {suites_dir}",
                err=True,
            )
            raise typer.Exit(2)
    else:
        # routing precedence, not accepts(). The fallback extractor
        # accepts every source type, so accepts() would hand it every domain
        # suite in the project and report the artifact as its score.
        suites = [s for s in suites if selects(extractor, s.source_types)]

    if not suites:
        typer.echo(
            f"No applicable calibration suites found for {extractor_id!r}. "
            f"Suites dir: {suites_dir}. Calibration reads its own suite family "
            ", whose gold sets deliberately cover only part of what an "
            "extractor emits — a §13.3 benchmark suite under tests/benchmark/suites/ "
            "will not be found here, and its total gold coverage could not fit a "
            "temperature anyway.",
            err=True,
        )
        raise typer.Exit(2)

    # the guard is per (extractor, current extraction provider_model) —
    # calibrating a *new* model doesn't trip on another model's record.
    sel_guard = get_config().llm.for_purpose("extraction")
    provider_model_guard = f"{sel_guard.provider}:{sel_guard.model}"
    async with session_scope() as session:
        existing_cal = await get_calibration(session, extractor_id, provider_model_guard)
        stored_cals = await get_calibrations(session, extractor_id)

    # the guard below is keyed on the *configured*
    # pairing, so it is structurally blind to every other stored pairing — a
    # re-fit for a new provider leaves an old pairing's record in place and
    # says nothing about it, which is exactly how narrowing
    # stranded records without anyone noticing. Report the ones whose suite
    # set no longer matches this extractor's. Emitted *before* the guard can
    # exit 1: an operator standing at the calibrate prompt is the one
    # positioned to act, and that path is the one they hit most.
    selected_now = _selected_suite_ids(extractor, suites_dir)
    if selected_now is not None:
        for cal in sorted(stored_cals, key=lambda c: c.provider_model or ""):
            key = cal.provider_model or LEGACY_PROVIDER_MODEL
            if key == provider_model_guard:
                continue  # the pairing this run is about; the guard speaks for it
            note = _staleness_note(cal, selected_now)
            if note is not None:
                typer.echo(
                    f"warning: another stored calibration for {extractor_id!r} "
                    f"({key}) is {note}. It is not re-fitted by this run and stays "
                    f"in effect for that pairing; retire it with "
                    f"`particles extractor calibration-forget {extractor_id} {key}`.",
                    err=True,
                )

    if existing_cal is not None and not regenerate:
        typer.echo(
            f"Extractor {extractor_id!r} already has a calibration for "
            f"{provider_model_guard} fitted on {existing_cal.fitted_at.isoformat()}. "
            "Use --regenerate to overwrite.",
            err=True,
        )
        raise typer.Exit(1)

    # Run every applicable suite once; take its own labelled population.
    #
    # the pairs come off the report the run just produced. The verb
    # used to re-run the extractor per case to recover raw confidences and
    # label each *fresh* particle by `p.id in matched_ids` — ids minted by the
    # second run, checked against the first run's set, so every label was False
    # and every fit ran to the optimizer bound. Reading run #1's own pairs is
    # both the correct labelling and half the LLM cost.
    # the calibration label IS the equivalence judge's verdict, so the
    # judge's error rate is the label's error rate. That is tolerable for §13.3
    # (a paraphrase miss is one precision point among many) and not tolerable
    # here: a well-behaved extractor's *only* negatives are judge misses, so they
    # are not noise around the signal — they are substantially all of it.
    #
    # Measured over `prose-calibration-001` (general-extractor 0.14.0,
    # anthropic:claude-sonnet-5), the embedding judge scored these pairs — each
    # two phrasings of ONE claim — below its 0.80 floor: 0.7915, 0.7872, 0.7769,
    # 0.5302, 0.4003. Every one became a label saying the extractor was wrong
    # when it was right, and across four runs of identical inputs the fitted T
    # moved 0.2486 → 0.7418 while the verdict flipped between refuse
    # and persist. What gets persisted is immutable at particle creation
    #, so that spread is not a reporting problem.
    #
    # Hence the default is the LLM judge, unlike `extractor benchmark`. It costs
    # calls only in the contested band — `equivalence.py` accepts above the
    # threshold on cosine alone and rejects below `llm_prefilter` (0.65) without
    # asking — so this buys label fidelity for a bounded number of calls on a
    # verb an operator runs rarely. `--judge embedding` restores the old
    # behaviour for a cost-free (and noisier) fit.
    #
    # Known residual: pairs below the 0.65 prefilter are still rejected without
    # adjudication, which is why two of the five misses above (0.5302, 0.4003)
    # need paraphrase twins in the suite rather than a better judge.
    judge_enum = EquivalenceJudge(judge.value)

    # the pairs come off the report the run just produced (see above).
    raw_values: list[float] = []
    labels: list[bool] = []
    suite_ids: list[str] = []
    for suite_obj in suites:
        report = await run_benchmark(
            suite_obj, extractor, fixture_dir=fixtures_dir, judge=judge_enum
        )
        suite_ids.append(suite_obj.suite_id)
        suite_raws, suite_labels = graded_pairs(report)
        raw_values.extend(suite_raws)
        labels.extend(suite_labels)

    sample_size = len(raw_values)
    if sample_size < 2:
        typer.echo(
            "Cannot fit a temperature on fewer than 2 (raw, label) pairs. "
            f"Collected {sample_size} pair(s) across {len(suites)} suite(s). "
            "Add more benchmark cases or check that the extractor emits particles "
            "against the suite fixtures.",
            err=True,
        )
        raise typer.Exit(1)

    scaler = TemperatureScaler().fit(raw_values, labels)
    # ECE is measured over the *full* emitted population, saturated pairs
    # included: the calibration is applied to them too (unchanged), so this is
    # what the operator's store would actually experience. The fit deliberately
    # saw a narrower set.
    #
    # Both figures are pooled over that one population with one estimator.
    # `ece_before` used to be an emission-weighted mean of the per-suite
    # reports' `calibration_error`, which is not the ECE of their union: ECE
    # bins by confidence, so two suites sharing a bin at different accuracies
    # do not average. That was a cosmetic discrepancy while the pair was only
    # printed; the comparison now decides whether the record is
    # persisted, and a refusal must not turn on which of two estimators
    # produced which half. Single-suite fits — every fit in tree today — are
    # unaffected: there the weighted mean was already the pooled value.
    calibrated = scaler.calibrate_batch(raw_values)
    ece_before = calibration_error(raw_values, labels, n_bins=10)
    ece_after = calibration_error(calibrated, labels, n_bins=10)
    diagnostics = scaler.diagnostics
    if diagnostics is not None:
        diagnostics = diagnostics.with_ece(ece_before, ece_after)

    suite_id_label = "+".join(suite_ids)
    n_correct = sum(1 for c in labels if c)
    typer.echo(
        f"T={scaler.temperature:.4f}, ECE: {ece_before:.4f} → {ece_after:.4f}, "
        f"sample N={sample_size}"
    )
    typer.echo(f"  suites: {suite_id_label}")
    # The judge decides every label, so two fits are only comparable when it
    # matches. Printed beside the labels it produced, not buried.
    typer.echo(f"  judge:  {judge_enum.value}")
    typer.echo(f"  labels: {n_correct} matched / {sample_size - n_correct} unmatched")
    if diagnostics is not None:
        typer.echo(
            f"  fittable: {diagnostics.n_fitted} of {diagnostics.n} pair(s), "
            f"{diagnostics.distinct_raw} distinct confidence value(s)"
        )
        # always say what was dropped and why. An operator told
        # "94 of 101 stated a saturated confidence" has been told the actual
        # state of their extractor, not just handed a smaller N.
        if diagnostics.n_saturated:
            typer.echo(
                f"  dropped: {diagnostics.n_saturated} pair(s) stated a saturated "
                "confidence (0.0 or 1.0), an exact fixed point of the transform — "
                "no temperature can move them, so they cannot inform one"
            )

    # a fit that cannot mean anything is never
    # persisted. The temperature is applied at particle creation and the value
    # it writes is immutable, so persisting an unfittable fit
    # silently rewrites every future confidence — the failure this guards.
    if diagnostics is not None and not diagnostics.is_trustworthy:
        for reason in diagnostics.reasons():
            typer.echo(f"error: {reason}", err=True)
        typer.echo(
            "Refusing to persist this calibration. An extractor with no stored "
            "calibration mints EXTRACTOR_DIRECT particles carrying their raw stated "
            "confidence, which is honest; a bad temperature is not.",
            err=True,
        )
        raise typer.Exit(1)

    if dry_run:
        typer.echo("--dry-run: calibration not persisted.")
        return

    # record the (provider, model) the benchmark ran under so the
    # pipeline only applies this temperature to that pairing's outputs.
    # `sample_size` is the *fitted* population — the pairs the T was actually
    # regressed from, not the wider set the ECE figures cover.
    sel = get_config().llm.for_purpose("extraction")
    calibration = ExtractorCalibration(
        temperature=scaler.temperature,
        transform=TRANSFORM_LOGIT,
        fitted_at=datetime.now(UTC),
        benchmark_suite_id=suite_id_label,
        sample_size=diagnostics.n_fitted if diagnostics is not None else sample_size,
        calibration_error_before=ece_before,
        calibration_error_after=ece_after,
        provider_model=f"{sel.provider}:{sel.model}",
    )

    # persist the calibration keyed by its provider_model in the
    # extractor_calibrations table, leaving other models' calibrations intact.
    # The registry record must exist (db init registers the built-ins).
    async with session_scope() as session:
        record = await get_extractor_record(session, extractor_id)
        if record is None:
            typer.echo(
                f"No extractor record found for {extractor_id!r}. "
                "Run `particles db init` first to register the built-in extractors.",
                err=True,
            )
            raise typer.Exit(1)
        await upsert_calibration(session, extractor_id, calibration)
        await session.commit()
    typer.echo(f"Calibration persisted for {extractor_id!r} ({calibration.provider_model}).")


@extractor_app.command("calibrations")
def extractor_calibrations_cmd(
    extractor_id: str = typer.Argument(..., help="Extractor id to list calibrations for"),
    suites_dir: Path | None = typer.Option(
        None,
        "--suites-dir",
        help="Override suite directory used to report suite-set staleness "
        "(default: tests/benchmark/calibration)",
    ),
) -> None:
    """List stored calibrations per (provider, model) for an extractor.

    Each `extractor calibrate` run stores one record keyed by the extraction
    model it ran under, so several models' calibrations coexist; the one matching
    the configured extraction model is applied at extraction time.

    Each record is checked for **suite-set staleness**: a fit whose
    contributing suites differ from the ones the extractor auto-matches today
    answers a question it is no longer asked. The check needs the benchmark
    suites, which ship in neither the wheel nor the sdist, so on an installed
    SDK the listing prints unannotated.
    """
    run(_extractor_calibrations(extractor_id, suites_dir))


async def _extractor_calibrations(extractor_id: str, suites_dir_override: Path | None) -> None:
    from particles.store.extractor_store import LEGACY_PROVIDER_MODEL, get_calibrations

    async with session_scope() as session:
        cals = await get_calibrations(session, extractor_id)
    if not cals:
        typer.echo(f"No calibrations stored for {extractor_id!r}.")
        return

    # a calibration record's `benchmark_suite_id` names calibration
    # suites, so staleness must be judged against the calibration directory.
    # Pointed at `suites/` — as it was before the family existed — every stored
    # record would read as permanently stale against a set it was never fitted
    # over, which is worse than not reporting at all.
    suites_dir = suites_dir_override or DEFAULT_CALIBRATION_SUITES_DIR
    extractor = next((e for e in get_extractors() if extractor_id == e.EXTRACTOR_ID), None)
    # Two distinct reasons the staleness column can be absent, and they are not
    # the same fact: no suites directory (an installed SDK), or no such
    # extractor registered (a record left behind by one that was removed).
    selected = _selected_suite_ids(extractor, suites_dir) if extractor is not None else None

    typer.echo(f"Calibrations for {extractor_id!r}:")
    stale_keys: list[str] = []
    unapplied_keys: list[str] = []
    for c in sorted(cals, key=lambda x: x.provider_model or ""):
        typer.echo(
            f"  {c.provider_model or 'LEGACY':<34} T={c.temperature:.4f}  "
            f"ECE {c.calibration_error_before:.4f}→{c.calibration_error_after:.4f}  "
            f"N={c.sample_size}  fitted {c.fitted_at.isoformat()}  "
            f"suites={c.benchmark_suite_id}"
        )
        # reported before staleness because it is the stronger fact —
        # a record that is never applied cannot be stale *at* anything.
        if (unapplied := _unapplied_note(c)) is not None:
            typer.echo(f"  {'':<34} {unapplied}")
            unapplied_keys.append(c.provider_model or LEGACY_PROVIDER_MODEL)
        if selected is None:
            continue
        note = _staleness_note(c, selected)
        if note is not None:
            typer.echo(f"  {'':<34} {note}")
            stale_keys.append(c.provider_model or LEGACY_PROVIDER_MODEL)

    if unapplied_keys:
        typer.echo(
            f"\n{len(unapplied_keys)} record(s) are not applied at all. "
            "Re-fit one by configuring its pairing for extraction and running "
            f"`particles extractor calibrate {extractor_id} --regenerate`, or retire it:"
        )
        for key in unapplied_keys:
            typer.echo(f"  particles extractor calibration-forget {extractor_id} {key}")

    if selected is None:
        reason = (
            f"no suites directory at {suites_dir}"
            if extractor is not None
            else f"{extractor_id!r} is not a registered extractor"
        )
        typer.echo(f"\n(suite-set staleness not checked — {reason}.)")
        return
    if stale_keys:
        typer.echo(
            f"\n{len(stale_keys)} stale record(s). A stale fit still applies whenever "
            "extraction runs under its pairing. Re-fit it "
            f"(`particles extractor calibrate {extractor_id} --regenerate`, under that "
            "pairing) or retire it:"
        )
        for key in stale_keys:
            typer.echo(f"  particles extractor calibration-forget {extractor_id} {key}")


@extractor_app.command("calibration-forget")
def extractor_calibration_forget_cmd(
    extractor_id: str = typer.Argument(..., help="EXTRACTOR_ID the calibration belongs to"),
    provider_model: str = typer.Argument(
        ...,
        help='The "<provider>:<model>" pairing to retire, as printed by `extractor calibrations`',
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt (for scripted use)"
    ),
) -> None:
    """Retire one stored calibration record.

    The counterpart to `extractor calibrate`. Before this verb a stored
    calibration could only be *replaced* — by re-fitting under the same
    pairing — so a record fitted against a model no longer reachable (a local
    endpoint since torn down) could not be retired at all without standing
    that model back up.

    Removing a record returns that pairing to `calibration_source=
    EXTRACTOR_DIRECT`, the documented fallback for an uncalibrated pairing.
    Particles already in the store keep the confidence they were minted with
    ; run `particles reindex --extractor-id <id>` to re-mint them.
    """
    run(_extractor_calibration_forget(extractor_id, provider_model, yes))


async def _extractor_calibration_forget(extractor_id: str, provider_model: str, yes: bool) -> None:
    from particles.store.extractor_store import delete_calibration, get_calibration

    async with session_scope() as session:
        existing = await get_calibration(session, extractor_id, provider_model)
    if existing is None:
        typer.echo(
            f"No calibration stored for {extractor_id!r} under {provider_model!r}. "
            f"List what is stored with `particles extractor calibrations {extractor_id}`.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"About to retire the calibration for {extractor_id!r} / {provider_model}:")
    typer.echo(
        f"  T={existing.temperature:.4f}  "
        f"ECE {existing.calibration_error_before:.4f}→{existing.calibration_error_after:.4f}  "
        f"N={existing.sample_size}  fitted {existing.fitted_at.isoformat()}"
    )
    typer.echo(f"  fitted over suites: {existing.benchmark_suite_id}")
    typer.echo(
        "Particles minted under this pairing from now on carry EXTRACTOR_DIRECT; "
        "particles already stored are unchanged."
    )
    if not yes:
        typer.confirm("Retire it?", abort=True)

    async with session_scope() as session:
        removed = await delete_calibration(session, extractor_id, provider_model)
        await session.commit()
    if removed is None:
        # Lost a race with a concurrent forget; the desired end state holds.
        typer.echo(f"Calibration for {extractor_id!r} / {provider_model} was already gone.")
        return
    typer.echo(f"Retired calibration for {extractor_id!r} / {provider_model}.")


def _print_benchmark_table(report: Any) -> None:
    """Render the table view documented."""
    typer.echo(
        f"Benchmark report — {report.extractor_id} {report.extractor_version}  "
        f"against  {report.suite_id} {report.suite_version}"
    )
    typer.echo(
        f"Cases: {report.cases_run}/{report.cases_total}     "
        f"Particles emitted: {report.particles_emitted}     "
        f"Required expected: {report.particles_required_total}"
    )
    typer.echo(f"Judge: {report.judge} @ ≥{report.equivalence_threshold:.2f}")
    typer.echo("")
    typer.echo(f"{'METRIC':<22}  VALUE")
    typer.echo("-" * 50)
    for name in ("recall", "precision", "calibration_error"):
        if name in report.metrics:
            typer.echo(f"  {name:<20}  {report.metrics[name]:.2f}")
    typer.echo("")
    typer.echo("Per-case detail")
    for c in report.per_case:
        typer.echo(f"  {c.case_id}")
        typer.echo(
            f"    emitted: {c.emitted_count}    "
            f"matched: {len(c.matched)} (req {c.matched_required_count})    "
            f"spurious: {len(c.spurious)}    "
            f"missed required: {len(c.missed_required)}    "
            f"under-confidence: {len(c.under_confidence)}"
        )
        for expected_content, _emitted_id in c.matched[:5]:
            typer.echo(f"      ✓ {expected_content[:80]}")
        for missed in c.missed_required:
            typer.echo(f"      ✗ MISSED REQUIRED: {missed[:80]}")
        for expected_content, stated, required_min in c.under_confidence:
            typer.echo(
                f"      ~ UNDER-CONFIDENCE ({stated:.2f} < {required_min:.2f}): "
                f"{expected_content[:80]}"
            )
    if report.quality_notes:
        typer.echo("")
        typer.echo("Notes:")
        for n in report.quality_notes:
            typer.echo(f"  {n}")


def _print_benchmark_aggregate_table(aggregate: Any) -> None:
    """Render the repeat-runs view — mean ± spread per metric.

    Deliberately does *not* print each run's per-case detail: N passes of the
    same cases would bury the distribution that is the whole point. The
    per-run reports are still persisted individually (and carried whole in
    ``--format json``) for anyone who wants them.
    """
    typer.echo(
        f"Benchmark report ({aggregate.runs} runs) — "
        f"{aggregate.extractor_id} {aggregate.extractor_version}  "
        f"against  {aggregate.suite_id} {aggregate.suite_version}"
    )
    emitted = [r.particles_emitted for r in aggregate.reports]
    first = aggregate.reports[0]
    typer.echo(
        f"Cases: {first.cases_run}/{first.cases_total} per run     "
        f"Particles emitted: {min(emitted)}–{max(emitted)}     "
        f"Required expected: {first.particles_required_total}"
    )
    typer.echo(f"Judge: {aggregate.judge} @ ≥{aggregate.equivalence_threshold:.2f}")
    typer.echo("")
    typer.echo(f"{'METRIC':<22}  {'MEAN':>6}  {'SPREAD':>7}  {'MIN':>6}  {'MAX':>6}  {'STDEV':>6}")
    typer.echo("-" * 66)
    for name in ("recall", "precision", "calibration_error"):
        stat = aggregate.metric_stats.get(name)
        if stat is None:
            continue
        typer.echo(
            f"  {name:<20}  {stat.mean:>6.2f}  {stat.spread:>7.2f}  "
            f"{stat.minimum:>6.2f}  {stat.maximum:>6.2f}  {stat.stdev:>6.2f}"
        )
    typer.echo("")
    typer.echo("Per-run values")
    for name in ("recall", "precision", "calibration_error"):
        stat = aggregate.metric_stats.get(name)
        if stat is None:
            continue
        typer.echo(f"  {name:<20}  {'  '.join(f'{v:.2f}' for v in stat.values)}")
    typer.echo("")
    typer.echo("--fail-on is evaluated against the MEAN across runs.")
    notes = [n for r in aggregate.reports for n in r.quality_notes]
    if notes:
        typer.echo("")
        typer.echo(f"Notes ({len(notes)} across {aggregate.runs} runs, deduplicated):")
        for n in dict.fromkeys(notes):
            typer.echo(f"  {n}")
