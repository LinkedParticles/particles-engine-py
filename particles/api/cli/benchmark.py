# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""`particles benchmark …` sub-Typer — whole-pipeline system benchmarks.

Deliberately **not** under ``particles extractor``: the system under test is
the pipeline (deposit → extract → reconcile → query), not an extractor, and
parking a system benchmark under the extractor noun would misdescribe what
the number means. The extractor-level ``benchmark`` / ``benchmark-modality`` /
``benchmark-polarity`` verbs stay where they are.

The verb is thin over :mod:`particles.benchmark.memory` — estimate/confirm
gating and rendering only; every measurement decision lives in the harness.
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import typer

from particles.api.cli import app, run

benchmark_app = typer.Typer(
    help="Whole-pipeline system benchmarks, distinct from the "
    "per-extractor `particles extractor benchmark*` verbs.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")

# ``benchmark memory`` is a group whose bare form runs the benchmark (the
# ``curate`` / ``rules`` shape: a callback with ``invoke_without_command``)
# so that ``benchmark memory rejudge`` can sit beside it as the verb that
# re-scores a saved report without re-answering.
memory_app = typer.Typer(
    help="The LongMemEval agent-memory benchmark. The bare verb runs "
    "it; `rejudge` re-scores a saved report under the current judge protocol.",
    invoke_without_command=True,
)
benchmark_app.add_typer(memory_app, name="memory")


class _Variant(StrEnum):
    oracle = "oracle"
    s = "s"
    m = "m"


class _Format(StrEnum):
    table = "table"
    json = "json"


class _Memory(StrEnum):
    particles = "particles"
    chunks = "chunks"
    notes = "notes"


@memory_app.callback(invoke_without_command=True)
def benchmark_memory_cmd(  # noqa: PLR0913 — CLI option list is the API
    ctx: typer.Context,
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Questions to run (stratified by type under the pinned seed; "
        "default: benchmark_memory.default_question_limit)",
    ),
    all_questions: bool = typer.Option(
        False, "--all", help="Run every question in the variant (mutually exclusive with --limit)"
    ),
    variant: _Variant | None = typer.Option(
        None,
        "--variant",
        help="LongMemEval variant: oracle | s | m (default: benchmark_memory.variant)",
    ),
    types: str | None = typer.Option(
        None,
        "--types",
        help="Comma-separated question-type filter (e.g. 'multi-session,knowledge-update')",
    ),
    estimate: bool = typer.Option(
        False,
        "--estimate",
        help="Print the projected LLM call count + token volume and exit: no LLM call.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-confirmation prompt."),
    output: Path | None = typer.Option(
        None, "--output", help="Write the rendered report to FILE as well as stdout."
    ),
    output_format: _Format = typer.Option(_Format.table, "--format", help="Output format"),
    store_dir: Path | None = typer.Option(
        None,
        "--store-dir",
        help="Directory for the per-question scratch stores (kept after the run; "
        "default: a deleted temp dir)",
    ),
    dataset_file: Path | None = typer.Option(
        None,
        "--dataset-file",
        help="Local LongMemEval-format JSON file (skips the pinned download; used "
        "by the checked-in fixture and pre-verified copies)",
    ),
    context_budget: int | None = typer.Option(
        None,
        "--context-budget",
        min=1,
        help="QA-at-budget clamp: cap condition ii's particle context "
        "at ~N tokens (rank order; baselines unclamped). Recorded on the run "
        "tuple; compare only against a matching run.",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Retrieval depth for condition i and the qa_particles context "
        "(default: benchmark_memory.top_k). Recorded on the run tuple; a "
        "top_k sweep is a sweep of this flag against one fixed store set.",
    ),
    qa: bool = typer.Option(
        True,
        "--qa/--no-qa",
        help="Run the end-to-end QA family (conditions ii-iv). --no-qa reports "
        "the retrieval stage alone and makes NO LLM call at all once the "
        "stores exist: the free tier for a retrieval-only ablation arm. The "
        "three QA rows then render `not run`.",
    ),
    consolidation: bool = typer.Option(
        False,
        "--consolidation",
        help="Ablation: run the dream cycle's pass list (reconcile, "
        "census, utility, abstraction) on each scratch store between extract "
        "and retrieve: the controlled instrument. LLM-priced "
        "(reconcile probes, contradiction probes); recorded on the run tuple. "
        "Mutually exclusive with --abstraction, which the cycle runs itself.",
    ),
    dedup_judge: bool = typer.Option(
        False,
        "--dedup-judge",
        help="Ablation: run the co-evidential LLM judge in APPLY mode on each "
        "scratch store before retrieval, linking PARAPHRASE pairs "
        "CO_EVIDENTIAL so the ranker collapses them inside top-k. "
        "LLM-priced (one judged cluster per Subject); recorded on the run tuple.",
    ),
    reuse_stores: bool = typer.Option(
        False,
        "--reuse-stores",
        help="Replay the scratch stores an earlier --store-dir run persisted "
        "instead of depositing and extracting again: zero write-time LLM "
        "calls. Requires --store-dir, and refuses unless that set's "
        "write-side tuple (dataset, selection, extraction + embedding model, "
        "write-time reconciliation knobs) matches this run's.",
    ),
    abstraction: bool = typer.Option(
        False,
        "--abstraction",
        help="Ablation: run the abstraction-promotion pass (auto mode, "
        "age gate 0) on each scratch store between extract and retrieve. "
        "Recorded on the run tuple.",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="Run up to N questions at once (each owns its scratch store; the "
        "report is identical to a sequential run's). Practical ceiling is "
        "your API rate tier; past ~4-8 the extra parallelism becomes 429 "
        "retries, not speed.",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard this experiment's checkpoint and start over. Runs are "
        "checkpointed per completed question by default, so an interrupted "
        "run resumes (and a completed run replays free) when re-invoked "
        "with identical knobs.",
    ),
    pooled: bool = typer.Option(
        False,
        "--pooled",
        help="Dispatch each question's haystack extractions as one pooled "
        "Message Batches job. This roughly halves the bill on a "
        "batch-eligible provider at the cost of latency (a batch's floor is "
        "one poll interval). Same model, prompt, and budget, so the report "
        "is comparable to an unpooled run's; degrades to sequential calls "
        "when llm.batch is off.",
    ),
    batch_qa: bool = typer.Option(
        False,
        "--batch-qa",
        help="Submit the QA answer + judge calls (conditions ii-iv) as Message "
        "Batches jobs, one answer batch and one judge batch per "
        "condition, all at 50% price, instead of one sequential call per "
        "question. The sibling of --pooled for the answerer/judge (the two "
        "compose); same model/prompt/budget, so the report is comparable. Off "
        "by default (a batch's floor is one poll interval: the right trade for "
        "a paid run, the wrong one for a small/interactive run); degrades to "
        "sequential calls when llm.batch is off.",
    ),
    memory: _Memory = typer.Option(
        _Memory.particles,
        "--memory",
        help="The memory under test: particles (the store; the default), or a "
        "COMPARATOR memory over the same questions, answer scaffold, judge "
        "and retrieval scoring: chunks (raw-transcript RAG, no write-time "
        "LLM call) or notes (LLM-written session notes by the extraction "
        "model). The report's selection.memory names which ran.",
    ),
    baselines: bool = typer.Option(
        True,
        "--baselines/--no-baselines",
        help="Run the qa_full_context / qa_no_memory baseline conditions. "
        "--no-baselines is for a comparator run reusing the particles run's "
        "baseline columns (same tuple ⇒ same calls); they render `not run`.",
    ),
) -> None:
    """Run the LongMemEval agent-memory benchmark.

    Reports four conditions in two labeled families: retrieval-stage
    Recall@k / Precision@k (provenance-scored), and end-to-end QA accuracy
    for qa_particles, qa_full_context (baseline), and qa_no_memory
    (baseline) under one pinned answer model.
    """
    if ctx.invoked_subcommand is not None:
        # ``benchmark memory rejudge …`` — the subcommand owns the invocation;
        # the group's own options are not consulted.
        return
    if limit is not None and all_questions:
        typer.echo("--limit and --all are mutually exclusive.", err=True)
        raise typer.Exit(2)
    if consolidation and abstraction:
        # The cycle's own pass list already contains the abstraction pass
        # (gated by consolidation.abstraction), so accepting both would run it
        # twice and leave the tuple claiming two independent knobs where the
        # arm has one. Refuse rather than silently pick a winner.
        typer.echo(
            "--consolidation and --abstraction are mutually exclusive: the "
            "consolidation cycle runs the abstraction pass itself (configure it "
            "under consolidation.abstraction).",
            err=True,
        )
        raise typer.Exit(2)
    if reuse_stores and store_dir is None:
        typer.echo("--reuse-stores requires --store-dir naming the persisted set.", err=True)
        raise typer.Exit(2)
    try:
        run(
            _benchmark_memory(
                limit=limit,
                all_questions=all_questions,
                variant=variant,
                types=types,
                estimate_only=estimate,
                yes=yes,
                output=output,
                output_format=output_format,
                store_dir=store_dir,
                dataset_file=dataset_file,
                context_budget=context_budget,
                abstraction=abstraction,
                top_k=top_k,
                qa=qa,
                consolidation=consolidation,
                dedup_judge=dedup_judge,
                reuse_stores=reuse_stores,
                concurrency=concurrency,
                fresh=fresh,
                pooled=pooled,
                batch_qa=batch_qa,
                memory=memory.value,
                baselines=baselines,
            )
        )
    except KeyboardInterrupt:
        # One ^C must be enough. Cancellation has already run (the pipeline's
        # IN_PROGRESS→PENDING resets, the checkpoint appends for completed
        # questions); what remains at interpreter shutdown is joining
        # non-daemon worker threads (the embedding to_thread pool), which can
        # block indefinitely and forced repeated ^C. Nothing needs flushing —
        # exit hard, skipping atexit.
        typer.echo(
            "\nInterrupted — completed questions are checkpointed; re-run the "
            "same command to resume.",
            err=True,
        )
        import os

        os._exit(130)


async def _benchmark_memory(  # noqa: PLR0913 — mirrors the CLI options
    *,
    limit: int | None,
    all_questions: bool,
    variant: _Variant | None,
    types: str | None,
    estimate_only: bool,
    yes: bool,
    output: Path | None,
    output_format: _Format,
    store_dir: Path | None,
    dataset_file: Path | None,
    context_budget: int | None,
    abstraction: bool,
    top_k: int | None,
    qa: bool,
    consolidation: bool,
    dedup_judge: bool,
    reuse_stores: bool,
    concurrency: int,
    fresh: bool,
    pooled: bool,
    batch_qa: bool,
    memory: str = "particles",
    baselines: bool = True,
) -> None:
    from particles.benchmark.memory import (
        ContextWindowExceeded,
        MemoryDatasetLoadError,
        check_context_window,
        ensure_dataset,
        estimate_run,
        load_dataset_file,
        render_context_window_check,
        render_estimate,
        render_report_table,
        run_memory_benchmark,
        select_questions,
    )
    from particles.benchmark.memory.runner import prepared_question_ids
    from particles.config import get_config

    cfg = get_config().benchmark_memory
    effective_variant = variant.value if variant is not None else cfg.variant
    selected_types = [t.strip() for t in types.split(",") if t.strip()] if types else []
    effective_limit: int | None
    if all_questions:
        effective_limit = None
    else:
        effective_limit = limit if limit is not None else cfg.default_question_limit

    # Acquire + parse the dataset (no LLM cost yet).
    try:
        path = dataset_file if dataset_file is not None else await ensure_dataset(effective_variant)
        all_parsed = load_dataset_file(path)
    except MemoryDatasetLoadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    # A replay draws from the questions the kept set was prepared for: the
    # stratified sampler does not nest, so a narrower --limit drawn from the
    # whole variant lands outside the set and is refused. At the preparing
    # run's own --limit this selects exactly what it selected.
    questions_total = len(all_parsed)
    if reuse_stores and store_dir is not None:
        prepared = prepared_question_ids(store_dir)
        if prepared is not None:
            all_parsed = [q for q in all_parsed if q.question_id in prepared]

    questions = select_questions(
        all_parsed,
        seed=cfg.sample_seed,
        limit=effective_limit,
        types=selected_types or None,
    )
    if not questions:
        typer.echo("No questions matched the selection (check --types / --variant).", err=True)
        raise typer.Exit(1)

    # Estimate ALWAYS printed before any LLM call.
    cost = estimate_run(
        questions,
        qa=qa,
        memory=memory,
        baselines=baselines,
        reuse_stores=reuse_stores,
        consolidation=consolidation,
        dedup_judge=dedup_judge,
        pooled=pooled,
        batch_qa=batch_qa,
    )
    typer.echo(render_estimate(cost))

    # ...and so is the context-window verdict, so `--estimate` is a complete
    # dry run: an operator trying a bigger variant learns it will not fit here,
    # before the confirm gate, rather than from the runner's refusal after the
    # dataset download. The runner re-checks and refuses regardless — this is
    # disclosure, not the enforcement.
    window = check_context_window(questions, variant=effective_variant, qa=qa, baselines=baselines)
    typer.echo(render_context_window_check(window))
    if estimate_only:
        typer.echo("--estimate: nothing was run.")
        return

    threshold = cfg.confirm_call_threshold
    # An unbounded component must gate too. Without this a --dedup-judge arm
    # over reused stores projects "~0 LLM calls" — every *boundable* component
    # really is zero — and sails past a threshold meant to stop exactly this
    # kind of spend.
    if (cost.estimated_llm_calls > threshold or cost.unbounded_components) and not yes:
        if not sys.stdin.isatty():
            typer.echo(
                f"Estimated LLM calls ({cost.estimated_llm_calls}) exceed "
                f"benchmark_memory.confirm_call_threshold ({threshold}) and no --yes "
                f"was given; aborting (non-interactive run).",
                err=True,
            )
            raise typer.Exit(1)
        dollars = (
            f" (~US${cost.estimated_cost_usd:,.2f})" if cost.estimated_cost_usd is not None else ""
        )
        if not typer.confirm(f"Proceed with ~{cost.estimated_llm_calls} LLM calls{dollars}?"):
            typer.echo("Aborted.")
            raise typer.Exit(1)

    # QA + extraction are LLM-priced; refuse up front when the key is missing
    # (mirrors the audit's §7 no-key refusal) rather than failing mid-run.
    _refuse_without_key()

    # Per-question progress on stderr (stdout stays clean for --format json).
    # A run at inaugural scale is hours of sequential LLM calls; silence
    # between the confirmation and the final table is not acceptable UX.
    concurrency_phrase = f"up to {concurrency} at a time" if concurrency > 1 else "one at a time"
    _progress_line(
        f"Running {len(questions)} question(s), {concurrency_phrase}: a status "
        f"heartbeat every 30s, plus one line per completed question. The run "
        f"checkpoints per question and resumes if interrupted."
    )

    try:
        report = await run_memory_benchmark(
            questions,
            variant=effective_variant,
            top_k=top_k,
            dataset_revision=cfg.dataset_revision,
            selection_seed=cfg.sample_seed,
            selection_limit=effective_limit,
            selection_types=selected_types,
            questions_total=questions_total,
            work_dir=store_dir,
            keep_stores=store_dir is not None,
            context_budget=context_budget,
            abstraction=abstraction,
            qa=qa,
            consolidation=consolidation,
            dedup_judge=dedup_judge,
            reuse_stores=reuse_stores,
            progress=_progress_line,
            concurrency=concurrency,
            checkpoint_dir=_checkpoint_dir(),
            fresh=fresh,
            heartbeat_seconds=30,
            pooled=pooled,
            batch_qa=batch_qa,
            memory=memory,
            baselines=baselines,
        )
    except ContextWindowExceeded as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output_format is _Format.json:
        rendered = report.model_dump_json(indent=2)
    else:
        rendered = render_report_table(report)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"))
        typer.echo(f"Report written to {output}")


@memory_app.command("rejudge")
def benchmark_memory_rejudge_cmd(
    report: Path = typer.Argument(
        ...,
        help="A saved `benchmark memory --format json` report whose QA rows carry "
        "the answering model's replies (written by v1.141.1 or later).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Write the re-judged report here as JSON: a complete report of "
        "record (retrieval stage copied, QA conditions re-scored, provenance "
        "in the first quality note), regardless of --format.",
    ),
    output_format: _Format = typer.Option(
        _Format.table, "--format", help="What to print on stdout: the table, or the JSON."
    ),
    dataset_file: Path | None = typer.Option(
        None,
        "--dataset-file",
        help="Local LongMemEval-format JSON file for the report's variant "
        "(default: the pinned download for the variant and revision the "
        "report records). The judge prompt needs each question's text and "
        "reference answer, which the report does not carry.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost-confirmation prompt."),
) -> None:
    """Re-score a saved report's stored answers under the current judge.

    Re-runs only the judge call (`llm.benchmark`, under the configured
    `benchmark_memory.judge_protocol`) over each QA row's recorded answer;
    no answer call is made, so the full-context baseline is not re-paid and
    the answers being judged do not change. Rows with no stored answer stay
    excluded, with the count disclosed. The output names the source report
    and both judge tuples in its first quality note.
    """
    run(
        _rejudge_report(
            report=report,
            output=output,
            output_format=output_format,
            dataset_file=dataset_file,
            yes=yes,
        )
    )


async def _rejudge_report(
    *,
    report: Path,
    output: Path,
    output_format: _Format,
    dataset_file: Path | None,
    yes: bool,
) -> None:
    from pydantic import ValidationError

    from particles.benchmark.memory import (
        MemoryBenchmarkReport,
        MemoryDatasetLoadError,
        RejudgeError,
        ensure_dataset,
        load_dataset_file,
        rejudge_report,
        render_report_table,
        stored_answer_count,
    )
    from particles.config import get_config

    try:
        source = MemoryBenchmarkReport.model_validate_json(report.read_text())
    except FileNotFoundError as exc:
        typer.echo(f"Error: report not found: {report}", err=True)
        raise typer.Exit(1) from exc
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Error: {report} is not a memory-benchmark JSON report: {exc}", err=True)
        raise typer.Exit(1) from exc

    calls = stored_answer_count(source)
    if calls == 0:
        typer.echo(
            f"Error: {report} holds no stored answers to re-judge (no QA condition "
            f"ran, or benchmark.record_claim_text was off).",
            err=True,
        )
        raise typer.Exit(1)

    cfg = get_config().benchmark_memory
    try:
        path = (
            dataset_file
            if dataset_file is not None
            else await ensure_dataset(
                source.selection.variant, revision=source.selection.dataset_revision
            )
        )
        questions = load_dataset_file(path)
    except MemoryDatasetLoadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"Re-judge: {calls} judge call(s) over stored answers, no answer call "
        f"(source judged under protocol {source.selection.judge_protocol} by "
        f"{source.selection.judge_model_id or 'not recorded'}; re-judging under "
        f"protocol {cfg.judge_protocol})."
    )
    threshold = cfg.confirm_call_threshold
    if calls > threshold and not yes:
        if not sys.stdin.isatty():
            typer.echo(
                f"Estimated LLM calls ({calls}) exceed "
                f"benchmark_memory.confirm_call_threshold ({threshold}) and no --yes "
                f"was given; aborting (non-interactive run).",
                err=True,
            )
            raise typer.Exit(1)
        if not typer.confirm(f"Proceed with ~{calls} LLM calls?"):
            typer.echo("Aborted.")
            raise typer.Exit(1)

    _refuse_without_key(purposes=("benchmark",))

    try:
        rejudged = await rejudge_report(
            source, questions, source_path=str(report), progress=_progress_line
        )
    except RejudgeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    as_json = rejudged.model_dump_json(indent=2)
    typer.echo(as_json if output_format is _Format.json else render_report_table(rejudged))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(as_json + "\n")
    typer.echo(f"Re-judged report written to {output}")


def _progress_line(line: str) -> None:
    """One wall-clock-stamped progress line on stderr.

    The timestamp removes the "is this line fresh or stale?" ambiguity on a
    multi-hour run — an "elapsed 42m" heartbeat alone still makes the
    operator do arithmetic against their clock. stderr keeps stdout clean
    for --format json / --output.
    """
    typer.echo(f"[{datetime.now():%H:%M:%S}] {line}", err=True)


def _checkpoint_dir() -> Path:
    """Interrupted-run checkpoints live beside the dataset cache."""
    from particles.benchmark.memory.loader import default_cache_dir

    return default_cache_dir() / "checkpoints"


def _refuse_without_key(
    purposes: tuple[str, ...] = ("extraction", "benchmark", "benchmark_answer"),
) -> None:
    """Exit 1 when a hosted purpose the run needs has no API key.

    ``purposes`` names the LLM purposes the invocation will call: the full
    run needs all three, a re-judge only the judge.
    """
    from particles.config import get_config
    from particles.secrets import get_anthropic_api_key_optional

    llm_cfg = get_config().llm
    needs_anthropic = any(
        llm_cfg.for_purpose(purpose).provider == "anthropic" for purpose in purposes
    )
    if needs_anthropic and not get_anthropic_api_key_optional():
        typer.echo(
            "ANTHROPIC_API_KEY is not set — the benchmark's LLM calls "
            f"({', '.join(purposes)}) need it. Set it and re-run:\n"
            "  export ANTHROPIC_API_KEY=sk-...",
            err=True,
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# `particles benchmark rot` — the memory-rot benchmark
# ---------------------------------------------------------------------------


class _RotArm(StrEnum):
    oracle = "oracle"
    probe = "probe"
    live = "live"


class _ObserverArm(StrEnum):
    lines = "lines"
    chunked = "chunked"


# ``benchmark rot`` is a group whose bare form runs the benchmark, so that
# ``benchmark rot rescore`` can sit beside it (the ``benchmark memory`` /
# ``rejudge`` shape).
@benchmark_app.command("observer")
def benchmark_observer_cmd(
    seed: list[int] | None = typer.Option(
        None, "--seed", help="World seed; repeat for several worlds (default: 1–8)."
    ),
    days: int = typer.Option(14, "--days", min=2, help="Simulated days per world."),
    output: Path | None = typer.Option(None, "--output", help="Write the report here."),
    output_format: str = typer.Option(
        "markdown", "--format", help='Output format: "markdown" (default) or "json".'
    ),
    store_dir: Path | None = typer.Option(
        None, "--store-dir", help="Keep the per-seed scratch stores under this directory."
    ),
    arm: _ObserverArm = typer.Option(
        _ObserverArm.lines,
        "--arm",
        help="How extraction reaches the store: `lines` re-emits every line (duplicate "
        "suppression); `chunked` sends two lines per chunk through carry-forward.",
    ),
) -> None:
    """The two-project observer fixture, with zero LLM calls.

    Two repositories' memory files, sharing generic subjects, evolve over
    `--days` and are harvested into one scratch store through the real
    pipeline with scripted extraction and a scripted contradiction probe.
    Each day, every line a project currently states is checked through that
    project's observer: in view, or not, and if not, which mechanism retired
    it (cross-project supersession, the generation cascade, or a surviving
    particle attested only by the other project) and whether the winner is in
    view. Report-only; the only number the default flip is decided
    against.
    """
    if output_format not in ("markdown", "json"):
        typer.echo("Error: --format must be markdown or json.", err=True)
        raise typer.Exit(2)
    seeds = seed or list(range(1, 9))

    async def _go() -> None:
        import tempfile

        from particles.benchmark.observer import (
            ObserverArm,
            ObserverFixtureError,
            render_report,
            run_observer_fixture,
        )

        with tempfile.TemporaryDirectory(prefix="particles-observer-") as tmp:
            work = store_dir or Path(tmp)
            try:
                report = await run_observer_fixture(
                    seeds=seeds,
                    days=days,
                    work_dir=work,
                    arm=ObserverArm(arm.value),
                    keep_stores=store_dir is not None,
                    progress=_progress_line,
                )
            except ObserverFixtureError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(2) from exc
        text = (
            report.model_dump_json(indent=2) if output_format == "json" else render_report(report)
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            typer.echo(f"Report written to {output}.")
        else:
            typer.echo(text)

    run(_go())


rot_app = typer.Typer(
    help="The memory-rot benchmark. The bare verb runs it; `rescore` "
    "re-classifies a saved report under the current scorer.",
    invoke_without_command=True,
)
benchmark_app.add_typer(rot_app, name="rot")


@rot_app.callback(invoke_without_command=True)
def benchmark_rot_cmd(  # noqa: PLR0913 — CLI option list is the API
    ctx: typer.Context,
    arm: _RotArm = typer.Option(
        _RotArm.oracle,
        "--arm",
        help="Perception arm: oracle (scripted extraction + scripted §6.6 probe; "
        "zero LLM calls, deterministic), probe (scripted extraction, live "
        "contradiction probe), or live (the general extractor, i.e. the product).",
    ),
    seed: list[int] | None = typer.Option(
        None,
        "--seed",
        help="World seed; repeat for several worlds (default: benchmark_rot.seeds).",
    ),
    days: int | None = typer.Option(
        None, "--days", min=30, help="Simulated world length (default: benchmark_rot.days)."
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", min=1, help="Probe top-k (default: benchmark_rot.top_k)."
    ),
    trust_policy: bool = typer.Option(
        True,
        "--trust-policy/--no-trust-policy",
        help="Write the domain rule demoting the untrusted source "
        "channel (the operator's policy). --no-trust-policy measures the "
        "neutral-when-silent default instead.",
    ),
    estimate: bool = typer.Option(
        False, "--estimate", help="Print the projected LLM calls and cost, then exit."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation above the call threshold."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the rendered report to this path as well."
    ),
    output_format: _Format = typer.Option(
        _Format.table, "--format", help="table (default) or json (the report of record)."
    ),
    store_dir: Path | None = typer.Option(
        None,
        "--store-dir",
        help="Keep each world's scratch store (and its blobs) here for inspection.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Persist the paid arms' extraction results here, so a "
        "re-run that changes only candidacy or the ladder pays probes alone. The "
        "key includes the extractor, the resolved model and the SDK version, so a "
        "prompt or model change is a miss, never a silent replay.",
    ),
    attribute: str | None = typer.Option(
        None,
        "--attribute",
        help="Stamp this author id on every session, which is what a multi-store "
        "needs before the attribution rule lets an update supersede.",
    ),
) -> None:
    """Run the memory-rot benchmark.

    Deposits a deterministic changing world in time order and, at each
    checkpoint, probes every attribute through the query op's selection half.
    Reports currency (recall_current@k, current_first), supersession
    (stale_over_current, stale_retained@k), and poison leakage across three
    untrusted channels, each reported separately, with no aggregate score.
    """
    if ctx.invoked_subcommand is not None:
        # ``benchmark rot rescore …`` owns the invocation.
        return
    run(
        _benchmark_rot(
            arm=arm.value,
            seeds=list(seed) if seed else None,
            days=days,
            top_k=top_k,
            trust_policy=trust_policy,
            estimate_only=estimate,
            yes=yes,
            output=output,
            output_format=output_format,
            store_dir=store_dir,
            cache_dir=cache_dir,
            attribute=attribute,
        )
    )


async def _benchmark_rot(  # noqa: PLR0913 — mirrors the CLI options
    *,
    arm: str,
    seeds: list[int] | None,
    days: int | None,
    top_k: int | None,
    trust_policy: bool,
    estimate_only: bool,
    yes: bool,
    output: Path | None,
    output_format: _Format,
    store_dir: Path | None,
    cache_dir: Path | None = None,
    attribute: str | None = None,
) -> None:
    from particles.benchmark.rot import (
        RotArmError,
        estimate_rot_run,
        render_estimate,
        render_report,
        run_rot_benchmark,
    )
    from particles.config import get_config

    cfg = get_config().benchmark_rot
    effective_seeds = seeds or list(cfg.seeds)
    effective_days = days if days is not None else cfg.days
    checkpoints = [c for c in cfg.checkpoints if c <= effective_days]

    # Estimate ALWAYS printed before any LLM call.
    est = estimate_rot_run(arm, seeds=effective_seeds, days=effective_days, checkpoints=checkpoints)
    typer.echo(render_estimate(est), err=True)
    if estimate_only:
        typer.echo("--estimate: nothing was run.", err=True)
        return
    if est.llm_calls > cfg.confirm_call_threshold and not yes:
        if not sys.stdin.isatty():
            typer.echo(
                f"Estimated LLM calls ({est.llm_calls}) exceed "
                f"benchmark_rot.confirm_call_threshold ({cfg.confirm_call_threshold}) "
                f"and no --yes was given; aborting (non-interactive run).",
                err=True,
            )
            raise typer.Exit(1)
        dollars = f" (~US${est.cost_usd:,.2f})" if est.cost_usd is not None else ""
        if not typer.confirm(f"Proceed with ~{est.llm_calls} LLM calls{dollars}?"):
            typer.echo("Aborted.")
            raise typer.Exit(1)
    if arm == "live":
        _refuse_without_key(purposes=("extraction", "semantic_lint"))
    elif arm == "probe":
        _refuse_without_key(purposes=("semantic_lint",))

    try:
        report = await run_rot_benchmark(
            arm=arm,
            seeds=effective_seeds,
            days=effective_days,
            checkpoints=checkpoints,
            top_k=top_k,
            trust_policy=trust_policy,
            work_dir=store_dir,
            keep_stores=store_dir is not None,
            cache_dir=cache_dir,
            attribute=attribute,
            progress=_progress_line,
        )
    except RotArmError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    rendered = (
        report.model_dump_json(indent=2) if output_format is _Format.json else render_report(report)
    )
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"))
        typer.echo(f"Report written to {output}", err=True)


@rot_app.command("rescore")
def benchmark_rot_rescore_cmd(
    report: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="A saved rot report (--format json output)."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Where to write the re-scored report (JSON)."
    ),
    output_format: _Format = typer.Option(
        _Format.table, "--format", help="What to print: table (default) or json."
    ),
) -> None:
    """Re-classify a saved rot report under the current scorer.

    Free: retrieval is taken as recorded, with no store, encoder, or LLM call. The
    output is a complete report of record with ``selection.scorer_version`` set
    to what ran and a first note naming the source and both versions.
    """
    from pydantic import ValidationError

    from particles.benchmark.rot import (
        RescoreError,
        RotBenchmarkReport,
        render_report,
        rescore_report,
    )

    try:
        source = RotBenchmarkReport.model_validate_json(report.read_text())
    except ValidationError as exc:
        typer.echo(f"Error: {report} is not a rot benchmark report: {exc}", err=True)
        raise typer.Exit(1) from exc
    try:
        rescored = rescore_report(source, source=report.name)
    except RescoreError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    as_json = rescored.model_dump_json(indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(as_json + "\n")
    typer.echo(as_json if output_format is _Format.json else render_report(rescored))
    typer.echo(f"Re-scored report written to {output}", err=True)


# ---------------------------------------------------------------------------
# `benchmark relevance-floor` — the query gate's error rates
# ---------------------------------------------------------------------------

# A group whose bare form runs the benchmark, so that ``harvest`` — the step
# that builds its private input — can sit beside it (the ``benchmark rot`` /
# ``rescore`` shape).
floor_app = typer.Typer(
    help="The relevance-floor benchmark: how often the query gate refuses an "
    "answerable question, on real questions. `harvest` builds the private "
    "held-out set; the bare verb replays it (free) and, with --judge, scores it; "
    "`resweep` re-renders a saved report over any floor list.",
    invoke_without_command=True,
)
benchmark_app.add_typer(floor_app, name="relevance-floor")


def _refuse_inside_repository(path: Path, *, what: str, allow: bool) -> None:
    """Refuse to write private question text into a git work tree.

    The held-out set and the JSON report of record are real questions a real
    person asked. A path under a ``.git``-bearing directory is one ``git add``
    from publication, so writing there needs an explicit opt-in (for a
    gitignored path); the rendered table carries no question text and is never
    subject to this.
    """
    if allow:
        return
    resolved = path.expanduser().resolve()
    for directory in resolved.parents:
        if (directory / ".git").exists():
            typer.echo(
                f"Error: {what} would be written inside the git work tree at "
                f"{directory}. It holds real question text; write it outside any "
                f"repository, or pass --allow-in-repo for a gitignored path.",
                err=True,
            )
            raise typer.Exit(1)


@floor_app.command("harvest")
def benchmark_floor_harvest_cmd(
    transcripts: Path | None = typer.Option(
        None,
        "--transcripts",
        help="Directory of agent transcripts (*.jsonl), searched recursively "
        "(default: benchmark_relevance_floor.transcripts_dir).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Held-out JSONL to write (default: benchmark_relevance_floor.heldout_path).",
    ),
    prompts: bool = typer.Option(
        True,
        "--prompts/--no-prompts",
        help="Harvest question-shaped sentences the operator typed to the agent "
        "as well: a proxy source, reported apart. --no-prompts keeps explicit "
        "memory queries only.",
    ),
    allow_in_repo: bool = typer.Option(
        False, "--allow-in-repo", help="Permit an --output inside a git work tree."
    ),
) -> None:
    """Build the private held-out question set from agent transcripts.

    Pure parsing: no store, encoder, or LLM call. Secrets are redacted before a
    question is kept. Prints a census only, never a question.
    """
    from particles.api.cli._claude_code import redact_secrets
    from particles.benchmark.relevance_floor import (
        QuestionSource,
        harvest_transcripts,
        heldout_fingerprint,
        write_heldout,
    )
    from particles.config import get_config

    cfg = get_config().benchmark_relevance_floor
    root = (transcripts or Path(cfg.transcripts_dir)).expanduser()
    target = (output or Path(cfg.heldout_path)).expanduser()
    if not root.is_dir():
        typer.echo(f"Error: no transcript directory at {root}", err=True)
        raise typer.Exit(1)
    _refuse_inside_repository(target, what="the held-out set", allow=allow_in_repo)
    sources = (
        frozenset(QuestionSource)
        if prompts
        else frozenset({QuestionSource.MCP_QUERY, QuestionSource.CLI_QUERY})
    )
    harvested = harvest_transcripts(
        root,
        sources=sources,
        min_chars=cfg.min_question_chars,
        max_chars=cfg.max_question_chars,
        redact=redact_secrets,
    )
    written = write_heldout(target, harvested.questions)
    by_source = ", ".join(f"{k} {v}" for k, v in sorted(harvested.by_source.items())) or "none"
    typer.echo(f"Scanned {harvested.transcripts_scanned} transcript(s) under {root}")
    typer.echo(f"Held-out questions: {written}  [{by_source}]")
    typer.echo(
        f"Explicit query calls with a recorded result: {harvested.historical_results}; "
        f"of those, refused by the floor at the time: {harvested.historical_refusals}"
    )
    typer.echo(f"Fingerprint {heldout_fingerprint(harvested.questions)}; written to {target}")


@floor_app.command("resweep")
def benchmark_floor_resweep_cmd(
    report: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="A saved report (--format json output)."
    ),
    floor: list[float] | None = typer.Option(
        None,
        "--floor",
        help="A floor to evaluate; repeat for several (default: benchmark_relevance_floor.floors).",
    ),
) -> None:
    """Re-sweep a saved report over a floor list and print the aggregate table.

    Free: the cosines and labels are taken as recorded (no store, encoder, or
    LLM call). This is also how a private JSON report of record becomes the
    publishable table, which carries no question text.
    """
    from pydantic import ValidationError

    from particles.benchmark.relevance_floor import (
        RelevanceFloorReport,
        build_report,
        render_report,
    )
    from particles.config import get_config

    try:
        source = RelevanceFloorReport.model_validate_json(report.read_text())
    except ValidationError as exc:
        typer.echo(f"Error: {report} is not a relevance-floor report: {exc}", err=True)
        raise typer.Exit(1) from exc
    floors = sorted(set(floor)) if floor else get_config().benchmark_relevance_floor.floors
    if any(not 0.0 <= f <= 1.0 for f in floors):
        typer.echo("Error: every --floor must lie within [0, 1].", err=True)
        raise typer.Exit(1)
    selection = source.selection.model_copy(update={"floors": list(floors)})
    typer.echo(render_report(build_report(selection, source.results)))


@floor_app.callback(invoke_without_command=True)
def benchmark_floor_cmd(  # noqa: PLR0913 — CLI option list is the API
    ctx: typer.Context,
    heldout: Path | None = typer.Option(
        None,
        "--heldout",
        help="Held-out JSONL from `harvest` (default: benchmark_relevance_floor.heldout_path).",
    ),
    store: str | None = typer.Option(
        None, "--store", help="Store handle to replay against (default: the default store)."
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        max=200,
        help="Retrieval depth (default: benchmark_relevance_floor.top_k). The floor "
        "reads the maximum cosine over the rendered top-k, so this is on the run tuple.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Replay a seeded sample of N questions, stratified by source.",
    ),
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Run the LLM-priced stage too: answer every question with the gate "
        "disabled, then judge the answer grounded-and-useful. Estimate-gated.",
    ),
    estimate: bool = typer.Option(
        False,
        "--estimate",
        help="With --judge: run the free replay, print the projection, and exit "
        "before any LLM call.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation above the call threshold."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the rendered report to this path as well."
    ),
    output_format: _Format = typer.Option(
        _Format.table,
        "--format",
        help="table (default; aggregate-only, no question text) or json (the "
        "report of record; carries question and answer text).",
    ),
    replay_from: Path | None = typer.Option(
        None,
        "--replay-from",
        exists=True,
        dir_okay=False,
        help="Reuse the free replay recorded in a saved JSON report instead of "
        "re-running it (the replay is free but slow on a large store). Refused "
        "unless top_k and the encoder match and it covers every question asked for.",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Judged-stage checkpoint file (default: beside the held-out set), so "
        "an interrupted run never re-pays a finished question.",
    ),
    allow_in_repo: bool = typer.Option(
        False,
        "--allow-in-repo",
        help="Permit a --format json --output inside a git work tree.",
    ),
) -> None:
    """Measure the query relevance floor on a held-out set of real questions.

    The bare verb replays every question through the query op's selection half
    and reports the refusal curve: no LLM call. --judge adds the answerable /
    unanswerable labels and the swept 2×2 table. There is no aggregate score.
    """
    if ctx.invoked_subcommand is not None:
        # ``benchmark relevance-floor harvest`` / ``resweep`` owns the invocation.
        return
    run(
        _benchmark_floor(
            heldout=heldout,
            store=store,
            top_k=top_k,
            limit=limit,
            judge=judge,
            estimate_only=estimate,
            yes=yes,
            output=output,
            output_format=output_format,
            checkpoint=checkpoint,
            allow_in_repo=allow_in_repo,
            replay_from=replay_from,
        )
    )


async def _benchmark_floor(  # noqa: PLR0913 — mirrors the CLI options
    *,
    heldout: Path | None,
    store: str | None,
    top_k: int | None,
    limit: int | None,
    judge: bool,
    estimate_only: bool,
    yes: bool,
    output: Path | None,
    output_format: _Format,
    checkpoint: Path | None,
    allow_in_repo: bool,
    replay_from: Path | None = None,
) -> None:
    from pydantic import ValidationError

    from particles.benchmark.relevance_floor import (
        RelevanceFloorError,
        RelevanceFloorReport,
        build_report,
        build_selection,
        estimate_judged_stage,
        load_heldout,
        render_estimate,
        render_report,
        replay_retrieval,
        reuse_replay,
        run_judged,
        sample_questions,
    )
    from particles.config import get_config
    from particles.db import DEFAULT_STORE

    cfg = get_config().benchmark_relevance_floor
    heldout_path = (heldout or Path(cfg.heldout_path)).expanduser()
    if not heldout_path.is_file():
        typer.echo(
            f"Error: no held-out set at {heldout_path}. Build one with "
            f"`particles benchmark relevance-floor harvest`.",
            err=True,
        )
        raise typer.Exit(1)
    if estimate_only and not judge:
        typer.echo("Error: --estimate projects the --judge stage; pass both.", err=True)
        raise typer.Exit(1)
    if output is not None and output_format is _Format.json:
        _refuse_inside_repository(output, what="the JSON report", allow=allow_in_repo)

    questions = sample_questions(load_heldout(heldout_path), limit, cfg.sample_seed)
    if not questions:
        typer.echo(f"Error: {heldout_path} holds no questions.", err=True)
        raise typer.Exit(1)
    handle = store or DEFAULT_STORE
    depth = top_k if top_k is not None else cfg.top_k

    try:
        selection = await build_selection(
            questions,
            store=handle,
            top_k=depth,
            floors=cfg.floors,
            sample_limit=limit,
            judged=judge and not estimate_only,
        )
        # Stage 1 is free and always happens, fresh or reused: it is the refusal
        # curve, and it is what makes the projection below a measurement
        # instead of a guess.
        if replay_from is not None:
            try:
                saved = RelevanceFloorReport.model_validate_json(replay_from.read_text())
            except ValidationError as exc:
                raise RelevanceFloorError(
                    f"{replay_from} is not a relevance-floor report: {exc}"
                ) from exc
            results = reuse_replay(saved, questions, selection)
        else:
            results = await replay_retrieval(
                questions, store=handle, top_k=depth, progress=_progress_line
            )
        if judge:
            est = estimate_judged_stage(results)
            typer.echo(render_estimate(est), err=True)
            if estimate_only:
                typer.echo("--estimate: no LLM call was made.", err=True)
            else:
                if est.llm_calls > cfg.confirm_call_threshold and not yes:
                    if not sys.stdin.isatty():
                        typer.echo(
                            f"Estimated LLM calls ({est.llm_calls}) exceed "
                            f"benchmark_relevance_floor.confirm_call_threshold "
                            f"({cfg.confirm_call_threshold}) and no --yes was given; "
                            f"aborting (non-interactive run).",
                            err=True,
                        )
                        raise typer.Exit(1)
                    dollars = f" (~US${est.cost_usd:,.2f})" if est.cost_usd is not None else ""
                    if not typer.confirm(f"Proceed with ~{est.llm_calls} LLM calls{dollars}?"):
                        typer.echo("Aborted.")
                        raise typer.Exit(1)
                _refuse_without_key(purposes=("query_response", "benchmark"))
                scoreable = {r.question_id for r in results if r.max_similarity is not None}
                results = await run_judged(
                    [q for q in questions if q.question_id in scoreable],
                    store=handle,
                    selection=selection,
                    checkpoint_path=(
                        checkpoint or heldout_path.with_name("judged-checkpoint.jsonl")
                    ).expanduser(),
                    progress=_progress_line,
                ) + [r for r in results if r.max_similarity is None]
    except RelevanceFloorError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    report = build_report(selection, results)
    rendered = (
        report.model_dump_json(indent=2) if output_format is _Format.json else render_report(report)
    )
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"))
        typer.echo(f"Report written to {output}", err=True)
