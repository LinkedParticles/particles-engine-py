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
    help="Whole-pipeline system benchmarks — distinct from the "
    "per-extractor `particles extractor benchmark*` verbs.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


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


@benchmark_app.command("memory")
def benchmark_memory_cmd(  # noqa: PLR0913 — CLI option list is the API
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
        help="Print the projected LLM call count + token volume and exit — no LLM call.",
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
        help="Local LongMemEval-format JSON file (skips the pinned download — used "
        "by the checked-in fixture and pre-verified copies)",
    ),
    context_budget: int | None = typer.Option(
        None,
        "--context-budget",
        min=1,
        help="QA-at-budget clamp: cap condition ii's particle context "
        "at ~N tokens (rank order; baselines unclamped). Recorded on the run "
        "tuple — compare only against a matching run.",
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
        "your API rate tier — past ~4-8 the extra parallelism becomes 429 "
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
        "Message Batches job — roughly halves the bill on a "
        "batch-eligible provider at the cost of latency (a batch's floor is "
        "one poll interval). Same model, prompt, and budget, so the report "
        "is comparable to an unpooled run's; degrades to sequential calls "
        "when llm.batch is off.",
    ),
    batch_qa: bool = typer.Option(
        False,
        "--batch-qa",
        help="Submit the QA answer + judge calls (conditions ii-iv) as Message "
        "Batches jobs — one answer batch and one judge batch per "
        "condition, all at 50% price — instead of one sequential call per "
        "question. The sibling of --pooled for the answerer/judge (the two "
        "compose); same model/prompt/budget, so the report is comparable. Off "
        "by default (a batch's floor is one poll interval — the right trade for "
        "a paid run, the wrong one for a small/interactive run); degrades to "
        "sequential calls when llm.batch is off.",
    ),
    memory: _Memory = typer.Option(
        _Memory.particles,
        "--memory",
        help="The memory under test: particles (the store — default), or a "
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
    if limit is not None and all_questions:
        typer.echo("--limit and --all are mutually exclusive.", err=True)
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
    concurrency: int,
    fresh: bool,
    pooled: bool,
    batch_qa: bool,
    memory: str = "particles",
    baselines: bool = True,
) -> None:
    from particles.benchmark.memory import (
        MemoryDatasetLoadError,
        ensure_dataset,
        estimate_run,
        load_dataset_file,
        render_estimate,
        render_report_table,
        run_memory_benchmark,
        select_questions,
    )
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
    cost = estimate_run(questions, memory=memory, baselines=baselines)
    typer.echo(render_estimate(cost))
    if estimate_only:
        typer.echo("--estimate: nothing was run.")
        return

    threshold = cfg.confirm_call_threshold
    if cost.estimated_llm_calls > threshold and not yes:
        if not sys.stdin.isatty():
            typer.echo(
                f"Estimated LLM calls ({cost.estimated_llm_calls}) exceed "
                f"benchmark_memory.confirm_call_threshold ({threshold}) and no --yes "
                f"was given; aborting (non-interactive run).",
                err=True,
            )
            raise typer.Exit(1)
        if not typer.confirm(f"Proceed with ~{cost.estimated_llm_calls} LLM calls?"):
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

    report = await run_memory_benchmark(
        questions,
        variant=effective_variant,
        dataset_revision=cfg.dataset_revision,
        selection_seed=cfg.sample_seed,
        selection_limit=effective_limit,
        selection_types=selected_types,
        questions_total=len(all_parsed),
        work_dir=store_dir,
        keep_stores=store_dir is not None,
        context_budget=context_budget,
        abstraction=abstraction,
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

    if output_format is _Format.json:
        rendered = report.model_dump_json(indent=2)
    else:
        rendered = render_report_table(report)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"))
        typer.echo(f"Report written to {output}")


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


def _refuse_without_key() -> None:
    """Exit 1 when a hosted purpose the run needs has no API key."""
    from particles.config import get_config
    from particles.secrets import get_anthropic_api_key_optional

    llm_cfg = get_config().llm
    needs_anthropic = any(
        llm_cfg.for_purpose(purpose).provider == "anthropic"
        for purpose in ("extraction", "benchmark", "benchmark_answer")
    )
    if needs_anthropic and not get_anthropic_api_key_optional():
        typer.echo(
            "ANTHROPIC_API_KEY is not set — the benchmark's extraction, answer, "
            "and judge calls need it. Set it and re-run:\n"
            "  export ANTHROPIC_API_KEY=sk-...",
            err=True,
        )
        raise typer.Exit(1)
