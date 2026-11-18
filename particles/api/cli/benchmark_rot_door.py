# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles benchmark rot door``: other memories through the rot world.

Registered on the ``benchmark rot`` group. ``run`` builds each memory and
scores its door, then has the pinned reader answer from it;
``read`` reads a saved report; ``rescore`` re-cuts a saved report and re-scores
its replies for free; ``reader-floor`` counts the reader's verdict flips;
``seeds`` and ``freeze`` make the held-out set and the record a milestone run
is pinned to; ``label-sample`` / ``label-score`` and ``label-replies`` /
``label-replies-score`` run the blind validations the freeze rests on. Every
paid build or read asks before its first call.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from particles.api.cli import run
from particles.api.cli._progress import confirm
from particles.api.cli.benchmark import _progress_line, _refuse_without_key, rot_app

if TYPE_CHECKING:
    from particles.benchmark.rot.door_schema import ReaderPin
    from particles.benchmark.rot.manifest import ReaderSpec

door_app = typer.Typer(
    help="The door arm: other memories through the rot world, scored on "
    "the text each memory's door hands the model.",
    no_args_is_help=True,
)
rot_app.add_typer(door_app, name="door")


class _Tier(StrEnum):
    quick = "quick"
    standard = "standard"
    milestone = "milestone"


def _refuse_inside_repository(path: Path, *, what: str, allow: bool) -> None:
    """Comparative results stay internal: never written into a work tree by default."""
    if allow:
        return
    for directory in path.expanduser().resolve().parents:
        if (directory / ".git").exists():
            typer.echo(
                f"Error: {what} would be written inside the git work tree at {directory}. "
                "Comparative results stay internal until the owner decides to publish "
                "; write outside any repository, or pass --allow-in-repo "
                "for a gitignored path.",
                err=True,
            )
            raise typer.Exit(1)


def _bench_lock() -> str | None:
    """The run lock's proof, as ``scripts/bench_lock.py`` exports it to the command."""
    return os.environ.get("PARTICLES_BENCH_LOCK")


def _readers(spec: ReaderSpec | None, *, sensitivity: bool) -> list[ReaderPin]:
    """The pinned readers, or exit 2 with the refusal."""
    from particles.benchmark.rot.answer_runner import ReaderRefused, resolve_readers

    try:
        return resolve_readers(spec, sensitivity=sensitivity, bench_lock=_bench_lock())
    except ReaderRefused as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _confirm_spend(paid_builds: int, paid_reads: int, *, yes: bool) -> None:
    """Ask before any paid build or read, unless --yes; refuse non-interactively."""
    if not (paid_builds or paid_reads) or yes:
        return
    parts = [f"{paid_builds} paid build(s)" if paid_builds else ""]
    parts.append(f"{paid_reads} paid reader call(s)" if paid_reads else "")
    what = " and ".join(p for p in parts if p)
    if not sys.stdin.isatty():
        typer.echo(f"{what} and no --yes was given; aborting (non-interactive run).", err=True)
        raise typer.Exit(1)
    if not confirm(f"Proceed with {what}?"):
        typer.echo("Aborted.")
        raise typer.Exit(1)


@door_app.command("run")
def door_run_cmd(  # noqa: PLR0913 — CLI option list is the API
    tier: _Tier = typer.Option(
        _Tier.quick,
        "--tier",
        help="quick (six-attribute world, one development seed, competitors cached), "
        "standard (full world, three development seeds) or milestone (held-out seeds, "
        "three builds per model-backed memory; needs --held-out and --freeze).",
    ),
    seed: list[int] | None = typer.Option(
        None, "--seed", help="Development seed; repeat for several (quick: 42; standard: 42 43 44)."
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        dir_okay=False,
        help="The arms manifest (default: benchmark_rot.door_manifest when it exists). "
        "Without one, Particles and the two baselines run alone.",
    ),
    memory: list[str] | None = typer.Option(
        None, "--memory", help="Run only these competitor memories from the manifest."
    ),
    baselines: bool = typer.Option(
        True, "--baselines/--no-baselines", help="Include the two zero-cost baselines."
    ),
    trust_row: bool | None = typer.Option(
        None,
        "--trust-row/--no-trust-row",
        help="Add a labelled Particles row with the trust rule on the untrusted domain "
        "(default: on at milestone).",
    ),
    parity: bool | None = typer.Option(
        None,
        "--parity/--no-parity",
        help="Add a labelled Particles row with every purpose on the manifest's parity model "
        "(default: on at milestone).",
    ),
    scripted: bool = typer.Option(
        False,
        "--particles-scripted",
        help="Run Particles with scripted perception instead of the product: a $0 "
        "plumbing check, labelled as not the product.",
    ),
    replay_extraction: Path | None = typer.Option(
        None,
        "--replay-extraction",
        file_okay=False,
        help="Replay Particles' extraction from this candidate cache (a read-side change "
        "only: ranking, the door's shape, the floor). The report says so.",
    ),
    cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Reuse cached competitor builds (keyed on pin and world)."
    ),
    held_out: Path | None = typer.Option(
        None, "--held-out", dir_okay=False, help="The private held-out seed file (milestone)."
    ),
    freeze: Path | None = typer.Option(
        None, "--freeze", dir_okay=False, help="The committed freeze record (milestone)."
    ),
    predictions: Path | None = typer.Option(
        None,
        "--predictions",
        dir_okay=False,
        help="A JSON file of metric → beat/trail/tie to print beside the results "
        "(milestone takes them from the freeze record).",
    ),
    read: bool | None = typer.Option(
        None,
        "--read/--no-read",
        help="Have the manifest's [reader] answer from every door after the builds "
        "(default: on when the manifest names a reader).",
    ),
    sensitivity: bool | None = typer.Option(
        None,
        "--sensitivity/--no-sensitivity",
        help="Read a second time with the free local [reader.sensitivity] model, under the "
        "benchmark host's run lock (default: on at the quick tier when the manifest names one).",
    ),
    estimate: bool = typer.Option(False, "--estimate", help="Print the estimate, then exit."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation before paid builds and reads."
    ),
    store_dir: Path | None = typer.Option(
        None, "--store-dir", file_okay=False, help="Keep each build's store here for inspection."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="The JSON report of record (default: under door_runs_dir)."
    ),
    allow_in_repo: bool = typer.Option(
        False, "--allow-in-repo", help="Permit --output inside a git work tree."
    ),
) -> None:
    """Build each memory on the rot world and score the text its door returns.

    When the manifest names a reader, the reader then answers from every door
    and the reply is scored. Prints the estimate first; a build or
    read that calls a model asks before it runs unless --yes. The Markdown
    table goes to stdout and the report of record to a JSON file outside any
    repository.
    """
    from particles.benchmark.rot.answer_runner import read_report, render_read_estimate
    from particles.benchmark.rot.door_render import render_door_report
    from particles.benchmark.rot.door_runner import (
        baseline_plans,
        competitor_plan,
        estimate_door_run,
        estimate_run_reads,
        particles_plan,
        render_door_estimate,
        run_door_benchmark,
    )
    from particles.benchmark.rot.freeze import (
        FreezeError,
        FreezeRecord,
        HeldOutSeeds,
        check_predictions,
        verify_against_freeze,
    )
    from particles.benchmark.rot.manifest import ManifestError, load_manifest
    from particles.config import get_config

    cfg = get_config().benchmark_rot
    milestone = tier is _Tier.milestone
    path = manifest
    if path is None:
        default = Path(cfg.door_manifest).expanduser()
        path = default if default.exists() else None
    try:
        arms = load_manifest(path)
    except ManifestError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc

    if read and arms.reader is None:
        typer.echo("Error: --read needs a [reader] table in the arms manifest.", err=True)
        raise typer.Exit(2)
    reading = arms.reader is not None if read is None else read
    named_sensitivity = arms.reader is not None and arms.reader.sensitivity is not None
    want_sensitivity = (
        (tier is _Tier.quick and named_sensitivity) if sensitivity is None else sensitivity
    )
    notes: list[str] = []
    preds: dict[str, str] = {}
    commitment: str | None = None
    if milestone:
        if held_out is None or freeze is None:
            typer.echo("Error: a milestone run needs --held-out and --freeze.", err=True)
            raise typer.Exit(2)
        if seed or scripted or replay_extraction is not None:
            typer.echo(
                "Error: a milestone run takes its seeds from --held-out and runs the product: "
                "no --seed, --particles-scripted or --replay-extraction.",
                err=True,
            )
            raise typer.Exit(2)
        try:
            seeds_file = HeldOutSeeds.model_validate_json(held_out.read_text())
            record = FreezeRecord.model_validate_json(freeze.read_text())
            notes += verify_against_freeze(seeds_file, record, manifest=arms)
        except (FreezeError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from exc
        seeds = seeds_file.seeds
        preds = record.predictions
        commitment = record.commitment
    else:
        if held_out is not None or freeze is not None:
            typer.echo("Error: held-out seeds are used only at the milestone tier.", err=True)
            raise typer.Exit(2)
        seeds = list(seed) if seed else ([42] if tier is _Tier.quick else list(cfg.seeds))
        if predictions is not None:
            preds = json.loads(predictions.read_text())
            try:
                check_predictions(preds, answer=reading)
            except FreezeError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(2) from exc

    want_trust = milestone if trust_row is None else trust_row
    want_parity = milestone if parity is None else parity
    p = arms.particles
    if want_parity and not (p.parity_provider and p.parity_model):
        if parity:
            typer.echo(
                "Error: --parity needs [particles] parity_provider and parity_model.", err=True
            )
            raise typer.Exit(2)
        if milestone:
            typer.echo("Error: a milestone run needs the manifest's parity model.", err=True)
            raise typer.Exit(2)
        want_parity = False

    plans = [
        particles_plan(
            role="variant" if scripted else "headline",
            variant_note="scripted perception, not the product" if scripted else None,
            scripted=scripted,
            cache_dir=replay_extraction,
            maintainer_review=p.maintainer_review,
        )
    ]
    if want_trust:
        plans.append(
            particles_plan(
                role="variant", variant_note="trust rule", scripted=scripted, trust_rule=True
            )
        )
    if want_parity and p.parity_provider and p.parity_model:
        plans.append(
            particles_plan(
                role="variant",
                variant_note="model parity",
                parity=(p.parity_provider, p.parity_model),
            )
        )
    chosen = set(memory or arms.memories)
    unknown = chosen - set(arms.memories)
    if unknown:
        typer.echo(f"Error: no such memory in the manifest: {', '.join(sorted(unknown))}", err=True)
        raise typer.Exit(2)
    plans += [competitor_plan(n, s, arms) for n, s in arms.memories.items() if n in chosen]
    if baselines:
        plans += baseline_plans()

    cache_dir = Path(cfg.door_cache_dir).expanduser() if cache else None
    reader_cache = Path(cfg.door_cache_dir).expanduser()
    readers = _readers(arms.reader, sensitivity=want_sensitivity) if reading else []
    est = estimate_door_run(plans, tier=tier.value, seeds=seeds, cache_dir=cache_dir)
    typer.echo(render_door_estimate(est), err=True)
    paid_reads = 0
    if readers:
        read_est = estimate_run_reads(
            plans,
            readers,
            tier=tier.value,
            seeds=seeds,
            cache_dir=cache_dir,
            reader_cache_dir=reader_cache,
        )
        typer.echo(render_read_estimate(read_est), err=True)
        paid_reads = read_est.paid_calls
    if estimate:
        typer.echo("--estimate: nothing was run.", err=True)
        return
    _confirm_spend(est.paid_builds, paid_reads, yes=yes)
    if not scripted:
        _refuse_without_key(purposes=("extraction", "semantic_lint", "query_response"))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = output or Path(cfg.door_runs_dir).expanduser() / f"door-{tier.value}-{stamp}.json"
    _refuse_inside_repository(target, what="the door report", allow=allow_in_repo)
    with tempfile.TemporaryDirectory(prefix="particles-rot-door-") as tmp:
        work = store_dir or Path(tmp)
        report = run_door_benchmark(
            plans,
            tier=tier.value,
            seeds=seeds,
            work_dir=work,
            manifest=arms,
            cache_dir=cache_dir,
            held_out=milestone,
            seed_commitment=commitment,
            predictions=preds,
            keep_stores=store_dir is not None,
            progress=_progress_line,
        )
    report.notes += notes
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if readers:
        # The door report is on disk before the first reader call, so a failed
        # read never costs the builds: `door read` can start again from it.
        report = run(
            read_report(
                report,
                readers,
                cache_dir=reader_cache,
                bench_lock=_bench_lock(),
                progress=_progress_line,
            )
        )
        target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(render_door_report(report))
    typer.echo(f"Report written to {target}", err=True)


@door_app.command("rescore")
def door_rescore_cmd(
    report: Path = typer.Argument(..., exists=True, dir_okay=False, help="A saved door report."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the re-scored report."),
    allow_in_repo: bool = typer.Option(
        False, "--allow-in-repo", help="Permit --output inside a git work tree."
    ),
) -> None:
    """Re-cut and re-classify a saved door report and its replies under the current versions; free.

    The headline figures under the versions it was scored with are kept and
    printed beside the new ones.
    """
    from particles.benchmark.rot.door import rescore_door_report
    from particles.benchmark.rot.door_render import render_door_report
    from particles.benchmark.rot.door_schema import DoorReport

    _refuse_inside_repository(output, what="the door report", allow=allow_in_repo)
    source = DoorReport.model_validate_json(report.read_text(encoding="utf-8"))
    rescored = rescore_door_report(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rescored.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(render_door_report(rescored))
    typer.echo(f"Re-scored report written to {output}", err=True)


@door_app.command("read")
def door_read_cmd(  # noqa: PLR0913 — CLI option list is the API
    report: Path = typer.Argument(..., exists=True, dir_okay=False, help="A saved door report."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the read report."),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        exists=True,
        dir_okay=False,
        help="The arms manifest naming the [reader] (default: benchmark_rot.door_manifest).",
    ),
    sensitivity: bool | None = typer.Option(
        None,
        "--sensitivity/--no-sensitivity",
        help="Read a second time with the free local [reader.sensitivity] model, under the "
        "run lock (default: on for a quick-tier report when the manifest names one).",
    ),
    cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Serve and keep replies in the reader cache. --no-cache bypasses it entirely: "
        "the second read of the reader's noise floor.",
    ),
    estimate: bool = typer.Option(False, "--estimate", help="Print the estimate, then exit."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation before paid reads."),
    allow_in_repo: bool = typer.Option(
        False, "--allow-in-repo", help="Permit --output inside a git work tree."
    ),
) -> None:
    """Have the pinned reader answer from every door of a saved report.

    No memory runs: the reader is handed each probe's question, the
    checkpoint's date and the recorded door text, and its reply is scored.
    Particles' headline row is also read without its composed answer and as
    the answer alone. Prints the estimate first and asks before paid calls.
    """
    from particles.benchmark.rot.answer_runner import (
        estimate_read,
        plan_reads,
        read_report,
        render_read_estimate,
    )
    from particles.benchmark.rot.door_render import render_door_report
    from particles.benchmark.rot.door_schema import DoorReport
    from particles.benchmark.rot.manifest import ManifestError, load_manifest
    from particles.config import get_config

    cfg = get_config().benchmark_rot
    _refuse_inside_repository(output, what="the door report", allow=allow_in_repo)
    path = manifest
    if path is None:
        default = Path(cfg.door_manifest).expanduser()
        path = default if default.exists() else None
    try:
        arms = load_manifest(path)
    except ManifestError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    source = DoorReport.model_validate_json(report.read_text(encoding="utf-8"))
    named_sensitivity = arms.reader is not None and arms.reader.sensitivity is not None
    want_sensitivity = (
        (source.selection.tier == "quick" and named_sensitivity)
        if sensitivity is None
        else sensitivity
    )
    readers = _readers(arms.reader, sensitivity=want_sensitivity)
    cache_dir = Path(cfg.door_cache_dir).expanduser()
    est = estimate_read(plan_reads(source, readers), cache_dir=cache_dir, use_cache=cache)
    typer.echo(render_read_estimate(est), err=True)
    if estimate:
        typer.echo("--estimate: nothing was run.", err=True)
        return
    _confirm_spend(0, est.paid_calls, yes=yes)
    read = run(
        read_report(
            source,
            readers,
            cache_dir=cache_dir,
            use_cache=cache,
            bench_lock=_bench_lock(),
            progress=_progress_line,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(read.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(render_door_report(read))
    typer.echo(f"Read report written to {output}", err=True)


@door_app.command("reader-floor")
def door_reader_floor_cmd(
    first: Path = typer.Argument(..., exists=True, dir_okay=False, help="A read report."),
    second: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="The same report read again with --no-cache."
    ),
) -> None:
    """Count the reader's verdict flips between two reads of one report.

    Run once per reader pin: the flips are the reader's floor, added to the
    door floor before a quick answer result reads as outside the noise.
    """
    from particles.benchmark.rot.answer_runner import reader_floor
    from particles.benchmark.rot.door_schema import DoorReport

    a = DoorReport.model_validate_json(first.read_text(encoding="utf-8"))
    b = DoorReport.model_validate_json(second.read_text(encoding="utf-8"))
    rows = reader_floor(a, b)
    if not rows:
        typer.echo("Error: the two reports share no scored reply.", err=True)
        raise typer.Exit(2)
    typer.echo("| row | reader/reading | compared | flips |")
    typer.echo("|---|---|---|---|")
    for r in rows:
        typer.echo(f"| {r.memory} | {r.key} | {r.compared} | {r.flips} |")
    compared = sum(r.compared for r in rows)
    flips = sum(r.flips for r in rows)
    typer.echo(f"\n{flips} flip(s) in {compared} paired replies.")


@door_app.command("seeds")
def door_seeds_cmd(
    output: Path = typer.Option(
        ..., "--output", "-o", help="The private held-out file (never in a repository)."
    ),
    count: int = typer.Option(3, "--count", min=1, help="How many held-out seeds."),
) -> None:
    """Generate a held-out seed set and print its commitment.

    The file stays private until the results are published; only the
    commitment goes into the freeze record.
    """
    from particles.benchmark.rot.freeze import new_held_out

    _refuse_inside_repository(output, what="the held-out seed file", allow=False)
    if output.exists():
        typer.echo(f"Error: {output} exists; a held-out set is never overwritten.", err=True)
        raise typer.Exit(1)
    held = new_held_out(count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(held.model_dump_json(indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    typer.echo(f"commitment: {held.commitment}")
    typer.echo(f"Held-out seeds written to {output}; keep it private.", err=True)


@door_app.command("freeze")
def door_freeze_cmd(
    held_out: Path = typer.Option(..., "--held-out", exists=True, dir_okay=False),
    predictions: Path = typer.Option(
        ..., "--predictions", exists=True, dir_okay=False, help="metric → beat/trail/tie (JSON)."
    ),
    validation: Path | None = typer.Option(
        None,
        "--validation",
        exists=True,
        dir_okay=False,
        help="The label-score result the freeze rests on (JSON).",
    ),
    reply_validation: Path | None = typer.Option(
        None,
        "--reply-validation",
        exists=True,
        dir_okay=False,
        help="The label-replies-score result the answer metrics rest on (JSON).",
    ),
    manifest: Path | None = typer.Option(None, "--manifest", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "-o", help="The freeze record to commit."),
) -> None:
    """Write the freeze record a milestone run is pinned to.

    It carries the seed commitment, every version the run depends on and the
    pre-registered predictions; with a [reader] in the manifest, also the
    answer stage's versions and a prediction for every answer metric.
    Refuses when a validation did not pass.
    """
    from particles.benchmark.rot.freeze import FreezeError, HeldOutSeeds, build_freeze_record
    from particles.benchmark.rot.manifest import ManifestError, load_manifest
    from particles.benchmark.rot.validation import ReplyValidationResult, ValidationResult
    from particles.config import get_config

    path = manifest
    if path is None:
        default = Path(get_config().benchmark_rot.door_manifest).expanduser()
        path = default if default.exists() else None
    validated = None
    if validation is not None:
        result = ValidationResult.model_validate_json(validation.read_text())
        if not result.passed:
            typer.echo("Error: the segmenter validation did not pass its bar.", err=True)
            raise typer.Exit(2)
        validated = result.model_dump(mode="json")
    replies = None
    if reply_validation is not None:
        reply_result = ReplyValidationResult.model_validate_json(reply_validation.read_text())
        if not reply_result.passed:
            typer.echo("Error: the reply validation did not pass its bar.", err=True)
            raise typer.Exit(2)
        replies = reply_result.model_dump(mode="json")
    try:
        arms = load_manifest(path)
        record = build_freeze_record(
            HeldOutSeeds.model_validate_json(held_out.read_text()),
            manifest=arms,
            predictions=json.loads(predictions.read_text()),
            segmenter_validation=validated,
            reply_validation=replies,
        )
    except (FreezeError, ManifestError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if validated is None:
        typer.echo("Warning: no segmenter validation attached.", err=True)
    if arms.reader is not None and replies is None:
        typer.echo("Warning: no reply validation attached.", err=True)
    typer.echo(f"Freeze record written to {output}; commit it before any held-out build.")


@door_app.command("label-sample")
def door_label_sample_cmd(
    report: Path = typer.Argument(..., exists=True, dir_okay=False, help="A door report."),
    items: Path = typer.Option(..., "--items", help="The labeller's JSONL (no memory names)."),
    key: Path = typer.Option(..., "--key", help="The key the labeller must not see (JSON)."),
    per_memory: int = typer.Option(150, "--per-memory", min=1),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Draw the blind, stratified segment sample for the segmenter validation."""
    from particles.benchmark.rot.door_schema import DoorReport
    from particles.benchmark.rot.validation import label_sample

    source = DoorReport.model_validate_json(report.read_text(encoding="utf-8"))
    sample, mapping = label_sample(source, per_memory=per_memory, seed=seed)
    items.parent.mkdir(parents=True, exist_ok=True)
    items.write_text("".join(i.model_dump_json() + "\n" for i in sample), encoding="utf-8")
    key.write_text(
        json.dumps({k: v.model_dump() for k, v in mapping.items()}, indent=2), encoding="utf-8"
    )
    typer.echo(f"{len(sample)} segment(s) to label in {items}; key in {key}.")


@door_app.command("label-score")
def door_label_score_cmd(
    items: Path = typer.Option(..., "--items", exists=True, help="The labelled JSONL."),
    key: Path = typer.Option(..., "--key", exists=True, help="The sample's key."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the result (JSON)."),
) -> None:
    """Score the labels against the scorer: 95 % agreement, 90 % STALE / HISTORY recall."""
    from particles.benchmark.rot.validation import LabelItem, LabelKey, score_labels

    labelled = [
        LabelItem.model_validate_json(line)
        for line in items.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping = {k: LabelKey.model_validate(v) for k, v in json.loads(key.read_text()).items()}
    result = score_labels(labelled, mapping)
    for m in result.memories:
        recall = ", ".join(
            f"{c} {('n/a' if r is None else f'{r:.1%}')}" for c, r in m.recall.items()
        )
        agreement = "n/a" if m.agreement is None else f"{m.agreement:.1%}"
        verdict = "pass" if m.passed else "FAIL"
        typer.echo(f"{m.memory}: {m.labelled} labelled, agreement {agreement}, {recall}: {verdict}")
    for note in result.notes:
        typer.echo(f"note: {note}")
    typer.echo("bar met" if result.passed else "bar NOT met")
    if output is not None:
        output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if not result.passed:
        raise typer.Exit(1)


@door_app.command("label-replies")
def door_label_replies_cmd(
    report: Path = typer.Argument(..., exists=True, dir_okay=False, help="A read door report."),
    items: Path = typer.Option(..., "--items", help="The labeller's JSONL (no memory names)."),
    key: Path = typer.Option(..., "--key", help="The key the labeller must not see (JSON)."),
    per_reader: int = typer.Option(100, "--per-reader", min=1),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Draw the blind, stratified reply sample for the reply scorer's validation."""
    from particles.benchmark.rot.door_schema import DoorReport
    from particles.benchmark.rot.validation import reply_label_sample

    source = DoorReport.model_validate_json(report.read_text(encoding="utf-8"))
    sample, mapping = reply_label_sample(source, per_reader=per_reader, seed=seed)
    items.parent.mkdir(parents=True, exist_ok=True)
    items.write_text("".join(i.model_dump_json() + "\n" for i in sample), encoding="utf-8")
    key.write_text(
        json.dumps({k: v.model_dump() for k, v in mapping.items()}, indent=2), encoding="utf-8"
    )
    typer.echo(f"{len(sample)} reply(ies) to label in {items}; key in {key}.")


@door_app.command("label-replies-score")
def door_label_replies_score_cmd(
    items: Path = typer.Option(..., "--items", exists=True, help="The labelled JSONL."),
    key: Path = typer.Option(..., "--key", exists=True, help="The sample's key."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the result (JSON)."),
) -> None:
    """Score reply labels: 95 % agreement, 90 % stale / hedged / poison recall, per reader."""
    from particles.benchmark.rot.validation import (
        ReplyLabelItem,
        ReplyLabelKey,
        score_reply_labels,
    )

    labelled = [
        ReplyLabelItem.model_validate_json(line)
        for line in items.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping = {k: ReplyLabelKey.model_validate(v) for k, v in json.loads(key.read_text()).items()}
    result = score_reply_labels(labelled, mapping)
    for r in result.readers:
        recall = ", ".join(
            f"{v} {('n/a' if x is None else f'{x:.1%}')}" for v, x in r.recall.items()
        )
        agreement = "n/a" if r.agreement is None else f"{r.agreement:.1%}"
        verdict = "pass" if r.passed else "FAIL"
        typer.echo(
            f"{r.reader}: {r.labelled} labelled from {r.memories} memories, agreement "
            f"{agreement}, {recall}: {verdict}"
        )
    for note in result.notes:
        typer.echo(f"note: {note}")
    typer.echo("bar met" if result.passed else "bar NOT met")
    if output is not None:
        output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if not result.passed:
        raise typer.Exit(1)
