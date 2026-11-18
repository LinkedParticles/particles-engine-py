# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The answer stage's calls, cache and estimate (§6, §7).

A read needs only a door report: no memory's code runs. Each probe's door is
handed to the pinned reader once per reading (every memory's ``door``;
Particles' headline row also ``hits_only`` and ``answer_only``), the reply is
recorded whole and scored by :mod:`.answer`. The reader is the manifest's
``[reader]`` table, installed as the ``benchmark_answer`` purpose through a
scoped provider override, so operator config is never changed; a read without
one refuses rather than fall back to ``llm.default``.

Replies are cached by content beside the builds: the key is everything the
reply is a function of, so rerunning a read is free, a competitor's door is
read once per reader pin, and a failed call is never cached.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from particles.benchmark.memory.runner import SameModelViolation
from particles.benchmark.rot.answer import (
    ANSWER_STRIP_VERSION,
    DIAGNOSTIC_READINGS,
    HEADLINE_READER,
    READER_PROMPT_VERSION,
    REPLY_SCORER_VERSION,
    SENSITIVITY_READER,
    derive_reading,
    finalize_answers,
    probe_date,
    prompt_sha256,
    reader_prompt,
    score_answers,
    strip_fields,
)
from particles.benchmark.rot.door import evaluate_predictions
from particles.benchmark.rot.door_schema import (
    AnswerPin,
    DoorBuild,
    DoorMemoryResult,
    DoorProbeResult,
    DoorReport,
    ReaderPin,
    ReplyResult,
)
from particles.benchmark.rot.generator import generate_world
from particles.benchmark.rot.manifest import ReaderSpec, SensitivityReaderSpec
from particles.benchmark.rot.schema import RotWorld
from particles.config import ProviderSelection, get_config
from particles.egress import current_policy
from particles.llm.registry import (
    CompletionProvider,
    complete_with_provider_model,
    override_providers,
    provider_for_selection,
    selection_endpoint,
)
from particles.llm.usage import track_usage
from particles.secrets import get_anthropic_api_key_optional

#: The purpose the reader is installed as (pinned answering model).
READER_PURPOSE = "benchmark_answer"


class ReaderRefused(RuntimeError):
    """A read that cannot start: no reader pinned, an endpoint the egress
    posture refuses, a reasoning setting that differs from what is sent, or a
    local reader without the benchmark host's run lock."""


# ---------------------------------------------------------------------------
# Resolving the readers
# ---------------------------------------------------------------------------


def _pin(spec: ReaderSpec | SensitivityReaderSpec, role: str) -> ReaderPin:
    selection = ProviderSelection(provider=spec.provider, model=spec.model)
    llm = get_config().llm
    if spec.provider == "anthropic":
        configured = None
    else:
        entry = llm.providers.get(spec.provider)
        if entry is None:
            raise ReaderRefused(
                f"[reader] provider {spec.provider!r} is neither 'anthropic' nor an "
                "llm.providers entry"
            )
        configured = entry.reasoning_effort
    if spec.reasoning_effort is not None and spec.reasoning_effort != configured:
        raise ReaderRefused(
            f"[reader] declares reasoning_effort {spec.reasoning_effort!r} but "
            f"llm.providers.{spec.provider}.reasoning_effort is {configured!r}, which is "
            "what the adapter sends; set the provider entry to match"
        )
    if spec.provider == "anthropic" and not get_anthropic_api_key_optional():
        raise ReaderRefused(f"the {role} reader needs ANTHROPIC_API_KEY, which is not set")
    policy = current_policy()
    if policy.refusing:
        host, port, name = selection_endpoint(selection)
        if not policy.allows(host, port):
            raise ReaderRefused(
                f"network.egress ({policy.posture}) does not allow the {role} reader "
                f"{name} at {host or '<unknown>'}{f':{port}' if port else ''}; nothing "
                "was asked"
            )
    provider = provider_for_selection(selection, context=f"the {role} reader")
    local = spec if isinstance(spec, SensitivityReaderSpec) else None
    return ReaderPin(
        role=role,
        provider=spec.provider,
        model=spec.model,
        provider_model=provider.provider_model,
        digest=spec.digest,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        reasoning_effort=configured,
        context_tokens=local.context_tokens if local else None,
        chars_per_token=local.chars_per_token if local else None,
    )


def resolve_readers(
    spec: ReaderSpec | None, *, sensitivity: bool, bench_lock: str | None
) -> list[ReaderPin]:
    """The pinned headline reader, then the sensitivity reader when asked for.

    Raises:
        ReaderRefused: Without a ``[reader]`` table (the instrument is never a
            fallback), when the sensitivity reader is asked for and the manifest
            names none or the run lock is not held, or on any check in
            :func:`_pin`.
    """
    if spec is None:
        raise ReaderRefused(
            "the arms manifest has no [reader] table; the answer stage never falls back "
            "to llm.default"
        )
    pins = [_pin(spec, HEADLINE_READER)]
    if sensitivity:
        if spec.sensitivity is None:
            raise ReaderRefused("--sensitivity needs a [reader.sensitivity] table")
        if not bench_lock:
            raise ReaderRefused(
                "the sensitivity reader runs only under the benchmark host's run lock: "
                'wrap the command in scripts/bench_lock.py --label "<task>" -- …'
            )
        pins.append(_pin(spec.sensitivity, SENSITIVITY_READER))
    return pins


def answer_pin(readers: list[ReaderPin], *, bench_lock: str | None) -> AnswerPin:
    """The answer stage's run tuple for a read under ``readers``."""
    return AnswerPin(
        readers=readers,
        reader_prompt_version=READER_PROMPT_VERSION,
        reader_prompt_sha256=prompt_sha256(),
        answer_strip_version=ANSWER_STRIP_VERSION,
        strip_fields=list(strip_fields()),
        reply_scorer_version=REPLY_SCORER_VERSION,
        read_at=datetime.now(UTC).isoformat(),
        bench_lock=bench_lock if any(r.role == SENSITIVITY_READER for r in readers) else None,
    )


# ---------------------------------------------------------------------------
# The plan: which readings of which doors
# ---------------------------------------------------------------------------


def readings_for(memory: DoorMemoryResult, role: str) -> tuple[str, ...]:
    """The readings a row gets: the diagnostic two on Particles' headline row only."""
    if role == HEADLINE_READER and memory.name == "particles" and memory.role == "headline":
        return ("door", *DIAGNOSTIC_READINGS)
    return ("door",)


@dataclass
class ReadJob:
    """One reader call to make (or serve from the cache)."""

    pin: ReaderPin
    reading: str
    probe: DoorProbeResult
    text: str | None
    why_not: str | None
    on: str

    @property
    def prompt(self) -> str:
        """The rendered prompt; empty for a reading that has no text."""
        if self.text is None:
            return ""
        return reader_prompt(self.probe.question, self.on, self.text)


def plan_reads(report: DoorReport, readers: list[ReaderPin]) -> list[ReadJob]:
    """Every call a read of ``report`` makes, in report order."""
    jobs: list[ReadJob] = []
    for memory in report.memories:
        for build in memory.builds:
            for probe in build.probes:
                if probe.error is not None:
                    continue
                for pin in readers:
                    for reading in readings_for(memory, pin.role):
                        text, why = derive_reading(probe.door_text, reading)
                        jobs.append(
                            ReadJob(
                                pin=pin,
                                reading=reading,
                                probe=probe,
                                text=text,
                                why_not=why,
                                on=probe_date(build, probe),
                            )
                        )
    return jobs


# ---------------------------------------------------------------------------
# The reader cache
# ---------------------------------------------------------------------------


class CachedReply(BaseModel):
    """One reply as cached: the reply and what the call that produced it cost."""

    reply: str
    provider_model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None


def reply_cache_key(job: ReadJob) -> str:
    """SHA-256 of everything a reply is a function of."""
    pin = job.pin
    payload = {
        "reader": pin.provider_model,
        "digest": pin.digest,
        "params": {
            "temperature": pin.temperature,
            "max_tokens": pin.max_tokens,
            "reasoning_effort": pin.reasoning_effort,
        },
        "prompt_version": READER_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "reading": job.reading,
        "strip_version": ANSWER_STRIP_VERSION,
        "text": job.text,
        "question": job.probe.question,
        "date": job.on,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "reader" / key[:2] / f"{key}.json"


def load_reply(cache_dir: Path | None, key: str) -> CachedReply | None:
    """A cached reply, or ``None``."""
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    return CachedReply.model_validate_json(path.read_text(encoding="utf-8"))


def store_reply(cache_dir: Path | None, key: str, cached: CachedReply) -> None:
    """Keep one successful reply (a failed call never reaches here)."""
    if cache_dir is None:
        return
    path = _cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(cached.model_dump_json(), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


class ReadEstimateRow(BaseModel):
    """One reader's projected calls and spend."""

    role: str
    provider_model: str
    calls: int = 0
    cached: int = 0
    overflow: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = 0.0
    basis: str = ""


class ReadEstimate(BaseModel):
    """What a read will call and roughly cost, before any call."""

    rows: list[ReadEstimateRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def paid_calls(self) -> int:
        """Uncached calls to a reader with a price (or no price known)."""
        return sum(r.calls - r.cached - r.overflow for r in self.rows if r.usd != 0.0)


def _overflows_chars(pin: ReaderPin, prompt_chars: int) -> bool:
    if pin.context_tokens is None or pin.chars_per_token is None:
        return False
    return prompt_chars / pin.chars_per_token + pin.max_tokens > pin.context_tokens


def _overflows(pin: ReaderPin, prompt: str) -> bool:
    return _overflows_chars(pin, len(prompt))


def _row(pin: ReaderPin) -> ReadEstimateRow:
    return ReadEstimateRow(role=pin.role, provider_model=pin.provider_model)


def _price_row(row: ReadEstimateRow, pin: ReaderPin) -> None:
    cfg = get_config()
    price = cfg.llm.price_for(ProviderSelection(provider=pin.provider, model=pin.model))
    if price is None:
        row.usd = None
        row.basis = "unpriced (no llm.price_per_mtok entry)"
        return
    row.usd = round(
        (row.input_tokens * price.input + row.output_tokens * price.output) / 1e6,
        4,
    )
    row.basis = "estimated from the door texts' size and llm.price_per_mtok"


@dataclass
class AssumedReads:
    """Readings of doors not built yet, priced at an assumed size (``door run``)."""

    pin: ReaderPin
    reading: str
    count: int


def _assumed_chars(reading: str) -> int:
    cfg = get_config().benchmark_rot
    if reading == "answer_only":
        return round(cfg.estimate_answer_output_tokens * cfg.estimate_reader_chars_per_token)
    return cfg.estimate_reader_door_chars


def estimate_read(
    jobs: list[ReadJob],
    *,
    cache_dir: Path | None,
    use_cache: bool = True,
    assumed: list[AssumedReads] | None = None,
) -> ReadEstimate:
    """Project the read's calls and spend; nothing is called.

    ``jobs`` are priced from the texts they hand over; ``assumed`` readings
    (doors a ``door run`` has not built yet) at
    ``benchmark_rot.estimate_reader_door_chars``.
    """
    cfg = get_config().benchmark_rot
    rows: dict[str, ReadEstimateRow] = {}
    pins: dict[str, ReaderPin] = {}

    def add(pin: ReaderPin, prompt_chars: int) -> None:
        row = rows[pin.role]
        if _overflows_chars(pin, prompt_chars):
            row.overflow += 1
            return
        row.input_tokens += round(prompt_chars / cfg.estimate_reader_chars_per_token)
        row.output_tokens += cfg.estimate_reader_output_tokens

    for job in jobs:
        rows.setdefault(job.pin.role, _row(job.pin))
        pins[job.pin.role] = job.pin
        if job.text is None:
            continue
        rows[job.pin.role].calls += 1
        if use_cache and load_reply(cache_dir, reply_cache_key(job)) is not None:
            rows[job.pin.role].cached += 1
            continue
        add(job.pin, len(job.prompt))
    overhead = len(reader_prompt("", "2026-01-01", ""))
    for a in assumed or []:
        rows.setdefault(a.pin.role, _row(a.pin))
        pins[a.pin.role] = a.pin
        for _ in range(a.count):
            rows[a.pin.role].calls += 1
            add(a.pin, overhead + _assumed_chars(a.reading))
    for role, row in rows.items():
        _price_row(row, pins[role])
    est = ReadEstimate(rows=list(rows.values()))
    est.notes.append(
        "Order-of-magnitude figures, not quotes: input tokens are the handed text at "
        "benchmark_rot.estimate_reader_chars_per_token, output "
        "benchmark_rot.estimate_reader_output_tokens per call"
        + (
            "; a door not built yet at benchmark_rot.estimate_reader_door_chars."
            if assumed
            else "."
        )
    )
    return est


def render_read_estimate(est: ReadEstimate) -> str:
    """The reader's estimate lines, printed apart from the build lines."""
    lines = ["Answer stage estimate:"]
    for r in est.rows:
        money = "unpriced" if r.usd is None else f"~US${r.usd:,.2f}"
        extra = f", {r.cached} cached" if r.cached else ""
        extra += f", {r.overflow} overflow" if r.overflow else ""
        lines.append(
            f"  {r.role} reader {r.provider_model}: {r.calls} call(s){extra}, "
            f"~{r.input_tokens:,} input / {r.output_tokens:,} output tokens, {money} ({r.basis})"
        )
    lines += [f"  note: {n}" for n in est.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


async def _call(job: ReadJob, provider: CompletionProvider) -> CachedReply:
    with override_providers({READER_PURPOSE: provider}), track_usage() as acc:  # type: ignore[dict-item]
        text, served = await complete_with_provider_model(
            READER_PURPOSE,  # type: ignore[arg-type]
            job.prompt,
            max_tokens=job.pin.max_tokens,
            temperature=job.pin.temperature,
        )
    usage = acc.snapshot()
    tokens_in = sum(
        r.input_tokens + r.cache_read_tokens + r.cache_creation_tokens for r in usage.rows
    )
    tokens_out = sum(r.output_tokens for r in usage.rows)
    return CachedReply(
        reply=text.strip(),
        provider_model=served,
        input_tokens=tokens_in if usage.rows else None,
        output_tokens=tokens_out if usage.rows else None,
        usd=usage.cost_usd if usage.rows else None,
    )


async def _read_one(
    job: ReadJob,
    provider: CompletionProvider,
    *,
    cache_dir: Path | None,
    use_cache: bool,
    gate: asyncio.Semaphore,
) -> ReplyResult:
    out = ReplyResult(reader=job.pin.role, reading=job.reading)
    if job.text is None:
        out.skipped = job.why_not or "no reading"
        return out
    out.chars = len(job.text)
    prompt = job.prompt
    if _overflows(job.pin, prompt):
        out.overflow = True
        return out
    key = reply_cache_key(job)
    cached = load_reply(cache_dir, key) if use_cache else None
    if cached is None:
        try:
            async with gate:
                cached = await _call(job, provider)
        except Exception as exc:  # noqa: BLE001 — a failed call is recorded, never cached
            out.error = f"{type(exc).__name__}: {exc}"
            return out
        if cached.provider_model != job.pin.provider_model:
            raise SameModelViolation(
                f"the {job.pin.role} reader resolved to {cached.provider_model} mid-read, "
                f"pinned {job.pin.provider_model}"
            )
        if use_cache:
            store_reply(cache_dir, key, cached)
    else:
        out.cached = True
    out.reply = cached.reply
    out.input_tokens = cached.input_tokens
    out.output_tokens = cached.output_tokens
    out.usd = cached.usd
    return out


def _worlds(report: DoorReport) -> dict[int, RotWorld]:
    s = report.selection
    seeds = {b.seed for m in report.memories for b in m.builds if b.probes}
    return {seed: generate_world(seed, days=s.days, profile=s.profile) for seed in seeds}


def rescore_answers(report: DoorReport, worlds: dict[int, RotWorld] | None = None) -> None:
    """Score every recorded reply and rebuild the answer families, in place; free."""
    worlds = worlds if worlds is not None else _worlds(report)
    strip = report.selection.answer.answer_strip_version if report.selection.answer else None
    for memory in report.memories:
        for build in memory.builds:
            if build.probes:
                score_answers(
                    build,
                    worlds[build.seed],
                    strip_version=strip or ANSWER_STRIP_VERSION,
                    scorer_version=report.selection.scorer_version,
                    segmenter_version=report.selection.door_segmenter_version,
                )
        finalize_answers(memory)
    report.prediction_results = evaluate_predictions(report.predictions, report.memories)


async def read_report(  # noqa: PLR0913 — the run tuple is the API
    report: DoorReport,
    readers: list[ReaderPin],
    *,
    cache_dir: Path | None,
    use_cache: bool = True,
    bench_lock: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> DoorReport:
    """Read every door of ``report`` with ``readers`` and score the replies.

    Returns a copy: the replies of each reader role read here replace any the
    report carried for that role, and the answer pins join the run tuple.
    With ``use_cache`` false the reader cache is neither read nor written (the
    second read of the noise floor).

    Raises:
        SameModelViolation: When a reader's resolved id changes mid-read.
    """
    out = report.model_copy(deep=True)
    roles = {pin.role for pin in readers}
    for memory in out.memories:
        for build in memory.builds:
            for probe in build.probes:
                probe.replies = [r for r in probe.replies if r.reader not in roles]
    jobs = plan_reads(out, readers)
    providers = {
        pin.role: provider_for_selection(
            ProviderSelection(provider=pin.provider, model=pin.model),
            context=f"the {pin.role} reader",
        )
        for pin in readers
    }
    gate = asyncio.Semaphore(get_config().benchmark_rot.reader_concurrency)
    done = 0

    async def run(job: ReadJob) -> tuple[ReadJob, ReplyResult]:
        nonlocal done
        reply = await _read_one(
            job, providers[job.pin.role], cache_dir=cache_dir, use_cache=use_cache, gate=gate
        )
        done += 1
        if progress is not None and done % 25 == 0:
            progress(f"answer stage: {done} of {len(jobs)} readings")
        return job, reply

    results = await asyncio.gather(*(run(job) for job in jobs))
    for job, reply in results:
        job.probe.replies.append(reply)
    previous = out.selection.answer.readers if out.selection.answer else []
    kept = [r for r in previous if r.role not in roles]
    out.selection.answer = answer_pin([*readers, *kept], bench_lock=bench_lock)
    rescore_answers(out)
    return out


# ---------------------------------------------------------------------------
# The reader's noise floor
# ---------------------------------------------------------------------------


class FloorRow(BaseModel):
    """Verdict flips between two reads of the same doors, for one memory and reading."""

    memory: str
    key: str
    compared: int = 0
    flips: int = 0


def _replies(build: DoorBuild) -> dict[tuple[int, str, str, str], ReplyResult]:
    out: dict[tuple[int, str, str, str], ReplyResult] = {}
    for probe in build.probes:
        for r in probe.replies:
            out[(probe.checkpoint, probe.slot, r.reader, r.reading)] = r
    return out


def reader_floor(first: DoorReport, second: DoorReport) -> list[FloorRow]:
    """Count verdict flips between two reads of one report (the second uncached).

    Pairs replies by memory row, build, checkpoint, slot, reader and reading;
    a reply with no verdict on either side is not compared.
    """
    rows: dict[tuple[str, str], FloorRow] = {}
    others = {(m.name, m.variant_note): m for m in second.memories}
    for memory in first.memories:
        twin = others.get((memory.name, memory.variant_note))
        if twin is None:
            continue
        twin_builds = {(b.seed, b.build): b for b in twin.builds}
        label = (
            memory.name if memory.variant_note is None else f"{memory.name} ({memory.variant_note})"
        )
        for build in memory.builds:
            other = twin_builds.get((build.seed, build.build))
            if other is None:
                continue
            theirs = _replies(other)
            for k, mine in _replies(build).items():
                peer = theirs.get(k)
                if peer is None or mine.verdict is None or peer.verdict is None:
                    continue
                row = rows.setdefault(
                    (label, f"{k[2]}/{k[3]}"), FloorRow(memory=label, key=f"{k[2]}/{k[3]}")
                )
                row.compared += 1
                row.flips += int(mine.verdict != peer.verdict)
    return list(rows.values())
