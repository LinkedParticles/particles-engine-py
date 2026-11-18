# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Report models for the rot benchmark's door arm.

The door arm drives any memory that implements a subset of the adebench
adapter contract through the rot world and scores the text its door hands the
model. These models are the report of record: every door text is kept (the
world is synthetic, so nothing in it is private), every rate carries its
denominator, and there is no aggregate score.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from particles.benchmark.rot.schema import FIRST_HIT_MISS, HitClass, Phrasing, PoisonChannel, Rate


class DoorSegment(BaseModel):
    """One value-bearing segment of a door text, at its character offset."""

    start: int
    text: str
    hit_class: HitClass


class DoorProbeResult(BaseModel):
    """One probe's door text and how it reads against the slot at the checkpoint."""

    checkpoint: int
    slot: str
    question: str
    negative: bool = False
    current: str | None = None
    superseded: list[str] = Field(default_factory=list)
    poison: list[str] = Field(default_factory=list)
    poison_channels: list[PoisonChannel] = Field(default_factory=list)
    last_phrasing: Phrasing | None = None
    #: The door scored (the memory's declared default).
    door: str = ""
    #: The text the client of that door literally received, uncut.
    door_text: str = ""
    #: What the adapter returned beside the text, kept for audit only; the
    #: scorer never reads it.
    raw: Any = None
    chars: int = 0
    door_ms: float = 0.0
    #: Set when the door call failed; the probe then leaves every denominator.
    error: str | None = None
    #: The first value-bearing segment's class, or ``"miss"``.
    first_class: str = FIRST_HIT_MISS
    first_offset: int | None = None
    #: On a negative probe: the door carried no value of any slot of the world.
    silent: bool | None = None
    segments: list[DoorSegment] = Field(default_factory=list)
    #: The reader's replies to this door, one per reader and reading.
    replies: list[ReplyResult] = Field(default_factory=list)


class ReplyResult(BaseModel):
    """One reader's reply to one reading of a probe's door (§4)."""

    #: ``headline`` or ``sensitivity`` (the free local reader).
    reader: str
    #: ``door`` (as delivered), ``hits_only`` or ``answer_only``.
    reading: str
    #: The size of the text handed to the reader.
    chars: int = 0
    #: The reply, recorded whole; ``None`` when no call was made.
    reply: str | None = None
    #: The verdict (``ReplyVerdict``), ``None`` when the reply was not scored.
    verdict: str | None = None
    #: On a ``poison`` verdict: ``asserted`` (rule 4) or ``adopted`` with its
    #: source (rule 8).
    poison_kind: str | None = None
    #: A ``current`` reply that also carries a poisoned value with its source.
    disclosed: bool = False
    #: Whether the handed text carried the current value (answerable probes).
    current_in_text: bool | None = None
    segments: list[DoorSegment] = Field(default_factory=list)
    #: Served from the reader cache; tokens and cost are the original call's.
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    #: The text did not fit the reader's context; no call was made.
    overflow: bool = False
    #: Why no reading was derived (a door that does not parse, no answer field).
    skipped: str | None = None
    #: The reader call failed; the reply leaves every denominator.
    error: str | None = None


class Stat(BaseModel):
    """A sample's size, median and 90th percentile; ``None`` on an empty sample."""

    n: int = 0
    median: float | None = None
    p90: float | None = None


class DoorMetrics(BaseModel):
    """The door arm's metric table over one set of probes."""

    # Currency: every non-negative probe whose door call succeeded.
    current_in_door: Rate = Field(default_factory=Rate)
    current_first: Rate = Field(default_factory=Rate)
    recall_current_at_2k: Rate = Field(default_factory=Rate)
    recall_current_at_8k: Rate = Field(default_factory=Rate)
    # Supersession: probes on slots with a superseded value by the checkpoint.
    stale_over_current: Rate = Field(default_factory=Rate)
    stale_in_door: Rate = Field(default_factory=Rate)
    # Source trust: probes on slots poisoned by the checkpoint.
    poison_first: Rate = Field(default_factory=Rate)
    poison_leak: Rate = Field(default_factory=Rate)
    poison_surfaced: Rate = Field(default_factory=Rate)
    # Abstention: negative probes.
    negative_silent: Rate = Field(default_factory=Rate)
    negative_chars: Stat = Field(default_factory=Stat)
    # The cost of the door and its speed.
    chars_per_door: Stat = Field(default_factory=Stat)
    door_ms: Stat = Field(default_factory=Stat)
    write_ms: Stat = Field(default_factory=Stat)
    #: First-segment class distribution over the currency probes.
    first_class_counts: dict[str, int] = Field(default_factory=dict)
    #: Door calls that failed (excluded from every denominator, disclosed).
    errors: int = 0


class AnswerMetrics(BaseModel):
    """The answer family over one reader and one reading.

    Rates over answerable probes count only replies that were scored; a reply
    that overflowed, failed or was skipped leaves every rate but
    ``reader_overflow`` and is counted below.
    """

    # Currency: answerable probes.
    answer_current: Rate = Field(default_factory=Rate)
    # Supersession: probes on slots with a superseded value by the checkpoint.
    answer_stale: Rate = Field(default_factory=Rate)
    answer_hedged: Rate = Field(default_factory=Rate)
    # Source trust: probes on slots poisoned by the checkpoint.
    answer_poison: Rate = Field(default_factory=Rate)
    answer_poison_asserted: Rate = Field(default_factory=Rate)
    answer_poison_adopted: Rate = Field(default_factory=Rate)
    answer_poison_disclosed: Rate = Field(default_factory=Rate)
    # Abstention: negative probes, and false abstention on answerable ones.
    answer_abstain: Rate = Field(default_factory=Rate)
    answer_false_abstain: Rate = Field(default_factory=Rate)
    #: Of the answerable probes whose handed text carried the current value,
    #: the share the reply did not answer with it.
    reader_missed: Rate = Field(default_factory=Rate)
    reader_overflow: Rate = Field(default_factory=Rate)
    reader_tokens_in: Stat = Field(default_factory=Stat)
    #: What the reading cost, cached replies at their original call's cost;
    #: ``None`` when any call was unpriced.
    reader_usd: float | None = 0.0
    #: Probe set (``currency``, ``supersession``, ``poison``, ``negative``) →
    #: verdict → count. Each set's counts sum to its scored replies.
    verdicts: dict[str, dict[str, int]] = Field(default_factory=dict)
    cached: int = 0
    errors: int = 0
    skipped: int = 0


class DoorBuild(BaseModel):
    """One memory fed one world once: a fresh store, written in time order."""

    seed: int
    build: int
    fingerprint: str
    profile: str
    #: The materialised world epoch (day 0), ISO date.
    epoch: str
    sessions_written: int = 0
    #: Which contract method wrote each session (``import_session`` /
    #: ``import_page`` / ``import_memory``), counted.
    write_methods: dict[str, int] = Field(default_factory=dict)
    write_ms_samples: list[float] = Field(default_factory=list)
    build_minutes: float = 0.0
    #: Spend on this build, and how it is known: ``measured`` (the memory
    #: reported its usage), ``estimated`` (the arms manifest's figure), or
    #: ``none`` (a memory with no model on its write path).
    usd: float | None = None
    usd_basis: str = "none"
    #: Served from the competitor cache rather than rebuilt.
    cached: bool = False
    #: Particles' extraction replayed from the candidate cache (read-side runs).
    extraction_replayed: bool = False
    probes: list[DoorProbeResult] = Field(default_factory=list)
    metrics: DoorMetrics = Field(default_factory=DoorMetrics)
    by_channel: dict[str, DoorMetrics] = Field(default_factory=dict)
    #: ``<reader>/<reading>`` → the answer family.
    answers: dict[str, AnswerMetrics] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class MetricRange(BaseModel):
    """One metric's mean and range over a memory's builds."""

    mean: float | None = None
    lo: float | None = None
    hi: float | None = None
    builds: int = 0


class DoorMemoryResult(BaseModel):
    """Every build of one memory, pooled, with its published configuration."""

    name: str
    #: ``headline``, ``variant`` (a labelled non-default row) or ``baseline``.
    role: str = "headline"
    #: Why a variant row is not the headline (``trust rule``, ``model parity``,
    #: ``patched adapter`` ...).
    variant_note: str | None = None
    adapter: str = ""
    version: str = ""
    model: str | None = None
    model_backed: bool = False
    door: str = ""
    #: Everything that is not the memory's own default, plus the env variable
    #: names it received (never secret values): the configuration published
    #: beside the row.
    configuration: dict[str, Any] = Field(default_factory=dict)
    #: The maintainer-review state printed beside the row.
    maintainer_review: str | None = None
    builds: list[DoorBuild] = Field(default_factory=list)
    metrics: DoorMetrics = Field(default_factory=DoorMetrics)
    ranges: dict[str, MetricRange] = Field(default_factory=dict)
    #: ``<reader>/<reading>`` → the answer family pooled over builds, and each
    #: answer metric's mean and range.
    answers: dict[str, AnswerMetrics] = Field(default_factory=dict)
    answer_ranges: dict[str, dict[str, MetricRange]] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    """One pre-registered prediction beside what the run observed."""

    metric: str
    predicted: str
    observed: str
    #: ``as predicted``, ``better than predicted`` or ``worse than predicted``;
    #: ``not observed`` when either side had nothing to compare.
    verdict: str


class PriorScoring(BaseModel):
    """A memory's headline figures under the versions a report was scored before."""

    scorer_version: int
    door_segmenter_version: int
    figures: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    #: The answer family's headline figures under the reply scorer then in force.
    reply_scorer_version: int | None = None
    answer_figures: dict[str, dict[str, float | None]] = Field(default_factory=dict)


class ReaderPin(BaseModel):
    """One reader as it was resolved for a read."""

    #: ``headline`` or ``sensitivity``.
    role: str
    provider: str
    model: str
    #: ``provider:model`` as the adapter reports it; a change mid-read refuses.
    provider_model: str
    #: The model digest or version the operator pinned (``ollama list``), if any.
    digest: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = 200
    #: The reasoning setting the provider entry sends, as configured.
    reasoning_effort: str | None = None
    #: The local reader's context, in tokens; a longer prompt is an overflow.
    context_tokens: int | None = None
    chars_per_token: float | None = None


class AnswerPin(BaseModel):
    """The answer stage's part of the run tuple (§6)."""

    readers: list[ReaderPin] = Field(default_factory=list)
    reader_prompt_version: int
    reader_prompt_sha256: str
    answer_strip_version: int
    strip_fields: list[str] = Field(default_factory=list)
    reply_scorer_version: int
    read_at: str = ""
    #: ``PARTICLES_BENCH_LOCK`` as the run lock exported it to the sensitivity read.
    bench_lock: str | None = None


class DoorRunSelection(BaseModel):
    """The run tuple: door reports are comparable only under the same one."""

    tier: str
    profile: str
    days: int = 90
    seeds: list[int]
    #: True only when the seeds came from a committed held-out set.
    held_out: bool = False
    seed_commitment: str | None = None
    builds_model_backed: int = 1
    builds_deterministic: int = 1
    generator_version: int
    scorer_version: int
    door_segmenter_version: int
    budgets_chars: list[int] = Field(default_factory=list)
    adebench_sha: str | None = None
    manifest_sha256: str | None = None
    #: The arms manifest, embedded whole.
    manifest: dict[str, Any] = Field(default_factory=dict)
    particles_version: str = ""
    #: The answer stage's pins, once the report has been read.
    answer: AnswerPin | None = None


class DoorReport(BaseModel):
    """The door arm's report of record."""

    selection: DoorRunSelection
    started_at: str
    finished_at: str
    #: The first lines of every report.
    disclosures: list[str] = Field(default_factory=list)
    #: metric → ``beat`` / ``trail`` / ``tie``, from the freeze record.
    predictions: dict[str, str] = Field(default_factory=dict)
    prediction_results: list[PredictionResult] = Field(default_factory=list)
    memories: list[DoorMemoryResult] = Field(default_factory=list)
    #: Headline figures under earlier scorer / segmenter versions, printed
    #: beside the current ones.
    prior_scorings: list[PriorScoring] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
