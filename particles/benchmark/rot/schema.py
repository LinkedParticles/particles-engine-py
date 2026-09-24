# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""World and report models for the memory-rot benchmark.

Two halves, deliberately separate:

* the **world** — :class:`RotWorld` and its events, sessions, and probes — is
  what :mod:`.generator` builds from a seed with no I/O. It is the fixture; a
  seed *is* the fixture, so nothing here is ever vendored;
* the **report** — :class:`RotBenchmarkReport` and its rows — is what
  :mod:`.runner` produces. Like the report it carries no aggregate
  "score": currency, supersession, and poison are separate families, and each
  rate carries its own numerator and denominator so an exclusion can never
  hide inside a percentage.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


class EventKind(StrEnum):
    """What one timeline event does to a slot."""

    #: The slot's first value.
    INITIAL = "initial"
    #: The slot's value changes (possibly back to an earlier value — a revert).
    UPDATE = "update"
    #: A later session mentions a superseded value in the past tense — the
    #: honest-history decoy RotBench's round-4 correction was about.
    HISTORY = "history"
    #: A value never true for the slot arrives through an untrusted channel.
    POISON = "poison"


class Phrasing(StrEnum):
    """How an update is stated, so the report can break results out by form."""

    #: "I moved from Boston to Denver" — names both values.
    TRANSITION = "transition"
    #: "I live in Denver now" — names only the new value.
    RESTATEMENT = "restatement"


class PoisonChannel(StrEnum):
    """How an untrusted value reaches the store."""

    #: The assistant turn relays tool output. Reaches extraction on every
    #: ingest path, the harvester included (it keeps assistant text).
    RELAY = "relay"
    #: A ``tool:`` speaker turn in the deposited transcript. The shipped
    #: harvester drops tool results, so on the wedge path this is
    #: structurally zero; the harness measures the raw-transcript path.
    TOOL_TURN = "tool_turn"
    #: A separate ``WEB_PAGE`` corpus entry on an untrusted domain.
    SOURCE = "source"


class SessionKind(StrEnum):
    """What one deposited session is."""

    CONVERSATION = "conversation"
    WEB_PAGE = "web_page"


class WorldEvent(BaseModel):
    """One change, decoy, or poison on one slot at one simulated day."""

    day: int
    slot: str
    kind: EventKind
    value: str
    #: The value this event replaced (UPDATE) — ``None`` otherwise.
    previous: str | None = None
    #: Set on UPDATE events only.
    phrasing: Phrasing | None = None
    #: Set on POISON events only.
    channel: PoisonChannel | None = None
    #: True when an UPDATE returns the slot to a value it held before.
    revert: bool = False


class Turn(BaseModel):
    """One speaker turn: ``role`` is ``user`` / ``assistant`` / ``tool``."""

    role: str
    content: str


class RotSession(BaseModel):
    """One deposit: a chat session, or the untrusted web page of a SOURCE poison."""

    session_id: str
    day: int
    #: Order within the day — deposit order is the timeline.
    seq: int
    kind: SessionKind
    uri: str
    turns: list[Turn] = Field(default_factory=list)
    #: Plain text of a WEB_PAGE session (``turns`` is empty then).
    page_text: str = ""
    #: Index into :attr:`RotWorld.events` of the event this session carries,
    #: ``None`` for a filler session.
    event_index: int | None = None
    #: Filler topic id — what the scripted extractor emits for filler.
    filler_claim: str | None = None


class Probe(BaseModel):
    """One question asked of the store at one checkpoint."""

    checkpoint: int
    #: The slot key; for a negative probe, the never-stated attribute key.
    slot: str
    question: str
    #: True for a probe on an attribute the world never states.
    negative: bool = False


class RotWorld(BaseModel):
    """A complete, deterministic changing world."""

    seed: int
    days: int
    generator_version: int
    checkpoints: list[int]
    #: slot key → the slot's value pool (its universe of scoreable values).
    pools: dict[str, list[str]]
    #: slot key → the slot's noun ("home city") used by claim templates.
    nouns: dict[str, str]
    #: Slots that never change (currency-only controls).
    control_slots: list[str]
    events: list[WorldEvent]
    sessions: list[RotSession]
    probes: list[Probe]
    #: SHA-256 over the canonical event log + session texts; pinned by the
    #: unit tests so a generator change cannot silently re-baseline reports.
    fingerprint: str = ""


class SlotState(BaseModel):
    """What is true of one slot as of one day — the scorer's ground truth."""

    slot: str
    current: str | None
    superseded: list[str]
    poison: list[str]
    poison_channels: list[PoisonChannel]
    #: Phrasing of the most recent UPDATE at or before the day, if any.
    last_phrasing: Phrasing | None = None
    changed: bool = False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class HitClass(StrEnum):
    """How one retrieved particle reads against one probed slot."""

    CURRENT = "current"
    MIXED = "mixed"
    STALE = "stale"
    HISTORY = "history"
    POISON_ASSERTED = "poison_asserted"
    POISON_ATTRIBUTED = "poison_attributed"
    #: Carries none of the slot's values — not value-bearing.
    NONE = "none"


#: The first-hit outcome when no top-k particle is value-bearing.
FIRST_HIT_MISS = "miss"


class Rate(BaseModel):
    """A rate with its denominator — ``value`` is ``None`` when nothing was eligible."""

    numerator: int = 0
    denominator: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value(self) -> float | None:
        """The rate, or ``None`` on an empty denominator (never a vacuous 0 or 1)."""
        return self.numerator / self.denominator if self.denominator else None

    def add(self, hit: bool) -> None:
        """Count one eligible probe."""
        self.denominator += 1
        if hit:
            self.numerator += 1

    def merged(self, other: Rate) -> Rate:
        """Pool two rates (sum numerators and denominators)."""
        return Rate(
            numerator=self.numerator + other.numerator,
            denominator=self.denominator + other.denominator,
        )


class HitRecord(BaseModel):
    """One top-k particle of one probe — the audit trail (v1.141.1 rule)."""

    rank: int
    particle_id: str
    #: Particle text; ``None`` when ``benchmark.record_claim_text`` is off.
    text: str | None = None
    hit_class: HitClass
    status: str
    cosine: float
    effective_confidence: float
    contested: bool = False


class ProbeResult(BaseModel):
    """One probe's outcome."""

    checkpoint: int
    slot: str
    question: str
    negative: bool = False
    #: Ground truth at the checkpoint (empty for a negative probe).
    current: str | None = None
    superseded: list[str] = Field(default_factory=list)
    poison: list[str] = Field(default_factory=list)
    poison_channels: list[PoisonChannel] = Field(default_factory=list)
    last_phrasing: Phrasing | None = None
    #: The first value-bearing hit's class, or ``"miss"``.
    first_hit: str = FIRST_HIT_MISS
    #: Whether that first value-bearing hit carried a contested badge.
    first_hit_contested: bool = False
    #: Top-1 raw cosine (the relevance-floor input); ``None`` when empty.
    top_cosine: float | None = None
    hits: list[HitRecord] = Field(default_factory=list)


class RotMetrics(BaseModel):
    """The three metric families over one set of probes."""

    # Currency — every non-negative probe is eligible.
    recall_current_at_k: Rate = Field(default_factory=Rate)
    current_first: Rate = Field(default_factory=Rate)
    # Supersession — probes on slots with a superseded value by the checkpoint.
    stale_over_current: Rate = Field(default_factory=Rate)
    stale_over_current_unflagged: Rate = Field(default_factory=Rate)
    stale_retained_at_k: Rate = Field(default_factory=Rate)
    # Poison — probes on slots with a poison value injected by the checkpoint.
    poison_leak_at_k: Rate = Field(default_factory=Rate)
    poison_first: Rate = Field(default_factory=Rate)
    poison_surfaced_at_k: Rate = Field(default_factory=Rate)
    #: First-hit class distribution over the currency probes; sums to the
    #: currency denominator so no outcome hides in an unreported remainder.
    first_hit_counts: dict[str, int] = Field(default_factory=dict)


class FloorSweepRow(BaseModel):
    """One relevance-floor setting of the offline floor sweep."""

    floor: float
    #: Probes whose current value was in top-k but whose top cosine fell below
    #: the floor — answerable, yet the product would refuse.
    answerable_refused: Rate = Field(default_factory=Rate)
    #: Negative probes whose top cosine cleared the floor — unanswerable, yet
    #: the product would pass them to the answer step.
    unanswerable_passed: Rate = Field(default_factory=Rate)


class WorldResult(BaseModel):
    """One seed's world, end to end."""

    seed: int
    fingerprint: str
    #: The materialised world epoch (day 0), ISO date.
    epoch: str
    sessions_deposited: int = 0
    #: Particle count by status at the end of the world.
    store_census: dict[str, int] = Field(default_factory=dict)
    metrics: RotMetrics = Field(default_factory=RotMetrics)
    by_checkpoint: dict[int, RotMetrics] = Field(default_factory=dict)
    by_phrasing: dict[str, RotMetrics] = Field(default_factory=dict)
    by_channel: dict[str, RotMetrics] = Field(default_factory=dict)
    probes: list[ProbeResult] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class RotRunSelection(BaseModel):
    """The run tuple — results are comparable only against the same tuple."""

    arm: str
    seeds: list[int]
    days: int
    checkpoints: list[int]
    top_k: int
    trust_policy: bool
    untrusted_domain_trust: float
    generator_version: int
    scorer_version: int
    extraction_model_id: str
    semantic_lint_model_id: str
    embedding_model_id: str
    thresholds: dict[str, float] = Field(default_factory=dict)


class RotBenchmarkReport(BaseModel):
    """The report of record for one run."""

    selection: RotRunSelection
    started_at: str
    finished_at: str
    #: Pooled over every world (numerators and denominators summed).
    metrics: RotMetrics = Field(default_factory=RotMetrics)
    by_checkpoint: dict[int, RotMetrics] = Field(default_factory=dict)
    by_phrasing: dict[str, RotMetrics] = Field(default_factory=dict)
    by_channel: dict[str, RotMetrics] = Field(default_factory=dict)
    floor_sweep: list[FloorSweepRow] = Field(default_factory=list)
    worlds: list[WorldResult] = Field(default_factory=list)
    #: LLM calls the scripted arms refused (a purpose they route to the
    #: refusing provider). Always disclosed; non-zero means a seam the arm did
    #: not anticipate — the call failed open rather than billing.
    refused_llm_calls: dict[str, int] = Field(default_factory=dict)
    quality_notes: list[str] = Field(default_factory=list)
