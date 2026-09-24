# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""World and report models for the two-project observer fixture (gate B)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from particles.benchmark.rot.schema import Rate

GENERATOR_VERSION = "1"

PROJECT_A = "-home-me-src-alpha"
PROJECT_B = "-home-me-src-beta"
PROJECTS = (PROJECT_A, PROJECT_B)


class LineKind(StrEnum):
    """What class of memory line a claim is — the axis the report breaks down on."""

    FACT = "fact"
    """A value for a slot about a *shared, generic* subject ("the default branch is
    `main`"). Two projects may state different values: the supersession
    hazard."""
    RULE = "rule"
    """A slot-less line about a shared subject, byte-identical wherever it appears
    ("every commit needs a sign-off"). Stated by two projects it becomes ONE
    particle: the §5 generation-cascade hazard."""
    OWN = "own"
    """A fact about a subject only this project has. The control: it shares
    nothing, so it must never vanish."""


class MemoryLine(BaseModel):
    """One line of a project's ``MEMORY.md``, as the scripted extractor will state it."""

    text: str
    subject: str
    kind: LineKind
    slot: str | None = None
    value: str | None = None


class EventKind(StrEnum):
    SET = "set"
    """A slot takes a (new) value in one project."""
    ADD_RULE = "add_rule"
    DROP_RULE = "drop_rule"
    SET_OWN = "set_own"


class WorldEvent(BaseModel):
    day: int
    project: str
    kind: EventKind
    slot: str | None = None
    value: str | None = None
    rule: str | None = None


class ObserverWorld(BaseModel):
    """A seed *is* the fixture: two projects' memory files evolving over ``days``."""

    seed: int
    days: int
    generator_version: str = GENERATOR_VERSION
    events: list[WorldEvent] = Field(default_factory=list)
    global_lines: list[MemoryLine] = Field(default_factory=list)
    fingerprint: str = ""


class VanishCause(StrEnum):
    SUPERSEDED_BY_UPDATE = "superseded_by_update"
    """Rung 2.5: another project's value for the same subject won."""
    CASCADE = "cascade"
    """The generation cascade: the other project dropped a line both stated."""
    QUARANTINED = "quarantined"
    """§6.6 fail-closed: born ``CONFLICT_PENDING``."""
    ACTIVE_ELSEWHERE = "active_elsewhere"
    """The store holds the claim ACTIVE, but only the other project's sources attest
    the surviving particle: this project's own attestation was retired earlier
    (rung 2.5 ping-pong) and its later restatement has not yet been re-deposited
    to fold onto the survivor. The one cost the lens adds over the store-wide
    view — the claim is believed, this project's file states it, and the lens
    hides it."""
    OTHER = "other"
    NEVER_MINTED = "never_minted"
    """No particle holds the line at all."""


class LineObservation(BaseModel):
    """One (checkpoint, project, own line) probe."""

    day: int
    project: str
    kind: LineKind
    text: str
    visible: bool
    visible_store_wide: bool
    cause: VanishCause | None = None
    cross_project: bool | None = None
    """For a vanished line: did the retiring event come from the OTHER project?"""
    winner_in_view: bool | None = None
    """For a superseded line: is the winning claim in view for this project?"""


class ObserverMetrics(BaseModel):
    own_visible: Rate = Field(default_factory=Rate)
    """Own lines in view for their project, over own-line checkpoints."""
    own_visible_store_wide: Rate = Field(default_factory=Rate)
    """The same lines, ACTIVE anywhere in the store — the no-lens control."""
    vanished_cross_project: Rate = Field(default_factory=Rate)
    """Own lines retired by the other project's activity, over own-line checkpoints."""
    winner_in_view: Rate = Field(default_factory=Rate)
    """Over cross-project supersessions: the winner is in view (it never should be)."""
    leaked: Rate = Field(default_factory=Rate)
    """Lines in view for a project that it never stated and are not global — the
    lens's correctness, over in-view lines. Must be 0."""
    vanished_by_cause: dict[str, int] = Field(default_factory=dict)
    by_kind: dict[str, Rate] = Field(default_factory=dict)
    """``own_visible`` per :class:`LineKind`."""


class ObserverArm(StrEnum):
    """How the scripted extractor reaches the store."""

    LINES = "lines"
    """Every line re-emitted on every extraction: an unchanged claim reaches the
    store through duplicate suppression."""
    CHUNKED = "chunked"
    """Two lines per chunk through the chunk-hash carry-forward: an
    unchanged chunk's claims are carried forward, never re-emitted."""


class WriteCensus(BaseModel):
    """What the write path did, attributed at the moment it did it.

    A retirement is **cross-project** when the retired claim was, just before
    the extraction that retired it, observed by a project other than the one
    whose deposit was being extracted — read through the same scope join the
    lens uses. That is the exact failure the precondition exists to prevent.
    """

    own_supersessions: int = 0
    """``SUPERSEDED_BY_UPDATE`` retirements of a claim only the depositing project observed."""
    cross_project_supersessions: int = 0
    own_cascades: int = 0
    """Generation-cascade retirements of a claim only the depositing project observed."""
    cross_project_cascades: int = 0
    candidates_born_superseded: int = 0
    """New claims stored already ``SUPERSEDED_BY_UPDATE`` (rung 2.5's mirror)."""
    declined_pairs: int = 0
    """Confirmed contradictions the precondition declined to reconcile."""
    divergences_recorded: int = 0
    """``CONTRADICTS`` relations written by ``OBSERVER_DIVERGENCE``."""
    declined_without_relation: int = 0
    """Declined pairs with no recorded relation. Must be 0."""
    global_contests: Rate = Field(default_factory=Rate)
    """A project contesting a global line, over contests: an ``INCONSISTENCY`` was
    filed and the global claim is still ``ACTIVE``."""
    chunks_carried: int = 0
    chunks_extracted: int = 0

    def merged(self, other: WriteCensus) -> WriteCensus:
        counts = {
            name: getattr(self, name) + getattr(other, name)
            for name in type(self).model_fields
            if name != "global_contests"
        }
        return WriteCensus(
            **counts, global_contests=self.global_contests.merged(other.global_contests)
        )


class WorldResult(BaseModel):
    seed: int
    fingerprint: str
    deposits: int = 0
    store_census: dict[str, int] = Field(default_factory=dict)
    write_census: WriteCensus = Field(default_factory=WriteCensus)
    metrics: ObserverMetrics = Field(default_factory=ObserverMetrics)
    observations: list[LineObservation] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class ObserverReport(BaseModel):
    seeds: list[int]
    days: int
    arm: ObserverArm = ObserverArm.LINES
    generator_version: str = GENERATOR_VERSION
    store_mode: str
    thresholds: dict[str, float] = Field(default_factory=dict)
    started_at: str
    finished_at: str
    metrics: ObserverMetrics = Field(default_factory=ObserverMetrics)
    write_census: WriteCensus = Field(default_factory=WriteCensus)
    worlds: list[WorldResult] = Field(default_factory=list)
    refused_llm_calls: dict[str, int] = Field(default_factory=dict)
    quality_notes: list[str] = Field(default_factory=list)
