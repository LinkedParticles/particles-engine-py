# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The two reports of record.

Neither is an extractor benchmark, so neither falls under techspec §13.3: both
are this SDK's own, non-normative, as ``RotBenchmarkReport`` is. Each carries
a ``report_version``; ``TripVsMoveReport`` also carries the ``checks_version``
it was scored under (:mod:`.checks`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from particles.llm.usage import LLMUsage

REPORT_VERSION = 1

Variant = Literal["move", "trip-alone"]
Arm = Literal["live", "scripted"]


class FixtureFile(BaseModel):
    """One fixture file a run read, by name and digest."""

    name: str
    sha256: str


class SavedRow(BaseModel):
    """One particle of a finished run's store: everything the checks read."""

    id: str
    content: str
    status: str
    reason: str | None = None
    #: The fixture note names of the particle's SOURCE refs.
    sources: list[str] = Field(default_factory=list)
    #: Whether the re-anchor pass wrote the claim.
    restatement: bool = False


class SavedEvent(BaseModel):
    """One checked operator event, with its full payload, in log order."""

    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class CaseRun(BaseModel):
    """One store: one variant, deposited, consolidated and checked."""

    variant: Variant
    index: int
    #: ``None`` when the run is incomplete or invalid: neither passed nor failed.
    passed: bool | None
    #: Each check's offending claim texts, and ``pass``.
    checks: dict[str, Any] = Field(default_factory=dict)
    #: Why the run stopped early (``budget``, a verb's exit), or ``None``.
    incomplete: str | None = None
    #: Why the run cannot count (``model leak: …``), or ``None``.
    invalid: str | None = None
    usage: LLMUsage | None = None
    rows: list[SavedRow] = Field(default_factory=list)
    events: list[SavedEvent] = Field(default_factory=list)
    #: Scripted arm only: calls no table entry answered, by purpose.
    refused_calls: dict[str, int] = Field(default_factory=dict)


class Verdict(BaseModel):
    """Runs passed per variant, and the bar when the protocol matches the record's."""

    move_passed: int
    move_runs: int
    trip_alone_passed: int
    trip_alone_runs: int
    #: ``None`` unless the run counts are the record's protocol (3 and 2).
    bar_met: bool | None


class Spend(BaseModel):
    """The estimate printed before the run, and what the runs measured."""

    #: ``profile``, ``record`` or ``bound``, or ``none``.
    estimate_source: str
    estimate_usd: float | None = None
    estimate_calls: int | None = None
    #: List price; ``None`` as soon as one usage row is unpriced.
    cost_usd: float | None = None
    priced_cost_usd: float = 0.0
    unpriced: list[str] = Field(default_factory=list)
    ceiling_usd: float | None = None


class TripVsMoveReport(BaseModel):
    """The trip-versus-move case end to end, one model configuration."""

    suite: Literal["trip-vs-move"] = "trip-vs-move"
    report_version: int = REPORT_VERSION
    checks_version: int
    sdk_version: str
    run_at: datetime
    #: ``provider:model`` of the one model, before any ``--pin``.
    model: str
    #: Every purpose's resolved ``provider:model``.
    purposes: dict[str, str]
    #: Whether ``--pin`` named a mixture.
    mixed: bool = False
    same_slot_admits: bool = True
    arm: Arm = "live"
    fixture: list[FixtureFile]
    runs: list[CaseRun]
    verdict: Verdict
    spend: Spend
    notes: list[str] = Field(default_factory=list)


class PairSample(BaseModel):
    """One sample of a pair: the contradiction verdict and the settled update verdict."""

    contradiction: bool | None = None
    slot: str | None = None


class PairRow(BaseModel):
    """One registered pair and its samples."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    row_class: str = Field(alias="class")
    expected: str
    older: str
    newer: str
    samples: list[PairSample] = Field(default_factory=list)


class SetResult(BaseModel):
    """One pair set: its rows, and every gate's tally from :mod:`.scoring`."""

    rows: list[PairRow]
    bars: dict[str, Any]


class UpdateGateReport(BaseModel):
    """The registered pair sets replayed through the production probes, one model."""

    suite: Literal["update-gate"] = "update-gate"
    report_version: int = REPORT_VERSION
    sdk_version: str
    run_at: datetime
    model: str
    samples_per_pair: int
    fixture: list[FixtureFile]
    sets: dict[str, SetResult]
    spend: Spend
    notes: list[str] = Field(default_factory=list)
