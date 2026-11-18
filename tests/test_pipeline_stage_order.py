# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The order of the extraction pipeline's stages, pinned before they are taken apart.

``ingest.pipeline._extract_claimed`` runs every reconciliation policy of one
snapshot in a fixed order, and much of that order is load-bearing: a stamp
that the pair search reads must land before the search, a phase that needs
every candidate's stored id must follow the write loop. The order is written
down only in the function's comments. The function is being split into
stages, one behaviour-preserving extraction at a time, and this
file is the check each of those extractions runs against.

Each named collaborator is wrapped wherever a ``particles`` module binds it, so
a stage moved into another module keeps its spy. ``ORDER`` lists the
constraints, each with the reason it holds. A ``barrier`` means every call of
the first stage precedes any call of the second; a ``lead`` means only the
first calls are ordered, for stages that interleave per candidate. An
extraction that renames a collaborator updates the table; one that reorders a
constrained pair fails here. ``TestHandoff`` checks what the pair search and
the embedder are handed, which holds whatever the stages are called.

What this file does not repeat: that no model, live authority or embedder is
called under the writer lock is ``tests/test_writer_lock_guard.py``.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import sys
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from particles.core.conflict_resolution import SlotVerdict
from particles.core.schema import RelationType, UncertaintyNature
from particles.core.stance import STANCE_HOLDER_KEY
from particles.corpus.rule_sources import RULE_SOURCE_TAG
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.scope import (
    SCOPE_ACTION_KEY,
    SCOPE_ACTION_SOURCE_EXEMPT,
    SCOPE_DOCUMENT_META,
    SCOPE_KEY,
)
from particles.ingest.second_reading import ProbeVerdict

SEED = "The podcast airs on Mondays."
CLAIM = "The podcast airs on Tuesdays."
META = "This page lists the schedule of the podcast."
STANCE = "Alice prefers the Tuesday slot for the podcast."
GATED_NAME = "config_loader"
AUTHOR = "alice"


@dataclass
class _Trace:
    """Every spied call in order, and what the hand-off stages were given."""

    calls: list[str] = field(default_factory=list)
    gathered: list[CandidateParticle] | None = None
    embedded: list[str] | None = None

    def first(self, label: str) -> int:
        return self.calls.index(label)

    def last(self, label: str) -> int:
        return len(self.calls) - 1 - self.calls[::-1].index(label)


Label = str | Callable[[tuple[Any, ...]], str]


def _spy(
    monkeypatch: pytest.MonkeyPatch,
    trace: _Trace,
    original: Any,
    label: Label,
    *,
    replace: Any = None,
    keep: Callable[[tuple[Any, ...]], None] | None = None,
) -> None:
    """Wrap ``original`` in every ``particles`` module that binds it."""
    target = replace if replace is not None else original

    def note(args: tuple[Any, ...]) -> None:
        trace.calls.append(label(args) if callable(label) else label)
        if keep is not None:
            keep(args)

    wrapper: Any
    if inspect.iscoroutinefunction(original):

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            note(args)
            return await target(*args, **kwargs)

    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            note(args)
            return target(*args, **kwargs)

    bound = 0
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("particles"):
            continue
        for attr, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, attr, wrapper)
                bound += 1
    assert bound, f"nothing binds {original!r}"


def _spy_lock(monkeypatch: pytest.MonkeyPatch, trace: _Trace) -> None:
    """Note the moment the writer lock is taken (``write_lock``)."""
    from particles import db

    original = db.write_lock

    @contextlib.asynccontextmanager
    async def write_lock(*args: Any, **kwargs: Any) -> AsyncGenerator[None, None]:
        async with original(*args, **kwargs):
            trace.calls.append("write_lock")
            yield

    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("particles"):
            continue
        for attr, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, attr, write_lock)


class _Constant:
    """Every text embeds to one vector, so every claim is every other's nearest."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _Scripted:
    """An extractor returning the candidates scripted for each deposited text."""

    EXTRACTOR_ID = "stage-order-extractor"
    EXTRACTOR_VERSION = "1"

    def __init__(self, trace: _Trace) -> None:
        self.script: dict[bytes, list[CandidateParticle]] = {}
        self.trace = trace

    def accepts(self, source_type: str) -> bool:
        return True

    async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> ExtractionResult:
        self.trace.calls.append("extract")
        return ExtractionResult(candidates=copy.deepcopy(self.script[content]))


def _candidate(content: str, *subjects: str, **extra: Any) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=list(subjects) or ["the podcast"],
        **extra,
    )


# (earlier stage, later stage, kind, why the order holds)
ORDER: list[tuple[str, str, str, str]] = [
    ("extract", "narrative_labels", "barrier", "the candidates exist"),
    (
        "narrative_labels",
        "embed",
        "barrier",
        "narrative fragments collapse before embedding",
    ),
    (
        "dedupe_exact_candidates",
        "embed",
        "barrier",
        "intra-pass duplicates fold before embedding",
    ),
    (
        "read_conflict_watermark",
        "load_active_candidates",
        "barrier",
        "AE2: the change marker is read before any candidate is loaded",
    ),
    (
        "is_scope_exempt_source",
        "gather_candidate_pairs",
        "barrier",
        "the rule-source exemption is stamped before the pair search",
    ),
    (
        "gate_candidate_subjects",
        "attach_speaker_subjects",
        "barrier",
        "the speaker is attached after the subject gate",
    ),
    (
        "gate_candidate_subjects",
        "gather_candidate_pairs",
        "barrier",
        "the subject search sees the gated subjects",
    ),
    (
        "attach_speaker_subjects",
        "gather_candidate_pairs",
        "barrier",
        "the subject search sees the speaker",
    ),
    (
        "gather_candidate_pairs",
        "load_generation_claims",
        "barrier",
        "the generation is read over the gathered pool",
    ),
    (
        "load_generation_claims",
        "match_restatements",
        "barrier",
        "restatements are matched against the generation",
    ),
    (
        "match_restatements",
        "probe_update_slots",
        "barrier",
        "a carried candidate takes no part in the probes",
    ),
    (
        "match_restatements",
        "probe_pairs",
        "barrier",
        "a carried candidate takes no part in the probes",
    ),
    (
        "probe_update_slots",
        "probe_pairs",
        "lead",
        "the update question is asked before the contradiction probe",
    ),
    ("probe_pairs", "select_pairs", "lead", "the selection reads the probes"),
    (
        "select_pairs",
        "read_confirmed_pairs",
        "barrier",
        "the second reading rereads the selected pairs",
    ),
    (
        "read_confirmed_pairs",
        "write_order",
        "barrier",
        "the write order follows the final selections",
    ),
    (
        "write_order",
        "prefetch_subjects",
        "barrier",
        "AD2: the live authorities are asked in write order",
    ),
    (
        "prefetch_subjects",
        "write_lock",
        "barrier",
        "AD2: the authorities are asked before the writer lock",
    ),
    (
        "build_duplicate_index",
        "route_particle",
        "barrier",
        "one suppression index, built before the write loop",
    ),
    (
        "build_retired_value_index",
        "route_particle",
        "barrier",
        "one retired-value index, built before the write loop",
    ),
    (
        "write_lock",
        "search_store_under_lock",
        "barrier",
        "AE2: the store is searched again under the lock",
    ),
    (
        "search_store_under_lock",
        "route_particle",
        "barrier",
        "AE2: every route decides on the search under the lock",
    ),
    (
        "route_particle",
        "record_divergences",
        "barrier",
        "a divergence is recorded once every claim has its id",
    ),
    (
        "route_particle",
        "plan_structure_edges",
        "barrier",
        "the edges bind stored ids",
    ),
    (
        "route_particle",
        "cascade_superseded_generation",
        "barrier",
        "the cascade excludes what the write loop suppressed into",
    ),
    (
        "cascade_superseded_generation",
        "status:COMPLETE",
        "barrier",
        "the cascade commits with the COMPLETE status",
    ),
    (
        "component_record",
        "status:COMPLETE",
        "barrier",
        "the components are stored with the COMPLETE status",
    ),
]


@pytest.fixture
async def traced(db_session: Any, monkeypatch: pytest.MonkeyPatch) -> _Trace:
    """Seed one claim, then trace the extraction of a second source about the same thing."""
    from particles import embeddings as ep
    from particles.corpus.deposit import deposit_text
    from particles.extraction import subject_gate
    from particles.ingest import (
        candidate_dedup,
        generation,
        pipeline,
        restatement,
        routing,
        second_reading,
        structure_edges,
        subject_resolver,
    )
    from particles.store import candidate_cache, extractor_store

    monkeypatch.setattr(ep, "_embedding_model", _Constant())
    trace = _Trace()
    extractor = _Scripted(trace)

    async def extract(text: str, candidates: list[CandidateParticle], **kwargs: Any) -> None:
        extractor.script[text.encode()] = candidates
        entry_id, snapshot_id = await deposit_text(
            db_session, text, source_type="LOCAL_MARKDOWN", **kwargs
        )
        await db_session.commit()
        await pipeline.extract_snapshot(db_session, entry_id, snapshot_id, extractor=extractor)
        await db_session.commit()

    async def judge(prompt: str, *, max_tokens: int, system: str | None = None) -> str:
        return json.dumps({"p0": "DISTINCT"})

    async def quiet_slot(earlier: str, later: str) -> Any:
        return restatement.UpdateGuard(verdict=SlotVerdict.DIFFERENT, any_changes=False)

    async def contradicts(a: str, b: str) -> bool:
        return True

    async def different(earlier: str, later: str) -> SlotVerdict:
        return SlotVerdict.DIFFERENT

    async def confirmed(*args: Any, **kwargs: Any) -> ProbeVerdict:
        return ProbeVerdict(usable=True, contradicts=True)

    for original, replace in (
        (pipeline._restatement_judge_call, judge),
        (pipeline._restatement_slot, quiet_slot),
        (pipeline._has_contradiction_signal, contradicts),
        (pipeline._has_update_signal, different),
        (second_reading.read_pair, confirmed),
    ):
        _spy(monkeypatch, trace, original, original.__name__.lstrip("_"), replace=replace)

    await extract(SEED, [_candidate(SEED)])

    def keep_gathered(args: tuple[Any, ...]) -> None:
        trace.gathered = copy.deepcopy(list(args[1]))

    def keep_embedded(args: tuple[Any, ...]) -> None:
        trace.embedded = list(args[0])

    def status(args: tuple[Any, ...]) -> str:
        return f"status:{args[2].name}"

    def start(args: tuple[Any, ...]) -> None:
        trace.calls.clear()

    spies: list[tuple[Any, Label, Callable[[tuple[Any, ...]], None] | None]] = [
        # The trace covers the claimed extraction only: the deposit and the
        # claim before it take the writer lock too.
        (pipeline._extract_claimed, "extract_claimed", start),
        (pipeline.plan_append, "plan_append", None),
        (pipeline.narrative_labels, "narrative_labels", None),
        (candidate_dedup.dedupe_exact_candidates, "dedupe_exact_candidates", None),
        (pipeline._embed_batch_async, "embed", keep_embedded),
        (pipeline.read_conflict_watermark, "read_conflict_watermark", None),
        (candidate_cache.load_active_candidates, "load_active_candidates", None),
        (extractor_store.get_calibration, "get_calibration", None),
        (pipeline.is_scope_exempt_source, "is_scope_exempt_source", None),
        (subject_gate.gate_candidate_subjects, "gate_candidate_subjects", None),
        (pipeline._attach_speaker_subjects, "attach_speaker_subjects", None),
        (pipeline._gather_candidate_pairs, "gather_candidate_pairs", keep_gathered),
        (pipeline.load_generation_claims, "load_generation_claims", None),
        (pipeline.match_restatements, "match_restatements", None),
        (pipeline._probe_update_slots, "probe_update_slots", None),
        (pipeline._probe_pairs, "probe_pairs", None),
        (pipeline.select_pairs, "select_pairs", None),
        (pipeline._read_confirmed_pairs, "read_confirmed_pairs", None),
        (pipeline.build_duplicate_index, "build_duplicate_index", None),
        (pipeline.build_retired_value_index, "build_retired_value_index", None),
        (pipeline.write_order, "write_order", None),
        (pipeline.prefetch_subjects, "prefetch_subjects", None),
        (pipeline.search_store_under_lock, "search_store_under_lock", None),
        (subject_resolver.resolve_subjects, "resolve_subjects", None),
        (routing.route_particle, "route_particle", None),
        (pipeline._record_divergences, "record_divergences", None),
        (structure_edges.plan_structure_edges, "plan_structure_edges", None),
        (generation.cascade_superseded_generation, "cascade_superseded_generation", None),
        (pipeline._component_record, "component_record", None),
        (pipeline.update_extraction_status, status, None),
    ]
    for original, label, keep in spies:
        _spy(monkeypatch, trace, original, label, keep=keep)
    _spy_lock(monkeypatch, trace)

    await extract(
        "\n".join((CLAIM, META, STANCE)),
        [
            _candidate(CLAIM),
            _candidate(META, properties={SCOPE_KEY: SCOPE_DOCUMENT_META}),
            _candidate(
                STANCE,
                "the podcast",
                GATED_NAME,
                stance_kind=RelationType.ENDORSES,
                stance_target_index=0,
            ),
            _candidate(CLAIM),
        ],
        tags=[RULE_SOURCE_TAG],
        author_id=AUTHOR,
    )
    return trace


class TestOrder:
    @pytest.mark.parametrize(
        ("earlier", "later", "kind", "why"),
        ORDER,
        ids=[f"{a}->{b}" for a, b, _, _ in ORDER],
    )
    async def test_the_stage_runs_in_its_declared_order(
        self, traced: _Trace, earlier: str, later: str, kind: str, why: str
    ) -> None:
        assert earlier in traced.calls, f"{earlier} never ran, so this pins nothing"
        assert later in traced.calls, f"{later} never ran, so this pins nothing"
        before = traced.last(earlier) if kind == "barrier" else traced.first(earlier)
        assert before < traced.first(later), why

    async def test_the_scenario_reaches_a_probed_pair(self, traced: _Trace) -> None:
        """The pass probes a real pair, so the probe stages ran on something."""
        assert "has_contradiction_signal" in traced.calls


class TestHandoff:
    async def test_the_embedder_sees_one_copy_of_a_repeated_claim(self, traced: _Trace) -> None:
        assert traced.embedded is not None
        assert traced.embedded.count(CLAIM) == 1

    async def test_the_pair_search_sees_what_the_embedder_saw(self, traced: _Trace) -> None:
        assert traced.gathered is not None
        assert [c.content for c in traced.gathered] == traced.embedded

    async def test_the_pair_search_sees_the_rule_source_exemption(self, traced: _Trace) -> None:
        assert traced.gathered is not None
        (meta,) = [c for c in traced.gathered if c.content == META]
        assert (meta.properties or {}).get(SCOPE_ACTION_KEY) == SCOPE_ACTION_SOURCE_EXEMPT

    async def test_the_pair_search_sees_the_stance_holder(self, traced: _Trace) -> None:
        assert traced.gathered is not None
        (stance,) = [c for c in traced.gathered if c.content == STANCE]
        assert (stance.properties or {}).get(STANCE_HOLDER_KEY) == AUTHOR

    async def test_the_pair_search_sees_the_gated_subjects(self, traced: _Trace) -> None:
        assert traced.gathered is not None
        (stance,) = [c for c in traced.gathered if c.content == STANCE]
        assert GATED_NAME not in stance.subjects
        assert "the podcast" in stance.subjects
