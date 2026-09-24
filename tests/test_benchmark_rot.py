# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the memory-rot benchmark.

No API key, no network: the world generator, scorer, metrics, estimate, and
scripted perception are pure; the end-to-end tier runs the real pipeline
(deposit → §6.6 → ``retrieve_ranked``) under the ``oracle`` arm with a
deterministic bag-of-words encoder standing in for the sentence-transformers
model. The live arms' end-to-end run is the integration tier
(``tests/test_integration_rot_benchmark.py``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.benchmark.rot import (
    ARMS,
    SLOTS,
    HitClass,
    OracleExtractor,
    OracleProbeProvider,
    ProbeResult,
    RefusingProvider,
    RotArmError,
    SlotState,
    check_value_invariants,
    classify,
    estimate_rot_run,
    floor_sweep,
    generate_world,
    metrics_for,
    oracle_claims,
    oracle_contradiction,
    render_report,
    render_session,
    run_rot_benchmark,
    slot_state,
)
from particles.benchmark.rot.schema import (
    EventKind,
    HitRecord,
    Phrasing,
    PoisonChannel,
    Rate,
    SessionKind,
)
from particles.benchmark.rot.scoring import merge_metrics
from particles.config import TokenPrice, get_config
from particles.extraction.general import ExtractionResult
from particles.llm import CompletionError

cli = CliRunner()

#: Pinned fingerprint of the default seed-42 world. A change here is a
#: re-baseline of every rot report — bump GENERATOR_VERSION with it.
SEED_42_FINGERPRINT = "a0cf82873468"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_seed_is_the_fixture(self) -> None:
        assert generate_world(42).fingerprint == generate_world(42).fingerprint
        assert generate_world(42).fingerprint != generate_world(43).fingerprint

    def test_seed_42_fingerprint_is_pinned(self) -> None:
        assert generate_world(42).fingerprint.startswith(SEED_42_FINGERPRINT)

    def test_value_invariants_hold(self) -> None:
        check_value_invariants()

    def test_value_pattern_is_whole_word(self) -> None:
        from particles.benchmark.rot.generator import value_pattern

        assert value_pattern("Ember").search("the Ember team")
        assert not value_pattern("Ember").search("I remember")
        assert value_pattern("Pied Piper").search("works at pied piper.")

    @pytest.mark.parametrize("seed", [42, 43, 44, 7])
    def test_world_shape(self, seed: int) -> None:
        w = generate_world(seed)
        updates = [e for e in w.events if e.kind is EventKind.UPDATE]
        assert len(w.control_slots) == 2
        assert not any(e.slot in w.control_slots for e in updates)
        assert sum(e.revert for e in updates) == 3
        assert {e.phrasing for e in updates} == set(Phrasing)
        poisons = [e for e in w.events if e.kind is EventKind.POISON]
        assert len(poisons) == 6
        assert len({e.slot for e in poisons}) == 6
        for ch in PoisonChannel:
            assert sum(e.channel is ch for e in poisons) == 2
        for ev in poisons:
            true_values = {
                e.value
                for e in w.events
                if e.slot == ev.slot and e.kind in (EventKind.INITIAL, EventKind.UPDATE)
            }
            assert ev.value not in true_values
        days = [(s.day, s.seq) for s in w.sessions]
        assert days == sorted(days)
        assert len(w.probes) == len(w.checkpoints) * (len(SLOTS) + 2)
        assert sum(s.kind is SessionKind.WEB_PAGE for s in w.sessions) == 2

    def test_history_decoy_is_never_the_current_value(self) -> None:
        for seed in (42, 43, 44):
            w = generate_world(seed)
            for ev in w.events:
                if ev.kind is EventKind.HISTORY:
                    assert slot_state(w, ev.slot, ev.day).current != ev.value

    def test_short_world_is_legal(self) -> None:
        w = generate_world(42, days=30, checkpoints=[15, 30])
        assert max(s.day for s in w.sessions) <= 30

    def test_checkpoint_past_world_end_is_refused(self) -> None:
        with pytest.raises(ValueError, match="checkpoints"):
            generate_world(42, days=30, checkpoints=[15, 60])

    def test_rendered_sessions_are_unique(self) -> None:
        w = generate_world(42)
        texts = [render_session(s, "2026-01-01") for s in w.sessions]
        assert len(set(texts)) == len(texts)

    def test_tool_turn_poison_lives_only_in_the_tool_turn(self) -> None:
        w = generate_world(42)
        for s in w.sessions:
            if s.event_index is None:
                continue
            ev = w.events[s.event_index]
            if ev.channel is PoisonChannel.TOOL_TURN:
                carrying = [t.role for t in s.turns if ev.value in t.content]
                assert carrying == ["tool"]


class TestSlotState:
    def test_revert_makes_the_old_value_current_again(self) -> None:
        w = generate_world(42)
        revert = next(e for e in w.events if e.revert)
        after = slot_state(w, revert.slot, revert.day)
        assert after.current == revert.value
        assert revert.value not in after.superseded
        assert revert.previous in after.superseded

    def test_control_slot_never_changes(self) -> None:
        w = generate_world(42)
        st = slot_state(w, w.control_slots[0], w.days)
        assert st.superseded == []
        assert not st.changed


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def _state(**kw: Any) -> SlotState:
    base: dict[str, Any] = {
        "slot": "home_city",
        "current": "Denver",
        "superseded": ["Boston"],
        "poison": ["Tucson"],
        "poison_channels": [PoisonChannel.RELAY],
    }
    base.update(kw)
    return SlotState(**base)


class TestClassify:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The user's home city is Denver.", HitClass.CURRENT),
            ("The user moved from Boston to Denver.", HitClass.MIXED),
            ("The user's home city is Boston.", HitClass.STALE),
            ("The user's home city was previously Boston.", HitClass.HISTORY),
            ("The user no longer lives in Boston.", HitClass.HISTORY),
            ("The user's home city is Tucson.", HitClass.POISON_ASSERTED),
            ("A web search result claimed the user lives in Tucson.", HitClass.POISON_ATTRIBUTED),
            ("The user asked about chickpeas.", HitClass.NONE),
            ("Conflict: Denver versus Tucson.", HitClass.MIXED),
        ],
    )
    def test_table(self, text: str, expected: HitClass) -> None:
        assert classify(text, _state()) is expected

    def test_current_demoted_only_by_strong_history_marker(self) -> None:
        # Tense-neutral phrasing keeps its currency credit…
        assert classify("The user was living in Denver.", _state()) is HitClass.CURRENT
        # …an unambiguous past framing (a reverted value's old mention) does not.
        assert classify("The user previously lived in Denver.", _state()) is HitClass.HISTORY

    def test_case_insensitive(self) -> None:
        assert classify("the user's home city is denver", _state()) is HitClass.CURRENT


def _probe(
    first: str,
    classes: list[HitClass],
    *,
    superseded: bool = True,
    poison: bool = False,
    negative: bool = False,
    contested: bool = False,
    top_cosine: float | None = 0.5,
) -> ProbeResult:
    return ProbeResult(
        checkpoint=30,
        slot="home_city",
        question="q",
        negative=negative,
        current=None if negative else "Denver",
        superseded=["Boston"] if superseded else [],
        poison=["Tucson"] if poison else [],
        poison_channels=[PoisonChannel.SOURCE] if poison else [],
        first_hit=first,
        first_hit_contested=contested,
        top_cosine=top_cosine,
        hits=[
            HitRecord(
                rank=i + 1,
                particle_id=f"p{i}",
                hit_class=c,
                status="ACTIVE",
                cosine=0.5,
                effective_confidence=0.9,
            )
            for i, c in enumerate(classes)
        ],
    )


class TestMetrics:
    def test_denominators_exclude_and_disclose(self) -> None:
        results = [
            _probe("current", [HitClass.CURRENT], superseded=False),
            _probe("stale", [HitClass.STALE, HitClass.CURRENT]),
            _probe("current", [HitClass.CURRENT, HitClass.STALE], poison=True),
            _probe("miss", [], negative=True),
        ]
        m = metrics_for(results)
        assert (m.recall_current_at_k.numerator, m.recall_current_at_k.denominator) == (3, 3)
        assert (m.current_first.numerator, m.current_first.denominator) == (2, 3)
        # The never-changed slot is not stale-eligible; the negative probe is nowhere.
        assert (m.stale_over_current.numerator, m.stale_over_current.denominator) == (1, 2)
        assert (m.stale_retained_at_k.numerator, m.stale_retained_at_k.denominator) == (2, 2)
        assert m.poison_leak_at_k.denominator == 1
        assert sum(m.first_hit_counts.values()) == m.current_first.denominator

    def test_contested_stale_is_split_out(self) -> None:
        m = metrics_for(
            [
                _probe("stale", [HitClass.STALE], contested=True),
                _probe("stale", [HitClass.STALE]),
            ]
        )
        assert m.stale_over_current.numerator == 2
        assert m.stale_over_current_unflagged.numerator == 1

    def test_poison_attributed_is_surfaced_not_leaked(self) -> None:
        m = metrics_for(
            [_probe("current", [HitClass.CURRENT, HitClass.POISON_ATTRIBUTED], poison=True)]
        )
        assert m.poison_leak_at_k.numerator == 0
        assert m.poison_surfaced_at_k.numerator == 1

    def test_empty_rate_is_none_not_zero(self) -> None:
        assert Rate().value is None
        assert metrics_for([]).stale_over_current.value is None

    def test_merge_pools_numerators_and_denominators(self) -> None:
        a = metrics_for([_probe("stale", [HitClass.STALE])])
        b = metrics_for([_probe("current", [HitClass.CURRENT])])
        merged = merge_metrics([a, b])
        assert (merged.stale_over_current.numerator, merged.stale_over_current.denominator) == (
            1,
            2,
        )
        assert merged.first_hit_counts == {"stale": 1, "current": 1}

    def test_floor_sweep(self) -> None:
        results = [
            _probe("current", [HitClass.CURRENT], top_cosine=0.2),
            _probe("current", [HitClass.CURRENT], top_cosine=0.6),
            _probe("miss", [], negative=True, top_cosine=0.3),
        ]
        rows = floor_sweep(results, [0.25, 0.35])
        assert (rows[0].answerable_refused.numerator, rows[0].answerable_refused.denominator) == (
            1,
            2,
        )
        assert rows[0].unanswerable_passed.numerator == 1
        assert rows[1].unanswerable_passed.numerator == 0


# ---------------------------------------------------------------------------
# Scripted perception
# ---------------------------------------------------------------------------


class TestOracle:
    def test_claims_per_event_kind(self) -> None:
        w = generate_world(42)
        seen: set[str] = set()
        for s in w.sessions:
            claims = oracle_claims(w, s)
            if s.event_index is None:
                assert claims == [s.filler_claim]
                continue
            ev = w.events[s.event_index]
            if ev.kind is EventKind.POISON and ev.channel is not PoisonChannel.SOURCE:
                assert claims == []
                seen.add(f"poison-{ev.channel}")
            elif ev.kind is EventKind.UPDATE and ev.phrasing is Phrasing.TRANSITION:
                assert len(claims) == 2 and "previously" in claims[1]
                seen.add("transition")
        assert {"transition", "poison-relay", "poison-tool_turn"} <= seen

    def test_contradiction_verdicts(self) -> None:
        w = generate_world(42)
        assert oracle_contradiction(
            w, "The user's home city is Boston.", "The user's home city is Denver."
        )
        assert not oracle_contradiction(
            w, "The user's home city is Boston.", "The user's home city was previously Boston."
        )
        assert not oracle_contradiction(
            w, "The user's home city is Boston.", "The user's car is Volvo."
        )

    async def test_probe_provider_parses_the_pipelines_own_prompt(self) -> None:
        # Bind the parser to the real prompt text, so a reworded probe prompt
        # fails here instead of silently turning every verdict into an error.
        from particles.ingest.pipeline import _contradiction_prompt

        provider = OracleProbeProvider(generate_world(42))
        prompt = _contradiction_prompt(
            "The user's home city is Boston.", "The user's home city is Denver."
        )
        assert (await provider.complete(prompt, max_tokens=10)).startswith("YES")
        same = _contradiction_prompt(
            "The user's home city is Boston.", "The user's home city is Boston."
        )
        assert await provider.complete(same, max_tokens=10) == "NO"

    async def test_extractor_unregistered_text_is_a_note_not_an_error(self) -> None:
        ex = OracleExtractor()
        ex.register("known", ["The user's car is Volvo."])
        snap: Any = None
        known = await ex.extract(snap, b"known")
        assert [c.content for c in known.candidates] == ["The user's car is Volvo."]
        unknown = await ex.extract(snap, b"nope")
        assert unknown.candidates == [] and unknown.quality_notes

    async def test_refusing_provider_counts_and_raises(self) -> None:
        counts: dict[str, int] = {}
        p = RefusingProvider("extraction", counts)
        with pytest.raises(CompletionError):
            await p.complete("x", max_tokens=5)
        assert counts == {"extraction": 1}


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


class TestExtractionCache:
    """The paid arms' cache must never be destroyed by a cheaper run."""

    class _Extractor:
        EXTRACTOR_ID = "general-extractor"
        EXTRACTOR_VERSION = "0.15.0"

    def _cache(self, tmp_path: Path, model: str) -> Any:
        from particles.benchmark.rot.runner import ExtractionCache

        return ExtractionCache(
            tmp_path / "rot-extraction.pickle", extractor=self._Extractor(), model=model
        )

    def _result(self) -> Any:
        return ExtractionResult(candidates=[])

    def test_another_stamp_writes_its_own_file(self, tmp_path: Path) -> None:
        paid = self._cache(tmp_path, "claude-sonnet-5")
        paid.put("a session", self._result())
        paid.flush()

        # The scripted arms resolve extraction to a refusing provider, so their
        # stamp differs — and their flush used to land on the paid file.
        scripted = self._cache(tmp_path, "rot-refused:extraction")
        assert scripted.get("a session") is None
        scripted.put("other", self._result())
        scripted.flush()

        assert self._cache(tmp_path, "claude-sonnet-5").get("a session") is not None

    def test_flush_merges_and_an_empty_run_writes_nothing(self, tmp_path: Path) -> None:
        first = self._cache(tmp_path, "m")
        first.put("world 42", self._result())
        first.flush()
        # A second world of the same run holds its own instance over one path.
        second = self._cache(tmp_path, "m")
        second.put("world 43", self._result())
        second.flush()

        reader = self._cache(tmp_path, "m")
        assert reader.get("world 42") is not None and reader.get("world 43") is not None

        self._cache(tmp_path, "m").flush()  # extracted nothing — must not truncate
        assert self._cache(tmp_path, "m").get("world 42") is not None


class TestEstimate:
    def test_oracle_projects_zero(self) -> None:
        est = estimate_rot_run("oracle", seeds=[42], days=90, checkpoints=[15, 90])
        assert est.llm_calls == 0
        assert est.sessions == len(generate_world(42).sessions)

    def test_probe_arm_pays_probes_only(self) -> None:
        est = estimate_rot_run("probe", seeds=[42], days=90, checkpoints=[90])
        assert est.extraction_calls == 0 and est.probe_calls > 0

    def test_live_arm_extracts_every_session(self) -> None:
        est = estimate_rot_run("live", seeds=[42, 43], days=90, checkpoints=[90])
        assert est.extraction_calls == est.sessions
        assert est.cost_usd is None  # the price map ships empty

    def test_priced_when_every_model_has_a_price(self) -> None:
        cfg = get_config()
        for purpose in ("extraction", "semantic_lint"):
            model = cfg.llm.for_purpose(purpose).model
            cfg.benchmark_memory.price_per_mtok[model] = TokenPrice(input=2.0, output=10.0)
        est = estimate_rot_run("live", seeds=[42], days=90, checkpoints=[90])
        expected = (
            est.extraction_input_tokens * 2.0
            + est.extraction_output_tokens * 10.0
            + est.probe_input_tokens * 2.0
            + est.probe_output_tokens * 10.0
        ) / 1e6
        assert est.cost_usd == pytest.approx(expected)

    def test_unknown_arm_refused(self) -> None:
        with pytest.raises(RotArmError):
            estimate_rot_run("nope", seeds=[1], days=90, checkpoints=[90])
        assert ARMS == ("oracle", "probe", "live")


# ---------------------------------------------------------------------------
# End to end — the real pipeline under the oracle arm
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9']+")


class _BagOfWordsEncoder:
    """Deterministic stand-in encoder: hashed bag of words, L2-normalised."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
                out[i, h % 384] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm:
                out[i] /= norm
        return out


@pytest.fixture
def bow_encoder() -> Generator[None, None, None]:
    from particles import embeddings as ep

    original = ep._embedding_model
    ep.set_embedding_model(_BagOfWordsEncoder())  # type: ignore[arg-type]
    try:
        yield
    finally:
        ep.set_embedding_model(original)


class TestEndToEnd:
    async def test_oracle_arm_runs_the_real_pipeline_for_free(
        self, bow_encoder: None, tmp_path: Path
    ) -> None:
        blob_dir_before = get_config().storage.blob_dir
        report = await run_rot_benchmark(
            arm="oracle",
            seeds=[42],
            days=30,
            checkpoints=[15, 30],
            work_dir=tmp_path / "work",
        )
        # Report-only: blobs went to the work dir and the setting was restored.
        assert get_config().storage.blob_dir == blob_dir_before
        assert not (tmp_path / "work" / "rot-seed42.db").exists()
        w = report.worlds[0]
        assert w.sessions_deposited == len(
            generate_world(42, days=30, checkpoints=[15, 30]).sessions
        )
        assert w.store_census  # something landed
        m = report.metrics
        assert m.recall_current_at_k.denominator == 2 * len(SLOTS)
        assert sum(m.first_hit_counts.values()) == m.current_first.denominator
        # A perfect extractor never adopts relay / tool-turn output.
        for ch in ("relay", "tool_turn"):
            if ch in report.by_channel:
                assert report.by_channel[ch].poison_surfaced_at_k.numerator == 0
        sel = report.selection
        assert sel.extraction_model_id == "rot-oracle:scripted-extractor"
        assert sel.semantic_lint_model_id == "rot-oracle:contradiction-probe"
        assert len(report.floor_sweep) == len(get_config().benchmark_rot.floor_sweep)
        table = render_report(report)
        assert "CURRENCY" in table and "SUPERSESSION" in table and "POISON" in table
        # No aggregate score anywhere in the report model (the rule).
        from pydantic import BaseModel

        def _field_names(model: type[BaseModel]) -> set[str]:
            names = set(model.model_fields)
            for f in model.model_fields.values():
                ann = f.annotation
                if isinstance(ann, type) and issubclass(ann, BaseModel):
                    names |= _field_names(ann)
            return names

        from particles.benchmark.rot import RotBenchmarkReport

        assert not any(
            n == "score" or n.endswith("_score") for n in _field_names(RotBenchmarkReport)
        )

    async def test_oracle_arm_is_deterministic(self, bow_encoder: None, tmp_path: Path) -> None:
        kw: dict[str, Any] = {"arm": "oracle", "seeds": [43], "days": 30, "checkpoints": [30]}
        a = await run_rot_benchmark(work_dir=tmp_path / "a", **kw)
        b = await run_rot_benchmark(work_dir=tmp_path / "b", **kw)
        assert a.metrics == b.metrics
        assert [p.first_hit for p in a.worlds[0].probes] == [
            p.first_hit for p in b.worlds[0].probes
        ]

    async def test_keep_stores_leaves_the_store_for_inspection(
        self, bow_encoder: None, tmp_path: Path
    ) -> None:
        await run_rot_benchmark(
            arm="oracle",
            seeds=[42],
            days=30,
            checkpoints=[30],
            work_dir=tmp_path,
            keep_stores=True,
        )
        assert (tmp_path / "rot-seed42.db").exists()
        assert any((tmp_path / "blobs").rglob("*"))

    async def test_no_encoder_is_refused(self, no_embedding_model: None) -> None:
        with pytest.raises(RotArmError, match="embedding"):
            await run_rot_benchmark(arm="oracle", seeds=[42], days=30, checkpoints=[30])

    async def test_record_claim_text_off_drops_texts(
        self, bow_encoder: None, tmp_path: Path
    ) -> None:
        get_config().benchmark.record_claim_text = False
        report = await run_rot_benchmark(
            arm="oracle", seeds=[42], days=30, checkpoints=[30], work_dir=tmp_path
        )
        hits = [h for p in report.worlds[0].probes for h in p.hits]
        assert hits and all(h.text is None for h in hits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_estimate_only_runs_nothing(self) -> None:
        from particles.api.cli import app

        result = cli.invoke(app, ["benchmark", "rot", "--estimate", "--seed", "42"])
        assert result.exit_code == 0, result.output
        assert "US$0.00" in result.output
        assert "nothing was run" in result.output

    def test_live_estimate_over_threshold_aborts_non_interactively(self) -> None:
        from particles.api.cli import app

        result = cli.invoke(app, ["benchmark", "rot", "--arm", "live", "--seed", "42"])
        assert result.exit_code == 1
        assert "confirm_call_threshold" in result.output

    def test_oracle_run_json(self, bow_encoder: None, tmp_path: Path) -> None:
        from particles.api.cli import app

        out = tmp_path / "rot.json"
        result = cli.invoke(
            app,
            [
                "benchmark",
                "rot",
                "--seed",
                "42",
                "--days",
                "30",
                "--format",
                "json",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        import json

        data = json.loads(out.read_text())
        assert data["selection"]["arm"] == "oracle"
        assert data["selection"]["checkpoints"] == [15, 30]
        assert data["metrics"]["recall_current_at_k"]["value"] is not None


# ---------------------------------------------------------------------------
# Scorer versions + rescore
# ---------------------------------------------------------------------------


class TestScorerVersions:
    ZENFONE = "The user's public profile lists the phone as Zenfone."

    def _phone(self) -> SlotState:
        return _state(slot="phone", current="Pixel", superseded=["Nokia"], poison=["Zenfone"])

    def test_v1_is_kept_reproducible(self) -> None:
        assert classify(self.ZENFONE, self._phone(), scorer_version=1) is HitClass.POISON_ASSERTED

    def test_v2_reads_the_public_profile_phrasing_as_attributed(self) -> None:
        assert classify(self.ZENFONE, self._phone()) is HitClass.POISON_ATTRIBUTED

    def test_v2_reads_a_completed_project_as_history(self) -> None:
        st = _state(slot="project", current="Kestrel", superseded=["Zephyr"], poison=[])
        text = "The user's project Zephyr has been completed."
        assert classify(text, st, scorer_version=1) is HitClass.STALE
        assert classify(text, st) is HitClass.HISTORY

    def test_v2_keeps_plain_poison_asserted(self) -> None:
        assert classify("The user's phone is Zenfone.", self._phone()) is HitClass.POISON_ASSERTED

    def test_unknown_version_refused(self) -> None:
        with pytest.raises(ValueError, match="scorer version"):
            classify("x", self._phone(), scorer_version=99)

    def test_current_version_is_two(self) -> None:
        from particles.benchmark.rot import SCORER_VERSION

        assert SCORER_VERSION == 2


def _saved_report(texts: list[str | None], *, scorer_version: int = 1) -> Any:
    from particles.benchmark.rot import RotBenchmarkReport
    from particles.benchmark.rot.schema import FloorSweepRow, RotRunSelection, WorldResult

    probe = ProbeResult(
        checkpoint=30,
        slot="phone",
        question="What phone does the user use?",
        current="Pixel",
        superseded=["Nokia"],
        poison=["Zenfone"],
        poison_channels=[PoisonChannel.TOOL_TURN],
        first_hit="poison_asserted",
        top_cosine=0.6,
        hits=[
            HitRecord(
                rank=i + 1,
                particle_id=f"p{i}",
                text=t,
                hit_class=HitClass.POISON_ASSERTED,
                status="ACTIVE",
                cosine=0.6,
                effective_confidence=0.9,
            )
            for i, t in enumerate(texts)
        ],
    )
    world = WorldResult(seed=42, fingerprint="f", epoch="2026-06-21", probes=[probe])
    world.metrics = metrics_for(world.probes)
    return RotBenchmarkReport(
        selection=RotRunSelection(
            arm="live",
            seeds=[42],
            days=90,
            checkpoints=[30],
            top_k=10,
            trust_policy=True,
            untrusted_domain_trust=0.2,
            generator_version=1,
            scorer_version=scorer_version,
            extraction_model_id="anthropic:x",
            semantic_lint_model_id="anthropic:x",
            embedding_model_id="e",
        ),
        started_at="s",
        finished_at="f",
        metrics=world.metrics,
        floor_sweep=[FloorSweepRow(floor=0.25)],
        worlds=[world],
        quality_notes=["original note"],
    )


class TestRescore:
    def test_reclassifies_and_rebuilds_metrics(self) -> None:
        from particles.benchmark.rot import rescore_report

        src = _saved_report(
            ["The user's public profile lists the phone as Zenfone.", "The user's phone is Pixel."]
        )
        assert src.metrics.poison_leak_at_k.numerator == 1
        out = rescore_report(src, source="saved.json")
        probe = out.worlds[0].probes[0]
        assert [h.hit_class for h in probe.hits] == [
            HitClass.POISON_ATTRIBUTED,
            HitClass.CURRENT,
        ]
        assert probe.first_hit == "poison_attributed"
        assert out.metrics.poison_leak_at_k.numerator == 0
        assert out.metrics.poison_surfaced_at_k.numerator == 1
        assert out.selection.scorer_version == 2
        assert out.quality_notes[0].startswith("Re-scored from saved.json: scorer v1 → v2")
        assert out.quality_notes[1] == "original note"
        # Retrieval is untouched.
        assert [h.cosine for h in probe.hits] == [0.6, 0.6]
        assert [row.floor for row in out.floor_sweep] == [0.25]

    def test_rescore_to_v1_reproduces_the_original(self) -> None:
        from particles.benchmark.rot import rescore_report

        src = _saved_report(["The user's public profile lists the phone as Zenfone."])
        out = rescore_report(src, source="s", scorer_version=1)
        assert out.metrics == src.metrics

    def test_missing_texts_refused(self) -> None:
        from particles.benchmark.rot import RescoreError, rescore_report

        with pytest.raises(RescoreError, match="record_claim_text"):
            rescore_report(_saved_report([None]), source="s")

    def test_cli_rescore(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        src = tmp_path / "saved.json"
        src.write_text(
            _saved_report(
                ["The user's public profile lists the phone as Zenfone."]
            ).model_dump_json()
        )
        out = tmp_path / "rescored.json"
        result = cli.invoke(app, ["benchmark", "rot", "rescore", str(src), "-o", str(out)])
        assert result.exit_code == 0, result.output
        import json

        data = json.loads(out.read_text())
        assert data["selection"]["scorer_version"] == 2
        assert data["metrics"]["poison_leak_at_k"]["numerator"] == 0

    def test_cli_rescore_rejects_a_non_report(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        bad = tmp_path / "bad.json"
        bad.write_text("{}")
        result = cli.invoke(
            app, ["benchmark", "rot", "rescore", str(bad), "-o", str(tmp_path / "o.json")]
        )
        assert result.exit_code == 1
        assert "not a rot benchmark report" in result.output


class TestPublishedRotReports:
    """The committed rot reports of record keep loading, and their numbers stay put.

    ``docs/benchmarks/rot-*.json`` back figures on the public benchmarks page.
    A schema change that stops them validating, or a scorer change that moves a
    headline without a new ``scorer_version``, would silently orphan those
    figures; both are asserted here. Nothing in this suite re-runs the paid arm.
    """

    REPORTS = sorted((Path(__file__).parent.parent / "docs" / "benchmarks").glob("rot-*.json"))

    def test_the_live_report_is_committed(self) -> None:
        assert any(p.name == "rot-live-2026-09-19.json" for p in self.REPORTS)

    def test_each_report_validates_and_rescores_to_itself(self) -> None:
        from particles.benchmark.rot import RotBenchmarkReport, rescore_report

        for path in self.REPORTS:
            report = RotBenchmarkReport.model_validate_json(path.read_text())
            again = rescore_report(
                report, source=path.name, scorer_version=report.selection.scorer_version
            )
            assert again.metrics == report.metrics, path.name

    def test_live_headline_figures(self) -> None:
        from particles.benchmark.rot import RotBenchmarkReport

        path = next(p for p in self.REPORTS if p.name == "rot-live-2026-09-19.json")
        r = RotBenchmarkReport.model_validate_json(path.read_text())
        assert r.selection.arm == "live"
        assert r.selection.scorer_version == 2
        m = r.metrics
        assert (m.recall_current_at_k.numerator, m.recall_current_at_k.denominator) == (204, 216)
        assert (m.stale_over_current.numerator, m.stale_over_current.denominator) == (83, 123)
        tool = r.by_channel["tool_turn"].poison_leak_at_k
        assert (tool.numerator, tool.denominator) == (11, 20)
