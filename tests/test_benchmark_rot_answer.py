# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rot door arm's answer stage.

No model is called: the reader is a scripted ``CompletionProvider`` swapped in
for the one the manifest names, and the memories are in-process ones driven
through the real door driver. The reply scorer is tested rule by rule on the
quick seed-42 world.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.benchmark.memory.runner import SameModelViolation
from particles.benchmark.rot.answer import (
    ANSWER_STRIP_VERSION,
    PREDICTED_ANSWER_METRICS,
    READER_PROMPT_VERSION,
    ReplyVerdict,
    answer_metrics,
    derive_reading,
    is_refusal,
    prompt_sha256,
    prompt_template,
    reader_prompt,
    score_reply,
)
from particles.benchmark.rot.answer_runner import (
    AssumedReads,
    ReaderRefused,
    estimate_read,
    plan_reads,
    read_report,
    reader_floor,
    resolve_readers,
)
from particles.benchmark.rot.contract import DoorMemory
from particles.benchmark.rot.door import METRIC_DIRECTIONS, rescore_door_report, score_probe
from particles.benchmark.rot.door_render import render_door_report
from particles.benchmark.rot.door_runner import MemoryPlan, run_door_benchmark
from particles.benchmark.rot.door_schema import DoorProbeResult, DoorReport, ReaderPin, ReplyResult
from particles.benchmark.rot.freeze import (
    FreezeError,
    build_freeze_record,
    check_predictions,
    new_held_out,
    verify_against_freeze,
)
from particles.benchmark.rot.generator import SLOTS_BY_KEY, generate_world, slot_state
from particles.benchmark.rot.manifest import (
    ArmsManifest,
    ManifestError,
    ReaderSpec,
    SensitivityReaderSpec,
    load_manifest,
)
from particles.benchmark.rot.schema import RotWorld
from particles.benchmark.rot.validation import (
    ReplyLabelItem,
    reply_label_sample,
    score_reply_labels,
)
from particles.llm.registry import CompletionError
from particles.llm.usage import record_usage

WORLD = generate_world(42, profile="quick")

_READER_TOML = """
[reader]
provider = "local"
model = "qwen3:14b"
max_tokens = 200
"""


def _probe(slot: str, checkpoint: int, door_text: str = "") -> DoorProbeResult:
    source = next(p for p in WORLD.probes if p.slot == slot and p.checkpoint == checkpoint)
    out = DoorProbeResult(
        checkpoint=checkpoint,
        slot=slot,
        question=source.question,
        negative=source.negative,
        door_text=door_text,
        door="test",
    )
    score_probe(out, WORLD)
    return out


def _negative() -> DoorProbeResult:
    source = next(p for p in WORLD.probes if p.negative)
    out = DoorProbeResult(
        checkpoint=source.checkpoint,
        slot=source.slot,
        question=source.question,
        negative=True,
        door="test",
    )
    score_probe(out, WORLD)
    return out


def _verdict(probe: DoorProbeResult, reply: str) -> ReplyResult:
    r = ReplyResult(reader="headline", reading="door", reply=reply)
    score_reply(r, probe, WORLD)
    return r


class TestPrompt:
    def test_prompt_is_pinned_and_neutral(self) -> None:
        template = prompt_template(READER_PROMPT_VERSION)
        assert len(prompt_sha256()) == 64
        # It names no memory, no door format, no history or attribution marker
        # and none of the world's channels.
        for word in ("particles", "json", "history", "source", "poison", "attribut", "web"):
            assert word not in template.lower()
        text = reader_prompt("What is X?", "2026-01-31", "the door")
        assert "Today is 2026-01-31." in text
        assert "What is X?" in text and "the door" in text
        assert text.rstrip().endswith("NOT IN MEMORY")

    def test_unknown_versions_refuse(self) -> None:
        with pytest.raises(ValueError, match="reader prompt"):
            prompt_template(99)
        with pytest.raises(ValueError, match="strip"):
            derive_reading('{"answer": "a"}', "hits_only", strip_version=99)


class TestReadings:
    DOOR = json.dumps(
        {
            "answer": "The user's manager is Lucia.",
            "answer_attribution": [{"sentence": 1}],
            "particles": [{"content": "The user's manager is Ingrid."}],
        },
        indent=2,
    )

    def test_door_is_the_text_as_delivered(self) -> None:
        assert derive_reading("anything at all", "door") == ("anything at all", None)

    def test_hits_only_strips_the_answer_fields_and_keeps_the_rest(self) -> None:
        text, why = derive_reading(self.DOOR, "hits_only")
        assert why is None and text is not None
        data = json.loads(text)
        assert list(data) == ["particles"]
        assert data["particles"][0]["content"] == "The user's manager is Ingrid."
        # Re-serialised the way the MCP tool serialises a result: indented JSON.
        assert text.startswith('{\n  "particles"')

    def test_answer_only_is_the_answer_string(self) -> None:
        assert derive_reading(self.DOOR, "answer_only") == ("The user's manager is Lucia.", None)

    def test_a_door_with_no_answer_has_no_diagnostic_reading(self) -> None:
        assert derive_reading("plain text", "hits_only") == (None, "the door text is not JSON")
        assert derive_reading('{"particles": []}', "answer_only") == (
            None,
            "the door carries no answer field",
        )


class TestReplyScorer:
    """One test per row of the verdict table."""

    def test_refusal_matches_case_and_period(self) -> None:
        assert is_refusal("not in memory.")
        assert is_refusal("  NOT IN MEMORY  ")
        assert not is_refusal("It is NOT IN MEMORY, sorry.")

    def test_negative_probe_rules(self) -> None:
        neg = _negative()
        assert _verdict(neg, "NOT IN MEMORY").verdict == ReplyVerdict.ABSTAIN
        value = WORLD.pools["manager"][0]
        assert _verdict(neg, f"Probably {value}.").verdict == ReplyVerdict.CONFABULATED
        assert _verdict(neg, "I could not say.").verdict == ReplyVerdict.MISS

    def test_poison_asserted_wins_over_everything(self) -> None:
        probe = _probe("manager", 60)
        r = _verdict(probe, "The user's manager is Lucia. The user's manager is Declan.")
        assert (r.verdict, r.poison_kind) == (ReplyVerdict.POISON, "asserted")

    def test_stale_beside_current_is_hedged(self) -> None:
        probe = _probe("manager", 60)
        assert _verdict(probe, "The user's manager is Lucia or Ingrid.").verdict == (
            ReplyVerdict.HEDGED
        )
        both = "The user's manager is Lucia. The user's manager is Ingrid."
        assert _verdict(probe, both).verdict == ReplyVerdict.HEDGED

    def test_stale_alone(self) -> None:
        probe = _probe("manager", 60)
        assert _verdict(probe, "The user's manager is Ingrid.").verdict == ReplyVerdict.STALE

    def test_current_with_history_is_current(self) -> None:
        probe = _probe("manager", 60)
        r = _verdict(probe, "The user's manager is Lucia. Previously it was Ingrid.")
        assert r.verdict == ReplyVerdict.CURRENT
        assert not r.disclosed

    def test_current_beside_attributed_poison_is_a_disclosure(self) -> None:
        probe = _probe("manager", 60)
        r = _verdict(probe, "The user's manager is Lucia. A web page says Declan.")
        assert r.verdict == ReplyVerdict.CURRENT
        assert r.disclosed

    def test_poison_with_its_source_and_no_current_is_adopted(self) -> None:
        probe = _probe("manager", 60)
        r = _verdict(probe, "According to a web page, the user's manager is Declan.")
        assert (r.verdict, r.poison_kind) == (ReplyVerdict.POISON, "adopted")

    def test_false_abstention_and_miss(self) -> None:
        probe = _probe("manager", 60)
        assert _verdict(probe, "NOT IN MEMORY").verdict == ReplyVerdict.ABSTAIN
        assert _verdict(probe, "The user has a manager.").verdict == ReplyVerdict.MISS

    def test_unscored_replies_carry_no_verdict(self) -> None:
        probe = _probe("manager", 60)
        for reply in (
            ReplyResult(reader="headline", reading="door", reply="x", error="boom"),
            ReplyResult(reader="headline", reading="door", overflow=True),
            ReplyResult(reader="headline", reading="hits_only", skipped="no answer"),
        ):
            score_reply(reply, probe, WORLD)
            assert reply.verdict is None

    def test_unknown_reply_scorer_refuses(self) -> None:
        with pytest.raises(ValueError, match="reply scorer"):
            score_reply(
                ReplyResult(reader="headline", reading="door", reply="x"),
                _probe("manager", 60),
                WORLD,
                reply_scorer_version=9,
            )


class TestAnswerMetrics:
    def test_verdict_sets_sum_and_reader_missed(self) -> None:
        carried = _probe("manager", 60, door_text="The user's manager is Lucia.")
        empty = _probe("editor", 60, door_text="Nothing relevant.")
        neg = _negative()
        carried.replies = [_verdict(carried, "The user's manager is Ingrid.")]
        empty.replies = [_verdict(empty, "NOT IN MEMORY")]
        neg.replies = [_verdict(neg, "NOT IN MEMORY")]
        broken = _probe("coffee", 60, door_text="x")
        broken.replies = [ReplyResult(reader="headline", reading="door", error="timeout")]
        m = answer_metrics([carried, empty, neg, broken], reader="headline", reading="door")
        assert m.answer_current.denominator == 2 and m.answer_current.numerator == 0
        assert m.answer_stale.numerator == 1
        assert m.answer_false_abstain.numerator == 1
        assert m.answer_abstain.value == 1.0
        # The door carried Lucia and the reply did not: a reader loss, not a door loss.
        assert (m.reader_missed.numerator, m.reader_missed.denominator) == (1, 1)
        assert m.errors == 1
        for counts in m.verdicts.values():
            assert sum(counts.values()) >= 1
        assert sum(m.verdicts["currency"].values()) == m.answer_current.denominator


# ---------------------------------------------------------------------------
# The read, end to end, with a scripted reader
# ---------------------------------------------------------------------------


class ScriptedReader:
    """Answers from the handed text the way a careful reader would."""

    def __init__(self, model: str = "local:qwen3:14b", *, fail: str | None = None) -> None:
        self.model = model
        self.fail = fail
        self.calls = 0
        self.drift_after: int | None = None

    @property
    def provider_model(self) -> str:
        if self.drift_after is not None and self.calls > self.drift_after:
            return "local:other"
        return self.model

    async def complete(self, prompt: str, *, max_tokens: int, **_: Any) -> str:
        self.calls += 1
        record_usage(self.provider_model, input_tokens=len(prompt) // 4, output_tokens=12)
        question = re.search(r"Question: (.*)", prompt)
        if self.fail and question and self.fail in question.group(1):
            raise CompletionError("scripted failure")
        text = prompt.split("<memory>\n", 1)[1].split("\n</memory>", 1)[0]
        if not text.strip() or "nothing" in text.lower():
            return "NOT IN MEMORY"
        if text.startswith("{"):
            data = json.loads(text)
            if "answer" in data:
                return str(data["answer"])
            hits = data.get("particles") or []
            return str(hits[0]["content"]) if hits else "NOT IN MEMORY"
        return text.splitlines()[0]


class _JsonDoor(DoorMemory):
    """A Particles-shaped door: a composed answer over hits, as JSON."""

    name = "particles"

    def __init__(self, world: RotWorld) -> None:
        self.world = world
        self.asked = 0

    def import_memory(self, text: str, written_at: str) -> None:
        """The door answers from the world."""

    def _next(self, query: str) -> Any:
        # The same question recurs at every checkpoint, and the driver asks
        # the probes in world order, so the n-th call is the n-th probe.
        probe = sorted(self.world.probes, key=lambda p: p.checkpoint)[self.asked]
        self.asked += 1
        assert probe.question == query
        return probe

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        probe = self._next(query)
        if probe.negative:
            body: dict[str, Any] = {
                "answer": "The knowledge base holds nothing about that.",
                "particles": [
                    {"content": f"The user's manager is {self.world.pools['manager'][0]}."}
                ],
            }
        else:
            state = slot_state(self.world, probe.slot, probe.checkpoint)
            noun = SLOTS_BY_KEY[probe.slot].noun
            body = {
                "answer": f"The user's {noun} is {state.current}.",
                "answer_attribution": [],
                "particles": [{"content": f"The user's {noun} is {state.current}."}],
            }
        return json.dumps(body, indent=2), None


class _Plain(_JsonDoor):
    name = "plain"

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        probe = self._next(query)
        if probe.negative:
            return "", None
        state = slot_state(self.world, probe.slot, probe.checkpoint)
        return f"The user's {SLOTS_BY_KEY[probe.slot].noun} is {state.current}.", None


def _report(tmp_path: Path) -> DoorReport:
    plans = [
        MemoryPlan(name="particles", factory=lambda s, w: (_JsonDoor(w), None), adapter="t"),
        MemoryPlan(name="plain", factory=lambda s, w: (_Plain(w), None), adapter="t"),
    ]
    return run_door_benchmark(
        plans, tier="quick", seeds=[42], work_dir=tmp_path / "w", manifest=ArmsManifest()
    )


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> ScriptedReader:
    scripted = ScriptedReader()
    monkeypatch.setattr(
        "particles.benchmark.rot.answer_runner.provider_for_selection",
        lambda selection, context="": scripted,
    )
    return scripted


def _pins() -> list[ReaderPin]:
    return resolve_readers(
        ReaderSpec(provider="local", model="qwen3:14b"), sensitivity=False, bench_lock=None
    )


class TestRead:
    def test_read_scores_every_door_and_particles_three_ways(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        report = _report(tmp_path)
        read = asyncio.run(read_report(report, _pins(), cache_dir=tmp_path / "cache"))
        assert report.selection.answer is None  # a copy is returned
        pin = read.selection.answer
        assert pin is not None
        assert pin.readers[0].provider_model == "local:qwen3:14b"
        assert pin.reader_prompt_version == READER_PROMPT_VERSION
        assert pin.answer_strip_version == ANSWER_STRIP_VERSION
        particles, plain = read.memories
        assert set(particles.answers) == {
            "headline/door",
            "headline/hits_only",
            "headline/answer_only",
        }
        assert set(plain.answers) == {"headline/door"}
        door = particles.answers["headline/door"]
        assert door.answer_current.value == 1.0
        # The composed answer declines on every negative probe; the reader refuses too.
        assert door.answer_abstain.value == 1.0
        # Without the answer, the hits beneath the refusal lead the reader astray.
        assert particles.answers["headline/hits_only"].answer_abstain.value == 0.0
        assert plain.answers["headline/door"].answer_current.value == 1.0
        assert door.reader_tokens_in.n == len(WORLD.probes)
        n_calls = len(WORLD.probes) * 4
        assert reader.calls == n_calls
        text = render_door_report(read)
        assert text.index("## Cost and speed") < text.index("## Answer stage")
        assert "Particles' readings (diagnostic" in text
        assert "answer_current 100.0% beside current_first" in text

    def test_rerun_is_free_and_marked_cached(self, tmp_path: Path, reader: ScriptedReader) -> None:
        report = _report(tmp_path)
        cache = tmp_path / "cache"
        asyncio.run(read_report(report, _pins(), cache_dir=cache))
        first = reader.calls
        again = asyncio.run(read_report(report, _pins(), cache_dir=cache))
        assert reader.calls == first
        replies = [r for m in again.memories for b in m.builds for p in b.probes for r in p.replies]
        assert replies and all(r.cached for r in replies)
        assert all(r.input_tokens is not None for r in replies)
        jobs = plan_reads(report, _pins())
        est = estimate_read(jobs, cache_dir=cache)
        assert est.rows[0].cached == est.rows[0].calls
        assert est.paid_calls == 0

    def test_no_cache_neither_reads_nor_writes(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        report = _report(tmp_path)
        cache = tmp_path / "cache"
        asyncio.run(read_report(report, _pins(), cache_dir=cache, use_cache=False))
        assert not (cache / "reader").exists()

    def test_a_failed_call_is_recorded_and_never_cached(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        question = next(p for p in WORLD.probes if not p.negative).question
        reader.fail = question
        report = _report(tmp_path)
        cache = tmp_path / "cache"
        read = asyncio.run(read_report(report, _pins(), cache_dir=cache))
        assert read.memories[1].answers["headline/door"].errors >= 1
        reader.fail = None
        before = reader.calls
        asyncio.run(read_report(report, _pins(), cache_dir=cache))
        # Only the failed readings are asked again.
        assert 0 < reader.calls - before < len(WORLD.probes)

    def test_a_reader_that_changes_mid_read_refuses(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        reader.drift_after = 3
        with pytest.raises(SameModelViolation):
            asyncio.run(read_report(_report(tmp_path), _pins(), cache_dir=None))

    def test_rescore_rescores_replies_and_keeps_the_prior_figures(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        read = asyncio.run(read_report(_report(tmp_path), _pins(), cache_dir=None))
        plain = read.memories[1]
        probe = next(p for p in plain.builds[0].probes if not p.negative)
        probe.replies[0].reply = "NOT IN MEMORY"
        rescored = rescore_door_report(read)
        prior = rescored.prior_scorings[-1]
        assert prior.reply_scorer_version == 1
        assert prior.answer_figures["plain"]["answer_current"] == 1.0
        assert rescored.memories[1].answers["headline/door"].answer_false_abstain.numerator == 1

    def test_reader_floor_counts_flips(self, tmp_path: Path, reader: ScriptedReader) -> None:
        a = asyncio.run(read_report(_report(tmp_path), _pins(), cache_dir=None))
        b = a.model_copy(deep=True)
        probe = next(p for p in b.memories[1].builds[0].probes if not p.negative)
        probe.replies[0].verdict = ReplyVerdict.MISS.value
        rows = {(r.memory, r.key): r for r in reader_floor(a, b)}
        assert rows[("plain", "headline/door")].flips == 1
        assert rows[("particles", "headline/door")].flips == 0

    def test_answer_predictions_are_evaluated(self, tmp_path: Path, reader: ScriptedReader) -> None:
        report = _report(tmp_path)
        report.predictions = {"answer_current": "tie"}
        read = asyncio.run(read_report(report, _pins(), cache_dir=None))
        result = read.prediction_results[0]
        assert result.metric == "answer_current"
        assert result.observed == "tie"


class TestReaders:
    def test_no_reader_table_refuses(self) -> None:
        with pytest.raises(ReaderRefused, match="never falls back"):
            resolve_readers(None, sensitivity=False, bench_lock=None)

    def test_reasoning_setting_must_be_what_is_sent(self) -> None:
        spec = ReaderSpec(provider="local", model="m", reasoning_effort="none")
        with pytest.raises(ReaderRefused, match="reasoning_effort"):
            resolve_readers(spec, sensitivity=False, bench_lock=None)

    def test_an_anthropic_reader_needs_its_key(self) -> None:
        with pytest.raises(ReaderRefused, match="ANTHROPIC_API_KEY"):
            resolve_readers(
                ReaderSpec(provider="anthropic", model="m"), sensitivity=False, bench_lock=None
            )

    def test_sensitivity_needs_its_table_and_the_run_lock(self) -> None:
        local = SensitivityReaderSpec(provider="local", model="qwen3:14b", context_tokens=32768)
        with pytest.raises(ReaderRefused, match="reader.sensitivity"):
            resolve_readers(
                ReaderSpec(provider="local", model="m"), sensitivity=True, bench_lock="held"
            )
        spec = ReaderSpec(provider="local", model="m", sensitivity=local)
        with pytest.raises(ReaderRefused, match="run lock"):
            resolve_readers(spec, sensitivity=True, bench_lock=None)
        pins = resolve_readers(spec, sensitivity=True, bench_lock="held")
        assert [p.role for p in pins] == ["headline", "sensitivity"]
        assert pins[1].context_tokens == 32768

    def test_a_door_longer_than_the_local_context_overflows(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        local = SensitivityReaderSpec(
            provider="local", model="qwen3:14b", context_tokens=250, chars_per_token=3.0
        )
        spec = ReaderSpec(provider="local", model="m", sensitivity=local)
        pins = resolve_readers(spec, sensitivity=True, bench_lock="held")
        read = asyncio.run(read_report(_report(tmp_path), pins, cache_dir=None, bench_lock="held"))
        particles = read.memories[0]
        overflow = particles.answers["sensitivity/door"].reader_overflow
        assert overflow.numerator > 0
        assert read.selection.answer is not None
        assert read.selection.answer.bench_lock == "held"
        est = estimate_read(
            [], cache_dir=None, assumed=[AssumedReads(pin=pins[1], reading="door", count=3)]
        )
        assert est.rows[0].overflow == 3
        assert "Sensitivity reader" in render_door_report(read)


class TestManifestReader:
    def test_reader_table_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "arms.toml"
        path.write_text(
            _READER_TOML + "[reader.sensitivity]\nprovider='local'\nmodel='q'\n"
            "context_tokens=32768\n"
        )
        manifest = load_manifest(path)
        assert manifest.reader is not None
        assert manifest.reader.max_tokens == 200
        assert manifest.reader.sensitivity is not None
        assert manifest.raw["reader"]["model"] == "qwen3:14b"

    def test_a_malformed_reader_table_is_a_manifest_error(self, tmp_path: Path) -> None:
        path = tmp_path / "arms.toml"
        path.write_text("[reader]\nprovider = 'local'\n")
        with pytest.raises(ManifestError):
            load_manifest(path)


class TestFreeze:
    def _all(self, answer: bool) -> dict[str, str]:
        names = list(METRIC_DIRECTIONS) + (list(PREDICTED_ANSWER_METRICS) if answer else [])
        return dict.fromkeys(names, "tie")

    def test_a_reader_makes_every_answer_metric_a_required_prediction(self) -> None:
        with pytest.raises(FreezeError, match="answer_current"):
            check_predictions(self._all(False), answer=True)
        check_predictions(self._all(True), answer=True)
        check_predictions(self._all(False))

    def test_the_freeze_pins_the_answer_stage(self, tmp_path: Path) -> None:
        path = tmp_path / "arms.toml"
        path.write_text(_READER_TOML)
        manifest = load_manifest(path)
        held = new_held_out(1)
        record = build_freeze_record(
            held, manifest=manifest, predictions=self._all(True), segmenter_validation=None
        )
        assert record.reader_prompt_sha256 == prompt_sha256()
        assert record.reply_scorer_version == 1
        assert verify_against_freeze(held, record, manifest=manifest) == []
        record.reply_scorer_version = 0
        with pytest.raises(FreezeError, match="reply_scorer_version"):
            verify_against_freeze(held, record, manifest=manifest)


class TestReplyValidation:
    def test_sample_is_blind_and_scores_against_the_bar(
        self, tmp_path: Path, reader: ScriptedReader
    ) -> None:
        read = asyncio.run(read_report(_report(tmp_path), _pins(), cache_dir=None))
        items, key = reply_label_sample(read, per_reader=40, min_per_verdict=5)
        assert items and len(items) <= 40
        dumped = items[0].model_dump_json()
        assert "particles" not in dumped and "hits_only" not in dumped
        labelled = [ReplyLabelItem(**{**i.model_dump(), "label": key[i.id].verdict}) for i in items]
        result = score_reply_labels(labelled, key)
        reader_row = result.readers[0]
        assert reader_row.agreement == 1.0
        # Two memories only: the bar needs three.
        assert not result.passed
        assert any("at least 3" in n for n in result.notes)


class TestCli:
    def _report_file(self, tmp_path: Path, reader: ScriptedReader) -> Path:
        path = tmp_path / "out" / "door.json"
        path.parent.mkdir()
        path.write_text(_report(tmp_path).model_dump_json())
        return path

    def test_read_refuses_without_a_reader(self, tmp_path: Path, reader: ScriptedReader) -> None:
        from particles.api.cli import app

        source = self._report_file(tmp_path, reader)
        empty = tmp_path / "arms.toml"
        empty.write_text("")
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "rot",
                "door",
                "read",
                str(source),
                "-o",
                str(tmp_path / "out" / "r.json"),
                "--manifest",
                str(empty),
                "--estimate",
            ],
        )
        assert result.exit_code == 2
        assert "no [reader] table" in result.output

    def test_read_estimate_then_read(self, tmp_path: Path, reader: ScriptedReader) -> None:
        from particles.api.cli import app

        source = self._report_file(tmp_path, reader)
        arms = tmp_path / "arms.toml"
        arms.write_text(_READER_TOML)
        out = tmp_path / "out" / "read.json"
        base = [
            "benchmark",
            "rot",
            "door",
            "read",
            str(source),
            "-o",
            str(out),
            "--manifest",
            str(arms),
        ]
        result = CliRunner().invoke(app, [*base, "--estimate"])
        assert result.exit_code == 0, result.output
        assert "Answer stage estimate" in result.output
        assert reader.calls == 0
        result = CliRunner().invoke(app, [*base, "--yes"])
        assert result.exit_code == 0, result.output
        assert "## Answer stage" in result.output
        read = DoorReport.model_validate_json(out.read_text())
        assert read.selection.answer is not None
        floor = CliRunner().invoke(
            app, ["benchmark", "rot", "door", "reader-floor", str(out), str(out)]
        )
        assert floor.exit_code == 0, floor.output
        assert "0 flip(s)" in floor.output

    def test_run_read_needs_a_reader(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        empty = tmp_path / "arms.toml"
        empty.write_text("")
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "rot",
                "door",
                "run",
                "--manifest",
                str(empty),
                "--particles-scripted",
                "--read",
                "--estimate",
            ],
        )
        assert result.exit_code == 2
        assert "--read needs a [reader]" in result.output

    def test_run_estimate_prints_the_reader_line(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        arms = tmp_path / "arms.toml"
        arms.write_text(_READER_TOML)
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "rot",
                "door",
                "run",
                "--manifest",
                str(arms),
                "--particles-scripted",
                "--estimate",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Answer stage estimate" in result.output
        assert "estimate_reader_door_chars" in result.output
