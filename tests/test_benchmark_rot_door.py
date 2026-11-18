# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rot benchmark's door arm.

No model is called and no competitor is installed: the contract is exercised
through a stdlib toy adapter run behind the real door shim, the driver through
in-process memories, and the Particles child's configuration as data. The
scripted Particles child end to end is in
``tests/test_integration_rot_door.py`` (it loads the real encoder).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from particles.benchmark.rot.baselines import (
    RawChunksMemory,
    RecentMentionsMemory,
    content_words,
)
from particles.benchmark.rot.contract import (
    BASE_ENV,
    ChildContractAdapter,
    DoorAdapterError,
    DoorMemory,
    ManagedServer,
    child_env,
)
from particles.benchmark.rot.door import (
    BUDGETS_CHARS,
    METRIC_DIRECTIONS,
    door_metrics,
    evaluate_predictions,
    rescore_door_report,
    score_probe,
    segment,
)
from particles.benchmark.rot.door_render import render_door_report
from particles.benchmark.rot.door_runner import (
    MemoryPlan,
    baseline_plans,
    estimate_door_run,
    particles_plan,
    run_door_benchmark,
)
from particles.benchmark.rot.door_schema import DoorMemoryResult, DoorProbeResult, MetricRange
from particles.benchmark.rot.freeze import (
    FreezeError,
    build_freeze_record,
    check_predictions,
    commitment_for,
    new_held_out,
    verify_against_freeze,
)
from particles.benchmark.rot.generator import (
    FULL_PROFILE,
    QUICK_PROFILE,
    SLOTS_BY_KEY,
    generate_world,
    slot_state,
)
from particles.benchmark.rot.manifest import ArmsManifest, ManifestError, load_manifest
from particles.benchmark.rot.schema import EventKind, HitClass, PoisonChannel, RotWorld
from particles.benchmark.rot.validation import LabelKey, score_labels

#: Pinned fingerprint of the quick seed-42 world.
QUICK_42_FINGERPRINT = "aedb6ff76097"

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


# ---------------------------------------------------------------------------
# The quick world
# ---------------------------------------------------------------------------


class TestQuickProfile:
    def test_full_world_is_unchanged_by_profiles(self) -> None:
        world = generate_world(42)
        assert world.profile == "full"
        assert world.fingerprint.startswith("a0cf82873468")
        assert len(world.probes) == 84

    def test_quick_world_is_pinned(self) -> None:
        assert generate_world(42, profile="quick").fingerprint.startswith(QUICK_42_FINGERPRINT)

    @pytest.mark.parametrize("seed", [1, 7, 42, 43, 44, 99])
    def test_quick_world_keeps_every_phenomenon(self, seed: int) -> None:
        world = generate_world(seed, profile="quick")
        assert world.checkpoints == list(QUICK_PROFILE.checkpoints)
        assert len(world.pools) == QUICK_PROFILE.slots
        assert len(world.probes) == 24
        assert len(world.control_slots) == 1
        assert sum(1 for e in world.events if e.revert) == 2
        channels = sorted(e.channel.value for e in world.events if e.channel is not None)
        assert channels == ["relay", "source", "tool_turn"]
        assert any(e.kind is EventKind.HISTORY for e in world.events)
        assert 40 <= len(world.sessions) <= 70
        assert len(generate_world(seed).sessions) > 2.5 * len(world.sessions)

    def test_unknown_profile_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown world profile"):
            generate_world(42, profile="tiny")

    def test_profiles_are_named_as_documented(self) -> None:
        assert FULL_PROFILE.slots == 12
        assert QUICK_PROFILE.filler_every == 3


# ---------------------------------------------------------------------------
# Segmenting and scoring a door text
# ---------------------------------------------------------------------------


def _changed_probe(world: RotWorld) -> tuple[Any, Any]:
    for probe in world.probes:
        if probe.negative:
            continue
        state = slot_state(world, probe.slot, probe.checkpoint)
        if state.superseded and state.current not in state.superseded:
            return probe, state
    raise AssertionError("no changed slot")


class TestSegment:
    def test_cuts_lines_and_sentences_with_offsets(self) -> None:
        text = "First one. Second one!\n\n  Third line?\nlast"
        segs = segment(text)
        assert [s for _, s in segs] == ["First one.", "Second one!", "Third line?", "last"]
        for offset, piece in segs:
            assert text[offset : offset + len(piece)] == piece

    def test_json_lines_are_segments(self) -> None:
        text = json.dumps({"answer": "a. b.", "particles": [{"content": "c."}]}, indent=2)
        assert any('"content": "c."' in s for _, s in segment(text))

    def test_json_escaped_line_breaks_cut(self) -> None:
        text = json.dumps({"answer": "Listing:\n- The user is X.\n- Y"}, indent=2)
        pieces = [s for _, s in segment(text)]
        assert "- The user is X." in pieces
        assert any(p.startswith("- Y") for p in pieces)

    def test_unknown_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="segmenter version"):
            segment("x", version=99)


class TestScoring:
    def _probe(self, world: RotWorld, text: str) -> DoorProbeResult:
        probe, _ = _changed_probe(world)
        result = DoorProbeResult(
            checkpoint=probe.checkpoint, slot=probe.slot, question=probe.question, door_text=text
        )
        result.chars = len(text)
        score_probe(result, world)
        return result

    def test_stale_first_then_current(self) -> None:
        world = generate_world(42, profile="quick")
        _, state = _changed_probe(world)
        noun = SLOTS_BY_KEY[state.slot].noun
        r = self._probe(
            world,
            f"The user's {noun} is {state.superseded[0]}.\nThe user's {noun} is {state.current}.",
        )
        assert r.first_class == HitClass.STALE.value
        m = door_metrics([r])
        assert m.stale_over_current.numerator == 1
        assert m.current_in_door.numerator == 1
        assert m.current_first.numerator == 0

    def test_honest_history_is_not_stale(self) -> None:
        world = generate_world(42, profile="quick")
        _, state = _changed_probe(world)
        noun = SLOTS_BY_KEY[state.slot].noun
        r = self._probe(world, f"The user's {noun} was previously {state.superseded[0]}.")
        assert r.first_class == HitClass.HISTORY.value
        assert door_metrics([r]).stale_in_door.numerator == 0

    def test_budget_is_measured_in_characters(self) -> None:
        world = generate_world(42, profile="quick")
        _, state = _changed_probe(world)
        noun = SLOTS_BY_KEY[state.slot].noun
        filler = "Nothing relevant here.\n" * 120
        r = self._probe(world, filler + f"The user's {noun} is {state.current}.")
        assert BUDGETS_CHARS[0] < r.segments[0].start < BUDGETS_CHARS[1]
        m = door_metrics([r])
        assert m.recall_current_at_2k.numerator == 0
        assert m.recall_current_at_8k.numerator == 1

    def test_negative_probe_silent_only_without_any_world_value(self) -> None:
        world = generate_world(42, profile="quick")
        some_value = next(iter(world.pools.values()))[0]
        neg = next(p for p in world.probes if p.negative)
        loud = DoorProbeResult(
            checkpoint=neg.checkpoint,
            slot=neg.slot,
            question=neg.question,
            negative=True,
            door_text=f"Maybe {some_value}?",
        )
        quiet = loud.model_copy(update={"door_text": "No memory matches."})
        score_probe(loud, world)
        score_probe(quiet, world)
        assert loud.silent is False
        assert quiet.silent is True
        m = door_metrics([loud, quiet])
        assert (m.negative_silent.numerator, m.negative_silent.denominator) == (1, 2)

    def test_failed_door_call_leaves_every_denominator(self) -> None:
        world = generate_world(42, profile="quick")
        probe, _ = _changed_probe(world)
        r = DoorProbeResult(
            checkpoint=probe.checkpoint, slot=probe.slot, question=probe.question, error="boom"
        )
        score_probe(r, world)
        m = door_metrics([r])
        assert m.errors == 1
        assert m.current_in_door.denominator == 0


class TestPredictions:
    def _mem(self, name: str, mean: float, role: str = "headline") -> DoorMemoryResult:
        return DoorMemoryResult(
            name=name,
            role=role,
            ranges={"current_first": MetricRange(mean=mean, lo=mean, hi=mean, builds=1)},
        )

    def test_verdicts(self) -> None:
        memories = [
            self._mem("particles", 0.9),
            self._mem("a", 0.5),
            self._mem("b", 0.6),
            self._mem("raw-chunks", 0.99, role="baseline"),
        ]
        out = {p.predicted: p for p in evaluate_predictions({"current_first": "trail"}, memories)}
        assert out["trail"].observed == "beat"
        assert out["trail"].verdict == "better than predicted"

    def test_overlapping_ranges_tie(self) -> None:
        p = self._mem("particles", 0.5)
        p.ranges["current_first"] = MetricRange(mean=0.5, lo=0.4, hi=0.7, builds=3)
        q = self._mem("a", 0.6)
        [res] = evaluate_predictions({"current_first": "tie"}, [p, q])
        assert res.verdict == "as predicted"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_content_words_drop_persona_and_stop_words(self) -> None:
        assert content_words("Which city does the user live in?") == {"city", "live"}

    def test_recent_mentions_newest_first(self) -> None:
        mem = RecentMentionsMemory()
        mem.import_memory("user: I live in Boston", "2026-01-01")
        mem.import_memory("user: I live in Denver now", "2026-02-01")
        text, _ = mem.door_text("Which city does the user live in?", "recent")
        assert text.splitlines()[0].endswith("Denver now")

    def test_raw_chunks_ranks_by_cosine(self) -> None:
        mem = RawChunksMemory(encoder=_BagOfWordsEncoder())  # type: ignore[arg-type]
        mem.import_memory("bread sourdough starter", "d")
        mem.import_memory("my phone is a Pixel", "d")
        text, raw = mem.door_text("What phone does the user use?", "chunks")
        assert text.startswith("my phone is a Pixel")
        assert raw == {"hits": 2}


# ---------------------------------------------------------------------------
# The contract over a child process
# ---------------------------------------------------------------------------

_TOY_ADAPTER = """
import os

class Toy:
    def __init__(self):
        self.items = []
    def health(self):
        return True
    def doors(self):
        return ["search", "other"]
    def door_text(self, query, door):
        print("noise on stdout must not break the protocol")
        hits = [t for t in self.items if any(w in t.lower() for w in query.lower().split())]
        return "\\n".join(hits), {"hits": len(hits), "env_key": os.environ.get("TOY_KEY")}
    def import_memory(self, text, written_at):
        self.items.append(text)
        return "id-%d" % len(self.items)
    def settle(self):
        return None
"""


@pytest.fixture
def toy_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "clone"
    (clone / "toy").mkdir(parents=True)
    (clone / "toy" / "__init__.py").write_text("")
    (clone / "toy" / "adapter.py").write_text(textwrap.dedent(_TOY_ADAPTER))
    return clone


class TestChildContract:
    def test_env_is_whitelisted_and_secrets_unrecorded(self, tmp_path: Path) -> None:
        parent = {"PATH": "/bin", "ANTHROPIC_API_KEY": "sk-x", "GEMINI_API_KEY": "g-1"}
        env, record = child_env(
            parent, {"KEY": "{env:GEMINI_API_KEY}", "STORE": "{store}/db"}, store=tmp_path
        )
        assert "ANTHROPIC_API_KEY" not in env
        assert env["KEY"] == "g-1"
        assert env["STORE"] == f"{tmp_path}/db"
        assert record["KEY"] == "<from env GEMINI_API_KEY>"
        assert set(env) <= {*BASE_ENV, "HOME", "KEY", "STORE"}
        assert env["HOME"] == str(tmp_path / "home")

    def test_home_is_never_the_operators(self, tmp_path: Path) -> None:
        env, _ = child_env({"HOME": "/Users/operator", "PATH": "/bin"}, {}, store=tmp_path)
        assert env["HOME"] == str(tmp_path / "home")
        declared, _ = child_env({"PATH": "/bin"}, {"HOME": "{store}/h2"}, store=tmp_path)
        assert declared["HOME"] == f"{tmp_path}/h2"

    def test_missing_passthrough_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DoorAdapterError, match="needs NOPE"):
            child_env({}, {"K": "{env:NOPE}"}, store=tmp_path)

    def test_adapter_round_trip(self, toy_clone: Path, tmp_path: Path) -> None:
        adapter = ChildContractAdapter(
            "toy",
            spec="toy.adapter:Toy",
            clone=toy_clone,
            python=Path(sys.executable),
            env={"PATH": "/usr/bin:/bin", "TOY_KEY": "k"},
            log=tmp_path / "toy.log",
        )
        try:
            assert adapter.health()
            assert adapter.doors() == ["search", "other"]
            assert adapter.supports("settle")
            assert not adapter.supports("import_session")
            adapter.import_memory("I drive a Saab", "2026-01-01")
            adapter.settle()
            text, raw = adapter.door_text("saab", "search")
            assert text == "I drive a Saab"
            assert raw == {"hits": 1, "env_key": "k"}
        finally:
            adapter.close()
        assert "noise on stdout" in (tmp_path / "toy.log").read_text()

    def test_unloadable_adapter_is_refused(self, toy_clone: Path, tmp_path: Path) -> None:
        with pytest.raises(DoorAdapterError, match="cannot load"):
            ChildContractAdapter(
                "nope",
                spec="toy.missing:Nope",
                clone=toy_clone,
                python=Path(sys.executable),
                env={"PATH": "/usr/bin:/bin"},
                log=tmp_path / "nope.log",
            )


# ---------------------------------------------------------------------------
# The driver, end to end with in-process memories
# ---------------------------------------------------------------------------


class _Perfect(DoorMemory):
    """A memory that always serves the current value, and stores the world."""

    name = "perfect"

    def __init__(self, world: RotWorld) -> None:
        self.world = world

    def import_memory(self, text: str, written_at: str) -> None:
        """Nothing to keep: the door answers from the world itself."""

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        probe = next(p for p in self.world.probes if p.question == query)
        if probe.negative:
            return "", None
        state = slot_state(self.world, probe.slot, 90)
        noun = SLOTS_BY_KEY[probe.slot].noun
        return f"The user's {noun} is {state.current}.", None


def _perfect_plan(*, cacheable: bool = False, model_backed: bool = False) -> MemoryPlan:
    return MemoryPlan(
        name="perfect",
        factory=lambda store, world: (_Perfect(world), None),
        adapter="tests:_Perfect",
        cacheable=cacheable,
        model_backed=model_backed,
    )


class TestDriver:
    def test_quick_tier_runs_and_renders(self, tmp_path: Path) -> None:
        report = run_door_benchmark(
            [_perfect_plan(), *baseline_plans()[1:]],
            tier="quick",
            seeds=[42],
            work_dir=tmp_path,
            manifest=ArmsManifest(),
        )
        assert report.selection.profile == "quick"
        assert any("tuned on" in d for d in report.disclosures)
        perfect = report.memories[0]
        assert perfect.metrics.negative_silent.value == 1.0
        assert perfect.metrics.write_ms.n == len(generate_world(42, profile="quick").sessions)
        assert perfect.builds[0].write_methods == {
            "import_memory": perfect.builds[0].sessions_written
        }
        text = render_door_report(report)
        assert text.startswith("> Written by the Particles authors")
        for metric in METRIC_DIRECTIONS:
            assert metric in text

    def test_competitor_cache_round_trip_and_rescore(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        plan = _perfect_plan(cacheable=True)
        first = run_door_benchmark(
            [plan],
            tier="quick",
            seeds=[42],
            work_dir=tmp_path / "w1",
            manifest=ArmsManifest(),
            cache_dir=cache,
        )
        second = run_door_benchmark(
            [plan],
            tier="quick",
            seeds=[42],
            work_dir=tmp_path / "w2",
            manifest=ArmsManifest(),
            cache_dir=cache,
        )
        assert not first.memories[0].builds[0].cached
        assert second.memories[0].builds[0].cached
        assert (
            second.memories[0].ranges["current_in_door"].mean
            == first.memories[0].ranges["current_in_door"].mean
        )
        rescored = rescore_door_report(second)
        assert len(rescored.prior_scorings) == 1
        assert "Headline under scorer" in render_door_report(rescored)

    def test_failed_build_is_reported_not_scored(self, tmp_path: Path) -> None:
        class _Broken(DoorMemory):
            name = "broken"

            def import_memory(self, text: str, written_at: str) -> None:
                raise DoorAdapterError("disk full")

        plan = MemoryPlan(name="broken", factory=lambda s, w: (_Broken(), None))
        report = run_door_benchmark(
            [plan], tier="quick", seeds=[42], work_dir=tmp_path, manifest=ArmsManifest()
        )
        build = report.memories[0].builds[0]
        assert build.probes == []
        assert "disk full" in build.notes[0]

    def test_health_is_checked_after_the_first_write(self, tmp_path: Path) -> None:
        # adebench adapters such as mem0's report unhealthy until they hold data
        class _NeedsData(_Perfect):
            name = "needs-data"

            def __init__(self, world: RotWorld) -> None:
                super().__init__(world)
                self.writes = 0

            def health(self) -> bool:
                return self.writes > 0

            def import_memory(self, text: str, written_at: str) -> None:
                self.writes += 1

        plan = MemoryPlan(name="needs-data", factory=lambda s, w: (_NeedsData(w), None), wait_s=0.1)
        report = run_door_benchmark(
            [plan], tier="quick", seeds=[42], work_dir=tmp_path, manifest=ArmsManifest()
        )
        build = report.memories[0].builds[0]
        assert build.notes == []
        assert build.probes

    def test_unhealthy_after_the_first_write_fails_before_probing(self, tmp_path: Path) -> None:
        class _NeverHealthy(_Perfect):
            name = "never"

            def health(self) -> bool:
                return False

        plan = MemoryPlan(name="never", factory=lambda s, w: (_NeverHealthy(w), None), wait_s=0.1)
        report = run_door_benchmark(
            [plan], tier="quick", seeds=[42], work_dir=tmp_path, manifest=ArmsManifest()
        )
        build = report.memories[0].builds[0]
        assert build.probes == []
        assert build.sessions_written == 1
        assert "not healthy" in build.notes[0]

    def test_particles_requires_import_session(self, tmp_path: Path) -> None:
        plan = MemoryPlan(
            name="particles",
            factory=lambda s, w: (_Perfect(w), None),
            requires_session=True,
        )
        report = run_door_benchmark(
            [plan], tier="quick", seeds=[42], work_dir=tmp_path, manifest=ArmsManifest()
        )
        assert "import_session is required" in report.memories[0].builds[0].notes[0]

    def test_milestone_builds_model_backed_memories_three_times(self, tmp_path: Path) -> None:
        report = run_door_benchmark(
            [_perfect_plan(model_backed=True)],
            tier="milestone",
            seeds=[42],
            work_dir=tmp_path,
            manifest=ArmsManifest(),
            held_out=True,
            seed_commitment="abc",
        )
        assert len(report.memories[0].builds) == 3
        assert report.memories[0].ranges["current_in_door"].builds == 3
        assert any("held-out seeds" in d for d in report.disclosures)

    def test_server_lifecycle(self, tmp_path: Path) -> None:
        server = ManagedServer(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            log=tmp_path / "s.log",
        )
        assert server.alive()
        server.stop()
        assert not server.alive()


class TestEstimate:
    def test_free_rows_cost_nothing(self) -> None:
        est = estimate_door_run(baseline_plans(), tier="quick", seeds=[42], cache_dir=None)
        assert est.paid_builds == 0
        assert est.total == (0.0, 0.0)

    def test_manifest_figure_scales_with_the_world(self) -> None:
        plan = _perfect_plan(model_backed=True)
        plan.usd_per_full_build = (1.0, 2.0)
        quick = estimate_door_run([plan], tier="quick", seeds=[42], cache_dir=None)
        full = estimate_door_run([plan], tier="standard", seeds=[42], cache_dir=None)
        assert quick.paid_builds == 1
        lo, hi = quick.total
        assert lo is not None and hi is not None
        assert 0.2 < lo < 0.5
        assert full.total == (1.0, 2.0)

    def test_particles_plan_records_its_configuration(self) -> None:
        plan = particles_plan(role="variant", variant_note="trust rule", trust_rule=True)
        assert plan.requires_session
        assert plan.configuration["compiled_defaults"] is True
        assert "domain_trust_rule" in plan.configuration
        # the door's shape rides the row, for the report's footnote.
        assert plan.configuration["mcp_query"] == {
            "detail": "compact",
            "list": "cited",
            "hits": 10,
            "grounded": True,
        }

    def test_a_compact_particles_row_gets_the_shape_footnote(self, tmp_path: Path) -> None:
        from particles.benchmark.rot.door_render import SHAPE_FOOTNOTE

        report = run_door_benchmark(
            [_perfect_plan(), *baseline_plans()[1:]],
            tier="quick",
            seeds=[42],
            work_dir=tmp_path,
            manifest=ArmsManifest(),
        )
        row = report.memories[0]
        assert SHAPE_FOOTNOTE not in render_door_report(report)

        row.name = "particles"
        row.configuration["mcp_query"] = {"detail": "full"}
        assert SHAPE_FOOTNOTE not in render_door_report(report)

        row.configuration["mcp_query"] = {"detail": "compact"}
        text = render_door_report(report)
        sections = text.split("\n## ")
        marked = [s.split("\n", 1)[0] for s in sections if SHAPE_FOOTNOTE in s]
        assert marked == ["Supersession", "Source trust"]


class TestParticlesChildConfig:
    def test_parity_pins_every_purpose(self, tmp_path: Path) -> None:
        from particles.benchmark.rot.particles_door import child_config
        from particles.llm import LLMPurpose

        config, pins = child_config(tmp_path, parity=("gemini", "gemini-3-flash-preview"))
        for purpose in ("default", *LLMPurpose.__args__):  # type: ignore[attr-defined]
            assert config["llm"][purpose] == {
                "provider": "gemini",
                "model": "gemini-3-flash-preview",
            }
            assert pins[purpose] == "gemini:gemini-3-flash-preview"
        assert config["agent_memory"]["projection"]["enabled"] is False

    def test_headline_pins_nothing(self, tmp_path: Path) -> None:
        from particles.benchmark.rot.particles_door import child_config

        config, pins = child_config(tmp_path, parity=None)
        assert pins == {}
        assert "extraction" not in config["llm"]


# ---------------------------------------------------------------------------
# Manifest, freeze, validation
# ---------------------------------------------------------------------------


class TestManifest:
    def test_none_means_particles_and_baselines(self) -> None:
        assert load_manifest(None).memories == {}

    def test_reserved_name_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.toml"
        path.write_text(
            '[adebench]\nclone="/c"\nsha="s"\npython="/p"\n[memories.particles]\nadapter="a:B"\n'
        )
        with pytest.raises(ManifestError, match="reserved"):
            load_manifest(path)

    def test_competitors_need_a_clone(self, tmp_path: Path) -> None:
        path = tmp_path / "m.toml"
        path.write_text('[memories.mem0]\nadapter="adebench.mem0:Mem0Adapter"\n')
        with pytest.raises(ManifestError, match="adebench"):
            load_manifest(path)

    def test_loads_and_hashes(self, tmp_path: Path) -> None:
        path = tmp_path / "m.toml"
        path.write_text(
            '[adebench]\nclone="/c"\nsha="6b23060"\npython="/p"\n'
            '[memories.mem0]\nadapter="adebench.mem0:Mem0Adapter"\nmodel_backed=true\n'
            "usd_per_full_build=[0.3, 0.8]\n[memories.mem0.env]\nK='{env:GEMINI_API_KEY}'\n"
        )
        m = load_manifest(path)
        assert m.memories["mem0"].usd_per_full_build == (0.3, 0.8)
        assert m.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions() -> dict[str, str]:
    return {k: "tie" for k in METRIC_DIRECTIONS}


class TestFreeze:
    def test_commitment_round_trip(self) -> None:
        held = new_held_out(3)
        assert commitment_for(held.seeds, held.salt) == held.commitment
        record = build_freeze_record(
            held, manifest=ArmsManifest(), predictions=_predictions(), segmenter_validation=None
        )
        assert verify_against_freeze(held, record, manifest=ArmsManifest()) == []

    def test_wrong_seeds_are_refused(self) -> None:
        held = new_held_out(3)
        record = build_freeze_record(
            held, manifest=ArmsManifest(), predictions=_predictions(), segmenter_validation=None
        )
        other = held.model_copy(update={"seeds": [1, 2, 3]})
        with pytest.raises(FreezeError, match="commitment"):
            verify_against_freeze(other, record, manifest=ArmsManifest())

    def test_moved_version_is_refused(self) -> None:
        held = new_held_out(3)
        record = build_freeze_record(
            held, manifest=ArmsManifest(), predictions=_predictions(), segmenter_validation=None
        )
        record.scorer_version = 1
        with pytest.raises(FreezeError, match="scorer_version"):
            verify_against_freeze(held, record, manifest=ArmsManifest())

    def test_every_metric_must_be_predicted(self) -> None:
        preds = _predictions()
        del preds["chars_per_door"]
        with pytest.raises(FreezeError, match="chars_per_door"):
            check_predictions(preds)
        with pytest.raises(FreezeError, match="not one of"):
            check_predictions(_predictions() | {"current_first": "win"})


class TestValidation:
    def _labelled(self, memory: str, n: int, wrong: int, cls: str = "stale") -> tuple[Any, Any]:
        from particles.benchmark.rot.validation import LabelItem

        items, key = [], {}
        for i in range(n):
            iid = f"{memory}-{i}"
            label = "history" if i < wrong else cls
            items.append(
                LabelItem(
                    id=iid,
                    question="q",
                    checkpoint=30,
                    current="X",
                    superseded=[],
                    poison=[],
                    segment="s",
                    label=label,
                )
            )
            key[iid] = LabelKey(memory=memory, scorer_class=cls)
        return items, key

    def test_bar_needs_two_competitors_and_particles(self) -> None:
        items, key = [], {}
        for m in ("particles", "mem0", "cognee"):
            i, k = self._labelled(m, 40, 0)
            items += i
            key |= k
        assert score_labels(items, key).passed
        assert not score_labels(
            items[:80], {k: v for k, v in key.items() if "cognee" not in k}
        ).passed

    def test_disagreement_fails_the_bar(self) -> None:
        items, key = [], {}
        for m in ("particles", "mem0", "cognee"):
            i, k = self._labelled(m, 40, 4 if m == "mem0" else 0)
            items += i
            key |= k
        result = score_labels(items, key)
        assert not result.passed
        assert [m.passed for m in result.memories if m.memory == "mem0"] == [False]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_estimate_without_manifest(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "rot",
                "door",
                "run",
                "--tier",
                "quick",
                "--manifest",
                str(tmp_path / "none.toml"),
                "--estimate",
            ],
        )
        assert result.exit_code == 2
        assert "no such manifest" in result.output

    def test_estimate_with_empty_manifest_lists_rows(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        manifest = tmp_path / "arms.toml"
        manifest.write_text("")
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "rot",
                "door",
                "run",
                "--tier",
                "quick",
                "--manifest",
                str(manifest),
                "--particles-scripted",
                "--estimate",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "raw-chunks [baseline]" in result.output
        assert "nothing was run" in result.output

    def test_milestone_needs_freeze(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        result = CliRunner().invoke(app, ["benchmark", "rot", "door", "run", "--tier", "milestone"])
        assert result.exit_code == 2
        assert "--held-out and --freeze" in result.output

    def test_seeds_refuse_a_work_tree(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        # A work tree of its own: the exported test trees are not git
        # checkouts, so a target beside this file would not be inside one.
        (tmp_path / ".git").mkdir()
        target = tmp_path / "private" / "held-out-should-not-exist.json"
        result = CliRunner().invoke(app, ["benchmark", "rot", "door", "seeds", "-o", str(target)])
        assert result.exit_code == 1
        assert not target.exists()

    def test_seeds_and_freeze(self, tmp_path: Path) -> None:
        from particles.api.cli import app

        held = tmp_path / "private" / "held.json"
        result = CliRunner().invoke(app, ["benchmark", "rot", "door", "seeds", "-o", str(held)])
        assert result.exit_code == 0, result.output
        assert "commitment:" in result.output
        preds = tmp_path / "preds.json"
        preds.write_text(json.dumps(_predictions()))
        empty = tmp_path / "arms.toml"
        empty.write_text("")
        record = tmp_path / "freeze.json"
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "rot",
                "door",
                "freeze",
                "--held-out",
                str(held),
                "--predictions",
                str(preds),
                "--manifest",
                str(empty),
                "-o",
                str(record),
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(record.read_text())["commitment"] in held.read_text()


def test_poison_channels_have_their_own_rows() -> None:
    from particles.benchmark.rot.door import channel_metrics

    world = generate_world(42, profile="quick")
    poisoned = [
        p for p in world.probes if not p.negative and slot_state(world, p.slot, p.checkpoint).poison
    ]
    results = []
    for p in poisoned:
        r = DoorProbeResult(checkpoint=p.checkpoint, slot=p.slot, question=p.question, door_text="")
        score_probe(r, world)
        results.append(r)
    rows = channel_metrics(results)
    assert set(rows) <= {c.value for c in PoisonChannel}
    assert rows
