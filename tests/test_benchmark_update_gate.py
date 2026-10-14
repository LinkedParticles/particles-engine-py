# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The trip-versus-move and update-gate suites.

Key-free throughout. The checks and the gate scoring are pure; the scripted
arm runs the real child process, the operator's own verbs and the real
pipeline with every model answer read from the package's table, which is the
only way to test the isolation across the process boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from particles.benchmark.update_gate import _child, case, checks, estimate, pairs, scoring
from particles.benchmark.update_gate.fixtures import (
    MOVE_NOTES,
    PAIR_SETS,
    TRIP_ALONE_NOTES,
    case_note,
    pair_set,
)
from particles.benchmark.update_gate.schema import (
    CaseRun,
    TripVsMoveReport,
    UpdateGateReport,
)
from particles.benchmark.update_gate.scripted import TABLE_PATH, ScriptedTable, source_key
from particles.config import ProviderSelection, get_config, reset_config
from particles.llm.usage import LLMUsage, UsageRow

_ROOT = Path(__file__).resolve().parent.parent
_BENCH = _ROOT / "docs" / "benchmarks"

S4, S11, S12 = ("session-04.md", "session-11.md", "session-12-moved.md")


def _row(content: str, status: str = "ACTIVE", *, src: str = S11, restated: bool = False) -> Any:
    return {"content": content, "status": status, "sources": [src], "restatement": restated}


#: A store the fix leaves behind on claude-sonnet-5-5 (2026-10-12, run 1).
_FIXED = [
    _row("The user lives in Lajpat Nagar, Delhi.", "PROVENANCE_STALE", src=S4),
    _row("The user's home is in Delhi.", "PROVENANCE_STALE"),
    _row("The user was in Mumbai for a week for work.", "PROVENANCE_STALE"),
    _row("The user plans to fly back to Delhi on Thursday, 2026-10-15.", "SUPERSEDED"),
    _row(
        "As of 2026-10-08, when the user was in Mumbai for a week for work, they were due "
        "to fly back to Delhi on Thursday, 2026-10-15.",
        restated=True,
    ),
    _row("The user moved to Mumbai.", src=S12),
    _row("Sandeep's Curry House is in Lajpat Nagar, Delhi.", src=S4),
]


# ---------------------------------------------------------------------------
# The checks (version 1 is the harness's evaluate(), unchanged)
# ---------------------------------------------------------------------------


class TestChecks:
    def test_the_fixed_store_passes(self) -> None:
        assert checks.evaluate(_FIXED, [], trip_alone=False)["pass"]

    @pytest.mark.parametrize(
        ("extra", "failed"),
        [
            (_row("The user lives in Lajpat Nagar, Delhi.", src=S4), "i_delhi_residence_or_flight"),
            (_row("The user flies back home to Delhi on Thursday."), "i_delhi_residence_or_flight"),
            (
                _row("The user is in Mumbai for a week for work, as of 2026-10-08."),
                "iii_trip_active",
            ),
            (
                _row("The user is due to fly back to Delhi on Thursday.", restated=True),
                "i_restatement_shape",
            ),
            (
                _row("As of 2026-10-08 the user was due to fly home.", "SUPERSEDED", restated=True),
                "ii_restatements_not_active",
            ),
        ],
    )
    def test_each_check_catches_its_failure(self, extra: Any, failed: str) -> None:
        got = checks.evaluate([*_FIXED, extra], [], trip_alone=False)
        assert not got["pass"] and got[failed] == [extra["content"]]

    def test_a_restatement_held_by_its_anchors_winner_fails(self) -> None:
        """failure: the check judged the restatement to contradict the move."""
        events: list[tuple[str, dict[str, Any]]] = [
            ("UPDATE_SUPERSESSION", {"retired_id": "trip", "winner_id": "move"}),
            (
                "DEPENDENT_CLAIM_EXAMINED",
                {
                    "outcome": "unrestated",
                    "retired_id": "trip",
                    "other_id": "move",
                    "why": "the restatement contradicts move",
                    "restatement": "As of 2026-10-08, when …",
                },
            ),
        ]
        got = checks.evaluate(_FIXED, events, trip_alone=False)
        assert got["ii_held_by_winner"] == ["As of 2026-10-08, when …"] and not got["pass"]
        # Held against another claim is not this check's failure.
        events[1][1]["other_id"] = "sandeep"
        assert checks.evaluate(_FIXED, events, trip_alone=False)["ii_held_by_winner"] == []

    def test_trip_alone_needs_the_residence_active(self) -> None:
        residence = _row("The user lives in Lajpat Nagar, Delhi.", src=S4)
        assert checks.evaluate([residence], [], trip_alone=True)["pass"]
        retired = {**residence, "status": "PROVENANCE_STALE"}
        assert not checks.evaluate([retired], [], trip_alone=True)["pass"]
        # A Delhi claim from the trip note does not count as the residence.
        assert not checks.evaluate([_row("The user's home is in Delhi.")], [], trip_alone=True)[
            "pass"
        ]

    def test_a_run_missing_a_field_the_version_reads_is_not_rescorable(self) -> None:
        bare = [{"content": "x", "status": "ACTIVE", "restatement": False}]
        with pytest.raises(checks.NotRescorable, match="sources"):
            checks.evaluate(bare, [], trip_alone=False)
        with pytest.raises(KeyError):
            checks.evaluate(_FIXED, [], trip_alone=False, version=99)

    def test_a_uri_is_read_as_its_note_name(self) -> None:
        assert checks.note_name("file:///tmp/x/notes/session-11.md") == S11
        assert checks.note_name("session-04.md") == S4


# ---------------------------------------------------------------------------
# The gates, against the record
# ---------------------------------------------------------------------------


def _gate_row(
    id_: str, cls: str, expected: str, samples: list[tuple[bool | None, str | None]]
) -> dict[str, Any]:
    return {
        "id": id_,
        "class": cls,
        "expected": expected,
        "older": "The user sings in a choir.",
        "newer": "b",
        "samples": [{"contradiction": c, "slot": s} for c, s in samples],
    }


class TestScoring:
    def test_each_gate_reads_the_two_verdicts_its_own_way(self) -> None:
        cleared = {"contradiction": False, "slot": "changes"}
        confirmed = {"contradiction": True, "slot": "changes"}
        marked = "The user had been a member of a choir."
        assert scoring.gate_retires(cleared, scoring.CHANGES_ONLY)
        assert not scoring.gate_retires(cleared, scoring.BEFORE_1_187)
        assert scoring.gate_retires(confirmed, scoring.BEFORE_1_187)
        assert scoring.gate_retires(cleared, scoring.PRODUCTION, "The user sings in a choir.")
        assert not scoring.gate_retires(cleared, scoring.PRODUCTION, marked)
        assert scoring.gate_retires(confirmed, scoring.PRODUCTION, marked)
        for slot in ("different", "fixed", None):
            assert not scoring.gate_retires({"contradiction": True, "slot": slot}, "changes-only")

    def test_case_and_report_rows_are_never_barred(self) -> None:
        dated = [(False, "changes")] * 3
        rows = [
            _gate_row("p0", "dated-update", "changes", dated),
            _gate_row("c0", "coexisting", "not_changes", [(False, "different")] * 3),
            _gate_row("k0", "case", "changes", [(False, "different")] * 3),
            _gate_row("u0", "past-unmarked", "report", dated),
        ]
        bars = scoring.set_bars(rows, heldout=False)
        assert bars["bar"] and bars["regression"]
        assert set(bars["reported"]) == {"k0", "u0"}
        # One retirement in one control sample fails the bar.
        rows[1]["samples"][1] = {"contradiction": False, "slot": "changes"}
        bars = scoring.set_bars(rows, heldout=False)
        assert not bars["bar"] and bars["controls_added_by_production"] == ["c0"]
        assert not bars["regression"]

    def test_a_held_out_set_lets_one_positive_miss(self) -> None:
        rows = [
            _gate_row(f"p{i}", "dated-update", "changes", [(False, "changes")] * 3)
            for i in range(3)
        ]
        rows[0]["samples"] = [{"contradiction": False, "slot": "different"}] * 3
        assert scoring.set_bars(rows, heldout=True)["bar"]
        assert not scoring.set_bars(rows, heldout=False)["bar"]


def _record_report(model: str) -> UpdateGateReport:
    """The 2026-10-12 samples, assembled from three pre-schema files.

    A converter for this test only: ``rescore`` reads ``UpdateGateReport`` and
    nothing else.
    """
    acc = json.loads((_BENCH / "update-gate-acceptance-2026-10-12.json").read_text())
    past = json.loads((_BENCH / "update-gate-past-value-2026-10-12.json").read_text())
    fallback = json.loads((_BENCH / "update-gate-fallback-2026-10-12.json").read_text())
    rows = {
        "gate": acc["gate"][model]["gate"]["rows"],
        "gate-heldout": acc["gate"][model]["gate-heldout"]["rows"],
        "past": past["runs"][model]["bounded"]["past"]["rows"],
        "past-marker": fallback["past_marker"][model]["rows"],
    }
    from particles.benchmark.update_gate.schema import Spend

    return pairs.build_report(
        rows, model=f"anthropic:{model}", samples=3, spend=Spend(estimate_source="none")
    )


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-sonnet-4-6"])
def test_rescore_reproduces_every_recorded_bar(model: str) -> None:
    """Every fallback, proposed and shipped tally of 2026-10-12, under the new names."""
    fallback = json.loads((_BENCH / "update-gate-fallback-2026-10-12.json").read_text())
    recorded = {
        "gate": fallback["rescore"][model]["gate"],
        "gate-heldout": fallback["rescore"][model]["gate-heldout"],
        "past": fallback["rescore"][model]["past"],
        "past-marker": fallback["past_marker"][model]["bars"],
    }
    report = pairs.rescore(_record_report(model))
    for name, result in report.sets.items():
        for gate, old in scoring.ADR_0313_NAMES.items():
            got = result.bars["gates"][gate]
            want = recorded[name][old]
            assert f"{got['positives_retired']}/{got['positives']}" == want["positives_retired"]
            assert (
                f"{got['control_retirements']}/{got['control_samples']}"
                == want["false_retirements"]
            )
            assert got["ok"] is want["ok"], (model, name, gate)
        # On 4-6 every control the production gate retires, the old gate retired too.
        assert result.bars["regression"], (model, name)


def test_the_pair_report_round_trips_by_alias() -> None:
    report = _record_report("claude-sonnet-5-5")
    again = UpdateGateReport.model_validate_json(report.model_dump_json(by_alias=True))
    assert again.sets["gate"].rows[0].row_class == report.sets["gate"].rows[0].row_class


# ---------------------------------------------------------------------------
# Fixtures, the scripted table, estimates
# ---------------------------------------------------------------------------


def test_the_fixtures_resolve_packaged_copy_first(tmp_path: Path) -> None:
    # Installed, the packaged copy; from a checkout, the source tree. The same bytes.
    source = _ROOT / "tests" / "fixtures" / "trip_vs_move" / S4
    assert case_note(S4).read_bytes() == source.read_bytes()
    packaged = tmp_path / "particles" / "_artifacts" / "benchmarks" / "update_probe"
    packaged.mkdir(parents=True)
    (packaged / "gate_pairs.json").write_text("{}")
    assert pair_set("gate", package_root=tmp_path / "particles") == packaged / "gate_pairs.json"
    assert pair_set("past", package_root=tmp_path / "particles") == (
        tmp_path / "tests" / "fixtures" / "update_probe" / PAIR_SETS["past"]
    )


def test_every_extraction_entry_names_its_note_by_digest() -> None:
    """A fixture edit fails here, naming the note, instead of as a table miss."""
    data = json.loads(TABLE_PATH.read_text())
    entries = {e["note"]: e["sha256"] for e in data["extraction"]}
    assert set(entries) == set(MOVE_NOTES) | set(TRIP_ALONE_NOTES)
    for note, digest in entries.items():
        assert source_key(case_note(note).read_text()) == digest, note
        assert hashlib.sha256(case_note(note).read_bytes()).hexdigest() == digest, note
    table = ScriptedTable.load()
    assert table.reanchor and table.update and table.contradiction and table.embedding


def _pins(model: str = "claude-sonnet-5-5") -> dict[str, ProviderSelection]:
    sel = ProviderSelection(provider="anthropic", model=model)
    return dict.fromkeys(_child.all_purposes(), sel)


@pytest.fixture
def no_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The estimate as it reads before any report of record supplied a profile."""
    monkeypatch.setattr(estimate, "PROFILE_PATH", tmp_path / "absent.json")


class TestEstimate:
    def test_the_shipped_profile_covers_both_record_models(self) -> None:
        profile = estimate.load_profile()
        assert profile is not None
        assert set(profile["models"]) >= set(estimate.RECORD_RUN_USD)
        for model in estimate.RECORD_RUN_USD:
            pins = _pins(model.split(":", 1)[1])
            est = estimate.estimate_case(pins, {"move": 3, "trip-alone": 2}, mixed=False)
            assert est.source == "profile" and "approximation" not in est.note
            assert est.total_usd and est.total_calls

    @pytest.mark.usefixtures("no_profile")
    def test_the_record_prices_the_two_measured_models(self) -> None:
        est = estimate.estimate_case(_pins(), {"move": 3, "trip-alone": 2}, mixed=False)
        assert est.source == "record"
        assert est.total_usd == pytest.approx(3 * 0.1384 + 2 * 0.0757)
        assert estimate.default_ceiling(est, "move", 4.0) == pytest.approx(4 * 0.1384)

    @pytest.mark.usefixtures("no_profile")
    def test_another_priced_model_gets_the_bound(self) -> None:
        est = estimate.estimate_case(_pins("claude-opus-5-5"), {"move": 1}, mixed=False)
        assert est.source == "bound" and est.total_usd and est.total_calls
        assert estimate.default_ceiling(est, "move", 4.0) == est.run_usd("move")

    @pytest.mark.usefixtures("no_profile")
    def test_a_mixture_never_reads_the_record(self) -> None:
        pins = _pins()
        pins["extraction"] = ProviderSelection(provider="anthropic", model="claude-sonnet-4-6")
        assert estimate.estimate_case(pins, {"move": 1}, mixed=True).source == "bound"

    @pytest.mark.usefixtures("no_profile")
    def test_an_unpriced_model_has_no_dollar_figure_and_no_ceiling(self) -> None:
        get_config().llm.price_per_mtok.clear()
        est = estimate.estimate_case(_pins("unpriced-model"), {"move": 1}, mixed=False)
        assert est.total_usd is None and est.total_calls
        assert estimate.default_ceiling(est, "move", 4.0) is None
        assert "No dollar total" in estimate.render_case_estimate(est, "anthropic:unpriced-model")

    def test_a_profile_wins_and_prices_each_purpose_at_its_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = tmp_path / "usage_profile.json"
        profile.write_text(
            json.dumps(
                {
                    "models": {
                        "anthropic:claude-sonnet-5-5": {
                            "source": "test",
                            "variants": {
                                "move": [
                                    {
                                        "purpose": "extraction",
                                        "calls": 3,
                                        "input_tokens": 1_000_000,
                                    },
                                    {
                                        "purpose": "semantic_lint",
                                        "calls": 9,
                                        "output_tokens": 1_000_000,
                                    },
                                ]
                            },
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(estimate, "PROFILE_PATH", profile)
        est = estimate.estimate_case(_pins(), {"move": 1}, mixed=False)
        price = get_config().llm.price_for(_pins()["default"])
        assert price is not None
        assert est.source == "profile" and est.per_run_calls["move"] == 12
        assert est.run_usd("move") == pytest.approx(price.input + price.output)
        assert "approximation" not in est.note
        other = estimate.estimate_case(_pins("claude-sonnet-4-6"), {"move": 1}, mixed=False)
        assert other.source == "profile" and "approximation" in other.note

    def test_the_profile_takes_the_costliest_live_run_per_variant(self) -> None:
        from datetime import UTC, datetime

        from particles.benchmark.update_gate.schema import Spend, Verdict

        def run(variant: str, index: int, tokens: int) -> CaseRun:
            row = UsageRow(
                purpose="extraction",
                provider="anthropic",
                model="claude-sonnet-5-5",
                calls=1,
                input_tokens=tokens,
                cost_usd=tokens / 1e6,
            )
            return CaseRun(
                variant=variant,  # type: ignore[arg-type]
                index=index,
                passed=True,
                usage=LLMUsage(rows=[row], cost_usd=tokens / 1e6),
            )

        report = TripVsMoveReport(
            checks_version=1,
            sdk_version="test",
            run_at=datetime(2026, 10, 13, tzinfo=UTC),
            model="anthropic:claude-sonnet-5-5",
            purposes={},
            fixture=[],
            runs=[run("move", 1, 100), run("move", 2, 300), run("trip-alone", 1, 50)],
            verdict=Verdict(
                move_passed=2, move_runs=2, trip_alone_passed=1, trip_alone_runs=1, bar_met=None
            ),
            spend=Spend(estimate_source="record"),
        )
        profile = estimate.profile_from_reports(
            [report, report.model_copy(update={"arm": "scripted"})]
        )
        move = profile["models"]["anthropic:claude-sonnet-5-5"]["variants"]["move"]
        assert move[0]["input_tokens"] == 300
        assert set(profile["models"]) == {"anthropic:claude-sonnet-5-5"}


# ---------------------------------------------------------------------------
# The parent's refusals and rules
# ---------------------------------------------------------------------------


def test_a_non_empty_store_dir_is_refused_and_nothing_is_deleted(tmp_path: Path) -> None:
    keep = tmp_path / "keep.txt"
    keep.write_text("mine")
    with pytest.raises(case.CaseRefused), case.work_root(tmp_path):
        pass
    assert keep.read_text() == "mine"


def test_resolve_pins_pins_every_purpose_and_refuses_an_unknown_one() -> None:
    base, pins = case.resolve_pins(
        provider="anthropic", model="claude-sonnet-5-5", pins={"extraction": "claude-sonnet-4-6"}
    )
    assert set(pins) == set(_child.all_purposes())
    assert case.selection_key(pins["extraction"]) == "anthropic:claude-sonnet-4-6"
    assert case.selection_key(pins["semantic_lint"]) == case.selection_key(base)
    with pytest.raises(case.CaseRefused, match="unknown purpose"):
        case.resolve_pins(provider=None, model=None, pins={"nonsense": "x"})


def test_the_model_leak_rule_checks_each_row_against_its_own_pin() -> None:
    pins = _pins()
    pins["extraction"] = ProviderSelection(provider="anthropic", model="claude-sonnet-4-6")

    def usage(purpose: str, model: str) -> LLMUsage:
        return LLMUsage(rows=[UsageRow(purpose=purpose, provider="anthropic", model=model)])

    assert case.model_leak(usage("extraction", "claude-sonnet-4-6"), pins) is None
    assert case.model_leak(usage("semantic_lint", "claude-sonnet-5-5"), pins) is None
    assert "model leak" in str(case.model_leak(usage("semantic_lint", "claude-sonnet-4-6"), pins))
    assert "model leak" in str(case.model_leak(usage("unattributed", "claude-opus-5-5"), pins))


def test_the_child_environment_drops_every_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PARTICLES_SOMETHING", "x")
    monkeypatch.setenv("HTML_CHUNK_SIZE", "5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    env = case.child_env(tmp_path, arm="live")
    assert "PARTICLES_SOMETHING" not in env and "HTML_CHUNK_SIZE" not in env
    assert env["ANTHROPIC_API_KEY"] == "sk-test"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["PARTICLES_CONFIG"] == str(tmp_path / "config.yaml")
    scripted = case.child_env(tmp_path, arm="scripted")
    assert scripted["ANTHROPIC_API_KEY"] == case.SCRIPTED_KEY_PLACEHOLDER


def test_the_child_config_carries_only_the_pins_and_the_readers_wiring(tmp_path: Path) -> None:
    cfg = case.child_config(tmp_path, _pins(), same_slot_admits=False, arm="live")
    assert cfg["agent_memory"] == {"projection": {"enabled": False}}
    assert cfg["reconciliation"]["update_supersession"]["same_slot_admits"] is False
    # The ceiling never reaches consolidation.budget_usd: that budget skips
    # passes by estimate, which would change what the case measures.
    assert "consolidation" not in cfg
    assert cfg["llm"]["extraction"] == {"provider": "anthropic", "model": "claude-sonnet-5-5"}
    assert "price_per_mtok" in cfg["llm"] and "providers" in cfg["llm"]
    scripted = case.child_config(tmp_path, _pins(), same_slot_admits=True, arm="scripted")
    assert scripted["network"] == {"egress": "none"}


class TestChildPreflight:
    def _config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: Path) -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{store}")
        reset_config()
        get_config().agent_memory.projection.enabled = False

    def _pins(self) -> dict[str, str]:
        cfg = get_config().llm
        return {
            p: f"{(cfg.default if p == 'default' else cfg.for_purpose(p)).provider}:"
            f"{(cfg.default if p == 'default' else cfg.for_purpose(p)).model}"
            for p in _child.all_purposes()
        }

    def test_it_runs_on_its_own_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path, monkeypatch, tmp_path / "run" / "particles.db")
        assert (
            _child.preflight(tmp_path / "run", "sqlite+aiosqlite:///elsewhere.db", self._pins())
            is None
        )

    def test_it_refuses_the_parents_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = tmp_path / "run" / "particles.db"
        self._config(tmp_path, monkeypatch, store)
        why = _child.preflight(tmp_path / "run", f"sqlite+aiosqlite:///{store}", self._pins())
        assert why is not None and "parent" in why

    def test_it_refuses_a_store_outside_its_run_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path, monkeypatch, tmp_path / "elsewhere.db")
        why = _child.preflight(tmp_path / "run", "sqlite+aiosqlite:///other.db", self._pins())
        assert why is not None and "not a SQLite file inside" in why

    def test_it_refuses_a_purpose_the_pin_missed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path, monkeypatch, tmp_path / "run" / "particles.db")
        pins = {**self._pins(), "verification": "anthropic:some-other-model"}
        why = _child.preflight(tmp_path / "run", "sqlite+aiosqlite:///other.db", pins)
        assert why is not None and "verification" in why

    def test_it_refuses_with_the_projection_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path, monkeypatch, tmp_path / "run" / "particles.db")
        get_config().agent_memory.projection.enabled = True
        why = _child.preflight(tmp_path / "run", "sqlite+aiosqlite:///other.db", self._pins())
        assert why == "the memory projection is on"


# ---------------------------------------------------------------------------
# The scripted arm, through a real child (§ 8 item 1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scripted(tmp_path_factory: pytest.TempPathFactory) -> Generator[dict[str, Any], None, None]:
    """One scripted run of the record's protocol, beside a sentinel parent store and HOME."""
    base = tmp_path_factory.mktemp("scripted")
    sentinel = base / "parent" / "particles.db"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"the parent's store; no run may touch it")
    parent_home = base / "parent-home"
    parent_home.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{sentinel}")
        mp.setenv("HOME", str(parent_home))
        reset_config()
        _base, pins = case.resolve_pins(provider=None, model=None, pins={})
        report = case.run_case(
            base=_base,
            pins=pins,
            runs={"move": 3, "trip-alone": 2},
            arm="scripted",
            same_slot_admits=True,
            estimate=estimate.estimate_case(pins, {"move": 3}, mixed=False),
            max_usd=None,
            store_dir=base / "runs",
        )
        old_gate = case.run_case(
            base=_base,
            pins=pins,
            runs={"move": 1, "trip-alone": 0},
            arm="scripted",
            same_slot_admits=False,
            estimate=estimate.estimate_case(pins, {"move": 1}, mixed=False),
            max_usd=None,
            store_dir=None,
        )
        yield {
            "report": report,
            "old_gate": old_gate,
            "sentinel": sentinel,
            "parent_home": parent_home,
            "runs": base / "runs",
        }
    reset_config()


class TestScriptedArm:
    def test_the_bar_is_met_with_every_answer_from_the_table(
        self, scripted: dict[str, Any]
    ) -> None:
        report: TripVsMoveReport = scripted["report"]
        assert report.verdict.bar_met is True
        assert [r.passed for r in report.runs] == [True] * 5
        assert all(r.refused_calls == {} for r in report.runs)
        assert report.arm == "scripted" and report.spend.estimate_source == "none"

    def test_the_old_gate_reproduces_the_failure_the_fix_addressed(
        self, scripted: dict[str, Any]
    ) -> None:
        run: CaseRun = scripted["old_gate"].runs[0]
        assert run.passed is False and run.refused_calls == {}
        assert run.checks["iii_trip_active"] and run.checks["i_delhi_residence_or_flight"]

    def test_the_parents_store_and_home_are_untouched(self, scripted: dict[str, Any]) -> None:
        assert scripted["sentinel"].read_bytes() == b"the parent's store; no run may touch it"
        assert list(scripted["parent_home"].iterdir()) == []

    def test_each_run_kept_its_store_inside_its_own_directory(
        self, scripted: dict[str, Any]
    ) -> None:
        runs = sorted(p.name for p in scripted["runs"].iterdir())
        assert runs == ["move-1", "move-2", "move-3", "trip-alone-1", "trip-alone-2"]
        for name in runs:
            assert (scripted["runs"] / name / "particles.db").is_file()

    def test_a_saved_report_rescores_to_the_same_verdict(self, scripted: dict[str, Any]) -> None:
        report: TripVsMoveReport = scripted["report"]
        saved = TripVsMoveReport.model_validate_json(report.model_dump_json())
        again = case.rescore(saved)
        assert [r.checks for r in again.runs] == [r.checks for r in report.runs]
        assert again.verdict == report.verdict

    def test_the_events_the_checks_read_are_saved(self, scripted: dict[str, Any]) -> None:
        move: CaseRun = scripted["report"].runs[0]
        kinds = {e.event_type for e in move.events}
        assert "UPDATE_SUPERSESSION" in kinds
        assert any(r.restatement for r in move.rows)
        assert all(r.sources for r in move.rows if not r.restatement)


def test_a_run_at_its_ceiling_stops_before_its_next_step(tmp_path: Path) -> None:
    """The ceiling is read against measured spend between steps (here, before the first)."""
    _base, pins = case.resolve_pins(provider=None, model=None, pins={})
    run = case.run_one(
        tmp_path / "move-1",
        variant="move",
        index=1,
        arm="scripted",
        pins=pins,
        same_slot_admits=True,
        ceiling=0.0,
        timeout=600,
    )
    assert run.incomplete == "budget" and run.passed is None
    result = json.loads((tmp_path / "move-1" / _child.RESULT_FILE).read_text())
    assert result["exits"] == {}


# ---------------------------------------------------------------------------
# Reports of record, and the CLI's spend guard
# ---------------------------------------------------------------------------

#: The records written before the schema existed.
PRE_SCHEMA = {"trip-vs-move-end-to-end-2026-10-12.json"}


def test_every_report_of_record_validates() -> None:
    for path in sorted(_BENCH.glob("trip-vs-move-*.json")):
        if path.name not in PRE_SCHEMA:
            TripVsMoveReport.model_validate_json(path.read_text())
    for path in sorted(_BENCH.glob("update-gate-*.json")):
        data = json.loads(path.read_text())
        if data.get("suite") == "update-gate":
            UpdateGateReport.model_validate(data)


def test_a_live_run_without_yes_spends_nothing_non_interactively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from particles.api.cli import app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def boom(**kwargs: object) -> None:
        raise AssertionError("a run started")

    monkeypatch.setattr(case, "run_case", boom)
    out = CliRunner().invoke(app, ["benchmark", "trip-vs-move", "--model", "claude-sonnet-5-5"])
    assert out.exit_code == 2, out.output
    assert "Estimate for anthropic:claude-sonnet-5-5" in out.output
    out = CliRunner().invoke(app, ["benchmark", "update-gate", "--estimate"])
    assert out.exit_code == 0 and "at most" in out.output


def test_the_child_entry_is_the_surface_module() -> None:
    import importlib.util

    assert _child.ENTRY_MODULE == "particles.api.cli._trip_vs_move_child"
    assert importlib.util.find_spec(_child.ENTRY_MODULE) is not None
