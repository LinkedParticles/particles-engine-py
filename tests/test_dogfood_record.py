"""Unit tests for the dogfood response recorder.

The live LLM run in ``tests.dogfood.record`` is exercised manually
(``python -m tests.dogfood.record``); these tests pin the pure reconstruction
core — the inverse-templating of synthesis bodies and the judge-verdict
parsing — against synthetic recorded calls, with no network.
"""

from __future__ import annotations

from pathlib import Path

from particles.core.status import Status
from tests.dogfood import DogfoodSubject, load_corpus
from tests.dogfood.record import (
    _merge_into_existing,
    reconstruct_responses,
)


def _corpus_with_two_active() -> tuple[list[DogfoodSubject], str, str, str]:
    """Return (subjects, name, short_id0, short_id1) for a corpus subject with
    at least two ACTIVE particles.

    The returned ``subjects`` list is the *same* instance whose particle IDs are
    referenced — IDs are random per ``load_corpus()`` call, so the recorded
    calls and the reconstruction must share one load.
    """
    subjects = load_corpus()
    for ds in subjects:
        active = [p for p in ds.particles if p.status == Status.ACTIVE]
        if len(active) >= 2:
            return subjects, ds.subject.canonical_name, active[0].id[:8], active[1].id[:8]
    raise AssertionError("dogfood corpus has no subject with two active particles")


class TestReconstructResponses:
    def test_synthesis_short_ids_become_placeholders(self) -> None:
        subjects, name, s0, s1 = _corpus_with_two_active()
        synth_prompt = f"...intro...\nSUBJECT: {name}\nPARTICLE LIST: ..."
        synth_text = f"# {name}\n\nClaim A [^p-{s0}]. Claim B [^p-{s1}].\n"
        out = reconstruct_responses(subjects, [(synth_prompt, synth_text)])
        assert out[name]["synthesis"] == f"# {name}\n\nClaim A [^p-{{p0}}]. Claim B [^p-{{p1}}].\n"

    def test_judge_verdicts_are_parsed(self) -> None:
        subjects, name, s0, s1 = _corpus_with_two_active()
        judge_prompt = f"Judge these.\nPAIRS:\nparticle [{s0}] vs text\nparticle [{s1}] vs text"
        judge_text = (
            '[{"id": 0, "verdict": "supports", "reason": "r0"}, '
            '{"id": 1, "verdict": "unrelated", "reason": "r1"}]'
        )
        out = reconstruct_responses(subjects, [(judge_prompt, judge_text)])
        assert out[name]["judge_verdicts"] == [
            {"id": 0, "verdict": "supports", "reason": "r0"},
            {"id": 1, "verdict": "unrelated", "reason": "r1"},
        ]

    def test_judge_tolerates_json_fence(self) -> None:
        subjects, name, s0, _s1 = _corpus_with_two_active()
        judge_prompt = f"PAIRS:\nparticle [{s0}] vs text"
        judge_text = '```json\n[{"id": 0, "verdict": "supports", "reason": "r"}]\n```'
        out = reconstruct_responses(subjects, [(judge_prompt, judge_text)])
        assert out[name]["judge_verdicts"] == [{"id": 0, "verdict": "supports", "reason": "r"}]

    def test_unknown_subject_synthesis_is_recorded_verbatim(self) -> None:
        subjects = load_corpus()
        synth_prompt = "SUBJECT: A Subject Not In The Corpus\n"
        # No active particle ids to template against; the entry is still recorded
        # under the prompt's subject name (the body just has no substitutions).
        out = reconstruct_responses(subjects, [(synth_prompt, "# X\n\nbody")])
        assert out["A Subject Not In The Corpus"]["synthesis"] == "# X\n\nbody"

    def test_unparseable_judge_is_dropped(self) -> None:
        subjects, name, s0, _s1 = _corpus_with_two_active()
        judge_prompt = f"PAIRS:\nparticle [{s0}] vs text"
        out = reconstruct_responses(subjects, [(judge_prompt, "not json at all")])
        # Unparseable judge text leaves no judge_verdicts entry for the subject.
        assert "judge_verdicts" not in out.get(name, {})

    def test_later_synthesis_call_wins(self) -> None:
        # Layer-A / Layer-B retries re-call synthesis; the final body is kept.
        subjects, name, s0, _s1 = _corpus_with_two_active()
        prompt = f"SUBJECT: {name}\n"
        out = reconstruct_responses(
            subjects,
            [(prompt, f"first [^p-{s0}]"), (prompt, f"final [^p-{s0}]")],
        )
        assert out[name]["synthesis"] == "final [^p-{p0}]"


class TestMergeIntoExisting:
    def test_preserves_untouched_keys(self, tmp_path: Path) -> None:
        existing = tmp_path / "llm_responses.yaml"
        existing.write_text(
            "Kept:\n  synthesis: keep me\nRefreshed:\n  synthesis: old\n",
            encoding="utf-8",
        )
        merged = _merge_into_existing({"Refreshed": {"synthesis": "new"}}, existing)
        assert merged["Kept"] == {"synthesis": "keep me"}  # untouched key preserved
        assert merged["Refreshed"] == {"synthesis": "new"}  # regenerated key overwritten

    def test_missing_file_returns_regenerated_only(self, tmp_path: Path) -> None:
        merged = _merge_into_existing({"A": {"synthesis": "x"}}, tmp_path / "nope.yaml")
        assert merged == {"A": {"synthesis": "x"}}
