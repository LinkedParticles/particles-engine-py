# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""An agent session's momentary working state is never extracted as a belief.

The positive cases are claims the live store actually held, extracted from
Claude Code session transcripts; the negative cases are durable claims about
worktrees that the filter must leave alone.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.config import get_config
from particles.core.schema import RelationType, UncertaintyNature
from particles.extraction.general import CandidateParticle, GeneralExtractor
from particles.extraction.session_state import (
    drop_session_state,
    is_session_state,
    worktree_names,
)

_WT = "/Users/jeff/src/particles-engine-py/.claude/worktrees"


def _cand(content: str, subjects: list[str] | None = None) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=subjects or [],
    )


@pytest.mark.parametrize(
    "content",
    [
        f"The session now operates on a fresh worktree at {_WT}/frosty-cray-4f8f06.",
        "The session now operates on the origin repository at /Users/jeff/src/particles-engine-py.",
        "The session operates on the branch `claude/adr-0218` checked out from `origin/main`.",
        f"The working directory for this session is {_WT}/unruffled-keller-25fe5b.",
        "The working directory is '/Users/jeff/src/particles-engine-py'.",
        "The current branch is spent (no further work will be done on it).",
        "The repo uses a branch naming convention and the current branch is 'claude/x-d77543'.",
        f"The file `{_WT}/tender-noyce-db0054/particles/secrets.py` was edited.",
        f"The git worktree at {_WT}/naughty-lewin-379fcd was recycled.",
        "The repository uses a worktree at .claude/worktrees/sleepy-mccarthy-cd1a57.",
    ],
)
def test_session_state_claims_are_recognised(content: str) -> None:
    assert is_session_state(content)


@pytest.mark.parametrize(
    "content",
    [
        "Git worktrees in this project are placed under `.claude/worktrees/<name>`.",
        "When the session runs in a git worktree, spawned subagents do not inherit the cwd.",
        "The default store resolves to a CWD-relative ./particles.db.",
        "The `guard-worktree-branch.py` hook keys off the session cwd.",
        "The main checkout at `/Users/jeff/src/particles-engine-py` is on branch `main`.",
        "PR #218 is on branch `claude/reverent-knuth-be52f0`.",
        "The tool-turn rule applies to CONVERSATION sources.",
    ],
)
def test_durable_claims_are_kept(content: str) -> None:
    assert not is_session_state(content)


def test_worktree_names_are_read_from_concrete_paths_only() -> None:
    text = (
        f"cd {_WT}/frosty-cray-4f8f06/particles && ls\n"
        "worktrees live under .claude/worktrees/<name>.\n"
        "moved to .claude/worktrees/Confident-Hawking-b0fea4."
    )
    assert worktree_names(text) == {"frosty-cray-4f8f06", "confident-hawking-b0fea4"}


class TestDropSessionState:
    def test_drops_in_order_and_strips_worktree_subjects(self) -> None:
        source = f"switching to {_WT}/frosty-cray-4f8f06 now"
        cands = [
            _cand("Particles uses SQLite.", ["Particles"]),
            _cand(f"The session now operates on {_WT}/frosty-cray-4f8f06.", ["frosty-cray-4f8f06"]),
            _cand("The fix landed in PR #631.", ["frosty-cray-4f8f06", "PR #631"]),
            _cand("A path subject.", [f"{_WT}/other-name-123abc/"]),
        ]
        out = drop_session_state(cands, source)
        assert [c.content for c in out.candidates] == [
            "Particles uses SQLite.",
            "The fix landed in PR #631.",
            "A path subject.",
        ]
        assert out.dropped == 1
        assert out.subjects_stripped == 2
        assert out.candidates[1].subjects == ["PR #631"]
        assert out.candidates[2].subjects == []

    def test_stance_indices_follow_the_surviving_list(self) -> None:
        target = _cand("The migration is safe.")
        stance = _cand("The operator agrees the migration is safe.")
        stance.stance_kind = RelationType.ENDORSES
        stance.stance_target_index = 2
        cands = [_cand("The current branch is main."), stance, target]
        out = drop_session_state(cands, "")
        assert out.candidates == [stance, target]
        assert stance.stance_target_index == 1
        assert stance.stance_kind is RelationType.ENDORSES

    def test_a_stance_whose_target_was_dropped_becomes_a_plain_claim(self) -> None:
        stance = _cand("I disagree.")
        stance.stance_kind = RelationType.DISPUTES
        stance.stance_target_index = 0
        stance.stance_magnitude = 0.8
        out = drop_session_state([_cand("The working directory is /tmp/x."), stance], "")
        assert out.candidates == [stance]
        assert stance.stance_kind is None
        assert stance.stance_target_index is None
        assert stance.stance_magnitude is None

    def test_nothing_to_drop_is_a_no_op(self) -> None:
        cands = [_cand("Particles uses SQLite.", ["Particles"])]
        out = drop_session_state(cands, "no worktrees here")
        assert out.candidates == cands
        assert (out.dropped, out.subjects_stripped) == (0, 0)


class TestExtractorWiring:
    """The filter runs on the extractor's output for configured source types."""

    _SOURCE = f"user: go\nassistant: now in {_WT}/frosty-cray-4f8f06\n".encode()

    def _reply(self) -> tuple[list[Any], list[str], bool]:
        return (
            [
                _cand(f"The session now operates on {_WT}/frosty-cray-4f8f06."),
                _cand("Particles stores beliefs in SQLite.", ["Particles", "frosty-cray-4f8f06"]),
            ],
            [],
            False,
        )

    async def _extract(self, source_type: str) -> Any:
        snapshot = MagicMock(content_published_at=None)
        with patch("particles.extraction.general._call_llm", AsyncMock(return_value=self._reply())):
            return await GeneralExtractor().extract(snapshot, self._SOURCE, source_type=source_type)

    @pytest.mark.asyncio
    async def test_conversation_sources_are_filtered_and_disclosed(self) -> None:
        result = await self._extract("CONVERSATION")
        assert [c.content for c in result.candidates] == ["Particles stores beliefs in SQLite."]
        assert result.candidates[0].subjects == ["Particles"]
        assert any("session state: 1 claim(s) dropped" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_other_source_types_are_untouched(self) -> None:
        result = await self._extract("WEB_PAGE")
        assert len(result.candidates) == 2
        assert not any("session state" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_an_empty_source_type_list_disables_the_filter(self) -> None:
        get_config().extraction.session_state_source_types = []
        result = await self._extract("CONVERSATION")
        assert len(result.candidates) == 2
