# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A session record names the artifact it is about, never a description of it.

The rewrite cases are claims the owner's store held, from Claude Code session
transcripts; the labelled-pairs test replays the rewrite over every pair the
contradiction census flagged and the owner labelled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.config import get_config
from particles.core.schema import ClaimTerm, StructuredClaim, TermKind, UncertaintyNature
from particles.extraction import general
from particles.extraction.general import CandidateParticle, GeneralExtractor
from particles.extraction.session_artifacts import (
    is_description_subject,
    name_pull_request,
    name_session_artifacts,
    pull_request_name,
)

_PAIRS = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "contradiction-probe-pairs-2026-10-07.json"
)
_PR = "https://github.com/LinkedParticles/particles-engine-py/pull"


def _cand(
    content: str,
    subjects: list[str] | None = None,
    triple_subject: str | None = None,
) -> CandidateParticle:
    candidate = CandidateParticle(
        content=content,
        confidence_value=0.9,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=subjects or [],
    )
    if triple_subject is not None:
        candidate.structured_claim = StructuredClaim(
            subject=ClaimTerm(kind=TermKind.TOKEN, value=triple_subject),
            predicate=ClaimTerm(kind=TermKind.TOKEN, value="was created at"),
            object=ClaimTerm(kind=TermKind.TOKEN, value=f"{_PR}/326"),
            structurizer_id="general-extractor",
            structurizer_version="0.19.0",
        )
    return candidate


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            f"The pull request was created at {_PR}/259.",
            f"Pull request LinkedParticles/particles-engine-py#259 was created at {_PR}/259.",
        ),
        (
            f"A pull request was opened at {_PR}/346.",
            f"Pull request LinkedParticles/particles-engine-py#346 was opened at {_PR}/346.",
        ),
        (
            f"The created pull request is located at {_PR}/337.",
            f"Pull request LinkedParticles/particles-engine-py#337 is located at {_PR}/337.",
        ),
        (
            f"The PR is located at {_PR}/388",
            f"Pull request LinkedParticles/particles-engine-py#388 is located at {_PR}/388",
        ),
        (
            f"The PR URL is {_PR}/296.",
            f"The URL of pull request LinkedParticles/particles-engine-py#296 is {_PR}/296.",
        ),
    ],
)
def test_a_described_pull_request_is_named_by_its_url(content: str, expected: str) -> None:
    rewrite = name_pull_request(content)
    assert rewrite is not None
    assert rewrite[0] == expected


@pytest.mark.parametrize(
    "content",
    [
        # Already named: the number follows the description.
        f"The PR #326 was created at {_PR}/326.",
        "The pull request number 326 was merged.",
        # Two different pull requests: which one the description means is unknown.
        f"The pull request at {_PR}/1 supersedes {_PR}/2.",
        # No URL: nothing to name it by.
        "The pull request was merged after CI passed.",
        # The description is not the claim's referent.
        f"CI failed on the pull request at {_PR}/12.",
        # A plural is not one artifact.
        f"The pull requests were opened at {_PR}/1.",
    ],
)
def test_claims_the_rule_cannot_settle_are_left_alone(content: str) -> None:
    assert name_pull_request(content) is None


def test_a_comment_fragment_is_not_part_of_the_name() -> None:
    url = f"{_PR}/801#issuecomment-6116764761"
    assert (
        pull_request_name(f"See {url} and {_PR}/801.") == "LinkedParticles/particles-engine-py#801"
    )


@pytest.mark.parametrize(
    "name",
    [
        "The pull request",
        "the assistant",
        "the commit",
        "this PR",
        "the session",
        "The repository",
        "the user",
        "`the branch`",
    ],
)
def test_bare_descriptions_are_recognised(name: str) -> None:
    assert is_description_subject(name)


@pytest.mark.parametrize(
    "name",
    [
        "The Lego Group",
        "The New Yorker",
        "the RDF extractor",
        "the main branch of LinkedParticles/particles-engine-py",
        "LinkedParticles/particles-engine-py#326",
        "the store",
        "user",
    ],
)
def test_names_and_specific_descriptions_are_kept(name: str) -> None:
    assert not is_description_subject(name)


class TestNameSessionArtifacts:
    def test_rewrites_names_subjects_and_the_triple(self) -> None:
        candidate = _cand(
            f"The pull request was created at {_PR}/326.",
            ["The pull request", "LinkedParticles/particles-engine-py"],
            triple_subject="The pull request",
        )
        out = name_session_artifacts([candidate])
        assert (out.named, out.subjects_stripped) == (1, 1)
        assert candidate.content.startswith("Pull request LinkedParticles/particles-engine-py#326 ")
        assert candidate.subjects == [
            "LinkedParticles/particles-engine-py",
            "LinkedParticles/particles-engine-py#326",
        ]
        assert candidate.structured_claim is not None
        assert candidate.structured_claim.subject.value == "LinkedParticles/particles-engine-py#326"

    def test_the_name_is_not_added_twice(self) -> None:
        candidate = _cand(
            f"A PR was created at {_PR}/336.", ["linkedparticles/particles-engine-py#336"]
        )
        name_session_artifacts([candidate])
        assert candidate.subjects == ["linkedparticles/particles-engine-py#336"]

    def test_description_subjects_go_even_on_an_unrewritten_claim(self) -> None:
        candidate = _cand("The assistant ran the unit suite.", ["the assistant", "unit suite"])
        out = name_session_artifacts([candidate])
        assert (out.named, out.subjects_stripped) == (0, 1)
        assert candidate.content == "The assistant ran the unit suite."
        assert candidate.subjects == ["unit suite"]

    def test_nothing_to_do_is_a_no_op(self) -> None:
        candidate = _cand("Particles uses SQLite.", ["Particles"])
        out = name_session_artifacts([candidate])
        assert (out.named, out.subjects_stripped) == (0, 0)
        assert candidate.content == "Particles uses SQLite."


def test_the_owner_labelled_pull_request_pairs_name_two_pull_requests() -> None:
    """The free check: replay the rewrite over the census's labelled pairs.

    Every flagged pair whose two sides open with a description of a pull
    request now names two different pull requests (79 of the 98 compatible
    pairs), and no pair the owner labelled a contradiction is rewritten.
    """
    pairs: list[dict[str, Any]] = json.loads(_PAIRS.read_text())["store_labelled"]["pairs"]
    named = 0
    for pair in pairs:
        a, b = name_pull_request(pair["a"]), name_pull_request(pair["b"])
        if pair["contradicts"]:
            assert a is None and b is None, pair["id"]
        if a is not None and b is not None:
            assert a[1] != b[1], pair["id"]
            assert not pair["contradicts"], pair["id"]
            named += 1
    assert named == 79


class TestPromptComponent:
    def test_the_rule_rides_a_session_source_only(self) -> None:
        on = {c.name for c in general._configured_prompt_components(session_source=True)}
        off = {c.name for c in general._configured_prompt_components(session_source=False)}
        assert general.PROMPT_SESSION_ARTIFACTS in on
        assert general.PROMPT_SESSION_ARTIFACTS not in off

    def test_the_rule_does_not_need_the_validity_rule(self) -> None:
        names = {
            c.name
            for c in general.extract_prompt_components(
                scope_enabled=False,
                modality_enabled=False,
                validity_enabled=False,
                session_source=True,
            )
        }
        assert general.PROMPT_SESSION_ARTIFACTS in names
        assert general.PROMPT_REFERENCE_DATE in names

    def test_every_other_prompt_is_unchanged(self) -> None:
        prompt = general._build_extract_prompt(scope_enabled=True, modality_enabled=True)
        assert "SESSION ARTIFACTS" not in prompt


class TestExtractorWiring:
    """The rewrite runs on the extractor's output for the session-state source types."""

    _SOURCE = b"user: open a PR\nassistant: done\n"

    def _reply(self) -> tuple[list[Any], list[str], bool]:
        return (
            [
                _cand(f"The pull request was created at {_PR}/259.", ["The pull request"]),
                _cand("Particles stores beliefs in SQLite.", ["Particles", "the session"]),
            ],
            [],
            False,
        )

    async def _extract(self, source_type: str) -> Any:
        snapshot = MagicMock(content_published_at=None)
        with patch("particles.extraction.general._call_llm", AsyncMock(return_value=self._reply())):
            return await GeneralExtractor().extract(snapshot, self._SOURCE, source_type=source_type)

    @pytest.mark.asyncio
    async def test_conversation_sources_are_rewritten_and_disclosed(self) -> None:
        result = await self._extract("CONVERSATION")
        first, second = result.candidates
        assert first.content.startswith("Pull request LinkedParticles/particles-engine-py#259 ")
        assert first.subjects == ["LinkedParticles/particles-engine-py#259"]
        assert second.subjects == ["Particles"]
        assert any(
            "session artifacts: 1 pull request(s) named, 2 description subject(s) stripped" in n
            for n in result.quality_notes
        )

    @pytest.mark.asyncio
    async def test_other_source_types_are_untouched(self) -> None:
        result = await self._extract("WEB_PAGE")
        assert result.candidates[0].content.startswith("The pull request was created")
        assert result.candidates[1].subjects == ["Particles", "the session"]
        assert not any("session artifacts" in n for n in result.quality_notes)

    @pytest.mark.asyncio
    async def test_an_empty_source_type_list_disables_the_rewrite(self) -> None:
        get_config().extraction.session_state_source_types = []
        result = await self._extract("CONVERSATION")
        assert result.candidates[0].content.startswith("The pull request was created")
