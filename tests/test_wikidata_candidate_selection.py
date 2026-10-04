# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Wikidata candidate selection for an unqualified name.

What's covered:
  * the pure chooser: the ``top_hit`` rule, and ``best_description``'s
    argmax, rank tie-break, unscoreable handling and floor
  * the authority under ``best_description``: the best-scoring candidate is
    adopted, a below-floor best keeps the extracted name with a low-confidence
    ref, and the resolver's abstention floor still drops the weakest
  * ``top_hit`` scores only the first candidate, as before
  * one search call per name and no alias call when nothing is adopted
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from particles.config import get_config
from particles.ingest.authorities.wikidata import UNSCOREABLE, CandidateChoice, choose_candidate

_CANDIDATES = "particles.ingest.authorities.wikidata._wikidata_candidates"
_ALIASES = "particles.ingest.authorities.wikidata._wikidata_aliases"
_SCORE = "particles.ingest.authorities.wikidata._wikidata_link_confidence"

# The live "Go" search of 2026-09-30, with the two scores that matter.
_GO = [
    {"id": "Q41587", "label": "Goiás", "description": "state in Brazil"},
    {"id": "Q37227", "label": "Go", "description": "programming language"},
    {"id": "Q12013", "label": "Google Maps", "description": "web mapping service"},
]
_SCORES = {"state in Brazil": 0.00, "programming language": 0.45, "web mapping service": 0.41}


class TestChooseCandidate:
    def test_top_hit_takes_the_first_whatever_its_score(self) -> None:
        assert choose_candidate([0.05, 0.9], 0.25, compare=False) == CandidateChoice(
            index=0, confidence=0.05, resolve=True
        )

    def test_top_hit_without_a_description_takes_the_sentinel(self) -> None:
        choice = choose_candidate([None, 0.9], 0.25, compare=False)
        assert choice == CandidateChoice(index=0, confidence=UNSCOREABLE, resolve=True)

    def test_best_description_takes_the_best_score(self) -> None:
        choice = choose_candidate([0.0, 0.45, 0.41], 0.25, compare=True)
        assert choice == CandidateChoice(index=1, confidence=0.45, resolve=True)

    def test_an_exact_tie_goes_to_the_higher_ranked(self) -> None:
        assert choose_candidate([0.3, 0.4, 0.4], 0.25, compare=True).index == 1  # type: ignore[union-attr]

    def test_an_undescribed_candidate_never_beats_a_scored_one(self) -> None:
        choice = choose_candidate([None, 0.3], 0.25, compare=True)
        assert choice == CandidateChoice(index=1, confidence=0.3, resolve=True)

    def test_nothing_scored_falls_back_to_the_top_hit(self) -> None:
        choice = choose_candidate([None, None], 0.25, compare=True)
        assert choice == CandidateChoice(index=0, confidence=UNSCOREABLE, resolve=True)

    def test_below_the_floor_is_returned_but_not_adopted(self) -> None:
        choice = choose_candidate([0.2, 0.1], 0.25, compare=True)
        assert choice == CandidateChoice(index=0, confidence=0.2, resolve=False)

    def test_no_candidates_is_no_choice(self) -> None:
        assert choose_candidate([], 0.25, compare=True) is None
        assert choose_candidate([], 0.25, compare=False) is None


def _score(description: str, content: str | None) -> float:
    return _SCORES.get(description, 0.0)


@pytest.fixture
def best_description(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_config().subjects, "wikidata_candidate_selection", "best_description")


class TestAuthority:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("best_description")
    async def test_best_description_adopts_the_best_scoring_candidate(
        self, db_session: object
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        search = AsyncMock(return_value=_GO)
        with (
            patch(_CANDIDATES, search),
            patch(_ALIASES, new_callable=AsyncMock, return_value=["Go", "golang"]),
            patch(_SCORE, side_effect=_score),
        ):
            resolved = await resolve_subject(
                db_session,  # type: ignore[arg-type]
                "Go",
                particle_content="The billing service was rewritten in Go.",
            )
        assert [(r.id, r.confidence) for r in resolved.external_ids] == [("Q37227", 0.45)]
        assert resolved.aliases == ["golang"]
        # The descriptions ride the one search response: no call per candidate.
        assert search.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("best_description")
    async def test_below_the_floor_keeps_the_name_and_records_the_ref(
        self, db_session: object
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        hits = [{"id": "Q862454", "label": "lantern", "description": "lighting device"}]
        aliases = AsyncMock(return_value=["lantern"])
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=hits),
            patch(_ALIASES, aliases),
            patch(_SCORE, return_value=0.20),  # between abstain 0.15 and floor 0.25
        ):
            resolved = await resolve_subject(
                db_session,  # type: ignore[arg-type]
                "Lantern",
                particle_content="Harbor has acquired Lantern.",
            )
        assert resolved.canonical_name == "Lantern"
        assert resolved.aliases == []
        assert resolved.description is None
        assert [(r.id, r.confidence) for r in resolved.external_ids] == [("Q862454", 0.20)]
        aliases.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("best_description")
    async def test_the_abstention_floor_still_drops_the_weakest(self, db_session: object) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        hits = [{"id": "Q1", "label": "x", "description": "y"}]
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=hits),
            patch(_SCORE, return_value=0.05),
        ):
            resolved = await resolve_subject(
                db_session,  # type: ignore[arg-type]
                "Harbor",
                particle_content="Harbor sells payroll software.",
            )
        assert resolved.canonical_name == "Harbor"
        assert resolved.external_ids == []

    @pytest.mark.asyncio
    async def test_top_hit_scores_only_the_first(
        self, db_session: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.ingest.subject_resolver import resolve_subject

        monkeypatch.setattr(get_config().subjects, "wikidata_candidate_selection", "top_hit")
        score = patch(_SCORE, side_effect=_score)
        with (
            patch(_CANDIDATES, new_callable=AsyncMock, return_value=_GO),
            patch(_ALIASES, new_callable=AsyncMock, return_value=["Goiás"]),
            score as scorer,
        ):
            resolved = await resolve_subject(
                db_session,  # type: ignore[arg-type]
                "Go",
                particle_content="The billing service was rewritten in Go.",
            )
        assert scorer.call_count == 1
        # Goiás scores 0.00, under the abstention floor: bare local, as before.
        assert resolved.canonical_name == "Go"
        assert resolved.external_ids == []
