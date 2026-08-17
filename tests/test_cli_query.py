"""Tests for the ``particles query`` verb (particles/api/cli/query.py).

The query operation is covered by ``tests/test_query.py``; the flag-validation
floor lives in ``tests/test_cli.py::TestQueryFlags`` and the ``--as-of`` parsing
in ``tests/test_as_of.py``. This file pins the rest of the wrapper: subject
resolution (local vs. remote), the ``--store`` federation guard, and the render
branches — the hit table, the refusal relabel, contestedness, the structural shapes, and the stderr disclosures.

The backend is patched at the module binding (``query.py`` imports
``get_backend`` at module top), so these run without a store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from particles.api.cli import app
from particles.core.schema import (
    AggregateBucket,
    AsOfNote,
    AsOfSuccessor,
    ClaimCoverage,
    Confidence,
    ContestednessReading,
    Particle,
    PredicateInfo,
    ProvenanceRef,
    ProvenanceRefType,
    QueryResponse,
    StructuralAggregate,
    StructuralGroupBy,
    Subject,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason

runner = CliRunner()

_P0 = "00000000-0000-0000-0000-00000000ab01"


def _claim(content: str = "A retrieved belief.", pid: str = _P0) -> Particle:
    return Particle(
        id=pid,
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        provenance=[ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id="ce-x")],
    )


def _response(**kwargs: Any) -> QueryResponse:
    defaults: dict[str, Any] = {
        "answer": "The answer.",
        "particles": [],
        "effective_confidences": [],
    }
    defaults.update(kwargs)
    return QueryResponse(**defaults)


@pytest.fixture
def backend() -> MagicMock:
    be = MagicMock()
    be.remote = False
    be.query = AsyncMock(return_value=_response())
    be.subject_show = AsyncMock(return_value=None)
    with patch("particles.api.cli.query.get_backend", return_value=be):
        yield be


# ---------------------------------------------------------------------------
# --subject resolution
# ---------------------------------------------------------------------------


class TestSubjectResolution:
    def test_unknown_subject_exits_one_without_querying(self, backend: MagicMock) -> None:
        result = runner.invoke(
            app, ["query", "anything?", "--subject", "nope"], catch_exceptions=False
        )
        assert result.exit_code == 1
        assert "Subject 'nope' not found." in result.output
        backend.query.assert_not_awaited()

    def test_resolved_subject_id_is_passed_to_the_query(self, backend: MagicMock) -> None:
        backend.subject_show.return_value = Subject(canonical_name="Deploys", asserted_by="test")
        result = runner.invoke(
            app, ["query", "anything?", "--subject", "Deploys"], catch_exceptions=False
        )
        assert result.exit_code == 0
        req = backend.query.await_args.args[0]
        assert req.subject_id == backend.subject_show.return_value.id

    def test_remote_backend_passes_the_subject_through_unresolved(self, backend: MagicMock) -> None:
        """graceful degradation: no local prefix/name lookup remotely."""
        backend.remote = True
        result = runner.invoke(
            app, ["query", "anything?", "--subject", "s-123"], catch_exceptions=False
        )
        assert result.exit_code == 0
        backend.subject_show.assert_not_awaited()
        assert backend.query.await_args.args[0].subject_id == "s-123"


# ---------------------------------------------------------------------------
# --store federation
# ---------------------------------------------------------------------------


class TestStoreFederation:
    def test_federation_against_a_remote_engine_is_refused(self, backend: MagicMock) -> None:
        backend.remote = True
        result = runner.invoke(
            app, ["query", "anything?", "--store", "a", "--store", "b"], catch_exceptions=False
        )
        assert result.exit_code == 1
        assert "--store federation runs locally" in result.output
        backend.query.assert_not_awaited()

    def test_local_federation_routes_to_query_federated(self, backend: MagicMock) -> None:
        federated = AsyncMock(return_value=_response(answer="Merged."))
        with patch("particles.operations.query.query_federated", new=federated):
            result = runner.invoke(
                app,
                ["query", "anything?", "--store", "work", "--store", "home"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Merged." in result.output
        assert federated.await_args.args[0] == ["work", "home"]
        backend.query.assert_not_awaited()


# ---------------------------------------------------------------------------
# Semantic-mode rendering
# ---------------------------------------------------------------------------


class TestSemanticRendering:
    def test_show_particles_prints_the_hit_table(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            particles=[_claim("Deploys happen on Fridays.")], effective_confidences=[0.72]
        )
        result = runner.invoke(
            app, ["query", "anything?", "--show-particles"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "CONF" in result.output and "EXTRACTOR" in result.output
        assert "0.90" in result.output and "0.72" in result.output
        # No publication date on this hit — the age column degrades to a dash.
        assert "—" in result.output
        assert "Deploys happen on Fridays." in result.output

    def test_age_column_is_days_since_publication(self, backend: MagicMock) -> None:
        published = datetime.now(UTC) - timedelta(days=10)
        backend.query.return_value = _response(
            particles=[_claim()],
            effective_confidences=[0.72],
            content_published_ats=[published.replace(tzinfo=None)],
        )
        result = runner.invoke(
            app, ["query", "anything?", "--show-particles"], catch_exceptions=False
        )
        assert result.exit_code == 0
        # A naive timestamp is read as UTC rather than skipped.
        assert "10d" in result.output

    def test_claim_prefiltered_semantic_query_carries_the_coverage_footer(
        self, backend: MagicMock
    ) -> None:
        """the footer rides below a *semantic* answer too."""
        backend.query.return_value = _response(
            claim_coverage=ClaimCoverage(
                active_total=100, with_claims=40, matched=3, not_normalizable_excluded=2
            ),
        )
        result = runner.invoke(
            app,
            ["query", "anything?", "--predicate", "schema:foundingDate"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "The answer." in result.output
        assert "40" in result.output

    def test_hits_are_omitted_without_show_particles(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            particles=[_claim("Deploys happen on Fridays.")], effective_confidences=[0.72]
        )
        result = runner.invoke(app, ["query", "anything?"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Deploys happen on Fridays." not in result.output
        assert "The answer." in result.output

    def test_refusal_renders_the_hits_relabelled(self, backend: MagicMock) -> None:
        """a refusal promises its nearest beliefs, so the table shows."""
        backend.query.return_value = _response(
            answer="I have no relevant knowledge.",
            particles=[_claim("Unrelated belief.")],
            effective_confidences=[0.3],
            answer_refused=True,
        )
        result = runner.invoke(app, ["query", "anything?"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Nearest beliefs — likely unrelated:" in result.output
        assert "Unrelated belief." in result.output

    def test_contestedness_prints_the_per_policy_spread(self, backend: MagicMock) -> None:
        p = _claim("Disputed belief.")
        backend.query.return_value = _response(
            particles=[p],
            effective_confidences=[0.8],
            contestedness=[
                ContestednessReading(
                    particle_id=p.id,
                    spread=0.25,
                    renderings=[
                        {"policy": "local", "effective_confidence": 0.80},
                        {"policy": "lens-a", "effective_confidence": 0.55},
                    ],
                )
            ],
        )
        result = runner.invoke(
            app, ["query", "anything?", "--contestedness"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Contestedness (spread" in result.output
        # Renderings sort by descending effective confidence.
        assert "[local:0.80, lens-a:0.55]" in result.output

    def test_contestedness_unavailable_is_disclosed(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            particles=[_claim()], effective_confidences=[0.8], contestedness=[]
        )
        result = runner.invoke(
            app, ["query", "anything?", "--contestedness"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Contestedness unavailable" in result.output
        assert "trust lens adopt" in result.output

    def test_as_of_hits_carry_their_supersession_crossing(self, backend: MagicMock) -> None:
        """one line per hit whose belief has since ended."""
        successor_id = "00000000-0000-0000-0000-00000000ab02"
        backend.query.return_value = _response(
            answer="As of then.",
            particles=[_claim("Deploys are on Fridays."), _claim("Still believed.")],
            effective_confidences=[0.9, 0.9],
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
            as_of_notes=[
                AsOfNote(
                    status=Status.SUPERSEDED,
                    status_reason=StatusReason.EXPLICIT_SUPERSESSION,
                    retired_at=datetime(2021, 6, 1, tzinfo=UTC),
                    basis="successor",
                    successor=AsOfSuccessor(
                        id=successor_id,
                        content="Deploys are on Tuesdays.",
                        asserted_at=datetime(2021, 6, 1, tzinfo=UTC),
                    ),
                ),
                None,
            ],
            as_of_excluded_undatable=2,
        )
        result = runner.invoke(
            app, ["query", "anything?", "--as-of", "2020-01-01"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "↳ Deploys are on Fridays. — now SUPERSEDED" in result.output
        assert "[basis: successor]" in result.output
        assert f"superseded by {successor_id[:8]}: Deploys are on Tuesdays." in result.output
        # The still-ACTIVE hit gets no crossing line.
        assert "↳ Still believed." not in result.output
        assert "2 retired particle(s)" in result.output

    def test_degradation_and_coverage_disclosures(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            answer="Fallback listing.",
            answer_generation_error="provider 503",
            truncation_warning="top_k cutoff may have excluded relevant particles",
            coverage_gaps=["ce-1", "ce-2"],
        )
        result = runner.invoke(app, ["query", "anything?"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Answer generation failed: provider 503" in result.output
        assert "top_k cutoff" in result.output
        assert "Coverage gap: 2 entries not yet extracted." in result.output


# ---------------------------------------------------------------------------
# structural modes
# ---------------------------------------------------------------------------


class TestStructuralRendering:
    def test_predicates_lists_the_vocabulary(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            answer="3 predicates.",
            predicate_vocabulary=[
                PredicateInfo(value="schema:author", kind=TermKind.URI, claim_count=12)
            ],
        )
        result = runner.invoke(app, ["query", "--predicates"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "PREDICATE" in result.output
        assert "schema:author" in result.output
        assert "12" in result.output

    def test_group_by_renders_the_bucket_table(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            answer="2 buckets.",
            structural_aggregate=StructuralAggregate(
                claim_count=5,
                group_by=StructuralGroupBy.SUBJECT,
                buckets=[
                    AggregateBucket(
                        key="s-1",
                        label="Deploys",
                        claim_count=3,
                        min_effective_confidence=0.4,
                        median_effective_confidence=0.6,
                        max_effective_confidence=0.9,
                    )
                ],
            ),
        )
        result = runner.invoke(app, ["query", "--group-by", "subject"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "SUBJECT" in result.output
        assert "Deploys  [s-1]" in result.output
        assert "0.40" in result.output and "0.90" in result.output

    def test_count_renders_the_confidence_distribution(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            answer="5 claims.",
            structural_aggregate=StructuralAggregate(
                claim_count=5,
                min_effective_confidence=0.4,
                median_effective_confidence=0.6,
                max_effective_confidence=0.9,
            ),
        )
        result = runner.invoke(app, ["query", "--count"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "effective confidence min 0.40 / median 0.60 / max 0.90" in result.output
        assert "5 claims." in result.output

    def test_structural_mode_discloses_a_failed_answer_generation(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            answer="Fallback listing.",
            answer_generation_error="provider 503",
            structural_aggregate=StructuralAggregate(claim_count=0),
        )
        result = runner.invoke(app, ["query", "--count"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Answer generation failed: provider 503" in result.output

    def test_structural_as_of_discloses_undatable_exclusions(self, backend: MagicMock) -> None:
        """fail-closed exclusions are never silently dropped."""
        backend.query.return_value = _response(
            answer="2 claims.",
            structural_aggregate=StructuralAggregate(claim_count=2),
            as_of=datetime(2020, 1, 1, tzinfo=UTC),
            as_of_excluded_undatable=3,
        )
        result = runner.invoke(
            app, ["query", "--count", "--as-of", "2020-01-01"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "3 retired particle(s)" in result.output
        assert "excluded from this as-of view" in result.output

    def test_claim_filter_listing_and_coverage_footer(self, backend: MagicMock) -> None:
        backend.query.return_value = _response(
            answer="1 claim.",
            particles=[_claim("Rome was founded in 753 BC.")],
            effective_confidences=[0.66],
            claim_coverage=ClaimCoverage(
                active_total=100, with_claims=40, matched=1, not_normalizable_excluded=2
            ),
        )
        result = runner.invoke(
            app, ["query", "--predicate", "schema:foundingDate"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "PREDICATE" in result.output and "OBJECT" in result.output
        assert "Rome was founded in 753 BC." in result.output
        # §2.6 footer plus the §2.2 non-normalizable disclosure.
        assert "40" in result.output
        assert "2" in result.output
