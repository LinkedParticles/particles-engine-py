# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reference memory-server migration extractor.

The parsing half is deterministic — fixed export bytes in, fixed candidate list
out — which is exactly the "extractor parsing" category ``tests/AGENTS.md``
marks required. The rest of these pin the **second-hand attribution rule**
(§4), which is the ADR's load-bearing invariant: a regression here would ship
manufactured provenance, which is worse than shipping nothing.

Reference fixture names (``John_Smith`` / ``Anthropic`` / ``works_at``) are the
reference server's own README example, reproduced with attribution
(the convention set); ``@modelcontextprotocol/server-memory`` is
MIT licensed.
"""

from __future__ import annotations

import json

import pytest

from particles.core.schema import Snapshot
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.general import candidate_to_particle
from particles.extraction.mcp_memory import (
    DEFAULT_TRUST_WEIGHT,
    EXTERNAL_REF_NAMESPACE,
    IMPORT_TAG,
    OBSERVATION_TAG,
    RELATION_TAG,
    SOURCE_TYPE,
    McpMemoryExtractor,
    parse_memory_jsonl,
    relation_content,
    relation_from_particle,
    relation_tags,
)

# ---------------------------------------------------------------------------
# Fixtures — the reference server's on-disk JSONL shape
# ---------------------------------------------------------------------------

_RECORDS: list[dict] = [
    {
        "type": "entity",
        "name": "John_Smith",
        "entityType": "person",
        "observations": ["Speaks fluent Spanish", "Graduated in 2019"],
    },
    {
        "type": "entity",
        "name": "Anthropic",
        "entityType": "organization",
        "observations": ["Builds Claude"],
    },
    {"type": "relation", "from": "John_Smith", "to": "Anthropic", "relationType": "works_at"},
]


def _export(records: list[dict] | None = None) -> bytes:
    return "\n".join(json.dumps(r) for r in (records if records is not None else _RECORDS)).encode()


async def _extract(content: bytes, **kwargs: object):
    return await McpMemoryExtractor().extract(
        Snapshot(entry_id="e", content_hash="h"), content, **kwargs
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_reads_entities_relations_and_line_numbers() -> None:
    graph = parse_memory_jsonl(_export())

    assert [e.name for e in graph.entities] == ["John_Smith", "Anthropic"]
    assert [e.entity_type for e in graph.entities] == ["person", "organization"]
    assert graph.entities[0].observations == ["Speaks fluent Spanish", "Graduated in 2019"]
    # Line numbers are 1-based and are what ProvenanceRef.location cites (§4b).
    assert [e.line for e in graph.entities] == [1, 2]
    assert graph.relations[0].line == 3
    assert not graph.notes


def test_parse_skips_bad_lines_without_losing_the_rest() -> None:
    content = b"\n".join(
        [
            json.dumps(_RECORDS[0]).encode(),
            b"this is not json",
            json.dumps({"type": "wat"}).encode(),
            b"[1, 2, 3]",
            json.dumps(_RECORDS[2]).encode(),
        ]
    )
    graph = parse_memory_jsonl(content)

    # A single bad line must never cost the user their migration.
    assert len(graph.entities) == 1
    assert len(graph.relations) == 1
    # ...and nothing is dropped silently: every skip is reported, with its line.
    assert len(graph.notes) == 3
    assert all(note.startswith("Line ") for note in graph.notes)


def test_parse_drops_non_string_observations_and_says_so() -> None:
    graph = parse_memory_jsonl(
        _export(
            [
                {
                    "type": "entity",
                    "name": "X",
                    "entityType": "t",
                    "observations": ["ok", 7, "", None],
                }
            ]
        )
    )
    assert graph.entities[0].observations == ["ok"]
    assert "dropped 3" in graph.notes[0]


def test_parse_skips_relation_missing_an_endpoint() -> None:
    graph = parse_memory_jsonl(_export([{"type": "relation", "from": "A", "relationType": "r"}]))
    assert not graph.relations
    assert "missing an endpoint" in graph.notes[0]


def test_parse_rejects_non_utf8_without_raising() -> None:
    graph = parse_memory_jsonl(b"\xff\xfe not utf-8")
    assert not graph.entities and not graph.relations
    assert "not valid UTF-8" in graph.notes[0]


def test_parse_tolerates_blank_lines_and_trailing_newline() -> None:
    graph = parse_memory_jsonl(_export() + b"\n\n")
    assert len(graph.entities) == 2
    assert not graph.notes


# ---------------------------------------------------------------------------
# Candidate production
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_candidate_per_observation_plus_one_per_relation() -> None:
    result = await _extract(_export())
    # 3 observations + 1 relation. Observations are asserted verbatim — never
    # re-paraphrased, because they are already claim-granular (§ Context).
    assert [c.content for c in result.candidates] == [
        "Speaks fluent Spanish",
        "Graduated in 2019",
        "Builds Claude",
        "John_Smith works_at Anthropic",
    ]


@pytest.mark.asyncio
async def test_every_candidate_is_imported_at_the_configured_floor() -> None:
    from particles.config import get_config

    result = await _extract(_export())
    floor = get_config().migration.import_confidence

    for candidate in result.candidates:
        # §5: the number is ours, flat, and honestly labelled. An incumbent
        # score must never reach confidence.value.
        assert candidate.calibration_source is CalibrationSource.IMPORTED
        assert candidate.confidence_value == floor


@pytest.mark.asyncio
async def test_provenance_location_cites_the_record_position() -> None:
    result = await _extract(_export())
    locations = [c.provenance_location for c in result.candidates]
    assert locations == [
        "line 1 observation 0",
        "line 1 observation 1",
        "line 2 observation 0",
        "line 3",
    ]


@pytest.mark.asyncio
async def test_importer_contributor_is_recorded_from_the_depositor() -> None:
    result = await _extract(_export(), deposited_by="jeff")
    for candidate in result.candidates:
        assert candidate.contributors is not None
        (contributor,) = candidate.contributors
        # §4c: the attributed act, using the role the catalogue
        # already reserves. The id is platform-scoped (§6.5).
        assert contributor.role == "importer"
        assert contributor.id == "import:jeff"


@pytest.mark.asyncio
async def test_contributor_id_keeps_an_already_scoped_actor() -> None:
    result = await _extract(_export(), deposited_by="github:torvalds")
    assert result.candidates[0].contributors[0].id == "github:torvalds"  # type: ignore[index]


@pytest.mark.asyncio
async def test_entity_identity_becomes_an_external_ref_in_its_own_namespace() -> None:
    result = await _extract(_export())
    ref = result.candidates[0].external_refs["John_Smith"]
    # §7b: the reference model has no entity ids — the name *is* the identity —
    # so this is what lets a second export re-attach instead of forking.
    assert (ref.namespace, ref.id) == (EXTERNAL_REF_NAMESPACE, "John_Smith")


@pytest.mark.asyncio
async def test_entity_type_becomes_the_subject_class() -> None:
    result = await _extract(_export())
    assert result.candidates[0].subject_classes == {"John_Smith": "person"}
    assert result.candidates[2].subject_classes == {"Anthropic": "organization"}


@pytest.mark.asyncio
async def test_relation_is_a_two_subject_candidate_carrying_the_triple() -> None:
    result = await _extract(_export())
    relation = result.candidates[-1]

    assert relation.subjects == ["John_Smith", "Anthropic"]
    assert set(relation.external_refs) == {"John_Smith", "Anthropic"}
    # The triple must survive losslessly: particle_subjects has no ordering
    # column, so direction rides the reserved tags.
    assert RELATION_TAG in (relation.tags or [])
    assert set(relation_tags("John_Smith", "Anthropic", "works_at")) <= set(relation.tags or [])


@pytest.mark.asyncio
async def test_every_candidate_carries_the_import_tag() -> None:
    result = await _extract(_export())
    assert all(IMPORT_TAG in (c.tags or []) for c in result.candidates)


@pytest.mark.asyncio
async def test_observation_candidates_carry_the_facade_observation_tag() -> None:
    result = await _extract(_export())
    for candidate in result.candidates[:3]:
        # This is what makes a migrated record visible through
        # `particles memory serve` — one encoding, not two (§9).
        assert OBSERVATION_TAG in (candidate.tags or [])


@pytest.mark.asyncio
async def test_entity_without_observations_produces_no_candidate_but_is_disclosed() -> None:
    result = await _extract(
        _export([{"type": "entity", "name": "Ghost", "entityType": "person", "observations": []}])
    )
    assert not result.candidates
    assert any("no observations" in note for note in result.quality_notes)


@pytest.mark.asyncio
async def test_empty_export_reports_rather_than_raising() -> None:
    result = await _extract(b"")
    assert not result.candidates
    assert result.quality_notes


@pytest.mark.asyncio
async def test_parse_notes_reach_the_extraction_quality_notes() -> None:
    result = await _extract(_export() + b"\nnot json\n")
    assert any("not valid JSON" in note for note in result.quality_notes)


# ---------------------------------------------------------------------------
# The relation tag codec round-trips (shared with the façade)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "dst", "rel"),
    [
        ("John_Smith", "Anthropic", "works_at"),
        # Reserved characters must survive as single tag tokens.
        ("A B", "C=D", "rel with spaces"),
        ("emoji 🙂", "tab\tsep", "a:b=c"),
    ],
)
def test_relation_tags_round_trip(src: str, dst: str, rel: str) -> None:
    class _P:
        id = "p-1"
        tags = relation_tags(src, dst, rel)

    assert relation_from_particle(_P()) == {"from": src, "to": dst, "relationType": rel}


def test_relation_from_particle_returns_none_when_a_tag_is_missing() -> None:
    class _P:
        id = "p-1"
        tags = [RELATION_TAG]

    # A damaged record degrades to omission, never to a malformed edge.
    assert relation_from_particle(_P()) is None


def test_relation_content_is_the_active_voice_the_reference_asks_for() -> None:
    assert (
        relation_content("John_Smith", "Anthropic", "works_at") == "John_Smith works_at Anthropic"
    )


def test_the_facade_reads_the_vocabulary_from_here() -> None:
    """One encoding, not two — the façade must not fork its own copy (§9)."""
    from particles.mcp.memory_compat import graph

    assert graph.OBSERVATION_TAG is OBSERVATION_TAG
    assert graph.RELATION_TAG is RELATION_TAG
    assert graph.relation_tags is relation_tags


# ---------------------------------------------------------------------------
# The attribution rule survives the conversion to a Particle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_to_particle_carries_the_whole_attribution_rule() -> None:
    result = await _extract(_export(), deposited_by="jeff")
    particle = candidate_to_particle(
        result.candidates[0], "entry-1", "snap-1", "mcp-memory-extractor", subject_ids=["s-1"]
    )

    (provenance,) = particle.provenance
    # §4a/§4b: provenance names the artifact the store actually holds, at the
    # record's position — never a synthesised reference into the source store.
    assert provenance.corpus_entry_id == "entry-1"
    assert provenance.snapshot_id == "snap-1"
    assert provenance.location == "line 1 observation 0"
    assert particle.confidence.calibration_source is CalibrationSource.IMPORTED
    assert particle.contributors is not None and particle.contributors[0].role == "importer"
    assert IMPORT_TAG in (particle.tags or [])


@pytest.mark.asyncio
async def test_a_declared_calibration_source_suppresses_temperature_scaling() -> None:
    """A flat import floor is not a logit; scaling it would be meaningless."""
    from datetime import UTC, datetime

    from particles.core.schema import ExtractorCalibration

    result = await _extract(_export())
    candidate = result.candidates[0]
    calibration = ExtractorCalibration(
        extractor_id="mcp-memory-extractor",
        temperature=2.0,
        transform="logit",
        fitted_at=datetime.now(UTC),
        benchmark_suite_id="suite-1",
        sample_size=100,
        calibration_error_before=0.2,
        calibration_error_after=0.05,
    )

    particle = candidate_to_particle(
        candidate, "entry-1", "snap-1", "mcp-memory-extractor", calibration=calibration
    )

    assert particle.confidence.value == candidate.confidence_value
    assert particle.confidence.calibration_source is CalibrationSource.IMPORTED
    assert particle.confidence.calibration_method is None


def test_llm_extractor_candidates_are_unaffected_by_the_new_fields() -> None:
    """Every pre-0258 producer leaves the four fields at their defaults."""
    from particles.core.schema import UncertaintyNature
    from particles.extraction.general import CandidateParticle

    candidate = CandidateParticle(
        content="c", confidence_value=0.8, uncertainty_nature=UncertaintyNature.EPISTEMIC
    )
    particle = candidate_to_particle(candidate, "entry-1", "snap-1")

    assert particle.confidence.calibration_source is CalibrationSource.EXTRACTOR_DIRECT
    assert particle.contributors is None
    assert particle.tags is None
    assert particle.provenance[0].location is None


# ---------------------------------------------------------------------------
# Registration and routing
# ---------------------------------------------------------------------------


def test_extractor_is_registered_ahead_of_the_general_fallback() -> None:
    from particles.extraction.registry import get_extractors

    accepting = [x.EXTRACTOR_ID for x in get_extractors() if x.accepts(SOURCE_TYPE)]
    assert accepting[0] == "mcp-memory-extractor"


def test_extractor_accepts_only_its_own_source_type() -> None:
    extractor = McpMemoryExtractor()
    assert extractor.accepts(SOURCE_TYPE)
    assert not extractor.accepts("WEB_PAGE")
    assert not extractor.accepts("CONVERSATION")


def test_trust_weight_sits_below_the_llm_extractors() -> None:
    """Second-hand by construction: a faithful translation of unverifiable claims."""
    from particles.extraction.general import DEFAULT_TRUST_WEIGHT as GENERAL_TRUST

    assert DEFAULT_TRUST_WEIGHT < GENERAL_TRUST


def test_live_authorities_are_skipped_for_the_export_source_type() -> None:
    """§7a: a bulk import must not rewrite canonical names the user never chose."""
    from particles.config import get_config

    assert SOURCE_TYPE in get_config().subjects.skip_live_authorities_source_types


# ---------------------------------------------------------------------------
# CLI: `particles import mcp-memory`
# ---------------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestImportMcpMemoryCli:
    """``particles import mcp-memory`` shape: routing, disclosure, error paths."""

    def test_help_lists_the_subcommand(self) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["import", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "mcp-memory" in _strip_ansi(result.output)

    def test_deposits_the_export_under_its_own_source_type(self, tmp_path, cli_db) -> None:
        import asyncio

        from typer.testing import CliRunner

        from particles.api.cli import app

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())

        result = CliRunner().invoke(
            app, ["import", "mcp-memory", str(export)], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "Deposited" in result.output

        async def _entry() -> tuple[str, str, str]:
            from particles.corpus.store import list_entries
            from particles.db import session_scope

            async with session_scope() as session:
                (entry,) = await list_entries(session)
                return entry.source_type, entry.mutability, entry.fetch_policy

        source_type, mutability, fetch_policy = asyncio.run(_entry())
        assert source_type == SOURCE_TYPE
        # §3: a dump is a record of what was seen, not a live handle.
        assert mutability == "STABLE"
        assert fetch_policy == "NEVER"

    def test_output_discloses_that_migrated_beliefs_are_second_hand(self, tmp_path, cli_db) -> None:
        """§ Consequences: the surprise must be stated, not discovered in a ranking."""
        from typer.testing import CliRunner

        from particles.api.cli import app

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())

        result = CliRunner().invoke(
            app, ["import", "mcp-memory", str(export)], catch_exceptions=False
        )
        output = _strip_ansi(result.output)
        assert "second-hand" in output
        assert "IMPORTED" in output
        assert "particles trust set" in output

    def test_missing_file_exits_nonzero(self, tmp_path, cli_db) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["import", "mcp-memory", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0

    def test_directory_argument_is_rejected(self, tmp_path, cli_db) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(app, ["import", "mcp-memory", str(tmp_path)])
        assert result.exit_code != 0

    def test_refuses_in_remote_mode(self, tmp_path, monkeypatch) -> None:
        """The verb walks the client's filesystem, which a remote engine cannot read."""
        from typer.testing import CliRunner

        from particles.api.cli import app
        from particles.config import reset_config

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())

        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8099")
        reset_config()
        try:
            result = CliRunner().invoke(app, ["import", "mcp-memory", str(export)])
            assert result.exit_code != 0
        finally:
            monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
            reset_config()
