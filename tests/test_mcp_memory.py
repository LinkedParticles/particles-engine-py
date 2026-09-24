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
    empty_entity_notes,
    empty_entity_report,
    parse_memory_jsonl,
    preview_memory_export,
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


# An isolated observation-less entity (``Ghost``) beside one a relation rescues
# (``Acme``). The two are different losses, which is the whole point of the
# split below: before it, ``Acme`` migrated with its entityType silently blanked.
_EMPTY_ENTITY_RECORDS: list[dict] = [
    {"type": "entity", "name": "John_Smith", "entityType": "person", "observations": ["Tall"]},
    {"type": "entity", "name": "Acme", "entityType": "organization", "observations": []},
    {"type": "entity", "name": "Ghost", "entityType": "person", "observations": []},
    {"type": "relation", "from": "John_Smith", "to": "acme", "relationType": "works_at"},
]


@pytest.mark.asyncio
async def test_entity_without_observations_produces_no_candidate_but_is_disclosed() -> None:
    result = await _extract(
        _export([{"type": "entity", "name": "Ghost", "entityType": "person", "observations": []}])
    )
    assert not result.candidates
    (note,) = result.quality_notes
    assert "no observations" in note
    assert "will not migrate" in note
    # Named, not just counted: the user is entitled to know *which* nodes.
    assert "'Ghost'" in note


def test_empty_entity_report_splits_dropped_from_relation_endpoints() -> None:
    report = empty_entity_report(parse_memory_jsonl(_export(_EMPTY_ENTITY_RECORDS)))
    assert report.dropped == ["Ghost"]
    # Matched the way the subject resolver will match it: case-insensitively.
    assert report.endpoint_only == ["Acme"]


def test_empty_entity_report_is_silent_for_an_export_without_empty_entities() -> None:
    report = empty_entity_report(parse_memory_jsonl(_export()))
    assert report.dropped == [] and report.endpoint_only == []
    assert empty_entity_notes(report) == []


def test_empty_entity_notes_cap_the_named_list_and_count_the_rest() -> None:
    records = [
        {"type": "entity", "name": f"E{i:02d}", "entityType": "x", "observations": []}
        for i in range(13)
    ]
    (note,) = empty_entity_notes(empty_entity_report(parse_memory_jsonl(_export(records))))
    assert note.startswith("13 entity/entities")
    assert "'E09'" in note and "'E10'" not in note
    assert "and 3 more" in note


@pytest.mark.asyncio
async def test_quality_notes_count_only_the_entities_that_truly_vanish() -> None:
    result = await _extract(_export(_EMPTY_ENTITY_RECORDS))
    (note,) = result.quality_notes
    assert note.startswith("1 entity/entities") and "'Ghost'" in note
    # ``Acme`` migrates whole as a relation endpoint, so it is no loss to disclose.
    assert "Acme" not in note


def test_the_dry_run_and_the_import_name_the_same_lost_entities() -> None:
    """Two reports of one loss must not disagree, down to the name match."""
    content = _export(_EMPTY_ENTITY_RECORDS)
    report = empty_entity_report(parse_memory_jsonl(content))
    assert preview_memory_export(content).entities_lost == report.dropped == ["Ghost"]


@pytest.mark.asyncio
async def test_relation_candidate_carries_its_endpoints_entity_types() -> None:
    """An observation-less endpoint is named by no other candidate, so its type rides here."""
    result = await _extract(_export(_EMPTY_ENTITY_RECORDS))
    relation = result.candidates[-1]
    assert relation.subjects == ["John_Smith", "acme"]
    # Keyed by the name the candidate itself uses, which is what the pipeline zips on.
    assert relation.subject_classes == {"John_Smith": "person", "acme": "organization"}


@pytest.mark.asyncio
async def test_relation_to_an_undeclared_endpoint_claims_no_type() -> None:
    result = await _extract(
        _export([{"type": "relation", "from": "A", "to": "B", "relationType": "knows"}])
    )
    assert result.candidates[0].subject_classes == {}


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
# Unplaced fields (§4e) and the dry-run preview
# ---------------------------------------------------------------------------

_MESSY: list[dict] = [
    *_RECORDS,
    # Named by a relation below, so it survives as a bare Subject.
    {"type": "entity", "name": "Acme", "entityType": "organization", "observations": []},
    # Named by nothing, so it would not survive the import.
    {"type": "entity", "name": "Orphan", "entityType": "thing", "observations": []},
    {
        "type": "relation",
        "from": "John_Smith",
        "to": "Acme",
        "relationType": "advises",
        "createdAt": "2025-01-01",
    },
]


def test_parse_counts_fields_the_format_does_not_define() -> None:
    graph = parse_memory_jsonl(_export(_MESSY))
    assert graph.unmapped_fields == {"relation.createdAt": 1}
    assert not parse_memory_jsonl(_export()).unmapped_fields


@pytest.mark.asyncio
async def test_unplaced_fields_are_disclosed_not_mapped() -> None:
    result = await _extract(_export(_MESSY))
    assert any("relation.createdAt (1)" in note for note in result.quality_notes)
    # Disclosed, never laundered into the particle (§4e).
    assert not any("createdAt" in " ".join(c.tags or []) for c in result.candidates)


@pytest.mark.asyncio
async def test_preview_reports_exactly_what_the_extractor_produces() -> None:
    """The dry run's whole value: it runs the mapping, it does not imitate it."""
    content = _export(_MESSY)
    result = await _extract(content, deposited_by="alice")
    preview = preview_memory_export(content, deposited_by="alice", sample_size=100)

    assert preview.particles == len(result.candidates)
    assert preview.dropped == result.quality_notes
    assert [s.content for s in preview.sample] == [c.content for c in result.candidates]
    assert preview.subjects == len({n for c in result.candidates for n in c.subjects})


def test_preview_counts_the_export_in_its_own_vocabulary() -> None:
    preview = preview_memory_export(_export(_MESSY))
    assert preview.records == {"entities": 4, "observations": 3, "relations": 2}
    assert preview.single_subject_particles == 3
    assert preview.multi_subject_particles == 2
    assert preview.source_type == SOURCE_TYPE


def test_preview_separates_lost_entities_from_relation_only_ones() -> None:
    preview = preview_memory_export(_export(_MESSY))
    assert preview.entities_without_records == ["Acme", "Orphan"]
    assert preview.entities_lost == ["Orphan"]


def test_preview_of_an_unreadable_export_reports_rather_than_raising() -> None:
    preview = preview_memory_export(b"\xff\xfe not utf-8")
    assert preview.particles == 0
    assert any("not valid UTF-8" in note for note in preview.dropped)


# ---------------------------------------------------------------------------
# CLI: `particles import mcp-memory`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrated_graph_keeps_an_empty_endpoints_type_and_drops_the_isolated_one(
    db_session, tmp_path, no_embedding_model
) -> None:
    """End to end: deposit → extract → the façade's ``read_graph``.

    This is the claim a migrating user can check for themselves, so it is
    pinned at the surface they will check it through rather than only at the
    candidate list.
    """
    from particles.core.schema import FetchPolicy, Mutability
    from particles.corpus.deposit import deposit_file
    from particles.ingest.pipeline import extract_snapshot
    from particles.mcp.memory_compat import ops

    export = tmp_path / "memory.jsonl"
    export.write_bytes(_export(_EMPTY_ENTITY_RECORDS))
    entry_id, snapshot_id = await deposit_file(
        db_session,
        export,
        deposited_by="test",
        mutability=Mutability.STABLE,
        fetch_policy=FetchPolicy.NEVER,
        source_type=SOURCE_TYPE,
    )
    await db_session.commit()
    await extract_snapshot(db_session, entry_id, snapshot_id)
    await db_session.commit()

    graph, _notes = await ops.read_graph(db_session)
    entities = {e["name"].lower(): e for e in graph["entities"]}
    # The observation-less relation endpoint survives, type intact.
    assert entities["acme"]["entityType"] == "organization"
    assert entities["acme"]["observations"] == []
    # The isolated one does not — the accepted, disclosed loss.
    assert "ghost" not in entities
    assert graph["relations"] == [{"from": "John_Smith", "to": "acme", "relationType": "works_at"}]


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

    def test_output_names_the_entities_that_will_not_migrate(self, tmp_path, cli_db) -> None:
        """The loss is announced at the door, before anything is extracted."""
        from typer.testing import CliRunner

        from particles.api.cli import app

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export(_EMPTY_ENTITY_RECORDS))

        result = CliRunner().invoke(
            app, ["import", "mcp-memory", str(export)], catch_exceptions=False
        )
        output = " ".join(_strip_ansi(result.output).split())
        assert result.exit_code == 0
        assert "will not migrate" in output
        assert "'Ghost'" in output
        assert "create_entities" in output

    def test_output_says_nothing_about_empty_entities_when_there_are_none(
        self, tmp_path, cli_db
    ) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())

        result = CliRunner().invoke(
            app, ["import", "mcp-memory", str(export)], catch_exceptions=False
        )
        assert "will not migrate" not in _strip_ansi(result.output)

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


class TestImportMcpMemoryDryRun:
    """``--dry-run``: a report, and a guarantee that nothing was touched."""

    def _invoke(self, export, *flags: str):
        from typer.testing import CliRunner

        from particles.api.cli import app

        return CliRunner().invoke(
            app, ["import", "mcp-memory", str(export), *flags], catch_exceptions=False
        )

    def test_touches_neither_the_store_nor_the_corpus(self, tmp_path, monkeypatch) -> None:
        """No ``cli_db``: the database is never created, so it must never be needed."""
        from particles.api.cli import import_vault
        from particles.config import reset_config

        db_path = tmp_path / "never.db"
        blob_dir = tmp_path / "blobs"
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
        monkeypatch.setenv("PARTICLES_BLOB_DIR", str(blob_dir))
        reset_config()

        def _forbidden(*args: object, **kwargs: object):
            raise AssertionError("--dry-run reached the store")

        monkeypatch.setattr(import_vault, "session_scope", _forbidden)
        monkeypatch.setattr(import_vault, "_import_mcp_memory", _forbidden)
        monkeypatch.setattr("particles.corpus.deposit.deposit_file", _forbidden)
        monkeypatch.setattr("particles.db.get_engine", _forbidden)

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export(_MESSY))
        before = export.read_bytes()

        result = self._invoke(export, "--dry-run")

        assert result.exit_code == 0, result.output
        assert not db_path.exists()
        assert not blob_dir.exists()
        assert export.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["memory.jsonl"]

    def test_a_dry_run_leaves_an_existing_store_empty(self, tmp_path, cli_db) -> None:
        import asyncio

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())
        assert self._invoke(export, "--dry-run").exit_code == 0

        async def _counts() -> tuple[int, int]:
            from sqlalchemy import func, select

            from particles.corpus.store import list_entries
            from particles.db import session_scope
            from particles.store.particle_store import ParticleRow

            async with session_scope() as session:
                particles = await session.scalar(select(func.count()).select_from(ParticleRow))
                return len(await list_entries(session)), int(particles or 0)

        assert asyncio.run(_counts()) == (0, 0)

    def test_report_names_counts_losses_and_a_sample(self, tmp_path) -> None:
        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export(_MESSY) + b"\nnot json")
        output = _strip_ansi(self._invoke(export, "--dry-run").output)

        assert "nothing was deposited or written" in output
        assert "4  entities" in output
        assert "5  particles" in output
        assert "Entities with no observations: 2" in output
        assert "would NOT survive the import: Orphan" in output
        assert "not valid JSON" in output
        assert "relation.createdAt" in output
        assert "Speaks fluent Spanish" in output

    def test_json_report_is_machine_readable(self, tmp_path) -> None:
        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export(_MESSY))
        result = self._invoke(export, "--dry-run", "--json", "--sample", "1")

        payload = json.loads(result.output)
        assert payload["particles"] == 5
        assert payload["entities_lost"] == ["Orphan"]
        assert payload["store_consulted"] is False
        assert len(payload["sample"]) == 1

    def test_json_without_dry_run_is_a_usage_error(self, tmp_path, cli_db) -> None:
        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())
        result = self._invoke(export, "--json")
        assert result.exit_code == 2
        assert "--dry-run" in result.output

    def test_an_export_that_maps_to_nothing_exits_nonzero(self, tmp_path) -> None:
        export = tmp_path / "memory.jsonl"
        export.write_text("not json\n")
        result = self._invoke(export, "--dry-run")
        assert result.exit_code == 1
        assert "Nothing in this export would become a particle" in result.output

    def test_negative_sample_is_rejected(self, tmp_path) -> None:
        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())
        assert self._invoke(export, "--dry-run", "--sample", "-1").exit_code == 2

    def test_works_in_remote_mode_because_it_reaches_no_store(self, tmp_path, monkeypatch) -> None:
        from particles.config import reset_config

        export = tmp_path / "memory.jsonl"
        export.write_bytes(_export())
        monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://127.0.0.1:8099")
        reset_config()
        try:
            assert self._invoke(export, "--dry-run").exit_code == 0
        finally:
            monkeypatch.delenv("PARTICLES_ENGINE_BASE_URL", raising=False)
            reset_config()
