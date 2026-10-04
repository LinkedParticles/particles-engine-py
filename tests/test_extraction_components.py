# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The extraction component table and its per-extraction record.

Two halves. The **table**: every section of the general and journal prompts is
covered by exactly one named component, the prompt is the concatenation of
those components and nothing else, and a change to one section's text changes
that component's digest only. The **record**: the tally an extraction fills,
how two records merge, and what one extraction actually records.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from particles.config import get_config
from particles.extraction import general, journal
from particles.extraction.components import (
    MIXED_DIGEST,
    ComponentRecord,
    ComponentTable,
    ComponentTally,
    PromptComponent,
    assemble_prompt,
    component_digest,
    declare_components,
    record_component,
    tally_components,
)
from particles.extraction.subject_gate import gate_digest
from particles.extraction.tool_turns import TOOL_TURN_RULE
from particles.operations.reindex_scope import component_change_reasons

_FLAGS = (
    "scope_enabled",
    "modality_enabled",
    "polarity_enabled",
    "stance_enabled",
    "validity_enabled",
    "structure_enabled",
    "tool_turns_present",
    "context_present",
)


def _all_flag_combinations() -> list[dict[str, bool]]:
    return [
        dict(zip(_FLAGS, combo, strict=True))
        for combo in itertools.product([False, True], repeat=len(_FLAGS))
    ]


def _general_section_texts() -> dict[str, str]:
    """Every text constant the general prompt is built from, by constant name.

    Found by name, so a new ``_FOO_RULE`` / ``_FOO_SCHEMA_FIELD`` added to the
    module joins the check without this test being edited.
    """
    sections = {
        name: value
        for name, value in vars(general).items()
        if isinstance(value, str)
        # PROMPT_* are the component names, not prompt text.
        and not name.startswith("PROMPT_")
        and (
            name.endswith(("_RULE", "_RULES", "_SCHEMA_FIELD"))
            or name in {"_EXTRACT_SCHEMA_HEAD", "_EXTRACT_SCHEMA_TAIL"}
        )
    }
    sections["TOOL_TURN_RULE"] = TOOL_TURN_RULE
    return sections


class TestGeneralPromptTable:
    def test_every_prompt_section_is_covered_by_exactly_one_component(self) -> None:
        """Each section constant is one component's rule or field text, never two.

        Over every flag combination, so the conditional sections (tool turns,
        append context, each config-gated rule) are all reached.
        """
        seen: dict[str, PromptComponent] = {}
        for flags in _all_flag_combinations():
            for component in general.extract_prompt_components(**flags):
                previous = seen.setdefault(component.name, component)
                assert previous == component, f"{component.name} changed text across flags"
        pieces = [(c.name, text) for c in seen.values() for text in (c.rule, c.field) if text]
        for constant, text in _general_section_texts().items():
            owners = [name for name, piece in pieces if piece == text]
            assert len(owners) == 1, f"{constant} is covered by {owners or 'no component'}"
        # And no component carries text that is not a known section.
        known = set(_general_section_texts().values())
        assert {piece for _, piece in pieces} <= known

    def test_the_prompt_is_its_components_and_nothing_else(self) -> None:
        for flags in _all_flag_combinations():
            components = general.extract_prompt_components(**flags)
            names = [c.name for c in components]
            assert len(names) == len(set(names)), flags
            prompt = general._build_extract_prompt(**flags)
            assert prompt == assemble_prompt(components), flags
            assert len(prompt) == sum(len(c.rule) + len(c.field) for c in components), flags

    def test_the_default_prompt_is_unchanged_by_the_table(self) -> None:
        """The pre-table formula, spelled out, gives the same bytes."""
        cfg = get_config()
        flags = {
            "scope_enabled": cfg.extraction_scope.enabled,
            "modality_enabled": cfg.extraction_modality.enabled,
            "polarity_enabled": cfg.extraction_polarity.enabled,
            "stance_enabled": cfg.extraction_stance.enabled,
            "validity_enabled": cfg.extraction_validity.enabled,
            "structure_enabled": cfg.structured_claim.enabled,
        }
        g = general
        legacy = (
            g._EXTRACT_RULES
            + g.REFERENCE_RULE
            + (g._SCOPE_RULE if flags["scope_enabled"] else "")
            + (g._MODALITY_RULE if flags["modality_enabled"] else "")
            + (g._POLARITY_RULE if flags["polarity_enabled"] else "")
            + (g._STANCE_RULE if flags["stance_enabled"] else "")
            + (g._VALIDITY_RULE if flags["validity_enabled"] else g._REFERENCE_DATE_RULE)
            + (g._STRUCTURE_RULE if flags["structure_enabled"] else "")
            + g._EXTRACT_SCHEMA_HEAD
            + (g._SCOPE_SCHEMA_FIELD if flags["scope_enabled"] else "")
            + (g._MODALITY_SCHEMA_FIELD if flags["modality_enabled"] else "")
            + (g._POLARITY_SCHEMA_FIELD if flags["polarity_enabled"] else "")
            + (g._STANCE_SCHEMA_FIELD if flags["stance_enabled"] else "")
            + (g._VALIDITY_SCHEMA_FIELD if flags["validity_enabled"] else "")
            + (g._STRUCTURE_SCHEMA_FIELD if flags["structure_enabled"] else "")
            + g._EXTRACT_SCHEMA_TAIL
        )
        assert g._build_extract_prompt(**flags) == legacy

    def test_changing_one_section_changes_only_its_digest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = general.general_component_table().digests
        monkeypatch.setattr(general, "_MODALITY_RULE", general._MODALITY_RULE + "\n- extra.")
        after = general.general_component_table().digests
        changed = {name for name in before if before[name] != after[name]}
        assert changed == {general.PROMPT_MODALITY}

    def test_table_always_set_is_the_prompt_the_parser_and_routing(self) -> None:
        table = general.general_component_table()
        always = {c.name for c in general._configured_prompt_components()}
        assert table.always == always | {general.CODE_PARSE, general.ROUTING}
        # Source-dependent components are in the table but not on every extraction.
        for name in (
            general.PROMPT_TOOL_TURNS,
            general.PROMPT_APPEND_CONTEXT,
            general.PATH_SINGLE_PASS,
            general.PATH_CHUNKED,
            general.PATH_APPEND_DELTA,
            general.PATH_PDF_PAGES,
            general.PATH_IMAGE,
            general.CHANNEL_VISION,
        ):
            assert name in table.digests
            assert name not in table.always

    def test_a_chunking_setting_changes_only_the_paths_it_shapes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = general.path_component_digests()
        monkeypatch.setattr(get_config().extraction, "append_chunk_chars", 9000)
        after = general.path_component_digests()
        assert {n for n in before if before[n] != after[n]} == {general.PATH_APPEND_DELTA}

    def test_a_routing_setting_changes_the_routing_component(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which sources get a source-dependent rule is not in any record, so it must select."""
        before = general.routing_component_digest()
        monkeypatch.setattr(
            get_config().extraction, "tool_turn_source_types", ["CONVERSATION", "WEB_PAGE"]
        )
        assert general.routing_component_digest() != before

    def test_parser_config_and_code_change_the_parse_component(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = general.parse_component_digest()
        monkeypatch.setattr(get_config().extraction_validity, "enabled", False)
        assert general.parse_component_digest() != before
        monkeypatch.undo()
        assert general.parse_component_digest() == before
        monkeypatch.setattr(general, "_CONFIDENCE_CEILING", 0.98)
        assert general.parse_component_digest() != before

    def test_a_function_with_no_source_falls_back_to_the_version(self) -> None:
        assert general.source_text(len) == f"no-source:{general.EXTRACTOR_VERSION}"


class TestJournalPromptTable:
    def test_the_journal_prompt_is_its_components(self) -> None:
        components = journal.journal_prompt_components()
        assert journal._build_journal_prompt() == assemble_prompt(components)
        sections = {journal._JOURNAL_RULES, journal.REFERENCE_RULE, journal._JOURNAL_SCHEMA}
        assert {c.rule for c in components} == sections
        assert all(c.field == "" for c in components)

    def test_the_shared_reference_rule_has_one_name_and_one_digest(self) -> None:
        general_ref = next(
            c for c in general._configured_prompt_components() if c.name == general.PROMPT_REFERENCE
        )
        journal_ref = next(
            c for c in journal.journal_prompt_components() if c.name == general.PROMPT_REFERENCE
        )
        assert general_ref.digest == journal_ref.digest


class TestComponentPrimitives:
    def test_digest_is_stable_and_separator_safe(self) -> None:
        assert component_digest("ab", "c") == component_digest("ab", "c")
        assert component_digest("ab", "c") != component_digest("a", "bc")
        assert len(component_digest("x")) == 16

    def test_table_refuses_an_always_name_it_does_not_hold(self) -> None:
        with pytest.raises(ValueError, match="not in the table"):
            ComponentTable(digests={"a": "1"}, always=frozenset({"b"}))

    def test_tally_records_and_marks_a_name_seen_under_two_texts(self) -> None:
        tally = ComponentTally()
        tally.add("prompt.a", "1")
        tally.add("prompt.a", "1")
        tally.add("prompt.b", "2")
        tally.add("prompt.b", "3")
        tally.declare(["path.x"])
        record = tally.record("general-extractor")
        assert record.exercised == {"prompt.a": "1", "prompt.b": MIXED_DIGEST}
        assert record.available == ["path.x", "prompt.a", "prompt.b"]
        assert record.extractor == "general-extractor"
        assert record.complete

    def test_recording_without_an_open_tally_is_a_no_op(self) -> None:
        record_component("prompt.a", "1")
        declare_components(ComponentTable(digests={"a": "1"}))

    def test_tally_sees_records_made_inside_its_block_only(self) -> None:
        with tally_components() as tally:
            record_component("prompt.a", "1")
            declare_components(ComponentTable(digests={"path.x": "2"}))
        record_component("prompt.b", "3")
        assert tally.exercised == {"prompt.a": "1"}
        assert tally.available == {"prompt.a", "path.x"}

    def test_merging_unions_exercised_and_intersects_available(self) -> None:
        mine = ComponentRecord(exercised={"a": "1", "b": "2"}, available=["a", "b"])
        theirs = ComponentRecord(exercised={"b": "9", "c": "3"}, available=["b", "c", "d"])
        merged = mine.merged(theirs)
        assert merged.exercised == {"a": "1", "b": MIXED_DIGEST, "c": "3"}
        assert merged.available == ["b"]
        assert merged.complete

    def test_a_component_only_the_newer_extraction_knew_is_still_a_reason(self) -> None:
        """A rule added between two extractions whose claims one record covers.

        The older extraction (a carried-forward source, or an earlier attempt)
        ran before the source-dependent rule ``rule.x`` existed; the newer one
        knew it and did not exercise it. The merged record must not say the
        older claims could have used it, so the selection rule still flags it.
        """
        older = ComponentRecord(exercised={"core": "1"}, available=["core"])
        newer = ComponentRecord(exercised={"core": "1"}, available=["core", "rule.x"])
        merged = newer.merged(older)
        table = ComponentTable(digests={"core": "1", "rule.x": "x1"}, always=frozenset({"core"}))
        assert component_change_reasons({"": table}, merged) == ("rule.x: new since the record",)

    def test_merging_an_unrecorded_source_makes_the_record_incomplete(self) -> None:
        assert not ComponentRecord(exercised={"a": "1"}).merged(None).complete

    def test_record_round_trips_through_json(self) -> None:
        record = ComponentRecord(
            extractor="general-extractor",
            exercised={"prompt.a": "1"},
            available=["path.x", "prompt.a"],
        )
        assert ComponentRecord.model_validate_json(record.model_dump_json()) == record


class TestWhatAnExtractionRecords:
    def test_the_request_builder_records_the_prompt_it_assembled(self) -> None:
        with tally_components() as tally:
            general._build_llm_request("Some source text.")
        expected = {c.name: c.digest for c in general._configured_prompt_components()}
        expected[general.CODE_PARSE] = general.parse_component_digest()
        assert tally.exercised == expected

    def test_tool_turns_and_images_add_their_components(self) -> None:
        from particles.llm import VisionImage

        image = VisionImage(media_type="image/png", data=b"\x89PNG\r\n\x1a\n")
        with tally_components() as tally:
            general._build_llm_request("text", [image], tool_turns_present=True)
        assert general.PROMPT_TOOL_TURNS in tally.exercised
        assert (
            tally.exercised[general.CHANNEL_VISION]
            == (general.path_component_digests()[general.CHANNEL_VISION])
        )

    @pytest.mark.asyncio
    async def test_extract_declares_its_table_and_records_its_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        from particles.core.schema import ExtractionStatus, Snapshot, WarcRecordType

        async def fake_call(text: str, *args: Any, **kwargs: Any) -> Any:
            general._build_llm_request(text)
            return [], [], False

        monkeypatch.setattr(general, "_call_llm", fake_call)
        snapshot = Snapshot(
            snapshot_id="s1",
            captured_at=datetime.now(UTC),
            content_hash="0" * 64,
            extraction_status=ExtractionStatus.PENDING,
            warc_record_type=WarcRecordType.RESPONSE,
        )
        with tally_components() as tally:
            await general.GeneralExtractor().extract(snapshot, b"A short note.")
        record = tally.record()
        assert general.PATH_SINGLE_PASS in record.exercised
        assert general.PATH_CHUNKED not in record.exercised
        assert set(general.general_component_table().digests) <= set(record.available)
        assert general.PROMPT_RULES in record.exercised
        assert record.exercised[general.ROUTING] == general.routing_component_digest()

    @pytest.mark.asyncio
    async def test_the_journal_records_its_own_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_complete(*args: Any, **kwargs: Any) -> tuple[str, str]:
            return "[]", "anthropic:test"

        monkeypatch.setattr("particles.llm.complete_with_provider_model", fake_complete)
        with tally_components() as tally:
            await journal._call_journal_llm("Dear diary.")
        assert set(tally.exercised) == {c.name for c in journal.journal_prompt_components()} | {
            journal.CODE_JOURNAL_PARSE
        }

    def test_the_journal_table_puts_its_prompt_parser_and_routing_on_every_extraction(
        self,
    ) -> None:
        table = journal.journal_component_table()
        prompt = {c.name for c in journal.journal_prompt_components()}
        assert table.always == prompt | {journal.CODE_JOURNAL_PARSE, general.ROUTING}
        assert general.PATH_CHUNKED in table.digests


class TestGateDigest:
    def test_a_disposition_change_changes_the_digest(self) -> None:
        base = gate_digest(cli_binaries=["particles"], allowlist=[], dispositions={})
        assert base == gate_digest(cli_binaries=["particles"], allowlist=[], dispositions={})
        assert base != gate_digest(
            cli_binaries=["particles"], allowlist=[], dispositions={"filename": "qualify"}
        )
        assert base != gate_digest(cli_binaries=["particles"], allowlist=["X-1"], dispositions={})
