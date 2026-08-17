"""Tests for the symbol-aware docstring extractor.

The extractor is deterministic and LLM-free, so the bulk of these are plain
unit tests over fixed source strings. Covers:
  - ``accepts(PYTHON_SOURCE)`` and registry ordering
  - dotted module-path resolution from ``entry_uri_r`` (package walk / standalone
    / absent / non-file)
  - the AST walk: module / class / function / method docstrings → one particle
    each; undocumented symbols emit nothing; nested symbols
  - Google-style section parsing → ``docstring:`` properties (args / returns /
    raises, multi-line continuations, ``*args`` keys)
  - claim granularity: one particle per documented symbol
  - modality / type / confidence / uncertainty
  - resilient error handling (syntax error, non-UTF-8, no docstrings)
  - the gate exemption for ``PYTHON_SOURCE`` (config + end-to-end)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from particles.core.schema import (
    AssertionModality,
    ExtractionStatus,
    ParticleType,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.extraction.docstrings import (
    DocstringExtractor,
    _module_path_from_uri,
    _parse_google_docstring,
    _walk_documented_symbols,
)
from particles.extraction.general import ExtractionResult


def _snap() -> Snapshot:
    return Snapshot(
        content_hash="a" * 64,
        extraction_status=ExtractionStatus.PENDING,
        warc_record_type=WarcRecordType.RESPONSE,
    )


def _extract(source: str, *, entry_uri_r: str | None = None) -> ExtractionResult:
    return asyncio.run(
        DocstringExtractor().extract(_snap(), source.encode("utf-8"), entry_uri_r=entry_uri_r)
    )


# ---------------------------------------------------------------------------
# accepts() + registry ordering
# ---------------------------------------------------------------------------


class TestAcceptsAndRegistry:
    def test_accepts_python_source_only(self) -> None:
        ex = DocstringExtractor()
        assert ex.accepts("PYTHON_SOURCE") is True
        assert ex.accepts("LOCAL_MARKDOWN") is False
        assert ex.accepts("WEB_PAGE") is False

    def test_registered_before_general(self) -> None:
        from particles.extraction.registry import get_extractors

        ids = [type(p).__name__ for p in get_extractors()]
        assert "DocstringExtractor" in ids
        assert ids.index("DocstringExtractor") < ids.index("GeneralExtractor")

    def test_registry_finds_extractor_for_python_source(self) -> None:
        from particles.extraction.registry import get_extractors

        for plugin in get_extractors():
            if plugin.accepts("PYTHON_SOURCE"):
                assert type(plugin).__name__ == "DocstringExtractor"
                return
        raise AssertionError("No extractor accepts PYTHON_SOURCE")

    def test_identity_constants(self) -> None:
        ex = DocstringExtractor()
        assert ex.EXTRACTOR_ID == "docstring-extractor"
        assert ex.EXTRACTOR_VERSION == "0.1.0"
        assert ex.DEFAULT_TRUST_WEIGHT == 0.80
        assert ex.APPLICABILITY[0].keyword == "MUST"
        assert "PYTHON_SOURCE" in ex.APPLICABILITY[0].source_types


# ---------------------------------------------------------------------------
# Module-path resolution
# ---------------------------------------------------------------------------


class TestModulePathFromUri:
    def test_package_walk(self, tmp_path: Path) -> None:
        """Walks up while ``__init__.py`` exists → full dotted path."""
        pkg = tmp_path / "particles" / "core" / "scoring"
        pkg.mkdir(parents=True)
        (tmp_path / "particles" / "__init__.py").write_text("")
        (tmp_path / "particles" / "core" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        mod = pkg / "confidence.py"
        mod.write_text("x = 1\n")
        assert _module_path_from_uri(mod.as_uri()) == "particles.core.scoring.confidence"

    def test_standalone_module_is_just_stem(self, tmp_path: Path) -> None:
        mod = tmp_path / "script.py"
        mod.write_text("x = 1\n")
        # No __init__.py sibling → just the file stem.
        assert _module_path_from_uri(mod.as_uri()) == "script"

    def test_absent_uri_is_empty(self) -> None:
        assert _module_path_from_uri(None) == ""
        assert _module_path_from_uri("") == ""

    def test_non_file_uri_is_empty(self) -> None:
        assert _module_path_from_uri("https://example.com/mod.py") == ""

    def test_partial_package_stops_at_first_missing_init(self, tmp_path: Path) -> None:
        """The walk stops at the first parent lacking ``__init__.py``."""
        pkg = tmp_path / "outer" / "pkg"
        pkg.mkdir(parents=True)
        # Only the immediate parent is a package.
        (pkg / "__init__.py").write_text("")
        mod = pkg / "mod.py"
        mod.write_text("x = 1\n")
        assert _module_path_from_uri(mod.as_uri()) == "pkg.mod"


# ---------------------------------------------------------------------------
# AST walk — granularity and skip rules
# ---------------------------------------------------------------------------


_SAMPLE = '''\
"""Module summary line."""

CONST = 1


def documented():
    """A documented function."""
    return 1


def undocumented():
    return 2


class Widget:
    """A documented class."""

    def method(self):
        """A documented method."""
        return 3

    def bare(self):
        return 4

    class Inner:
        """A nested documented class."""
'''


class TestAstWalk:
    def test_one_particle_per_documented_symbol(self) -> None:
        result = _extract(_SAMPLE, entry_uri_r="file:///tmp/widgets.py")
        subjects = [c.subjects[0] for c in result.candidates]
        # Module, documented func, class, method, nested class — NOT the two
        # undocumented functions/methods.
        assert subjects == [
            "widgets",
            "widgets.documented",
            "widgets.Widget",
            "widgets.Widget.method",
            "widgets.Widget.Inner",
        ]

    def test_kinds_are_recorded(self) -> None:
        result = _extract(_SAMPLE, entry_uri_r="file:///tmp/widgets.py")
        kinds = {c.subjects[0]: c.properties["docstring:kind"] for c in result.candidates}  # type: ignore[index]
        assert kinds["widgets"] == "module"
        assert kinds["widgets.documented"] == "function"
        assert kinds["widgets.Widget"] == "class"
        assert kinds["widgets.Widget.method"] == "function"
        assert kinds["widgets.Widget.Inner"] == "class"

    def test_async_function_documented(self) -> None:
        src = '''\
async def fetch():
    """Fetch something asynchronously."""
    return 1
'''
        result = _extract(src, entry_uri_r="file:///tmp/aio.py")
        assert len(result.candidates) == 1
        assert result.candidates[0].subjects == ["aio.fetch"]
        assert result.candidates[0].content == "Fetch something asynchronously."

    def test_no_docstrings_yields_no_candidates(self) -> None:
        result = _extract("x = 1\n\ndef f():\n    return 2\n", entry_uri_r="file:///tmp/empty.py")
        assert result.candidates == []
        assert any("No documented symbols" in n for n in result.quality_notes)

    def test_subjectless_when_module_path_absent(self) -> None:
        """A module docstring with no resolvable path becomes a subjectless claim."""
        result = _extract('"""Just a module doc."""\n', entry_uri_r=None)
        assert len(result.candidates) == 1
        assert result.candidates[0].subjects == []
        assert result.candidates[0].content == "Just a module doc."

    def test_walk_helper_document_order(self) -> None:
        import ast

        tree = ast.parse(_SAMPLE)
        walked = _walk_documented_symbols(tree, "widgets")
        kinds = [kind for _subj, kind, _doc in walked]
        assert kinds == ["module", "function", "class", "function", "class"]


# ---------------------------------------------------------------------------
# Google-style section parsing
# ---------------------------------------------------------------------------


class TestSectionParsing:
    def test_summary_is_first_paragraph(self) -> None:
        doc = (
            "Return the SHA-256 hex digest of data.\n\n"
            "A longer description that should not be the summary.\n"
        )
        summary, sections = _parse_google_docstring(doc)
        assert summary == "Return the SHA-256 hex digest of data."
        assert sections == {}

    def test_multiline_summary_joined(self) -> None:
        doc = "First line of the summary\ncontinues on the second line.\n\nMore."
        summary, _ = _parse_google_docstring(doc)
        assert summary == "First line of the summary continues on the second line."

    def test_args_returns_raises_folded_into_sections(self) -> None:
        doc = (
            "Hash some bytes.\n\n"
            "Args:\n"
            "    data: bytes to hash.\n"
            "    salt: optional salt value.\n\n"
            "Returns:\n"
            "    the SHA-256 hex digest.\n\n"
            "Raises:\n"
            "    FileNotFoundError: if no blob exists.\n"
        )
        summary, sections = _parse_google_docstring(doc)
        assert summary == "Hash some bytes."
        assert sections["args"] == {"data": "bytes to hash.", "salt": "optional salt value."}
        assert sections["returns"] == "the SHA-256 hex digest."
        assert sections["raises"] == {"FileNotFoundError": "if no blob exists."}

    def test_typed_arg_and_star_args(self) -> None:
        doc = (
            "Do a thing.\n\n"
            "Args:\n"
            "    path (Path): the input path.\n"
            "    *args: extra positionals.\n"
            "    **kwargs: extra keywords.\n"
        )
        _summary, sections = _parse_google_docstring(doc)
        assert sections["args"] == {
            "path": "the input path.",
            "*args": "extra positionals.",
            "**kwargs": "extra keywords.",
        }

    def test_multiline_arg_description_continues(self) -> None:
        doc = "Summary.\n\nArgs:\n    data: bytes to hash\n        spanning two lines.\n"
        _summary, sections = _parse_google_docstring(doc)
        assert sections["args"] == {"data": "bytes to hash spanning two lines."}

    def test_attributes_section(self) -> None:
        doc = "A class.\n\nAttributes:\n    count: how many.\n    name: the label.\n"
        _summary, sections = _parse_google_docstring(doc)
        assert sections["attributes"] == {"count": "how many.", "name": "the label."}

    def test_summary_only_no_kind_pollution(self) -> None:
        """A summary-only docstring yields no section keys (only kind is added later)."""
        _summary, sections = _parse_google_docstring("Just a summary.")
        assert sections == {}


# ---------------------------------------------------------------------------
# Properties + modality / type / confidence
# ---------------------------------------------------------------------------


class TestCandidateShape:
    def test_falsifiable_claim_high_confidence_epistemic(self) -> None:
        result = _extract('"""Module doc."""\n', entry_uri_r="file:///tmp/m.py")
        c = result.candidates[0]
        assert c.particle_type == ParticleType.CLAIM
        assert c.assertion_modality == AssertionModality.FALSIFIABLE
        assert c.uncertainty_nature == UncertaintyNature.EPISTEMIC
        assert c.confidence_value == 0.95

    def test_properties_carry_docstring_prefix(self) -> None:
        src = '''\
def f(data):
    """Hash bytes.

    Args:
        data: bytes to hash.

    Returns:
        the digest.
    """
    return 1
'''
        result = _extract(src, entry_uri_r="file:///tmp/h.py")
        props = result.candidates[0].properties
        assert props is not None
        assert props["docstring:kind"] == "function"
        assert props["docstring:args"] == {"data": "bytes to hash."}
        assert props["docstring:returns"] == "the digest."
        assert "docstring:raises" not in props  # omitted when empty

    def test_content_falls_back_to_full_docstring_when_no_summary(self) -> None:
        """A docstring that opens directly with a section still gets non-empty content."""
        src = '''\
def f():
    """Args:
        x: a value.
    """
    return 1
'''
        result = _extract(src, entry_uri_r="file:///tmp/h.py")
        c = result.candidates[0]
        assert c.content  # never empty
        assert c.properties is not None and "docstring:args" in c.properties


# ---------------------------------------------------------------------------
# Resilient error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_syntax_error_yields_note_not_exception(self) -> None:
        result = _extract("def f(:\n    pass\n", entry_uri_r="file:///tmp/bad.py")
        assert result.candidates == []
        assert any("parse error" in n.lower() for n in result.quality_notes)

    def test_non_utf8_yields_note(self) -> None:
        bad = b"\xff\xfe\x00bad bytes"
        result = asyncio.run(
            DocstringExtractor().extract(_snap(), bad, entry_uri_r="file:///tmp/x.py")
        )
        assert result.candidates == []
        assert any("UTF-8" in n for n in result.quality_notes)


# ---------------------------------------------------------------------------
# gate exemption for PYTHON_SOURCE
# ---------------------------------------------------------------------------


class TestGateExemption:
    def test_python_source_exempt_by_default(self) -> None:
        from particles.config import get_config

        assert "PYTHON_SOURCE" in get_config().subject_gate.exempt_source_types

    def test_bare_snake_module_name_would_be_gated(self) -> None:
        """Proves the exemption matters: a bare snake_case module subject is
        exactly the shape the lexical gate strips."""
        from particles.extraction.subject_gate import classify_non_entity

        assert classify_non_entity("subject_store") == "snake_case"

    @pytest.mark.asyncio
    async def test_code_symbol_subjects_survive_extraction(
        self, db_session: object, tmp_path: Path
    ) -> None:
        """End-to-end: a PYTHON_SOURCE entry's snake_case module subject survives
        the pipeline because the gate exempts the source type."""
        import anthropic

        from particles import embeddings as ep
        from particles.corpus.deposit import deposit_file
        from particles.ingest.pipeline import extract_snapshot
        from particles.llm import set_client
        from particles.store.subject_store import find_by_name

        # `subject_store.py` → module subject "subject_store" (snake_case): the
        # gate would strip it without the PYTHON_SOURCE exemption.
        mod = tmp_path / "subject_store.py"
        mod.write_text(
            '"""The subject store module."""\n\n\n'
            'def helper():\n    """A helper."""\n    return 1\n'
        )

        mock_content = MagicMock()
        mock_content.text = "NO"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_client = MagicMock(spec=anthropic.Anthropic)
        mock_client.messages = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_resp)

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        original_model = ep._embedding_model
        ep.set_embedding_model(mock_model)
        set_client(mock_client)
        try:
            session = db_session  # type: ignore[assignment]
            entry_id, snapshot_id = await deposit_file(session, mod, deposited_by="test")  # type: ignore[arg-type]
            await session.commit()  # type: ignore[union-attr]

            particles = await extract_snapshot(session, entry_id, snapshot_id)  # type: ignore[arg-type]
            await session.commit()  # type: ignore[union-attr]

            assert len(particles) == 2  # module + helper
            # The snake_case module subject survived (was not gated).
            module_subj = await find_by_name(session, "subject_store")  # type: ignore[arg-type]
            assert module_subj is not None
            # The dotted symbol subject also resolved.
            helper_subj = await find_by_name(session, "subject_store.helper")  # type: ignore[arg-type]
            assert helper_subj is not None
        finally:
            ep.set_embedding_model(original_model)
            set_client(None)
