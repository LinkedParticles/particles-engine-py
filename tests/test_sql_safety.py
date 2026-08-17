"""Tests for SQL LIKE wildcard escaping (particles/sql_safety.py).

Direct unit tests for escape_like_pattern, plus a regression test against
the real DB confirming that a user-supplied prefix containing ``%`` no
longer expands the match in repository helpers that take user input.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from particles.sql_safety import LIKE_ESCAPE, escape_like_pattern


class TestEscapeLikePattern:
    def test_plain_string_unchanged(self) -> None:
        assert escape_like_pattern("abcdef") == "abcdef"

    def test_percent_is_escaped(self) -> None:
        assert escape_like_pattern("foo%bar") == "foo\\%bar"

    def test_underscore_is_escaped(self) -> None:
        assert escape_like_pattern("foo_bar") == "foo\\_bar"

    def test_escape_char_is_escaped_first(self) -> None:
        # If we escape % first, then \, we'd produce \\\% which is wrong.
        # Escaping \ first yields \\% which is correct.
        assert escape_like_pattern("a\\b") == "a\\\\b"

    def test_combined(self) -> None:
        assert escape_like_pattern("a%b_c\\d") == "a\\%b\\_c\\\\d"

    def test_empty_string(self) -> None:
        assert escape_like_pattern("") == ""


# ---------------------------------------------------------------------------
# Regression: a user-supplied prefix containing '%' must NOT broaden the match
# ---------------------------------------------------------------------------


class TestPrefixSearchRespectsEscape:
    @pytest.mark.asyncio
    async def test_find_entry_ids_by_prefix_does_not_match_unrelated_rows(
        self, db_session: Any
    ) -> None:
        from particles.core.schema import CorpusEntry
        from particles.corpus.store import CorpusEntryRow, find_entry_ids_by_prefix

        # Two entries with unrelated IDs
        e1 = CorpusEntry(
            entry_id="abcd1234-0000-0000-0000-000000000001",
            source_type="WEB_PAGE",
            uri_r="https://a",
            deposited_by="test",
        )
        e2 = CorpusEntry(
            entry_id="9999ffff-0000-0000-0000-000000000002",
            source_type="WEB_PAGE",
            uri_r="https://b",
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(e1))
        db_session.add(CorpusEntryRow.from_model(e2))
        await db_session.commit()

        # User types literal "%" as the prefix — before the escape fix this
        # would match every row; after the fix it matches none (no entry_id
        # actually starts with a literal "%").
        result = await find_entry_ids_by_prefix(db_session, "%")
        assert result == []

    @pytest.mark.asyncio
    async def test_underscore_in_prefix_does_not_match_any_character(self, db_session: Any) -> None:
        from particles.core.schema import CorpusEntry
        from particles.corpus.store import CorpusEntryRow, find_entry_ids_by_prefix

        e = CorpusEntry(
            entry_id="abcdef-" + str(uuid.uuid4())[:8],
            source_type="WEB_PAGE",
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(e))
        await db_session.commit()

        # User types "a_cdef" — unescaped LIKE would match "abcdef..." because
        # "_" is the "any single character" wildcard. With the fix, "_" is
        # treated literally and finds nothing.
        result = await find_entry_ids_by_prefix(db_session, "a_cdef")
        assert result == []

    @pytest.mark.asyncio
    async def test_plain_prefix_still_works(self, db_session: Any) -> None:
        from particles.core.schema import CorpusEntry
        from particles.corpus.store import CorpusEntryRow, find_entry_ids_by_prefix

        e = CorpusEntry(
            entry_id="deadbeef-1111-2222-3333-444444444444",
            source_type="WEB_PAGE",
            deposited_by="test",
        )
        db_session.add(CorpusEntryRow.from_model(e))
        await db_session.commit()

        result = await find_entry_ids_by_prefix(db_session, "deadbeef")
        assert result == ["deadbeef-1111-2222-3333-444444444444"]


def test_like_escape_constant_is_backslash() -> None:
    # Smoke check: callers that pass `escape=LIKE_ESCAPE` rely on this.
    assert LIKE_ESCAPE == "\\"
