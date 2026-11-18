# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Day marks in a distilled session transcript (§2).

* The pure helpers: the heading, the escape of a mark-shaped body line, the
  day in force at an offset, and the reduction with its offset map.
* The distiller writes a mark only on a forward day change, none on the first
  day, none for an untimed turn or a clock that went back; a single-day
  transcript distills exactly as version 1 did, and ``reduce_marks`` of any
  version-2 text is its version-1 text.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from particles.api.cli._claude_code import (
    distill_transcript,
    redact_secrets,
    transcript_days,
)
from particles.extraction.day_marks import (
    day_at,
    escape_mark_lines,
    first_mark_offset,
    format_turn_heading,
    mark_days,
    reduce_marks,
)

SEP = "·"


def _jsonl(*records: dict[str, Any]) -> str:
    return "\n".join(json.dumps(r) for r in records)


def _user(text: str, stamp: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {"type": "user", "message": {"content": text}}
    if stamp is not None:
        record["timestamp"] = stamp
    return record


def _assistant(text: str, stamp: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "timestamp": stamp,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _version_one(jsonl_text: str, session_id: str) -> str:
    """The version-1 distiller, as it stood before day marks: the oracle."""
    out = [f"# Claude Code session {session_id}"]
    for line in jsonl_text.splitlines():
        entry = json.loads(line)
        if entry.get("type") not in ("user", "assistant"):
            continue
        content = entry["message"]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        out.append("## User" if entry["type"] == "user" else "## Assistant")
        out.append(text)
    return "\n\n".join(out) + "\n"


class TestHelpers:
    def test_heading_plain_and_marked(self) -> None:
        assert format_turn_heading("User") == "## User"
        assert (
            format_turn_heading("Assistant", date(2026, 9, 14)) == f"## Assistant {SEP} 2026-09-14"
        )

    def test_escape_adds_one_backslash_and_only_to_mark_shaped_lines(self) -> None:
        text = f"a\n## User {SEP} 2026-01-01\n\\## User {SEP} 2026-01-02\n## User\nz"
        assert escape_mark_lines(text) == (
            f"a\n\\## User {SEP} 2026-01-01\n\\\\## User {SEP} 2026-01-02\n## User\nz"
        )
        assert escape_mark_lines("no marks here") == "no marks here"

    def test_day_in_force_at_an_offset(self) -> None:
        text = (
            f"## User\n\nx\n\n## User {SEP} 2026-09-02\n\ny\n\n## Assistant {SEP} 2026-09-05\n\nz"
        )
        marks = mark_days(text)
        assert [d for _, d in marks] == [date(2026, 9, 2), date(2026, 9, 5)]
        assert day_at(marks, 0) is None
        assert day_at(marks, text.index("y")) == date(2026, 9, 2)
        assert day_at(marks, marks[1][0]) == date(2026, 9, 2)
        assert day_at(marks, len(text)) == date(2026, 9, 5)

    def test_an_escaped_line_is_not_a_mark(self) -> None:
        assert mark_days(f"\\## User {SEP} 2026-09-02\n") == []
        assert first_mark_offset(f"\\## User {SEP} 2026-09-02\n".encode()) is None

    def test_reduction_and_its_offset_map(self) -> None:
        marked = (
            f"## User\n\na\n\n## User {SEP} 2026-09-02\n\nb\n\n\\## User {SEP} 2026-01-01\n"
        ).encode()
        reduced = reduce_marks(marked)
        assert (
            reduced.text == f"## User\n\na\n\n## User\n\nb\n\n## User {SEP} 2026-01-01\n".encode()
        )
        b_raw = marked.index(b"b")
        assert reduced.text[reduced.to_reduced(b_raw)] == ord("b")
        assert reduced.to_raw(reduced.to_reduced(b_raw)) == b_raw
        # At the start of an escaped line the raw offset keeps the backslash.
        escaped = marked.index(b"\\##")
        assert reduced.to_raw(reduced.to_reduced(escaped)) == escaped
        assert first_mark_offset(marked) == marked.index(f"## User {SEP}".encode())

    def test_reduction_is_the_identity_without_marks(self) -> None:
        plain = b"## User\n\nhello\n"
        reduced = reduce_marks(plain)
        assert reduced.text == plain and reduced.cuts == ()
        assert reduced.to_raw(5) == 5 and reduced.to_reduced(5) == 5


_BODY_LINES = st.sampled_from(
    [
        "plain words",
        "",
        f"## User {SEP} 2026-01-01",
        f"\\## Assistant {SEP} 2026-02-02",
        f"## User {SEP} \u0662\u0660\u0662\u0666-01-01",
        "## User",
        "tail line",
    ]
)
_TURNS = st.lists(
    st.tuples(
        st.sampled_from(["User", "Assistant"]),
        st.lists(_BODY_LINES, min_size=1, max_size=4).map("\n".join),
        st.integers(min_value=0, max_value=2),
    ),
    min_size=1,
    max_size=12,
)


def _render(turns: list[tuple[str, str, int]]) -> tuple[str, str]:
    """A version-1 and a version-2 text of the same turns, as the distiller writes them."""
    start = date(2026, 9, 1)
    v1, v2 = ["# Claude Code session p"], ["# Claude Code session p"]
    day = 0
    for speaker, body, step in turns:
        v1 += [f"## {speaker}", body]
        mark = None
        if step and len(v2) > 1:
            day += step
            mark = start + timedelta(days=day)
        v2 += [format_turn_heading(speaker, mark), escape_mark_lines(body)]
    return "\n\n".join(v1) + "\n", "\n\n".join(v2) + "\n"


class TestReductionProperties:
    """The offset map holds over random mark placements."""

    @given(_TURNS)
    def test_reduced_version_two_is_version_one(self, turns: list[tuple[str, str, int]]) -> None:
        v1, v2 = _render(turns)
        assert reduce_marks(v2.encode()).text == v1.encode()

    @given(_TURNS)
    def test_every_paragraph_boundary_maps_both_ways(
        self, turns: list[tuple[str, str, int]]
    ) -> None:
        v1, v2 = _render(turns)
        plain, marked = v1.encode(), v2.encode()
        reduced = reduce_marks(marked)
        boundaries = [0] + [i + 2 for i in range(len(plain)) if plain[i : i + 2] == b"\n\n"]
        for boundary in boundaries:
            raw = reduced.to_raw(boundary)
            assert reduced.to_reduced(raw) == boundary
            # What lies before the mapped offset reduces to what lies before
            # the boundary, so a delta cut there neither skips nor repeats text.
            assert reduce_marks(marked[:raw]).text == plain[:boundary]

    def test_a_line_with_other_digits_is_never_escaped(self) -> None:
        line = f"## User {SEP} \u0662\u0660\u0662\u0666-01-01"
        assert escape_mark_lines(line) == line
        assert reduce_marks(line.encode()).text == line.encode()


class TestDistiller:
    def test_a_single_day_transcript_is_byte_identical_to_version_one(self) -> None:
        jsonl = _jsonl(
            _user("run the tests", "2026-09-01T09:00:00Z"),
            _assistant("the tests passed", "2026-09-01T09:05:00Z"),
        )
        assert distill_transcript(jsonl, "s") == _version_one(jsonl, "s")
        assert transcript_days(jsonl) == [date(2026, 9, 1)]

    def test_marks_only_forward_day_changes(self) -> None:
        jsonl = _jsonl(
            {"type": "summary", "timestamp": "2026-09-01T23:00:00Z"},
            _user("first", "2026-09-01T23:30:00Z"),
            _assistant("after midnight", "2026-09-02T00:10:00Z"),
            _user("clock went back", "2026-08-30T10:00:00Z"),
            _user("untimed", None),
            _assistant("later", "2026-09-05T10:00:00Z"),
        )
        out = distill_transcript(jsonl, "s")
        assert "## User\n\nfirst" in out
        assert f"## Assistant {SEP} 2026-09-02\n\nafter midnight" in out
        assert "## User\n\nclock went back" in out
        assert "## User\n\nuntimed" in out
        assert f"## Assistant {SEP} 2026-09-05\n\nlater" in out
        assert transcript_days(jsonl) == [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 5)]

    def test_a_mark_shaped_body_line_is_escaped_and_never_dates(self) -> None:
        jsonl = _jsonl(
            _user("hello", "2026-09-01T09:00:00Z"),
            _assistant(f"quoting:\n## User {SEP} 2099-01-01\nend", "2026-09-01T09:01:00Z"),
        )
        out = distill_transcript(jsonl, "s")
        assert f"\\## User {SEP} 2099-01-01" in out
        assert mark_days(out) == []

    def test_reducing_version_two_gives_version_one(self) -> None:
        jsonl = _jsonl(
            _user("day one", "2026-09-01T09:00:00Z"),
            _assistant("day two", "2026-09-02T09:00:00Z"),
            _user("day nine", "2026-09-09T09:00:00Z"),
        )
        v2 = distill_transcript(jsonl, "s")
        assert v2 != _version_one(jsonl, "s")
        assert reduce_marks(v2.encode()).text.decode() == _version_one(jsonl, "s")

    def test_appending_turns_appends_text(self) -> None:
        records = [
            _user("day one", "2026-09-01T09:00:00Z"),
            _assistant("day two", "2026-09-02T09:00:00Z"),
        ]
        shorter = distill_transcript(_jsonl(*records), "s")
        longer = distill_transcript(
            _jsonl(*records, _user("day three", "2026-09-03T09:00:00Z")), "s"
        )
        assert longer.startswith(shorter.rstrip("\n"))

    def test_a_mark_survives_redaction(self) -> None:
        jsonl = _jsonl(
            _user("day one", "2026-09-01T09:00:00Z"),
            _assistant("day two", "2026-09-02T09:00:00Z"),
        )
        assert mark_days(redact_secrets(distill_transcript(jsonl, "s"))) == mark_days(
            distill_transcript(jsonl, "s")
        )
