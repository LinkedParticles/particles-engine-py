# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A frontmatter header that cannot be built reads as no frontmatter at every reader.

The bounded loader turns deep nesting and an over-long number into a
``yaml.YAMLError``, so the capture split, the supersession read, the
extractor's strip and the memory-file type all degrade rather than raise
(security review 2026-10-27-1, G17, G18, G33).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

_DEEP = "k: " + "[" * 3000 + "]" * 3000
_LONG_INT = "k: " + "1" * 5000
_HEADERS = pytest.mark.parametrize("header", [_DEEP, _LONG_INT], ids=["deep", "long-int"])


@_HEADERS
def test_a_capture_header_that_fails_to_construct_reads_as_no_frontmatter(header: str) -> None:
    from particles.corpus.deposit import _split_frontmatter

    text = f"---\n{header}\n---\nbody\n"
    assert _split_frontmatter(text) == ({}, text)


@_HEADERS
def test_supersession_frontmatter_that_fails_to_construct_is_no_relation(header: str) -> None:
    from particles.corpus.supersession import _parse_frontmatter

    assert _parse_frontmatter(f"---\n{header}\n---\nbody\n".encode()) is None


@_HEADERS
def test_the_general_frontmatter_strip_never_raises(header: str) -> None:
    from particles.extraction.general import _strip_obsidian_frontmatter

    meta, body = _strip_obsidian_frontmatter(f"---\n{header}\n---\nbody\n")
    assert meta == {}
    assert body == "body\n"


@_HEADERS
def test_a_memory_file_header_that_fails_to_construct_has_no_type(
    header: str, tmp_path: Path
) -> None:
    from particles.api.cli._claude_code import memory_file_type

    md = tmp_path / "note.md"
    md.write_text(f"---\n{header}\n---\nbody\n", encoding="utf-8")
    assert memory_file_type(md) is None


@pytest.mark.parametrize(
    "block",
    [
        "type: ADR\nid: " + "9" * 5000 + "\n",
        "type: ADR\nid: " + "[" * 3000 + "]" * 3000 + "\n",
    ],
    ids=["long-integer", "deep-nesting"],
)
def test_unreadable_frontmatter_is_malformed_not_an_error(block: str) -> None:
    """A declaring document's frontmatter that cannot be read is no declaration."""
    from particles.corpus.supersession import document_relation_for_content

    content = f"---\n{block}---\n\nbody\n".encode()
    assert document_relation_for_content(content) is None


def _header_parse_seconds(parts: int) -> float:
    from particles.corpus.deposit import _split_frontmatter

    text = "---\nk: 1" + ":1" * parts + "\n---\nbody\n"
    start = time.perf_counter()
    _split_frontmatter(text)
    return time.perf_counter() - start


def test_a_capture_header_parse_is_linear_in_its_length() -> None:
    """Parsing a capture header four times longer costs at most about four times as much."""
    small = min(_header_parse_seconds(16_000) for _ in range(2))
    large = _header_parse_seconds(64_000)
    assert large < 8 * max(small, 0.005)


def test_a_capture_header_closes_within_the_head() -> None:
    """A header whose closing fence lies past the 8 KiB head is no frontmatter."""
    from particles.corpus.deposit import _split_frontmatter

    text = "---\nsource: https://example.org/\nnote: " + "x" * 9000 + "\n---\nbody\n"
    assert _split_frontmatter(text) == ({}, text)
    short = "---\nsource: https://example.org/\n---\nbody\n"
    assert _split_frontmatter(short) == ({"source": "https://example.org/"}, "body\n")


@pytest.mark.asyncio
async def test_one_unbuildable_capture_header_does_not_abort_the_folder_scan(
    db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.corpus.deposit import deposit_web_clipper

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "a-good.md").write_text(
        "---\nsource: https://example.org/good\n---\nGood body.\n", encoding="utf-8"
    )
    (captures / "b-bad.md").write_text(f"---\n{_DEEP}\n---\nBad body.\n", encoding="utf-8")
    results = await deposit_web_clipper(db_session, captures)  # type: ignore[arg-type]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_one_hard_linked_capture_does_not_abort_the_folder_scan(
    db_session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard-linked capture is skipped, its writes rolled back, and the rest deposited."""
    import os

    from particles.corpus.deposit import deposit_web_clipper
    from particles.corpus.store import list_entries

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "a-good.md").write_text(
        "---\nsource: https://example.org/good\n---\nGood body.\n", encoding="utf-8"
    )
    outside = tmp_path / "elsewhere.md"
    outside.write_text("Linked body.\n", encoding="utf-8")
    os.link(outside, captures / "b-linked.md")
    (captures / "c-good.md").write_text(
        "---\nsource: https://example.org/other\n---\nOther body.\n", encoding="utf-8"
    )
    results = await deposit_web_clipper(db_session, captures)  # type: ignore[arg-type]
    assert len(results) == 2
    entries = list(await list_entries(db_session))  # type: ignore[arg-type]
    assert sorted(e.uri_r or "" for e in entries) == [
        "https://example.org/good",
        "https://example.org/other",
    ]


def test_a_memory_file_type_is_read_from_its_head_only(tmp_path: Path) -> None:
    """A memory file's header must close within its first 8 KiB, and a link is not
    followed (G18)."""
    from particles.api.cli._claude_code import memory_file_type

    md = tmp_path / "feedback_x.md"
    md.write_text("---\nmetadata:\n  type: Feedback\n---\nbody\n" + "x" * 100_000)
    assert memory_file_type(md) == "feedback"
    long = tmp_path / "long.md"
    long.write_text("---\ntype: user\npad: '" + "y" * 9000 + "'\n---\nbody\n")
    assert memory_file_type(long) is None
    link = tmp_path / "link.md"
    link.symlink_to(md)
    assert memory_file_type(link) is None
