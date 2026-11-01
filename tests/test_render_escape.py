# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Each escaping function keeps stored text inert in its own output context.

The functions of :mod:`particles.render.escape` (security review 2026-10-28-1,
O1, O2, O37, O38), each checked by parsing what it writes in its context.
Pure; no store.
"""

from __future__ import annotations

import csv
import io

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from markdown_it import MarkdownIt

from particles.render.escape import (
    anki_deck,
    anki_field,
    identifier,
    logseq_alias,
    logseq_page_ref,
    logseq_property_key,
    logseq_property_value,
    logseq_tag,
    logseq_text,
    md_code_span,
    md_link,
    md_table_cell,
    md_text,
    prose_text,
    synthesized_prose,
    wikilink,
    wikilink_target,
    yaml_scalar,
)

# Text with no surrogate code points, which no file the exporters write can hold.
_TEXT = st.text(st.characters(blacklist_categories=("Cs",)), max_size=80)
_MD = MarkdownIt("commonmark").enable("table")


def _blocks(markdown: str) -> list[str]:
    return [t.type for t in _MD.parse(markdown)]


class TestMarkdownText:
    @given(_TEXT)
    @settings(max_examples=300)
    def test_any_text_is_at_most_one_paragraph_with_no_html_or_image(self, text: str) -> None:
        tokens = _MD.parse(md_text(text))
        assert {t.type for t in tokens} <= {"paragraph_open", "paragraph_close", "inline"}
        kinds = {c.type for t in tokens for c in t.children or []}
        assert not kinds & {"html_inline", "image"}, kinds

    @given(_TEXT)
    @settings(max_examples=200)
    def test_any_text_stays_inside_its_list_item(self, text: str) -> None:
        tokens = _MD.parse(f"- {md_text(text)}\n")
        assert [t.type for t in tokens].count("list_item_open") == 1
        assert [t.type for t in tokens].count("bullet_list_open") == 1

    @given(_TEXT)
    @settings(max_examples=200)
    def test_any_text_stays_inside_one_callout(self, text: str) -> None:
        tokens = _MD.parse(f"> [!info] Title\n> {md_text(text)}\n>\n> {md_text(text)}\n")
        assert [t.type for t in tokens].count("blockquote_open") == 1
        inner = {t.type for t in tokens} - {"blockquote_open", "blockquote_close"}
        assert inner <= {"paragraph_open", "paragraph_close", "inline"}, inner

    def test_line_leading_block_markers_read_as_text(self) -> None:
        for text in ("# h", "> q", "- i", "+ i", "* i", "1. i", "2) i", "---", "===", "| a |"):
            assert _blocks(md_text(text)) == ["paragraph_open", "inline", "paragraph_close"], text

    def test_a_wikilink_stays_a_link(self) -> None:
        assert md_text("see [[Home]]") == "see [[Home]]"

    def test_a_callout_header_and_a_block_id_are_escaped(self) -> None:
        assert md_text("[!success] ok").startswith("\\[")
        assert md_text("text ^p-deadbeef").endswith("\\^p-deadbeef")


class TestYamlScalar:
    @given(_TEXT)
    @settings(max_examples=300)
    def test_any_text_round_trips_as_one_key(self, text: str) -> None:
        doc = f"tags:\n  - {yaml_scalar(text)}\nkey: {yaml_scalar(text)}\n"
        assert "\n" not in yaml_scalar(text)
        assert yaml.safe_load(doc) == {"tags": [text], "key": text}

    def test_numbers_and_booleans_keep_their_type(self) -> None:
        assert yaml.safe_load(f"k: {yaml_scalar(0.75)}") == {"k": 0.75}
        assert yaml.safe_load(f"k: {yaml_scalar(12)}") == {"k": 12}
        assert yaml.safe_load(f"k: {yaml_scalar(True)}") == {"k": True}
        assert yaml.safe_load(f"k: {yaml_scalar(float('nan'))}") == {"k": "nan"}

    def test_a_line_break_yaml_alone_reads_is_escaped(self) -> None:
        for ch in ("\x85", " ", " ", "\x7f"):
            assert ch not in yaml_scalar(f"a{ch}b")


class TestTableCell:
    @given(_TEXT)
    @settings(max_examples=200)
    def test_any_text_stays_in_its_own_column(self, text: str) -> None:
        cell = md_table_cell(text) or "x"
        table = f"| a | b |\n|---|---|\n| {cell} | z |\n"
        tokens = _MD.parse(table)
        row = [t.type for t in tokens][[t.type for t in tokens].index("tbody_open") :]
        assert row.count("td_open") == 2, table


class TestLinks:
    def test_a_wikilink_target_cannot_close_the_link_or_point_elsewhere(self) -> None:
        assert wikilink_target("a]] ![[b|c#d^e") == "a)) !((b-c-d-e"
        assert wikilink("x]]", "y]]") == "[[x))|y))]]"

    def test_only_an_http_uri_becomes_a_link_target(self) -> None:
        assert md_link("x", "javascript:x") == "x (javascript:x)"
        link = md_link("x", "https://a.invalid/ ){{b}}")
        assert link == "[x](https://a.invalid/%20%29%7B%7Bb%7D%7D)"

    def test_a_code_span_cannot_be_left(self) -> None:
        assert md_code_span("a`b\nc") == "`a'b c`"


class TestLogseq:
    @given(_TEXT)
    @settings(max_examples=200)
    def test_block_text_and_property_values_hold_no_property_macro_or_break(
        self, text: str
    ) -> None:
        for out in (logseq_text(text), logseq_property_value(text), logseq_alias(text)):
            assert "\n" not in out and "\r" not in out
            assert "::" not in out and "{{" not in out

    def test_an_alias_is_one_alias(self) -> None:
        assert "," not in logseq_alias("Smith, John")

    def test_a_property_key_is_one_key(self) -> None:
        assert logseq_property_key("nmo:hasShape") == "nmo:hasShape"
        assert logseq_property_key("a:: b\nc") == "a:-b-c"

    def test_a_page_ref_and_a_tag_cannot_close_early(self) -> None:
        # Brackets and braces become parentheses, and a run of them one, so
        # the name also holds no ``((uuid))`` block reference (AA32).
        assert logseq_page_ref("a]] {{q}} <i> k:: v") == "a) (q) -i- k: v"
        assert logseq_tag("ml/optimizers") == "#ml/optimizers"
        assert logseq_tag("two words]]") == "#[[two words)]]"


class TestAnki:
    @given(_TEXT, _TEXT)
    @settings(max_examples=200)
    def test_any_fields_make_one_card_of_three_fields(self, front: str, back: str) -> None:
        line = "\t".join((anki_field(front), anki_field(back, line_separator="<br>"), "t"))
        assert "\n" not in line and "\r" not in line
        rows = list(csv.reader(io.StringIO(line), delimiter="\t", quotechar='"'))
        assert len(rows) == 1 and len(rows[0]) == 3, rows

    def test_a_deck_name_adds_no_sub_deck(self) -> None:
        assert anki_deck("a::b\nc") == "a:b c"


class TestEntityForm:
    @given(_TEXT)
    @settings(max_examples=200)
    def test_prose_text_is_one_paragraph_the_backstop_leaves_unchanged(self, text: str) -> None:
        out = prose_text(text)
        assert synthesized_prose(out) == out
        assert {t.type for t in _MD.parse(out)} <= {"paragraph_open", "paragraph_close", "inline"}


def test_an_identifier_keeps_a_uuid_and_drops_markup() -> None:
    assert identifier("0a1b2c3d-4e5f") == "0a1b2c3d-4e5f"
    assert identifier("<img src") == "-img-src"
