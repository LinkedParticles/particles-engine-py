# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""No stored value becomes structure in any exporter's output (security review 2026-10-28-1).

One scratch store holds a hostile value in every stored text field an exporter
writes: subject names (two of them colliding, so a disambiguation note and its
qualifiers are written), descriptions, classes, aliases, external refs,
property names and values, taxonomy tags, extractor and asserter names, a
corpus URI, and claim text. Every Markdown, Logseq and Anki exporter runs over
it, and what each wrote is parsed: frontmatter with a YAML parser, Markdown
with a CommonMark parser, tables and card lines by column count, Logseq pages
line by line. No hostile value may add a frontmatter key, end the frontmatter,
write an HTML node, an image or a code block, open a heading, quote, callout
or list item the engine did not write, add a table column, a Logseq property,
block or macro, or an Anki field or card (O1, O2, O37, O38; G4, G5).

The rendering functions get the same checks directly below, as the audit's
proofs drove them (ported from its render-structure proofs).
"""

from __future__ import annotations

import csv
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from markdown_it import MarkdownIt

from particles.core.schema import (
    Confidence,
    ExternalRef,
    LintFinding,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    RelationCreatedBy,
    RelationType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

# Markers a hostile value carries; each names a construct the engine never
# writes itself, so finding one as structure means stored text became markup.
FORGED_HEADING = "Forged heading"
FORGED_ITEM = "Forged item"
FORGED_CALLOUT = "forged-callout"
FORGED_KEY = "cssclasses"

#: A value that tries every construct at once: HTML, a remote image, an embed,
#: a macro, a fence, a table pipe, a link close, a Logseq property, a line
#: break that adds a YAML key, ends the frontmatter, and opens a quote, a
#: callout, a heading and a list item.
#: A footnote reference to another claim's citation and a Logseq block
#: reference to another block (security review 2026-10-28-2, AA27, AA32).
FORGED_FOOTNOTE = "[^p-deadbeef]"
FORGED_BLOCK_REF = "((00000000-0000-4000-8000-000000000000))"
#: A right-to-left override, which reorders what a reader sees (AA34).
BIDI = "\u202e"

HOSTILE = (
    'Name <img src="https://collector.invalid/p.png"> ![[Home]] {{embed [[Home]]}} '
    f"{FORGED_FOOTNOTE} {FORGED_BLOCK_REF} {BIDI}txt "
    "```dataviewjs | col | col ]] [[Home key:: value\n"
    f"{FORGED_KEY}: injected\n"
    "---\n"
    "<div>forged</div>\n"
    f"> [!{FORGED_CALLOUT}] Verified\n"
    f"# {FORGED_HEADING}\n"
    f"- {FORGED_ITEM}\n"
    "| a | b |\n"
    "|---|---|"
)


#: The same constructs, short enough to stay inside a filename's 255 bytes
#: once a subject name becomes one.
HOSTILE_NAME = (
    f"N <img src=x> ![[H]] {{{{embed [[H]]}}}} {BIDI}``` | c ]] k:: v\n"
    f"{FORGED_KEY}: i\n---\n> [!{FORGED_CALLOUT}] V\n# {FORGED_HEADING}\n- {FORGED_ITEM}"
)


def _hostile(tag: str) -> str:
    return f"{tag} {HOSTILE}"


def _hostile_name(tag: str) -> str:
    return f"{tag} {HOSTILE_NAME}"


def _particle(content: str, subject_ids: list[str], **kw: Any) -> Particle:
    return Particle(
        id=str(uuid.uuid4()),
        content=content,
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id="hostile-entry", snapshot_id="s1"
            )
        ],
        asserted_by=_hostile("asserter"),
        asserted_at=datetime.now(UTC),
        status=Status.ACTIVE,
        extractor_ref={"name": _hostile("extractor"), "version": _hostile("v")},
        subject_ids=subject_ids,
        **kw,
    )


async def _seed_hostile_store() -> None:
    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope
    from particles.store.particle_store import insert_particle
    from particles.store.relation_store import create_relation
    from particles.store.subject_store import insert_subject

    refs = [
        ExternalRef(
            namespace=_hostile("ns"),
            id=_hostile("id"),
            uri="https://example.invalid/" + HOSTILE,
            confidence=0.99,
        ),
        ExternalRef(namespace="wikidata", id=_hostile("Q"), confidence=0.1),
    ]
    coin = Subject(
        canonical_name=_hostile_name("Coin"),
        subject_class="nmo:NumismaticObject",
        description=_hostile("desc"),
        aliases=[_hostile("alias")],
        external_ids=refs,
        asserted_by="test",
    )
    pivot = Subject(
        canonical_name=_hostile_name("Pivot"),
        subject_class="nmo:Material",
        external_ids=refs,
        asserted_by="test",
    )
    generic = Subject(
        canonical_name=_hostile_name("Generic"),
        subject_class=_hostile("class"),
        description=_hostile("desc"),
        aliases=[_hostile("alias")],
        external_ids=refs,
        asserted_by="test",
    )
    twins = [
        Subject(
            canonical_name=_hostile_name("Twin"),
            description=f"{_hostile('desc')} {kind}",
            asserted_by="test",
        )
        for kind in ("software", "titan")
    ]
    subjects = [coin, pivot, generic, *twins]
    every = [s.id for s in subjects]
    props = {
        "nmo:hasShape": HOSTILE,
        "nmo:hasIssuer": HOSTILE,
        "nmo:hasWeight": HOSTILE,
        "nuds:references": [HOSTILE, HOSTILE],
        "numista:url": "data:text/html," + HOSTILE,
        HOSTILE: HOSTILE,
    }
    particles = [
        _particle(_hostile("claim"), every, tags=[HOSTILE, "plain/tag"]),
        _particle(_hostile("other claim"), every, properties=props, tags=[HOSTILE]),
        *(_particle(_hostile(f"claim {i}"), [s.id]) for i, s in enumerate(subjects)),
        _particle(_hostile("valued"), [coin.id], properties={"nmo:hasShape": "Round"}),
    ]
    # A NARRATIVE whose label is hostile text, over three of the claims: the wiki
    # writes its article, and its label reaches the Logseq Narratives block
    # (AA28).
    narrative = _particle(_hostile("narrative"), []).model_copy(
        update={"particle_type": ParticleType.NARRATIVE}
    )
    async with session_scope() as session:
        session.add(
            CorpusEntryRow(
                entry_id="hostile-entry",
                uri_r="https://example.invalid/" + HOSTILE,
                source_type="WEB_PAGE",
                mutability="MUTABLE",
                fetch_policy="LAZY",
                created_at=datetime.now(UTC),
                deposited_by="test",
            )
        )
        await session.flush()
        for s in subjects:
            await insert_subject(session, s)
        for p in particles:
            await insert_particle(session, p)
        await insert_particle(session, narrative)
        for c in particles[:3]:
            await create_relation(
                session, c.id, narrative.id, RelationType.PART_OF, RelationCreatedBy.MANUAL_CLI
            )
        await session.commit()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_ENGINE_FRONTMATTER_KEYS = {
    # Obsidian notes
    "subject_id",
    "tags",
    "aliases",
    "particle_count",
    "phantom",
    "article_input_hash",
    "article_synthesis",
    "Instance of",
    "References",
    "Numista URL",
    "Issuer",
    "Period",
    "Type",
    "Years",
    "Value",
    "Currency",
    "Composition",
    "Weight",
    "Diameter",
    "Thickness",
    "Shape",
    "Technique",
    "Orientation",
    "Demonetized",
    "Numista ID",
    # Wiki articles
    "generated_at",
    "input_hash",
    "sources",
    "contradictions",
    "synthesis",
    "layer_a_passed",
    "layer_b_passed",
    "layer_b_unrelated_count",
    "layer_b_contradicts_count",
    "min_particle_confidence",
    "dropped_below_threshold",
}


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    return (yaml.safe_load(text[4:end]) or {}), text[end + 5 :]


def _markdown_problems(path: Path, text: str) -> list[str]:
    """Every way the note's body holds structure no engine construct explains."""
    problems: list[str] = []
    fm, body = _split_frontmatter(text)
    if extra := set(fm) - _ENGINE_FRONTMATTER_KEYS:
        problems.append(f"{path.name}: frontmatter keys {sorted(extra)}")
    for key, value in fm.items():
        if key == "tags" and not isinstance(value, list):
            problems.append(f"{path.name}: tags is {type(value).__name__}")
    tokens = MarkdownIt("commonmark").enable("table").parse(body)
    depth = 0
    for i, t in enumerate(tokens):
        if t.type == "blockquote_open":
            depth += 1
            if depth > 1:
                problems.append(f"{path.name}: nested quote")
        elif t.type == "blockquote_close":
            depth -= 1
        if t.type in ("html_block", "fence", "code_block"):
            problems.append(f"{path.name}: {t.type} {t.content[:40]!r}")
        if t.type == "heading_open" and tokens[i + 1].content.startswith(FORGED_HEADING):
            problems.append(f"{path.name}: forged heading")
        if t.type == "list_item_open" and tokens[i + 2].content.startswith(FORGED_ITEM):
            problems.append(f"{path.name}: forged list item")
        if t.type == "tr_open":
            cells = 0
            j = i + 1
            while tokens[j].type != "tr_close":
                cells += tokens[j].type in ("td_open", "th_open")
                j += 1
            if cells != 4:
                problems.append(f"{path.name}: table row of {cells} cells")
        for child in t.children or []:
            if child.type in ("html_inline", "image"):
                problems.append(f"{path.name}: {child.type} {child.content[:40]!r}")
    if re.search(rf"^>+\s*\[!{FORGED_CALLOUT}\]", body, re.MULTILINE):
        problems.append(f"{path.name}: forged callout header")
    # ``{{`` is a macro only in Logseq; a Markdown vault may still hold one
    # inside quoted frontmatter, a code span or a link target or URL, none of
    # which is rendered as markup, so only the rest of the body is checked.
    rendered = re.sub(r"(?<!\\)`[^`\n]*`|\[\[[^\]\n]*\]\]|\]\([^)\n]*\)", "", body)
    if "{{" in rendered:
        problems.append(f"{path.name}: live macro")
    return problems


_LOGSEQ_ENGINE_PROPERTIES = {
    "type",
    "alias",
    "external-id",
    "id",
    "confidence",
    "tags",
    "article_input_hash",
    "article_synthesis",
}
_LOGSEQ_LINE = re.compile(r"^(?:  )*(?:- (?P<block>.*)|(?P<key>[^\s:][^\s]*?):: (?P<value>.*))$")


def _logseq_problems(path: Path, text: str, property_keys: set[str]) -> list[str]:
    problems: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _LOGSEQ_LINE.match(line)
        if m is None:
            problems.append(f"{path.name}: line outside the outline {line[:60]!r}")
            continue
        if m.group("key") is not None and m.group("key") not in _LOGSEQ_ENGINE_PROPERTIES:
            problems.append(f"{path.name}: property {m.group('key')!r}")
        block = m.group("block") or ""
        if block.startswith((FORGED_ITEM, "{{", f"# {FORGED_HEADING}")):
            problems.append(f"{path.name}: forged block {block[:40]!r}")
        # A property block the engine writes is ``- <key>:: <value>``, keyed by
        # a sanitised property name; any other ``::`` in a block is live.
        if ":: " in block and block.split(":: ", 1)[0] not in property_keys:
            problems.append(f"{path.name}: property in block {block[:60]!r}")
        if re.search(r"(?<!\\)<[a-z]", line, re.I):
            problems.append(f"{path.name}: HTML {line[:60]!r}")
    if "{{" in text:
        problems.append(f"{path.name}: live macro")
    return problems


# ---------------------------------------------------------------------------
# The whole-export test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_stored_value_becomes_structure_in_any_exporter(
    db_session: object, tmp_path: Path
) -> None:
    """Every exporter writes every hostile stored value as text, never as structure."""
    from particles.db import session_scope
    from particles.exporters.anki import AnkiExporter
    from particles.exporters.logseq import LogseqExporter
    from particles.exporters.obsidian import ObsidianExporter
    from particles.exporters.wiki import WikiExporter
    from particles.render.escape import logseq_property_key

    await _seed_hostile_store()

    obsidian, logseq, wiki = tmp_path / "obsidian", tmp_path / "logseq", tmp_path / "wiki"
    anki = tmp_path / "deck.txt"
    async with session_scope() as session:
        await ObsidianExporter().export(session, obsidian, min_links=0, force=True)
    async with session_scope() as session:
        await LogseqExporter().export(session, logseq, min_links=0, force=True)
    async with session_scope() as session:
        await WikiExporter().export(session, wiki, without_synthesis=True, min_particles=1)
    async with session_scope() as session:
        summary = await AnkiExporter().export(session, anki)

    problems: list[str] = []
    markdown = sorted([*obsidian.rglob("*.md"), *wiki.rglob("*.md")])
    # The coin, pivot, generic and two qualified notes, the disambiguation
    # note and the index, in each Markdown vault.
    assert len(list(obsidian.rglob("*.md"))) >= 7, sorted(p.name for p in obsidian.rglob("*"))
    assert any("disambiguation" in p.name for p in obsidian.rglob("*.md"))
    assert len(list(wiki.rglob("*.md"))) >= 6, sorted(p.name for p in wiki.rglob("*"))
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        problems += _markdown_problems(path, text)
        # A stored footnote reference is never a live citation (AA27). A
        # frontmatter scalar and a code span are inert, so they are not read.
        body = _split_frontmatter(text)[1] if text.startswith("---\n") else text
        body = re.sub(r"(?<!\\)`[^`\n]*`", "", body)
        if re.search(r"(?<![\\;])\[\^p-deadbeef\]", body):
            problems.append(f"{path.name}: a stored footnote reference is live")
    # A wiki narrative article is written from the hostile label.
    assert any(p.parent.name == "Narratives" for p in wiki.rglob("*.md"))
    every_file = [*obsidian.rglob("*"), *wiki.rglob("*"), *logseq.rglob("*"), anki]
    for path in every_file:
        # No bidirectional override reaches a file name or a file (AA34).
        if BIDI in path.name or (path.is_file() and BIDI in path.read_text(encoding="utf-8")):
            problems.append(f"{path.name}: a bidirectional override was written")
    # The lint pass runs on every export, so a finding's detail (the stored
    # name and description) reaches a callout and is checked with the rest.
    assert any("] WIKIDATA_LINK_MISMATCH" in p.read_text(encoding="utf-8") for p in markdown)

    pages = sorted((logseq / "pages").glob("*.md"))
    assert len(pages) >= 7, [p.name for p in pages]
    keys = {logseq_property_key(k) for k in ("nmo:hasShape", "nmo:hasIssuer", "nmo:hasWeight")}
    keys |= {logseq_property_key(k) for k in ("nuds:references", "numista:url", HOSTILE)}
    for path in pages:
        text = path.read_text(encoding="utf-8")
        problems += _logseq_problems(path, text, keys)
        # A stored block reference is never live (AA32).
        if FORGED_BLOCK_REF in text:
            problems.append(f"{path.name}: a stored block reference is live")

    header, cards = [], []
    for line in anki.read_text(encoding="utf-8").splitlines():
        (header if line.startswith("#") else cards).append(line)
    for line in header:
        if not line.startswith(
            ("#separator:", "#html:", "#notetype:", "#columns:", "#tags column:", "#comment:")
        ) and not line.startswith("#deck:"):
            problems.append(f"anki: header line {line[:60]!r}")
    if len(cards) != summary.cards_written:
        problems.append(f"anki: {len(cards)} card lines for {summary.cards_written} cards")
    for row in csv.reader(cards, delimiter="\t", quotechar='"'):
        if len(row) != 3:
            problems.append(f"anki: card of {len(row)} fields")
    decks = [line for line in header if line.startswith("#deck:")]
    if any(line.count("::") != 1 for line in decks):
        problems.append("anki: a subject name added a sub-deck")

    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# The rendering functions, driven directly (the audit's proofs)
# ---------------------------------------------------------------------------


def _subject(name: str, cls: str | None = None, **kw: Any) -> Subject:
    return Subject(canonical_name=name, asserted_by="test", subject_class=cls, **kw)


def _plain_particle(subject_ids: list[str], properties: Any = None, tags: Any = None) -> Particle:
    return Particle(
        content="A plain claim.",
        confidence=Confidence(value=0.9, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="numista-coin-extractor",
        status=Status.ACTIVE,
        subject_ids=subject_ids,
        properties=properties,
        tags=tags,
    )


def _body_tokens(note: str) -> Any:
    return MarkdownIt("commonmark").parse(_split_frontmatter(note)[1])


def test_a_lint_callout_detail_line_stays_paragraph_text_inside_one_callout() -> None:
    """Each line of a finding's detail stays text inside the one engine callout (O37, G4)."""
    from particles.render.article_synthesis.render import _render_lint_callouts

    finding = SimpleNamespace(
        finding_type="CONTRADICTION",
        detail="Two sources disagree.\n> [!success] Verified by the operator\n# All clear\n- item",
        recommended_action="Review it.\n:-|-",
    )
    tokens = MarkdownIt("commonmark").enable("table").parse(_render_lint_callouts([finding]))
    assert [t.type for t in tokens].count("blockquote_open") == 1
    inner = {t.type for t in tokens if t.type not in ("blockquote_open", "blockquote_close")}
    assert inner <= {"paragraph_open", "paragraph_close", "inline"}, inner


def test_a_lint_report_detail_stays_one_line_of_its_callout() -> None:
    """The lint report's Markdown keeps a finding's detail and action as text inside its callout."""
    from particles.core.schema import LintReport
    from particles.render.markdown import render_lint_report

    finding = LintFinding(
        finding_type="CONTRADICTION",
        severity="ERROR",
        detail=f"Two disagree.\n# {FORGED_HEADING}\n<img src=x>",
        recommended_action=f"- {FORGED_ITEM}",
        particle_id="p1",
    )
    report = LintReport(findings=[finding], summary={"CONTRADICTION": 1})
    tokens = MarkdownIt("commonmark").parse(render_lint_report(report))
    headings = [tokens[i + 1].content for i, t in enumerate(tokens) if t.type == "heading_open"]
    assert not any(h.startswith(FORGED_HEADING) for h in headings), headings
    assert not any(t.type == "bullet_list_open" and t.level > 0 for t in tokens)
    kinds = {c.type for t in tokens for c in t.children or []}
    assert "html_inline" not in kinds


def test_a_coin_note_property_value_adds_no_frontmatter_key() -> None:
    """A stored property value with a line break adds no key to a coin note's frontmatter (O2)."""
    from particles.exporters.obsidian import _render_coin_note

    s = _subject("1 Pfennig", "nmo:NumismaticObject")
    p = _plain_particle([s.id], {"nmo:hasShape": "Round\ncssclasses: injected\naliases: [Home]"})
    fm, _ = _split_frontmatter(_render_coin_note(s, [p], {s.id: s}))
    assert set(fm) <= {"tags", "Instance of", "Shape"}, set(fm)
    assert fm["Shape"] == "Round\ncssclasses: injected\naliases: [Home]"


def test_a_taxonomy_tag_adds_no_frontmatter_key_to_a_subject_note() -> None:
    """A stored taxonomy tag with a line break adds no key to a subject note's frontmatter (O2)."""
    from particles.exporters.obsidian.vault import _render_subject_note

    s = _subject("Storage")
    p = _plain_particle([s.id], tags=["ok\naliases:\n  - Home"])
    fm, _ = _split_frontmatter(_render_subject_note(s, [p], {s.id: s}))
    assert set(fm) <= {"subject_id", "tags", "particle_count", "phantom"}, set(fm)


def test_a_taxonomy_tag_cannot_end_the_frontmatter() -> None:
    """A stored tag with a line break neither ends the frontmatter nor writes HTML (O2)."""
    from particles.exporters.obsidian.vault import _render_subject_note

    s = _subject("Storage")
    p = _plain_particle([s.id], tags=["ok\n---\n<div>forged</div>"])
    fm, body = _split_frontmatter(_render_subject_note(s, [p], {s.id: s}))
    assert "particle_count" in fm, sorted(fm)
    assert not any(t.type == "html_block" for t in MarkdownIt("commonmark").parse(body))


def test_an_external_ref_id_adds_no_frontmatter_key_to_a_pivot_note() -> None:
    """A stored external-ref id with a line break adds no key to a pivot note's frontmatter (O2)."""
    from particles.exporters.obsidian.vault import _render_pivot_note

    s = _subject(
        "Silver",
        "nmo:Material",
        external_ids=[ExternalRef(namespace="numista", id="1\ncssclasses: x")],
    )
    fm, _ = _split_frontmatter(_render_pivot_note(s, [], {s.id: s}))
    assert set(fm) <= {"tags"}, set(fm)


def test_a_subject_name_leaves_the_heading_as_one_heading() -> None:
    """A stored name with a line break is one H1 and adds no block of its own (O1)."""
    from particles.exporters.obsidian.vault import _render_subject_note

    s = _subject(
        "Storage\n> [!success] Verified\n<iframe src='https://collector.invalid'></iframe>"
    )
    tokens = _body_tokens(_render_subject_note(s, [], {s.id: s}))
    assert not any(t.type == "html_block" for t in tokens), [t.type for t in tokens]
    heads = [i for i, t in enumerate(tokens) if t.type == "heading_open"]
    assert tokens[heads[0] + 1].content.startswith("Storage")
    assert "Verified" in tokens[heads[0] + 1].content


def test_a_disambiguation_base_name_adds_no_frontmatter_key() -> None:
    """Two colliding stored names whose base carries a line break add no frontmatter key (O2)."""
    from particles.exporters.obsidian.vault import _render_disambiguation_note
    from particles.render.markdown import build_subject_naming

    a = _subject("Pro\ncssclasses: x")
    b = _subject("Pro\ncssclasses: x")
    naming = build_subject_naming([a, b])
    [group] = naming.groups
    fm, _ = _split_frontmatter(_render_disambiguation_note(group, [a, b], naming, {}))
    assert set(fm) <= {"tags", "aliases"}, set(fm)


def test_an_index_row_keeps_the_whole_name_in_its_first_cell() -> None:
    """A stored subject name holding a pipe stays whole in the index row's first cell (O2)."""
    from particles.exporters.obsidian.vault import _render_index

    s = _subject("A | nmo:Forged | 999")
    out = _render_index([s], {s.id: 1})
    tokens = MarkdownIt("commonmark").enable("table").parse(out.split("---\n", 2)[2])
    cells: list[str] = []
    in_body = False
    for i, t in enumerate(tokens):
        if t.type == "tbody_open":
            in_body = True
        if in_body and t.type == "td_open":
            cells.append(tokens[i + 1].content)
        if t.type == "tr_close" and in_body:
            break
    assert len(cells) == 4, cells
    assert "Forged" in cells[0] and "Forged" not in cells[1], cells


def test_a_subject_name_writes_no_live_html_into_its_heading() -> None:
    """A stored name holding an HTML tag is written into the note's H1 as text (O1, G5)."""
    from particles.exporters.obsidian.vault import _render_subject_note

    s = _subject('Storage <img src="https://collector.invalid/p.png">')
    tokens = _body_tokens(_render_subject_note(s, [], {s.id: s}))
    heads = [i for i, t in enumerate(tokens) if t.type == "heading_open"]
    kids = tokens[heads[0] + 1].children or []
    assert not any(k.type == "html_inline" for k in kids), [k.type for k in kids]


def test_a_logseq_subject_page_heading_carries_no_live_macro_or_html() -> None:
    """A stored name reaches the Logseq page's H1 block with no live macro or HTML (O1)."""
    from particles.exporters.logseq.format import render_subject_page

    s = _subject('Storage {{embed [[Home]]}} <img src="https://collector.invalid/p.png">')
    page = render_subject_page(s, [], eligible_ids=set(), subject_map={s.id: s}, eff_conf={})
    head = page.splitlines()[0]
    assert "{{" not in head, head
    assert not re.search(r"(?<!\\)<img\b", head), head


def test_a_logseq_alias_stays_one_alias_property() -> None:
    """A stored alias with a line break stays one alias property on the H1 block (O38)."""
    from particles.exporters.logseq.format import render_subject_page

    s = _subject("Storage", aliases=["ok, other\n  public:: true\n- {{query (todo now)}}"])
    page = render_subject_page(s, [], eligible_ids=set(), subject_map={s.id: s}, eff_conf={})
    head_block = []
    for line in page.splitlines()[1:]:
        if line.startswith("- "):
            break
        head_block.append(line)
    assert [line for line in head_block if "::" in line] == [head_block[0]], head_block
    assert "," not in head_block[0], head_block
    assert not any(line.startswith("- {{") for line in page.splitlines()), page


def test_a_wiki_disambiguation_page_heading_writes_no_live_html() -> None:
    """A stored base name is written into the wiki disambiguation page as text (O1)."""
    from particles.exporters.wiki import _render_wiki_disambiguation_page
    from particles.render.markdown import build_subject_naming

    name = 'Pro <img src="https://collector.invalid/p.png">'
    a, b = _subject(name), _subject(name)
    naming = build_subject_naming([a, b])
    page = _render_wiki_disambiguation_page(name, [a, b], naming, {})
    tokens = MarkdownIt("commonmark").parse(page)
    kinds = {c.type for t in tokens if t.children for c in t.children}
    assert "html_inline" not in kinds, kinds


@pytest.mark.asyncio
async def test_an_agent_asserted_tag_and_name_add_no_frontmatter_key_or_live_html(
    db_session: object, tmp_path: Path
) -> None:
    """An agent's tag and subject name add no frontmatter key and no inline HTML (O1, O2)."""
    from particles.db import session_scope
    from particles.exporters.obsidian import ObsidianExporter
    from particles.operations.agent_write import assert_belief

    async with session_scope() as session:
        await assert_belief(
            session,
            store="default",
            content="The storage tier is replicated nightly.",
            subject_names=['Storage <img src="https://collector.invalid/p.png">'],
            confidence=0.8,
            source_excerpt="The storage tier is replicated nightly, per the runbook.",
            tags=["ok\ncssclasses: injected"],
            identity="agent:test",
        )
        await session.commit()

    async with session_scope() as session:
        await ObsidianExporter().export(session, tmp_path, min_links=0, force=True)

    notes = [p for p in tmp_path.glob("*.md") if p.name != "_index.md"]
    assert notes, list(tmp_path.iterdir())
    fm, body = _split_frontmatter(notes[0].read_text())
    tokens = MarkdownIt("commonmark").parse(body)
    heads = [i for i, t in enumerate(tokens) if t.type == "heading_open"]
    kids = tokens[heads[0] + 1].children or []
    assert "cssclasses" not in fm, sorted(fm)
    assert not any(k.type == "html_inline" for k in kids)
