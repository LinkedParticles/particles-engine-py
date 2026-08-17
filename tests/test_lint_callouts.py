"""lint callouts are a write-time layer, not cached content.

These tests pin ``apply_lint_callouts`` / ``strip_lint_callouts`` — the helper
that exporters call at write time so a finding's text (or a finding appearing /
resolving) surfaces on the next export without regenerating the cached body.

The callout block is identified structurally (a `> [!sev] FINDING_TYPE` run),
not by a sentinel: original HTML-comment fence rendered as visible
grey text in Obsidian Live Preview, so it is no longer emitted — only stripped
on sight to migrate notes written by 0.61.x.
"""

from __future__ import annotations

from particles.core.schema import LintFinding
from particles.exporters.article_synthesis import (
    _parse_frontmatter,
    apply_lint_callouts,
    strip_lint_callouts,
)

_WIKIDATA_HEADER = "> [!warning] WIKIDATA_LINK_MISMATCH"
_CONTRA_HEADER = "> [!danger] CONTRADICTION"


def _wikidata_finding() -> LintFinding:
    return LintFinding(
        particle_id=None,
        subject_id="543b86be-0000-0000-0000-000000000000",
        corpus_entry_id=None,
        finding_type="WIKIDATA_LINK_MISMATCH",
        severity="WARNING",
        detail="Subject 'Douglas B. Lenat' has Wikidata link Q559334. "
        "If the link is wrong, remove it: "
        "`particles subjects unlink 543b86be wikidata:Q559334`.",
    )


def _contradiction_finding() -> LintFinding:
    return LintFinding(
        particle_id="p1",
        subject_id=None,
        corpus_entry_id=None,
        finding_type="CONTRADICTION",
        severity="ERROR",
        detail="Disagrees with another source.",
    )


_ARTICLE = (
    "---\n"
    "particle_count: 2\n"
    "contradictions: 0\n"
    "synthesis: llm\n"
    "---\n\n"
    "# Douglas B. Lenat\n\n"
    "Douglas Bruce Lenat was an American computer scientist. [^p-aabbccdd]\n\n"
    "## References\n\n"
    "### p-aabbccdd\n\n"
    "[^p-aabbccdd]: (no public URI)\n"
)


class TestApplyLintCallouts:
    def test_splices_callout_after_h1_without_html_comment(self) -> None:
        out = apply_lint_callouts(_ARTICLE, [_wikidata_finding()])
        assert _WIKIDATA_HEADER in out
        # No HTML-comment sentinel — it renders as visible text in Obsidian.
        assert "<!--" not in out
        # Warning sits between the H1 and the prose.
        h1 = out.index("# Douglas B. Lenat")
        callout = out.index(_WIKIDATA_HEADER)
        prose = out.index("Douglas Bruce Lenat was an American")
        assert h1 < callout < prose
        # The current resolution command is present verbatim.
        assert "particles subjects unlink 543b86be wikidata:Q559334" in out
        # Frontmatter and references survive untouched.
        assert out.startswith("---\n")
        assert "## References" in out

    def test_idempotent(self) -> None:
        once = apply_lint_callouts(_ARTICLE, [_wikidata_finding()])
        twice = apply_lint_callouts(once, [_wikidata_finding()])
        assert once == twice
        # Exactly one callout — no accumulation.
        assert twice.count(_WIKIDATA_HEADER) == 1

    def test_changed_finding_text_replaces_old(self) -> None:
        """The motivating case: the cached body carries an old callout; a
        re-export with the new finding text replaces it, not appends."""
        old = apply_lint_callouts(
            _ARTICLE,
            [
                LintFinding(
                    particle_id=None,
                    subject_id="s",
                    corpus_entry_id=None,
                    finding_type="WIKIDATA_LINK_MISMATCH",
                    severity="WARNING",
                    detail="Use: `particles subjects merge 543b86be TARGET_ID`",
                )
            ],
        )
        assert "TARGET_ID" in old
        new = apply_lint_callouts(old, [_wikidata_finding()])
        assert "TARGET_ID" not in new
        assert "particles subjects unlink 543b86be wikidata:Q559334" in new
        assert new.count(_WIKIDATA_HEADER) == 1

    def test_resolved_finding_is_removed(self) -> None:
        """No findings → any prior callout block is stripped, leaving clean body."""
        with_callout = apply_lint_callouts(_ARTICLE, [_wikidata_finding()])
        cleared = apply_lint_callouts(with_callout, [])
        assert "WIKIDATA_LINK_MISMATCH" not in cleared
        # The article body is otherwise intact.
        assert "Douglas Bruce Lenat was an American" in cleared
        assert "## References" in cleared

    def test_refreshes_contradictions_frontmatter_count(self) -> None:
        out = apply_lint_callouts(_ARTICLE, [_contradiction_finding()])
        fm = _parse_frontmatter(out)
        assert fm is not None
        assert fm["contradictions"] == 1
        # Clearing the finding resets the count.
        cleared = apply_lint_callouts(out, [])
        fm2 = _parse_frontmatter(cleared)
        assert fm2 is not None
        assert fm2["contradictions"] == 0

    def test_migrates_legacy_unfenced_callout(self) -> None:
        """A body cached before has an unfenced callout baked in; the
        first export strips it (no duplicate) and re-emits the current one."""
        legacy = (
            "---\ncontradictions: 0\n---\n\n"
            "# Douglas B. Lenat\n\n"
            "> [!warning] WIKIDATA_LINK_MISMATCH\n"
            "> Use: `particles subjects merge 543b86be TARGET_ID`\n\n"
            "Douglas Bruce Lenat was an American computer scientist. [^p-aabbccdd]\n"
        )
        out = apply_lint_callouts(legacy, [_wikidata_finding()])
        assert out.count("WIKIDATA_LINK_MISMATCH") == 1
        assert "TARGET_ID" not in out
        assert "particles subjects unlink 543b86be wikidata:Q559334" in out

    def test_migrates_legacy_0_61_html_comment_fence(self) -> None:
        """Notes written by 0.61.x carry an HTML-comment fence that renders as
        visible grey text in Obsidian. The next export strips the fence and the
        callout it wrapped, replacing them with a clean structural callout."""
        fenced = (
            "---\ncontradictions: 0\n---\n\n"
            "# Douglas B. Lenat\n\n"
            "<!-- particles:lint-callouts -->\n"
            "> [!warning] WIKIDATA_LINK_MISMATCH\n"
            "> Use: `particles subjects merge 543b86be TARGET_ID`\n"
            "<!-- /particles:lint-callouts -->\n\n"
            "Douglas Bruce Lenat was an American computer scientist. [^p-aabbccdd]\n"
        )
        out = apply_lint_callouts(fenced, [_wikidata_finding()])
        # The visible HTML comment is gone.
        assert "<!--" not in out
        assert "particles:lint-callouts" not in out
        # Exactly one callout, with the new text; prose preserved.
        assert out.count("WIKIDATA_LINK_MISMATCH") == 1
        assert "particles subjects unlink 543b86be wikidata:Q559334" in out
        assert "Douglas Bruce Lenat was an American" in out

    def test_preserves_non_lint_callouts(self) -> None:
        """The structural banners (Unverified link, Structured-listing) are not
        lint-finding callouts — leave them be."""
        body = (
            "# GDR\n\n"
            "> [!warning] Unverified Wikidata link\n"
            "> Candidate: `wikidata:Q1` (confidence 0.10).\n\n"
            "> [!note] Structured-listing render\n"
            "> This article was rendered without LLM synthesis.\n\n"
            "## Claims\n\n- a claim [^p-aabbccdd]\n"
        )
        out = apply_lint_callouts(body, [_contradiction_finding()])
        assert "> [!warning] Unverified Wikidata link" in out
        assert "> [!note] Structured-listing render" in out
        assert _CONTRA_HEADER in out

    def test_no_findings_no_callout_on_clean_body(self) -> None:
        out = apply_lint_callouts(_ARTICLE, [])
        assert "[!" not in out
        # A clean body is returned essentially unchanged.
        assert "# Douglas B. Lenat" in out
        assert "## References" in out


class TestStripLintCallouts:
    def test_strips_structural_block_idempotently(self) -> None:
        with_callout = apply_lint_callouts(_ARTICLE, [_wikidata_finding()])
        stripped = strip_lint_callouts(with_callout)
        assert "WIKIDATA_LINK_MISMATCH" not in stripped
        assert strip_lint_callouts(stripped) == stripped

    def test_strips_legacy_html_comment_fence(self) -> None:
        fenced = (
            "# X\n\n"
            "<!-- particles:lint-callouts -->\n"
            "> [!warning] WIKIDATA_LINK_MISMATCH\n> detail\n"
            "<!-- /particles:lint-callouts -->\n\n"
            "prose\n"
        )
        stripped = strip_lint_callouts(fenced)
        assert "<!--" not in stripped
        assert "WIKIDATA_LINK_MISMATCH" not in stripped
        assert "prose" in stripped

    def test_leaves_callout_free_text_unchanged(self) -> None:
        assert strip_lint_callouts(_ARTICLE) == _ARTICLE
