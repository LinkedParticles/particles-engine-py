"""Layer A — deterministic citation ID-membership + density validation.

Layer A is the first of the two synthesis guard rails.
It is purely regex-based: scan the LLM body for ``[^p-xxxxxxxx]``
footnote references, confirm every short-ID belongs to the allowed set
(invented IDs hard-fail), and count uncited content paragraphs (LLM
padding hard-fails too).

This module also owns ``_short_id`` (the canonical 8-char particle-ID
prefix used in citations) and the regexes for citation references /
inline footnote definitions. Putting all "what does a citation look
like?" code in one place keeps layer_b, render, and orchestrate free of
shared regex constants and lets the package's dependency graph stay
acyclic: layer_a has no internal deps, everything else may depend on it.
"""

from __future__ import annotations

import re


def _short_id(particle_id: str) -> str:
    """Eight-char prefix used in the ``[^p-xxxxxxxx]`` footnote IDs."""
    return particle_id[:8]


# Footnote citation pattern. Matches ``[^p-abcdef12]`` — eight hex chars
# after the ``p-`` prefix. Layer A is exactly this regex
# plus a set-membership check. The ``p-`` separator (not ``p:``) keeps the
# identifier inside the GFM/Pandoc footnote-id character class
# (``[A-Za-z0-9._-]``); colons broke parsing in Obsidian.
_CITATION_RE = re.compile(r"\[\^p-([0-9a-fA-F]{8})\]")

# Inline footnote *definition* pattern — the lines the LLM keeps emitting
# despite the prompt forbidding it. Pandoc footnote definitions are a
# line starting with ``[^id]:`` plus optional indented continuation
# lines. We strip the whole block; the canonical definition is the one
# the exporter appends in the References section.
_INLINE_FOOTNOTE_DEF_RE = re.compile(
    r"^\[\^p-[0-9a-fA-F]{8}\]:.*(?:\n[ \t]+.*)*\n?",
    re.MULTILINE,
)


def _strip_inline_footnote_defs(body: str) -> str:
    """Remove ``[^p-xxxxxxxx]: …`` definition lines from an LLM body.

    Belt-and-suspenders to the prompt rule against inline definitions.
    Duplicate definitions (LLM-emitted + exporter-appended) cause Obsidian
    and other GFM-flavoured parsers to silently drop the footnote link.
    Stripping here means a non-compliant LLM cannot break the link no
    matter how often it ignores the prompt.
    """
    cleaned = _INLINE_FOOTNOTE_DEF_RE.sub("", body)
    # Collapse the blank-line runs the stripped definitions leave behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def count_uncited_paragraphs(body: str, *, allow_leading: int = 1) -> int:
    """Count body paragraphs over the uncited-allowance quota.

    The synthesis prompt forbids unsourced claims ("Do not invent
    claims; do not use any external knowledge"), but LLMs sometimes
    pad with general-knowledge prose anyway — paragraphs of plausible
    text with zero ``[^p-xxxxxxxx]`` citations. Layer A (ID
    membership) and Layer B (semantic alignment) only inspect *cited*
    sentences, so uncited padding slips through both gates.

    This check counts paragraphs that contain content but no citation,
    excluding:

    * Headings (``# …`` lines) — structural, not content.
    * Empty paragraphs from split artifacts.
    * The first ``allow_leading`` uncited content paragraphs — intros
      typically restate the subject identity in paraphrased prose that
      genuinely doesn't introduce a new cite-worthy fact.

    Returns the number of over-quota uncited paragraphs. Zero is
    acceptable; any positive integer means the LLM produced unbacked
    content and the article should be rejected for retry / fallback.
    """
    uncited_count = 0
    leading_skipped = 0
    for raw in body.split("\n\n"):
        para = raw.strip()
        if not para:
            continue
        # Skip headings (any ATX-style heading: `#`, `##`, …).
        if para.lstrip().startswith("#"):
            continue
        if _CITATION_RE.search(para) is not None:
            continue
        # Uncited content paragraph. Spend the leading-allowance budget
        # first; anything after counts toward the failure tally.
        if leading_skipped < allow_leading:
            leading_skipped += 1
            continue
        uncited_count += 1
    return uncited_count


def validate_citations(body: str, allowed_short_ids: set[str]) -> tuple[set[str], set[str]]:
    """Scan an LLM-synthesised body for ``[^p-xxxx]`` references.

    Returns ``(seen, invalid)``:
      * ``seen``   — every short-ID that appears in the body
      * ``invalid`` — IDs that the LLM used but were not in the input set

    An empty ``invalid`` set means Layer A passes. Layer A is mandatory:
    even if Layer B (semantic alignment) is disabled, a non-empty
    ``invalid`` set triggers retry-then-fallback. A claim that cites an
    invented ID is *uncited* — it is a regression on Particles' core
    promise.
    """
    seen = {m.group(1).lower() for m in _CITATION_RE.finditer(body)}
    invalid = seen - {sid.lower() for sid in allowed_short_ids}
    return seen, invalid
