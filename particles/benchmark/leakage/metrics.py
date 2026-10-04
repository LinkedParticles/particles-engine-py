# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure functions of the leakage benchmark: the sentence splitter and the rates.

Nothing here calls a model or reads a store, so a saved report re-aggregates
for free and the splitter's output for a given answer is fixed: the same text
always yields the same sentences, in the same order, which is what makes two
runs' rates comparable sentence for sentence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from particles.benchmark.rot.schema import Rate
from particles.core.schema import AttributionKind

from .schema import LeakageReport, QueryLeakage, SentenceCounts, SentenceVerdict

# ---------------------------------------------------------------------------
# Sentence splitting — deterministic, markdown-aware, no model
# ---------------------------------------------------------------------------

#: Tokens whose trailing period does not end a sentence. Lower-cased, without
#: the final period. "etc." ending a real sentence is merged with the next one:
#: a deterministic over-merge, which costs a finer split, never a verdict.
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "cf",
        "dr",
        "e.g",
        "eg",
        "etc",
        "fig",
        "i.e",
        "ie",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "u.k",
        "u.s",
        "vs",
    }
)
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_ITEM = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_QUOTE = re.compile(r"^\s*>\s?")
#: A terminal run, any closing quotes or brackets, then whitespace.
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
_LAST_TOKEN = re.compile(r"(\S+)$")
_ALNUM = re.compile(r"[^\W_]")


def _blocks(text: str) -> list[tuple[str, bool]]:
    """Paragraphs, list items, headings, table rows, and code fences, in order.

    A list item, a heading, a table row, and a quoted line each start a new
    block; an indented continuation line joins the block above it; a blank
    line ends one. A fenced code block is kept whole as one block, flagged
    ``True`` so it is never split at a period.
    """
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []
    fenced: list[str] | None = None

    def flush() -> None:
        if current:
            blocks.append((" ".join(current), False))
            current.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if fenced is not None:
            if _FENCE.match(line):
                blocks.append(("\n".join(fenced), True))
                fenced = None
            else:
                fenced.append(line)
            continue
        if _FENCE.match(line):
            flush()
            fenced = []
            continue
        if not line.strip() or _RULE.match(line) or _TABLE_SEPARATOR.match(line):
            flush()
            continue
        if _HEADING.match(line):
            flush()
            blocks.append((_HEADING.sub("", line), False))
            continue
        if line.lstrip().startswith("|"):
            flush()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            blocks.append((" | ".join(c for c in cells if c), False))
            continue
        if _ITEM.match(line) or _QUOTE.match(line):
            flush()
            current.append(_QUOTE.sub("", _ITEM.sub("", line)).strip())
            continue
        current.append(line.strip())
    if fenced:
        blocks.append(("\n".join(fenced), True))
    flush()
    return blocks


def _is_abbreviation(prefix: str) -> bool:
    """Whether the token ending ``prefix`` (before its period) is not a sentence end."""
    found = _LAST_TOKEN.search(prefix)
    if found is None:
        return False
    token = found.group(1).lstrip("(\"'“‘*_").lower().rstrip(".")
    return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def _split_block(block: str) -> list[str]:
    """Sentence boundaries inside one block."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(block):
        end = match.end()
        following = block[end : end + 1]
        if following.islower():
            continue  # "approx. three", "i.e. the": the sentence goes on
        terminal = match.group(0).lstrip()[:1]
        if terminal == "." and _is_abbreviation(block[: match.start()]):
            continue
        sentences.append(block[start:end].strip())
        start = end
    sentences.append(block[start:].strip())
    return sentences


def _clean(sentence: str) -> str:
    """Drop emphasis markers; keep every word."""
    return sentence.replace("**", "").replace("__", "").strip()


def split_sentences(text: str) -> list[str]:
    """Split an answer into the units the judge scores, deterministically.

    Markdown-aware, because answers are markdown: each list item, heading,
    table row and fenced code block is its own unit, and prose inside a block
    splits at ``.``, ``!`` or ``?`` followed by whitespace — except after a
    known abbreviation or a single-letter initial, or before a lower-case
    letter. Decimals and version strings (``0.85``, ``v1.2.3``) carry no
    whitespace after the point and are never split. A unit with no letter or
    digit is dropped. No model is involved, so the same text always yields the
    same sentences.
    """
    out: list[str] = []
    for block, code in _blocks(text):
        for sentence in [block] if code else _split_block(block):
            cleaned = _clean(sentence)
            if cleaned and _ALNUM.search(cleaned):
                out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def _judged(results: Sequence[QueryLeakage]) -> list[QueryLeakage]:
    """Answers that were composed and judged — not refused, not excluded."""
    return [r for r in results if not r.excluded and not r.refused]


def unsupported_sentence_rate(results: Sequence[QueryLeakage]) -> Rate:
    """Silent unsupported over claim-bearing sentences, pooled over every judged answer.

    Ungrounded, every sentence is silent and this is the plain unsupported
    rate. Grounded, a sentence the composer labelled its own
    inference or background is disclosed: it stays in the denominator and
    out of the numerator, and :func:`labelled_rate` reports it.
    """
    rate = Rate()
    for result in _judged(results):
        rate = rate.merged(Rate(numerator=result.unsupported, denominator=result.claim_bearing))
    return rate


def labelled_rate(results: Sequence[QueryLeakage], kind: AttributionKind) -> Rate:
    """Claim-bearing sentences labelled ``kind``, over every claim-bearing sentence."""
    rate = Rate()
    for result in _judged(results):
        count = result.inference if kind is AttributionKind.INFERENCE else result.background
        rate = rate.merged(Rate(numerator=count, denominator=result.claim_bearing))
    return rate


def unsupported_rate_by_source(results: Sequence[QueryLeakage]) -> dict[str, Rate]:
    """The pooled rate per question source — reported apart, never blended silently."""
    by_source: dict[str, list[QueryLeakage]] = {}
    for result in results:
        by_source.setdefault(result.source.value, []).append(result)
    return {source: unsupported_sentence_rate(rows) for source, rows in sorted(by_source.items())}


def mean_answer_fraction(results: Sequence[QueryLeakage]) -> float | None:
    """Mean of the per-answer fractions, over answers with a claim-bearing sentence."""
    fractions = [f for f in (r.unsupported_fraction for r in _judged(results)) if f is not None]
    return sum(fractions) / len(fractions) if fractions else None


def answers_with_unsupported(results: Sequence[QueryLeakage]) -> Rate:
    """Answers with at least one unsupported sentence, over answers with a claim."""
    rate = Rate()
    for result in _judged(results):
        if result.claim_bearing:
            rate.add(result.unsupported > 0)
    return rate


def refusal_rate(results: Sequence[QueryLeakage]) -> Rate:
    """Refused answers over every question the op answered or refused."""
    rate = Rate()
    for result in results:
        if not result.excluded:
            rate.add(bool(result.refused))
    return rate


def sentence_counts(results: Sequence[QueryLeakage]) -> SentenceCounts:
    """Every sentence of every judged answer, by outcome."""
    counts = SentenceCounts()
    for result in _judged(results):
        counts.total += len(result.sentences)
        counts.supported += result.supported
        counts.unsupported += result.unsupported
        counts.no_claim += result.no_claim
        counts.judge_excluded += result.judge_excluded
        counts.inference += result.inference
        counts.background += result.background
        for sentence in result.sentences:
            counts.invalid_citations += sentence.invalid_citations
            entailed = sentence.verdict is SentenceVerdict.SUPPORTED
            unsupported = sentence.verdict is SentenceVerdict.UNSUPPORTED
            match sentence.attribution:
                case AttributionKind.INFERENCE:
                    counts.inference_entailed += entailed
                case AttributionKind.BACKGROUND:
                    counts.background_entailed += entailed
                case AttributionKind.CITED:
                    counts.cited += 1
                    counts.unsupported_cited += unsupported
                case AttributionKind.UNATTRIBUTED:
                    counts.unattributed += 1
                    counts.unsupported_unattributed += unsupported
                case None:
                    pass
    return counts


def aggregate(report: LeakageReport) -> LeakageReport:
    """Recompute every rate of ``report`` from its rows — pure, so re-runnable."""
    rows = report.results
    excluded: dict[str, int] = {}
    for row in rows:
        if row.excluded:
            excluded[row.excluded] = excluded.get(row.excluded, 0) + 1
    return report.model_copy(
        update={
            "unsupported_sentences": unsupported_sentence_rate(rows),
            "unsupported_sentences_by_source": unsupported_rate_by_source(rows),
            "inference_sentences": labelled_rate(rows, AttributionKind.INFERENCE),
            "background_sentences": labelled_rate(rows, AttributionKind.BACKGROUND),
            "mean_answer_fraction": mean_answer_fraction(rows),
            "answers_with_unsupported": answers_with_unsupported(rows),
            "refused": refusal_rate(rows),
            "sentences": sentence_counts(rows),
            "excluded": dict(sorted(excluded.items())),
        }
    )


__all__ = [
    "aggregate",
    "answers_with_unsupported",
    "labelled_rate",
    "mean_answer_fraction",
    "refusal_rate",
    "sentence_counts",
    "split_sentences",
    "unsupported_rate_by_source",
    "unsupported_sentence_rate",
]
