# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Grounded answers: per-sentence citation and attribution.

In grounded mode the composer is shown each retrieved particle under a short
handle (``p-1a2b3c4d``, the CLI's display-id form) and must end every
sentence with one tag: the handles of the particles the sentence rests on,
``[inference]`` for a sentence it drew over cited claims itself, or
``[background]`` for one that comes from nothing in the store. This module
builds the handles, writes the grounded prompt rules, and parses the reply.

The parser never drops a sentence and never guesses a label. Every cited
handle is resolved against the particles the composer was actually given; a
citation that names none of them is recorded as invalid, and a unit left with
no valid citation and no declaration is ``UNATTRIBUTED``. The model's own
contribution is labelled, not banned, and it is never written to the store:
this is response-level attribution only.

Pure: no model call, no store.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from particles.core.schema import (
    AnswerAttribution,
    AttributedSentence,
    AttributionKind,
    Particle,
    ParticleType,
)

#: Hex characters of the id a handle carries — the CLI's display-id length.
_HANDLE_CHARS = 8

#: Trusted instructions appended to the composer's system turn in grounded
#: mode. The handles themselves ride the fenced particle list.
GROUNDED_RULES = """\
- Grounded answer: every particle above carries an id such as [p-1a2b3c4d].
  End every sentence, and every list item, with exactly one tag, placed
  before its final punctuation:
  - the ids of the particles it rests on, e.g. [p-1a2b3c4d] or
    [p-1a2b3c4d, p-5e6f7a8b]. Cite only ids from the list you were given, and
    cite every particle the sentence draws on;
  - [inference] when no particle states it and it is your own inference over
    the cited claims: a motive, a status, a conclusion, a link between claims,
    or a remark about what the knowledge base does or does not hold;
  - [background] when it comes from your own general knowledge and from no
    particle at all.
- If part of a sentence goes beyond the particles it cites, split that part
  into its own sentence with its own tag. Never let an inference or
  background ride on a citation.
- Write sentences or list items only: no headings, no tables, and no lead-in
  line such as "Here is what I found:".
- The NO_RELEVANT_KNOWLEDGE line, when it applies, takes no tag."""

# One tag: a declaration, or a citation list. A citation token is a ``p-`` /
# ``p:`` handle or a bare UUID; requiring that shape keeps ordinary bracketed
# prose ("[sic]") and markdown links ("[text](url)") out of the parse.
_CITE_TOKEN = r"(?:p[-:][^\[\]\s,]+|[0-9a-f]{8}-[0-9a-f-]{27,})"
_TAG = re.compile(
    r"\[\s*(inference|background|" + _CITE_TOKEN + r"(?:\s*,\s*" + _CITE_TOKEN + r")*)\s*\](?!\()",
    re.IGNORECASE,
)
#: Terminal punctuation (and any closing quote) a tag placed before the
#: sentence end leaves behind; it belongs to the unit the tag closes.
_TRAILING_PUNCT = re.compile(r"[.!?:;]+[\"'”’)\]]*")
_LINE_MARK = re.compile(r"(?m)^\s*(?:[-*+•]|\d{1,3}[.)]|#{1,6}|>)\s+")
_ALNUM = re.compile(r"[^\W_]")


def handle_for(particle_id: str, length: int = _HANDLE_CHARS) -> str:
    """The display handle of ``particle_id`` (``p-`` plus its first hex chars)."""
    return "p-" + particle_id.replace("-", "")[:length].lower()


def composed_particles(
    particles: Sequence[Particle],
    narrative_constituents: dict[str, list[Particle]] | None = None,
) -> list[Particle]:
    """Every particle the composer is shown, in the order it sees them.

    Each hit, and under a NARRATIVE hit its constituents, which the
    response step expands. A particle shown twice appears once.
    """
    narrative_constituents = narrative_constituents or {}
    seen: set[str] = set()
    out: list[Particle] = []
    for particle in particles:
        group = [particle]
        if particle.particle_type == ParticleType.NARRATIVE:
            group.extend(narrative_constituents.get(particle.id, []))
        for p in group:
            if p.id not in seen:
                seen.add(p.id)
                out.append(p)
    return out


def particle_handles(particles: Iterable[Particle]) -> dict[str, str]:
    """``{particle id: handle}``, unique within the set.

    Handles are eight hex chars; a set where two ids share a prefix gets the
    full hex id for those two, so a handle never names two particles.
    """
    ids = list(dict.fromkeys(p.id for p in particles))
    short = {pid: handle_for(pid) for pid in ids}
    counts: dict[str, int] = {}
    for handle in short.values():
        counts[handle] = counts.get(handle, 0) + 1
    return {
        pid: handle if counts[handle] == 1 else handle_for(pid, length=32)
        for pid, handle in short.items()
    }


@dataclass(frozen=True)
class GroundedAnswer:
    """A parsed grounded reply: the labelled answer text and its attribution."""

    answer: str
    attribution: AnswerAttribution


@dataclass
class _Group:
    """Adjacent tags read as one: everything the composer attached to a unit."""

    start: int
    end: int
    cited: list[str]
    invalid: list[str]
    declared: AttributionKind | None


def _resolver(handles: dict[str, str]) -> dict[str, str]:
    """Lower-cased handle (without prefix) or full id → particle id."""
    lookup: dict[str, str] = {}
    for pid, handle in handles.items():
        lookup[handle[2:].lower()] = pid
        lookup[pid.lower()] = pid
        lookup[pid.replace("-", "").lower()] = pid
    return lookup


def _normalise_token(token: str) -> str:
    token = token.strip().lower()
    for prefix in ("p-", "p:"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return token


def _groups(text: str, lookup: dict[str, str]) -> list[_Group]:
    groups: list[_Group] = []
    for match in _TAG.finditer(text):
        body = match.group(1)
        joins = bool(groups) and not text[groups[-1].end : match.start()].strip()
        group = groups[-1] if joins else _Group(match.start(), match.end(), [], [], declared=None)
        group.end = match.end()
        lowered = body.strip().lower()
        if lowered in ("inference", "background"):
            label = AttributionKind(lowered)
            # Both declared on one unit: the stronger disclosure wins.
            if group.declared is not AttributionKind.BACKGROUND:
                group.declared = label
        else:
            for raw in body.split(","):
                pid = lookup.get(_normalise_token(raw))
                if pid is None:
                    group.invalid.append(raw.strip())
                elif pid not in group.cited:
                    group.cited.append(pid)
        if not joins:
            groups.append(group)
    return groups


def _clean(unit: str) -> str:
    """The unit's prose: line markers, emphasis and spacing normalised."""
    unit = _LINE_MARK.sub("", unit).replace("**", "").replace("__", "")
    # A clause-level citation ("… [p-a], with … [p-b].") leaves the next unit
    # opening on the clause's comma.
    return " ".join(unit.split()).lstrip(",;: ")


def _tag(kind: AttributionKind, cited: Sequence[str], handles: dict[str, str]) -> str:
    if kind is AttributionKind.CITED:
        return "[" + ", ".join(handles[pid] for pid in cited) + "]"
    return f"[{kind.value}]"


def parse_grounded(text: str, handles: dict[str, str]) -> GroundedAnswer:
    """Split a grounded reply into attributed units and relabel it.

    A unit is the text between one tag (or run of adjacent tags) and the
    previous one, plus the punctuation that follows its tag. Its kind is the
    composer's declaration when it made one (keeping any ids it also cited as
    the premises it reasoned from), else ``CITED`` when at least one cited id
    is in ``handles``, else ``UNATTRIBUTED``. Text after the last tag that
    says anything is one more ``UNATTRIBUTED`` unit.

    The returned answer is the reply with every tag rewritten to its
    canonical form (valid handles only, or the label), and an
    ``[unattributed]`` label wherever the composer left a unit without one,
    so a reader of the plain text sees the same attribution the structure
    carries.
    """
    lookup = _resolver(handles)
    sentences: list[AttributedSentence] = []
    pieces: list[str] = []
    cursor = 0
    for group in _groups(text, lookup):
        unit = text[cursor : group.start]
        punct = _TRAILING_PUNCT.match(text, group.end)
        after = punct.end() if punct else group.end
        if group.declared is not None:
            kind = group.declared
        elif group.cited:
            kind = AttributionKind.CITED
        else:
            kind = AttributionKind.UNATTRIBUTED
        prose = _clean(unit.rstrip() + (punct.group(0) if punct else ""))
        if _ALNUM.search(prose):
            sentences.append(
                AttributedSentence(
                    text=prose,
                    kind=kind,
                    cited_ids=list(group.cited),
                    invalid_citations=list(group.invalid),
                )
            )
            pieces.append(unit + _tag(kind, group.cited, handles))
        else:
            # A tag with no prose of its own ("… [p-a]. [p-b]"): it belongs
            # to the unit before it, so its citations join that unit.
            pieces.append(unit + _tag(kind, group.cited, handles))
            if sentences:
                prior = sentences[-1]
                prior.cited_ids.extend(c for c in group.cited if c not in prior.cited_ids)
                prior.invalid_citations.extend(group.invalid)
                if prior.kind is AttributionKind.UNATTRIBUTED and group.declared is not None:
                    prior.kind = group.declared
                elif prior.kind is AttributionKind.UNATTRIBUTED and prior.cited_ids:
                    prior.kind = AttributionKind.CITED
        pieces.append(text[group.end : after])
        cursor = after
    tail = text[cursor:]
    prose = _clean(tail)
    if _ALNUM.search(prose):
        sentences.append(AttributedSentence(text=prose, kind=AttributionKind.UNATTRIBUTED))
        stripped = tail.rstrip()
        pieces.append(stripped + " [unattributed]" + tail[len(stripped) :])
    else:
        pieces.append(tail)
    return GroundedAnswer(
        answer="".join(pieces).strip(),
        attribution=AnswerAttribution(sentences=sentences),
    )


__all__ = [
    "GROUNDED_RULES",
    "GroundedAnswer",
    "composed_particles",
    "handle_for",
    "parse_grounded",
    "particle_handles",
]
