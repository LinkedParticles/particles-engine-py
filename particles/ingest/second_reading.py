# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The second reading of a contradiction the first probe confirmed.

The first contradiction probe sees two claim texts and nothing else. The second
reading sees, for each claim, its source's name, kind and date and the passage
the claim came from, on ``llm.verification``, under the instruction the pair's
source kinds choose. Two callers share it:

* the nightly census and the memory audit read a probe-flagged pair before it
  becomes a finding or a record (``operations.lint.contradictions``);
* extraction reads a probe-confirmed pair before the write loop acts on it
  (``ingest.pipeline``), and pass 3b re-reads the records
  extraction opened without one (``operations.contradiction_disclosure``).

It lives in ``ingest`` so extraction can call it with no import from
``operations``. Two ways of building a claim's context live here:

* :func:`claim_context` is the census's: the claim's stored passage
  (``particles particle source``: the chunk it hashed to, else the first
  best-matching paragraph);
* :func:`in_flight_context` is extraction's: the chunk the claim was read
  from, windowed around both claims' words, dated by the snapshot. It never
  picks a single paragraph, so a transcript that repeats a command is not read
  at its first run (§ Context). :func:`snapshot_claim_context` builds
  the same view of a stored claim from its own snapshot.

Gather (store and blob reads) and the call are separate steps, so a caller can
end its read transaction before the LLM round trip.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.contradiction_disclosure import READING_OBSERVED, READING_STANDING
from particles.core.probe_verdict import prompt_hash
from particles.core.schema import Mutability, Particle, ProvenanceRefType, SourceType
from particles.corpus.deposit import load_blob
from particles.corpus.store import get_entry, get_snapshot, resolve_snapshot_for_blob
from particles.extraction.general import sniff_image_media_type
from particles.ingest.source_passage import (
    PassageMatch,
    extractor_view_text,
    find_chunk,
    focus_window,
    hydrate_source_passage,
)
from particles.llm.breaker import _llm_call, llm_circuit_open, record_unusable_reply

if TYPE_CHECKING:
    from particles.llm import CompletionRequest

__all__ = [
    "READING_OBSERVED",
    "READING_STANDING",
    "ClaimContext",
    "ProbeVerdict",
    "SourceKind",
    "claim_context",
    "in_flight_context",
    "one_line",
    "read_pair",
    "reading_for",
    "second_reading_prompt_hash",
    "snapshot_claim_context",
    "snapshot_text",
    "source_kind",
    "source_name",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The verdict: shared with the first probe
# ---------------------------------------------------------------------------

# The probe's verdict shape: a schema-enforcing adapter (the
# OpenAICompatProvider when the entry's ``structured_output`` enforces) constrains the
# reply to this object; the Anthropic adapter ignores it and keeps answering
# in the YES/NO text protocol. The parser below accepts both.
_PROBE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradicts": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["contradicts"],
    "additionalProperties": False,
}

#: A rendered contradiction description is one line of at most this many characters.
_DESCRIPTION_MAX_CHARS = 160

_VERDICT_RE = re.compile(r"^\W*VERDICT\W*:\W*(YES|NO)\W*$", re.IGNORECASE)


def one_line(text: str, limit: int = _DESCRIPTION_MAX_CHARS) -> str:
    """``text`` on one line, cut at a word boundary with an ellipsis past ``limit``."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "…"


@dataclass(frozen=True)
class ProbeVerdict:
    """One parsed probe reply. ``usable`` is False for a reply with no clear verdict."""

    usable: bool
    contradicts: bool = False
    description: str = ""


def _parse_probe_verdict(response: str) -> ProbeVerdict:
    """Parse a probe reply, refusing to read a verdict into a cut or off-protocol reply.

    Two dialects. The enforced JSON verdict object
    (``{"contradicts": true, "description": …}``), where a reply
    cut mid-object is invalid JSON; and the text protocol, whose verdict is the
    reply's final line (``VERDICT: YES`` / ``VERDICT: NO``) after the reason,
    so a reply cut at its budget has no verdict line. Either way a truncated,
    ambiguous (both or neither), or free-form reply is ``usable=False`` and is
    never counted as a contradiction.
    """
    stripped = response.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return ProbeVerdict(usable=False)
        if not isinstance(data, dict) or not isinstance(data.get("contradicts"), bool):
            return ProbeVerdict(usable=False)
        description = one_line(str(data.get("description") or "")) or "contradiction detected"
        return ProbeVerdict(usable=True, contradicts=data["contradicts"], description=description)

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    verdicts = [m.group(1).upper() for line in lines if (m := _VERDICT_RE.match(line))]
    if len(set(verdicts)) != 1 or not lines or not _VERDICT_RE.match(lines[-1]):
        return ProbeVerdict(usable=False)
    reason = " ".join(line for line in lines if not _VERDICT_RE.match(line))
    reason = re.sub(r"^\W*REASON\W*:\s*", "", reason, flags=re.IGNORECASE)
    if verdicts[0] == "NO":
        # The reason is kept: a re-reading that withdraws a pair records why.
        return ProbeVerdict(usable=True, contradicts=False, description=one_line(reason))
    return ProbeVerdict(
        usable=True,
        contradicts=True,
        description=one_line(reason) or "contradiction detected",
    )


# ---------------------------------------------------------------------------
# The claim's context
# ---------------------------------------------------------------------------

#: Characters of source passage shown per claim, taken around the claim's own
#: words (:func:`~particles.ingest.source_passage.focus_window`). A memory
#: note's located passage is a line or a paragraph; a long one used to be cut
#: from its start, which hid a fix stated after the bug it fixed (the second
#: held-out audit store, 2026-09-27).
_VERIFY_PASSAGE_CHARS = 900


class SourceKind(StrEnum):
    """What kind of statement a claim's source makes, for the second reading."""

    NOTE = "note"
    """A note kept current (a memory file, a document): it states standing facts."""
    RECORD = "record"
    """A time-stamped, append-only record (a session transcript, a log): it
    reports what was the case when it was written."""


def source_kind(source_type: str | None, mutability: str | None) -> SourceKind:
    """Classify a claim's source (pure).

    An ``APPEND_ONLY`` entry is a record: the platform never rewrites it, so
    what it says is dated to when it was written. A
    ``CONVERSATION`` is a record whatever its declared mutability. Everything
    else, and a claim with no readable source, reads as a note: the rule the
    second reading was measured under.
    """
    if mutability == Mutability.APPEND_ONLY or source_type == SourceType.CONVERSATION:
        return SourceKind.RECORD
    return SourceKind.NOTE


@dataclass(frozen=True)
class ClaimContext:
    """What the second reading sees of one claim: its text and where it came from."""

    content: str
    #: The source's name: the tail of its URI (a note's file name), or ``"unknown"``.
    source: str
    #: The source's date as ``YYYY-MM-DD`` (content date, else capture date), or ``"unknown"``.
    date: str
    #: The passage the claim was located in, or ``""`` when none was found.
    passage: str
    #: Whether the source states standing facts or records a moment.
    kind: SourceKind = SourceKind.NOTE
    #: When a record observed the claim, as ``YYYY-MM-DD HH:MM UTC`` (the capture
    #: time of the snapshot that states it), or ``"unknown"``.
    observed: str = "unknown"
    #: The source's type, for naming a record (``CONVERSATION`` is a session transcript).
    source_type: str | None = None


def reading_for(a: ClaimContext, b: ClaimContext) -> str:
    """Which second-reading instruction a pair gets (pure).

    Two notes get the standing-facts instruction, the one the audit's
    precision was measured under; a pair with a claim from a record
    gets the observed-state instruction.
    """
    if SourceKind.RECORD in (a.kind, b.kind):
        return READING_OBSERVED
    return READING_STANDING


@dataclass(frozen=True)
class _ClaimSource:
    """One claim's source, read once: its note's name, date and full passage."""

    source: str
    date: str
    #: The located passage, or the whole note (capped) when none was located;
    #: ``""`` when the source is unavailable. Windowed per pair.
    text: str
    kind: SourceKind = SourceKind.NOTE
    observed: str = "unknown"
    source_type: str | None = None


def _observed_at(when: datetime | None) -> str:
    return when.strftime("%Y-%m-%d %H:%M UTC") if when is not None else "unknown"


def source_name(uri_r: str | None) -> str:
    """A source's name as the second reading shows it: the tail of its URI (pure)."""
    if not uri_r:
        return "unknown"
    return uri_r.rstrip("/").rsplit("/", 1)[-1] or uri_r


def _content_of(partner: Particle | str | None) -> str | None:
    return partner.content if isinstance(partner, Particle) else partner


async def _claim_source(session: AsyncSession, particle: Particle) -> _ClaimSource:
    """Read a claim's source name, source date and passage (the gather step).

    Read-only: the source-passage hydration (the ``particles particle source``
    path, one blob read), the snapshot row, and the entry row for its
    mutability. A claim with no readable source still gets one, with
    ``unknown`` fields and no passage, so a store with missing blobs degrades
    to a reading of the bare claims rather than failing.
    """
    passage = await hydrate_source_passage(session, particle.id)
    source, date, text, observed = "unknown", "unknown", "", "unknown"
    source_type: str | None = None
    mutability: str | None = None
    if passage is not None:
        source_type = passage.source_type
        if passage.uri_r:
            source = source_name(passage.uri_r)
        # A WHOLE match (no paragraph cleared the overlap floor, as for a
        # paraphrased claim) is the whole note; its window is still chosen by
        # the claim's own words, or is empty when none occur.
        if passage.match is not PassageMatch.UNAVAILABLE:
            text = passage.text
        if passage.corpus_entry_id:
            entry = await get_entry(session, passage.corpus_entry_id)
            mutability = entry.mutability if entry is not None else None
        if passage.snapshot_id:
            snapshot = await get_snapshot(session, passage.snapshot_id)
            when = (snapshot.content_published_at or snapshot.captured_at) if snapshot else None
            if when is not None:
                date = when.date().isoformat()
                observed = _observed_at(when)
    return _ClaimSource(
        source=source,
        date=date,
        text=text,
        kind=source_kind(source_type, mutability),
        observed=observed,
        source_type=source_type,
    )


def _claim_context_for(
    particle: Particle, src: _ClaimSource, partner: Particle | str | None
) -> ClaimContext:
    """The pair-specific view of one claim (the decide step; pure).

    The passage window is chosen by the claim's own words and, second, by its
    partner's, so each side shows the part of its note that bears on the
    comparison.
    """
    passage = focus_window(
        src.text,
        particle.content,
        _VERIFY_PASSAGE_CHARS,
        also=_content_of(partner),
    )
    return ClaimContext(
        content=particle.content,
        source=src.source,
        date=src.date,
        passage=passage,
        kind=src.kind,
        observed=src.observed,
        source_type=src.source_type,
    )


async def claim_context(
    session: AsyncSession, particle: Particle, partner: Particle | str | None = None
) -> ClaimContext:
    """Gather a claim's source name, source date and passage for the second reading.

    ``partner``, when given, is the other claim of the pair being read (a
    particle, or the text of a candidate not yet stored): the passage window
    then leans toward the part of the note that bears on it.
    """
    return _claim_context_for(particle, await _claim_source(session, particle), partner)


# ---------------------------------------------------------------------------
# The in-flight context
# ---------------------------------------------------------------------------


def snapshot_text(content: bytes, source_type: str | None) -> str:
    """The text the extractor was shown for a snapshot's bytes; ``""`` for an image (pure)."""
    if sniff_image_media_type(content) is not None:
        return ""
    return extractor_view_text(content, source_type)


def in_flight_context(
    content: str,
    *,
    text: str,
    chunk_hash: str | None,
    partner: str | None,
    source: str,
    source_type: str | None,
    mutability: str | None,
    when: datetime | None,
) -> ClaimContext:
    """One claim's context as extraction holds it (pure).

    ``text`` is the snapshot's extractor-view text (:func:`snapshot_text`).
    The passage is the chunk the claim was read from: the re-derived chunk
    that hashes to ``chunk_hash`` when the claim carries one (the chunked
    carry-forward path), else the whole text. It is windowed around
    the claim's words and, second, its partner's, so no single paragraph is
    chosen for it: in a transcript that repeats a command, the first
    best-matching paragraph can be an earlier run than the one the claim
    describes. ``when`` is the snapshot's content date, else its capture time:
    for an append-only record, when the record said it.
    """
    chunk = (find_chunk(text, chunk_hash) if chunk_hash else None) or text
    passage = focus_window(chunk, content, _VERIFY_PASSAGE_CHARS, also=partner) if chunk else ""
    return ClaimContext(
        content=content,
        source=source,
        date=when.date().isoformat() if when is not None else "unknown",
        passage=passage,
        kind=source_kind(source_type, mutability),
        observed=_observed_at(when),
        source_type=source_type,
    )


async def snapshot_claim_context(
    session: AsyncSession, particle: Particle, partner: Particle | str | None = None
) -> ClaimContext:
    """A stored claim's in-flight context, read from its own snapshot.

    The view :func:`in_flight_context` gives a candidate, built for a claim
    already stored: its first SOURCE ref names the entry, the snapshot and,
    on the chunked path, the chunk hash. Read-only (the entry and snapshot
    rows, one blob read). A claim with no readable source degrades to a
    reading of its bare text, as :func:`claim_context` does.
    """
    ref = next(
        (
            r
            for r in particle.provenance
            if r.type is ProvenanceRefType.SOURCE and r.corpus_entry_id
        ),
        None,
    )
    text = ""
    source, source_type, mutability, when = "unknown", None, None, None
    if ref is not None:
        entry = await get_entry(session, ref.corpus_entry_id)
        if entry is not None:
            source, source_type, mutability = (
                source_name(entry.uri_r),
                entry.source_type,
                entry.mutability,
            )
        snapshot = (
            await get_snapshot(session, ref.snapshot_id)
            if ref.snapshot_id
            else await resolve_snapshot_for_blob(session, ref.corpus_entry_id)
        )
        if snapshot is not None:
            when = snapshot.content_published_at or snapshot.captured_at
            try:
                text = snapshot_text(load_blob(snapshot.content_hash), source_type)
            except FileNotFoundError:
                text = ""
    return in_flight_context(
        particle.content,
        text=text,
        chunk_hash=ref.chunk_hash if ref is not None else None,
        partner=_content_of(partner),
        source=source,
        source_type=source_type,
        mutability=mutability,
        when=when,
    )


_VERIFY_INSTRUCTION = """\
You are checking an agent's memory, a set of notes, for claims that contradict each other. \
Decide whether the two claims in the user message contradict: whether an agent that trusted \
both notes would hold two beliefs that cannot both be true of the same thing.

Each claim comes with the name of its note, the note's date, and the passage the claim was \
extracted from. Read each claim in the light of its passage.

Most notes state standing facts: how something is, how it works, or how to do it. Two standing \
facts about the same thing that disagree are a contradiction even when the notes carry \
different dates, because the older note still asserts its claim. One of the two is then wrong \
or out of date, which is exactly what this check exists to find.

These are not contradictions:
- A change that a passage itself describes: a series of iterations, runs, versions or steps, \
an earlier value and a later one within one record, a plan and what later shipped. Count this \
only when a passage frames its claim as one point in such a sequence. Different note dates \
alone never make a change.
- Items of one list, or alternatives a passage offers together ("X or Y", "happened twice, in \
X and in Y").
- One claim narrower than the other, or a consequence, cause or restatement of it.
- Claims about different things, including a word such as "this", "it" or "the issue" that \
refers to different things in the two passages.
- A claim that reports what someone said, believed or got wrong, set against what was \
actually the case.
- Two quantities that differ by design, such as a stored value and a value derived from it.

Answer YES only when both claims cannot be true of the same thing, so at least one note is \
wrong or out of date.

Reply with exactly two lines and nothing else:
REASON: <one sentence of at most 25 words naming the conflicting detail, or why there is none>
VERDICT: YES or VERDICT: NO
"""

#: The instruction for a pair with a claim from a time-stamped record
#: (:data:`READING_OBSERVED`). The standing-facts rule above is true
#: of memory notes and false of a session transcript, which is full of
#: moment-in-time state: "the version is now X", "lint passes with N". Read
#: under that rule, two sessions' snapshots of a value that moved between them
#: were confirmed as contradictions (the owner's store, 2026-09-28).
_VERIFY_OBSERVED_INSTRUCTION = """\
You are checking an agent's memory for claims that contradict each other. Decide whether the \
two claims in the user message contradict: whether an agent that trusted both would hold two \
beliefs that cannot both be true.

Each claim comes with its source and the passage the claim was extracted from. Read each claim \
in the light of its passage. At least one source is a record: a session transcript or another \
append-only log, stamped with the time it was observed. A record reports what was the case \
when it was written: a count, a version, a status, what a check printed, what the session was \
working on. A note, by contrast, is a file kept current that states standing facts.

A value observed at one time and a different value observed at a later time is a change, not a \
contradiction: counts grow, versions advance, a bug seen in one session is fixed before the \
next, a check that failed later passes, a note is updated after a session. When the two \
observation times differ, read a difference in such a current state as a change, even when \
neither passage mentions one.

Answer YES only in one of two cases:
1. The same moment. Both claims describe the state of the same thing at the same moment and \
disagree. The moment is the same only when both claims pin it to one point (the same release, \
run, commit, date or event), not merely because the observation times are close.
2. A fixed fact. Both claims answer a question that has one answer for all time, and answer it \
differently: which pull request implemented a given change, which number a given item was \
given, when a given version was released, or a permanent property. An event that recurs, \
such as a version bump, a test run or a lint pass, is not one event: two of them can both \
have happened.

Read each claim as the agent will, by its own words. Two claims that name the same thing by \
the same explicit identifier (a record number, a file name, a pull request number) are about \
the same thing even when a passage shows the identifier named something else when it was \
written: a reused or renumbered identifier leaves one of the claims wrong as it now reads.

These are not contradictions either:
- Items of one list, or alternatives a passage offers together ("X or Y", "happened twice, in \
X and in Y").
- One claim narrower than the other, or a consequence, cause or restatement of it.
- Claims about different things, including a word such as "this", "it", "the PR" or "the \
issue" that refers to different things in the two passages.
- A claim that reports what someone said, believed or got wrong, set against what was \
actually the case.
- Two quantities that differ by design, such as a stored value and a value derived from it.

Reply with exactly two lines and nothing else:
REASON: <one sentence of at most 25 words naming the conflicting detail, or why there is none>
VERDICT: YES or VERDICT: NO
"""


def _source_line(c: ClaimContext) -> str:
    """How the observed-state reading names one claim's source."""
    if c.kind is SourceKind.RECORD:
        what = "session transcript" if c.source_type == SourceType.CONVERSATION else "record"
        return f"record: {c.source} ({what}, observed {c.observed})"
    return f"note: {c.source} (a note kept current, dated {c.date})"


def _verify_request(
    a: ClaimContext, b: ClaimContext, *, nonce: str | None = None
) -> CompletionRequest:
    """Build one second reading: the instruction trusted, both claims and passages fenced.

    Everything but the instruction was extracted from, or is, untrusted source
    text, so it all goes behind the per-call nonce fence (F3 hardening, as in
    ``operations.lint.contradictions._probe_request``). Two notes are read under the standing-facts
    instruction; a pair with a record, under the observed-state one, with
    each record's observation time. ``nonce`` is fixed only by
    :func:`second_reading_prompt_hash`; a real reading always mints its own.
    """
    from particles.llm import CompletionRequest, data_fence_instruction, fence, make_nonce

    nonce = nonce or make_nonce()
    observed = reading_for(a, b) == READING_OBSERVED

    def block(c: ClaimContext) -> str:
        passage = c.passage or "(not found)"
        if observed:
            return f"claim: {c.content}\n{_source_line(c)}\npassage: {passage}"
        return f"claim: {c.content}\nnote: {c.source} (dated {c.date})\npassage: {passage}"

    user = (
        f"Claim A:\n{fence(block(a), nonce, label='claim_a')}\n\n"
        f"Claim B:\n{fence(block(b), nonce, label='claim_b')}"
    )
    instruction = _VERIFY_OBSERVED_INSTRUCTION if observed else _VERIFY_INSTRUCTION
    return CompletionRequest(prompt=user, system=instruction + "\n" + data_fence_instruction(nonce))


def second_reading_prompt_hash() -> str:
    """The second reading's prompt version: both instructions, rendered.

    Built by :func:`_verify_request` itself over placeholder claims, once per
    instruction, so a change to either instruction or to the claim blocks
    changes the hash. One hash covers both instructions: which one a pair gets
    is decided by its sources after the ledger is consulted.
    """
    parts: list[str] = []
    for kind in (SourceKind.NOTE, SourceKind.RECORD):
        claim = ClaimContext(
            content="{claim}",
            source="{source}",
            date="{date}",
            passage="{passage}",
            kind=kind,
            observed="{observed}",
        )
        request = _verify_request(claim, claim, nonce="{nonce}")
        parts += [request.system or "", request.prompt]
    return prompt_hash(*parts)


async def _llm_verify_contradiction(a: ClaimContext, b: ClaimContext) -> ProbeVerdict | None:
    """The second reading's verdict, or ``None`` when no usable verdict came back.

    The budget is ``audit.verify_max_tokens``. A reply with no usable verdict
    (cut at the budget, empty because a thinking model spent it all, or off
    protocol) is re-issued once at ``audit.verify_retry_max_tokens``: the same
    call at the same budget tends to repeat the cut, and an unread flag is a
    contradiction the agent is never warned about. Only the settled outcome is counted, so a
    reading the retry recovered is not disclosed as a failed call. A failed
    call (network, refusal, billing) is not retried.
    """
    request = _verify_request(a, b)
    cfg = get_config().audit
    budgets = [cfg.verify_max_tokens]
    if cfg.verify_retry_max_tokens > cfg.verify_max_tokens:
        budgets.append(cfg.verify_retry_max_tokens)
    response = ""
    for attempt, max_tokens in enumerate(budgets):
        if attempt:
            if llm_circuit_open():
                break
            log.warning(
                "Contradiction second reading at max_tokens=%d carried no usable "
                "verdict; retrying once at %d (audit.verify_retry_max_tokens)",
                budgets[0],
                max_tokens,
            )
        reply = await _llm_call(
            request.prompt,
            max_tokens=max_tokens,
            system=request.system,
            response_schema=_PROBE_RESPONSE_SCHEMA,
            purpose="verification",
            empty_reply_as_text=True,
        )
        if reply is None:
            return None  # a failed call, already counted by the seam
        response = reply
        verdict = _parse_probe_verdict(response)
        if verdict.usable:
            return verdict
    # A cut reply is never read as a verdict (see _parse_probe_verdict).
    record_unusable_reply("contradiction second reading", response)
    return None


async def read_pair(a: ClaimContext, b: ClaimContext) -> ProbeVerdict | None:
    """Read one pair a second time: its verdict, or ``None`` when the reading failed.

    ``None`` covers every way a reading can fail: the breaker is open, the
    call failed, or the reply was still cut off after the retry. The verdict's
    instruction is :func:`reading_for` of the same two contexts.
    """
    return await _llm_verify_contradiction(a, b)


async def _claim_kind(session: AsyncSession, particle: Particle) -> SourceKind:
    """A claim's source kind from its first SOURCE ref, with no blob read."""
    ref = next(
        (
            r
            for r in particle.provenance
            if r.type is ProvenanceRefType.SOURCE and r.corpus_entry_id
        ),
        None,
    )
    if ref is None:
        return SourceKind.NOTE
    entry = await get_entry(session, ref.corpus_entry_id)
    if entry is None:
        return SourceKind.NOTE
    return source_kind(entry.source_type, entry.mutability)
