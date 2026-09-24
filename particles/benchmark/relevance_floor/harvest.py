# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Harvest a held-out question set from an operator's agent transcripts.

The product keeps no query log, so the only record of what a real person asked
is the agent transcript. Three sources are read, tagged, and never blended
silently (:class:`~.schema.QuestionSource`):

* the MCP ``query`` tool's ``question`` input;
* ``particles query "<question>"`` in a shell tool call — skipped when the
  command addresses another store (a ``DATABASE_URL`` / ``PARTICLES_CONFIG``
  override or ``--store``), because a smoke-test question put to a scratch
  store is not a question about this one;
* question-shaped sentences the operator typed to the agent — a proxy (nobody
  addressed them to the store), which is why the report breaks them out.

Everything here is pure parsing — no store, encoder, or LLM — and the output is
private by construction: the CLI writes it outside the repository and redacts
secrets first. Nothing in this module decides where the file goes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .schema import HeldOutQuestion, QuestionSource

log = logging.getLogger(__name__)

#: The phrase the deterministic below-floor refusal always carries.
#: A unit test holds it against the query op's own builder, so the two cannot drift.
REFUSAL_PHRASE = "below the relevance floor"

_MCP_QUERY_TOOL = re.compile(r"^mcp__.*particles.*__query$", re.IGNORECASE)
_CLI_QUERY = re.compile(r"\bparticles\s+query\s+(?P<q>'[^']+'|\"[^\"]+\")")
_OTHER_STORE = re.compile(r"DATABASE_URL=|PARTICLES_CONFIG=|--store\b")
# Harness-injected wrappers around a typed prompt: not the operator's words.
_WRAPPER_BLOCK = re.compile(r"<([a-z][a-z0-9_-]*)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")
_MIN_QUESTION_WORDS = 4


class HarvestResult(BaseModel):
    """A harvested, de-duplicated held-out set plus the census that produced it."""

    questions: list[HeldOutQuestion] = Field(default_factory=list)
    transcripts_scanned: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    #: Explicit query calls whose transcript result recorded the floor refusal.
    historical_refusals: int = 0
    #: Explicit query calls whose result the transcript captured at all.
    historical_results: int = 0


def question_id(question: str) -> str:
    """Stable id for a question — a digest of its case- and space-normalised text."""
    normalised = " ".join(question.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:12]


def heldout_fingerprint(questions: Iterable[HeldOutQuestion]) -> str:
    """Digest over the sorted ids: names a held-out set without disclosing it."""
    joined = "\n".join(sorted(q.question_id for q in questions))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return "" if content is None else str(content)


def prompt_questions(text: str, *, min_chars: int, max_chars: int) -> list[str]:
    """Question-shaped sentences in one typed prompt.

    Wrapper blocks the harness injects and fenced code are dropped first; a
    sentence qualifies when it ends in ``?`` and sits inside the length bounds.
    Slash commands and wrapper-only prompts yield nothing.
    """
    cleaned = _CODE_FENCE.sub(" ", _WRAPPER_BLOCK.sub(" ", text)).strip()
    if not cleaned or cleaned.startswith(("/", "<")):
        return []
    found: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        candidate = sentence.strip()
        if (
            candidate.endswith("?")
            and min_chars <= len(candidate) <= max_chars
            and len(candidate.split()) >= _MIN_QUESTION_WORDS
        ):
            found.append(candidate)
    return found


def _records(jsonl_text: str) -> Iterator[dict[str, Any]]:
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def transcript_questions(
    jsonl_text: str,
    *,
    sources: frozenset[QuestionSource],
    min_chars: int,
    max_chars: int,
) -> list[HeldOutQuestion]:
    """Every held-out question in one transcript, in order, not yet de-duplicated."""
    pending: dict[str, HeldOutQuestion] = {}
    found: list[HeldOutQuestion] = []
    for record in _records(jsonl_text):
        if record.get("isSidechain") or record.get("isMeta"):
            continue
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            blocks: list[Any] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = content
        else:
            continue
        is_tool_return = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_use":
                harvested = _from_tool_use(block, sources)
                if harvested is not None:
                    pending[str(block.get("id"))] = harvested
                    found.append(harvested)
            elif kind == "tool_result":
                call = pending.pop(str(block.get("tool_use_id")), None)
                if call is not None:
                    call.historical_refused = REFUSAL_PHRASE in _result_text(block.get("content"))
            elif (
                kind == "text"
                and record.get("type") == "user"
                and not is_tool_return
                and QuestionSource.USER_PROMPT in sources
            ):
                for question in prompt_questions(
                    str(block.get("text", "")), min_chars=min_chars, max_chars=max_chars
                ):
                    found.append(
                        HeldOutQuestion(
                            question_id=question_id(question),
                            question=question,
                            source=QuestionSource.USER_PROMPT,
                        )
                    )
    return found


def _from_tool_use(
    block: dict[str, Any], sources: frozenset[QuestionSource]
) -> HeldOutQuestion | None:
    name = str(block.get("name", ""))
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        return None
    if QuestionSource.MCP_QUERY in sources and _MCP_QUERY_TOOL.match(name):
        question = tool_input.get("question")
        if isinstance(question, str) and question.strip():
            text = question.strip()
            return HeldOutQuestion(
                question_id=question_id(text), question=text, source=QuestionSource.MCP_QUERY
            )
        return None
    if QuestionSource.CLI_QUERY in sources and name == "Bash":
        command = str(tool_input.get("command", ""))
        match = _CLI_QUERY.search(command)
        if match is None or _OTHER_STORE.search(command):
            return None
        text = match.group("q")[1:-1].strip()
        if text:
            return HeldOutQuestion(
                question_id=question_id(text), question=text, source=QuestionSource.CLI_QUERY
            )
    return None


# An explicit query outranks a typed prompt when the same text arrives both ways.
_SOURCE_PRECEDENCE = {
    QuestionSource.MCP_QUERY: 0,
    QuestionSource.CLI_QUERY: 1,
    QuestionSource.USER_PROMPT: 2,
}


def _merge_refused(a: bool | None, b: bool | None) -> bool | None:
    """Pool two recordings of one question: a refusal seen once is never lost."""
    if a is None or b is None:
        return b if a is None else a
    return a or b


def harvest_transcripts(
    root: Path,
    *,
    sources: frozenset[QuestionSource] = frozenset(QuestionSource),
    min_chars: int,
    max_chars: int,
    redact: Callable[[str], str] | None = None,
) -> HarvestResult:
    """Walk ``root`` for ``*.jsonl`` transcripts and build the de-duplicated set.

    De-duplication is by :func:`question_id`; on a collision the more explicit
    source wins, and a recorded historical refusal is never lost to a duplicate
    that lacks one. An unreadable file is logged and skipped. ``redact`` is
    applied to each question before its id is taken, so a secret pasted into a
    prompt never reaches the held-out file (the caller supplies the redactor;
    this package sits below the surface that owns it).
    """
    result = HarvestResult()
    by_id: dict[str, HeldOutQuestion] = {}
    for path in sorted(root.rglob("*.jsonl")):
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            log.warning("relevance-floor harvest: skipping %s (%r)", path, exc)
            continue
        result.transcripts_scanned += 1
        for question in transcript_questions(
            text, sources=sources, min_chars=min_chars, max_chars=max_chars
        ):
            if redact is not None:
                clean = redact(question.question)
                if clean != question.question:
                    question = question.model_copy(
                        update={"question": clean, "question_id": question_id(clean)}
                    )
            if question.historical_refused is not None:
                result.historical_results += 1
                result.historical_refusals += int(question.historical_refused)
            held = by_id.get(question.question_id)
            if held is None:
                by_id[question.question_id] = question
                continue
            explicit = _SOURCE_PRECEDENCE[question.source] < _SOURCE_PRECEDENCE[held.source]
            winner, other = (question, held) if explicit else (held, question)
            winner.historical_refused = _merge_refused(
                winner.historical_refused, other.historical_refused
            )
            by_id[question.question_id] = winner
    result.questions = sorted(by_id.values(), key=lambda q: q.question_id)
    for question in result.questions:
        result.by_source[question.source.value] = result.by_source.get(question.source.value, 0) + 1
    return result


def write_heldout(path: Path, questions: Iterable[HeldOutQuestion]) -> int:
    """Write the set as JSONL (one question per line); return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [q.model_dump_json() for q in questions]
    path.write_text("".join(row + "\n" for row in rows))
    return len(rows)


def load_heldout(path: Path) -> list[HeldOutQuestion]:
    """Read a held-out JSONL file. A malformed line is an error, not a skip."""
    return [
        HeldOutQuestion.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
