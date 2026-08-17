"""Comparator memories for the LongMemEval harness (§ comparators).

The four-condition table answers *how good is Particles as a memory* against
two anchors — the whole haystack in context, and no memory at all. It cannot
answer *how good is Particles compared with the memory an agent harness
already gives you*, because that memory is not a condition. This module
supplies two such conditions, each a **drop-in for the particle store** in
the run loop: the same 150 questions, the same answer scaffold, the same
judge, the same retrieval-stage scoring against labeled evidence sessions —
only the memory differs. ``selection.memory`` names which one ran, so a
report can never be read as a Particles number by mistake.

* ``chunks`` — **raw-transcript RAG**, the standard retrieval baseline. Each
  haystack session is cut into turn-aligned chunks of at most
  ``CHUNK_MAX_CHARS`` characters, every chunk is embedded with the *same*
  embedding model the store uses, and the top-k chunks by cosine become the
  answering model's context. No LLM call is made at write time. This
  isolates one question: *does claim extraction add anything over
  retrieving the transcript itself?*
* ``notes`` — **LLM-written session notes**, the harness-memory pattern
  (Claude Code's auto-memory file, a "summarise each conversation into a
  notes file" agent): each session is summarised once by the ``extraction``
  purpose's model (so the writer is the same model that wrote the
  particles), the notes are embedded, and the top-k notes become the
  context. This isolates the second question: *does Particles' epistemic
  layer beat plain distillation?*

Both score retrieval at session granularity exactly as the particle path
does — a chunk or a note stands on exactly one session, so provenance is the
item's own session id — and both leave the QA family untouched: the
``qa_full_context`` and ``qa_no_memory`` baselines are the *same calls* under
the same tuple, so a comparator run may skip them (``baselines=False``) and
the table reuses the particles run's columns, disclosed as such.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from particles.benchmark.memory.metrics import precision_at_k, recall_at_k
from particles.benchmark.memory.schema import (
    MemoryQuestion,
    MemorySession,
    RetrievalQuestionResult,
)
from particles.embeddings import get_embedding_model
from particles.llm.registry import CompletionRequest, complete_many, get_provider

log = logging.getLogger(__name__)

#: The memories the run loop can put under test. ``particles`` is the store.
MEMORY_KINDS: tuple[str, ...] = ("particles", "chunks", "notes")

#: Turn-aligned chunk ceiling for the ``chunks`` comparator (~400 tokens at
#: the ~4 chars/token heuristic the estimate uses) — the ordinary RAG chunk
#: size, and roughly the context a top-10 particle list occupies, so the two
#: read-time contexts are of comparable size.
CHUNK_MAX_CHARS = 1500

#: Output budget for one session's notes. Loose for the same reason as the
#: answer/judge budgets: an adaptive-thinking writer spends from it.
NOTES_MAX_TOKENS = 2048

_NOTES_SYSTEM = (
    "You are the long-term memory of an AI assistant. You will be shown one "
    "past chat session between the user and the assistant. Write concise notes "
    "that a future assistant could rely on to answer questions about this "
    "session weeks later: every fact the user stated about themselves, their "
    "plans, possessions, relationships, and preferences; every commitment or "
    "recommendation the assistant made; every date, number, name, and place. "
    "Keep the session date. Use short bullet points; do not editorialise; do "
    "not omit specifics."
)


@dataclass
class MemoryItem:
    """One retrievable unit of a comparator memory: text standing on one session."""

    session_id: str
    text: str
    date: str | None = None


@dataclass
class ComparatorQuestionResult:
    """What the run loop needs back: the scored retrieval row + the QA context."""

    retrieval: RetrievalQuestionResult
    context: str
    notes: list[str] = field(default_factory=list)


def chunk_session(session: MemorySession, *, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Cut one session into turn-aligned chunks of at most ``max_chars``.

    Turns are packed greedily in order; a single turn longer than the ceiling
    is split on character boundaries so nothing is dropped. Every chunk is
    prefixed with the session date so the answering model sees the same
    temporal cue the transcript carries.
    """
    header = f"Session date: {session.date}\n" if session.date else ""
    budget = max(64, max_chars - len(header))
    pieces: list[str] = []
    for turn in session.turns:
        line = f"{turn.role}: {turn.content}"
        while len(line) > budget:
            pieces.append(line[:budget])
            line = line[budget:]
        pieces.append(line)

    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for piece in pieces:
        if current and used + len(piece) + 1 > budget:
            chunks.append(header + "\n".join(current))
            current, used = [], 0
        current.append(piece)
        used += len(piece) + 1
    if current:
        chunks.append(header + "\n".join(current))
    return chunks


def session_hash(session: MemorySession) -> str:
    """Content hash of one session — the notes cache key (date + turns)."""
    payload = json.dumps(
        {"date": session.date, "turns": [(t.role, t.content) for t in session.turns]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NotesWriter:
    """Session → notes, written once per distinct session (in-process + on-disk cache).

    Mirrors the particle path's ``CachingExtractor``: repeated sessions across
    questions are summarised once, concurrent questions that reach the same
    unwritten session share one in-flight call, and an optional JSONL
    ``cache_path`` persists every note so an interrupted run resumes without
    re-paying (the on-disk cache is keyed by session hash *and* stamped with
    the writer's resolved model id, so a model change never replays stale
    notes). Requests are dispatched through ``complete_many`` on the
    ``extraction`` purpose with ``latency_tolerant=True`` — one Message
    Batches job per question at the half price when the provider
    supports it, the same sequential calls otherwise.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path
        self.model_id = get_provider("extraction").provider_model
        self._notes: dict[str, str] = {}
        self._inflight: dict[str, asyncio.Future[str]] = {}
        self.hits = 0
        self.misses = 0
        self.failures = 0
        if cache_path is not None and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    raw = json.loads(line)
                except ValueError:
                    continue
                if raw.get("model_id") == self.model_id and "hash" in raw and "notes" in raw:
                    self._notes[str(raw["hash"])] = str(raw["notes"])
            if self._notes:
                log.info(
                    "Notes cache: %d session note(s) restored from %s", len(self._notes), cache_path
                )

    def _persist(self, key: str, notes: str) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as fh:
            fh.write(json.dumps({"model_id": self.model_id, "hash": key, "notes": notes}) + "\n")

    @staticmethod
    def _prompt(session: MemorySession) -> str:
        # Deferred: runner imports this module at top (cycle break, case 1).
        from particles.benchmark.memory.runner import render_session_text

        return f"Chat session:\n\n{render_session_text(session)}\n\nWrite the memory notes."

    async def notes_for(self, sessions: list[MemorySession]) -> list[str | None]:
        """Notes for each session, positionally aligned; ``None`` marks a failed write."""
        keys = [session_hash(s) for s in sessions]
        results: dict[str, str | None] = {}
        to_write: list[tuple[str, MemorySession]] = []
        awaiting: dict[str, asyncio.Future[str]] = {}
        loop = asyncio.get_running_loop()
        for key, session in zip(keys, sessions, strict=True):
            if key in results or key in awaiting or any(k == key for k, _ in to_write):
                continue
            cached = self._notes.get(key)
            if cached is not None:
                self.hits += 1
                results[key] = cached
            elif key in self._inflight:
                self.hits += 1
                awaiting[key] = self._inflight[key]
            else:
                self.misses += 1
                fut: asyncio.Future[str] = loop.create_future()
                self._inflight[key] = fut
                to_write.append((key, session))

        if to_write:
            requests = [
                CompletionRequest(prompt=self._prompt(s), system=_NOTES_SYSTEM) for _, s in to_write
            ]
            try:
                replies = await complete_many(
                    "extraction",
                    requests,
                    max_tokens=NOTES_MAX_TOKENS,
                    latency_tolerant=True,
                )
            except BaseException as exc:
                for key, _ in to_write:
                    fut = self._inflight.pop(key)
                    if not fut.done():
                        fut.set_exception(exc)
                raise
            for (key, _), reply in zip(to_write, replies, strict=True):
                fut = self._inflight.pop(key)
                if reply is None or not reply.strip():
                    self.failures += 1
                    results[key] = None
                    if not fut.done():
                        fut.set_exception(RuntimeError("notes call failed"))
                    continue
                self._notes[key] = reply
                self._persist(key, reply)
                results[key] = reply
                if not fut.done():
                    fut.set_result(reply)

        for key, fut in awaiting.items():
            try:
                results[key] = await fut
            except Exception:  # noqa: BLE001 — the owning call already counted the failure
                results[key] = None
        return [results.get(key) for key in keys]


async def _embed(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    if model is None:
        raise RuntimeError("embedding model unavailable — the comparator memories need it")
    vectors = await asyncio.to_thread(model.encode, texts)
    return np.asarray(vectors, dtype=np.float32)


def _context_from_items(
    items: list[MemoryItem], kind: str, budget_tokens: int | None = None
) -> str:
    """The comparator's QA context: retrieved items in rank order.

    ``budget_tokens`` is the same clamp the particle path applies
    (``_particles_context``): items are appended in rank order
    until the next would exceed the budget at ~4 chars/token, the first item
    always included. It is how a budget-matched comparison is run
    — retrieval is still scored at ``top_k``; only the answerer's context
    shrinks — and it rides ``selection.context_budget_tokens`` exactly as
    for particles, so a clamped comparator can only be compared against a
    matching run.
    """
    if not items:
        return f"(no memory {kind} were retrieved)"
    lines: list[str] = []
    used_chars = 0
    budget_chars = None if budget_tokens is None else budget_tokens * 4
    for item in items:
        date = item.date or "undated"
        body = item.text if kind == "notes" else item.text.replace("\n", "\n  ")
        line = f"- [{date}] {body}"
        if budget_chars is not None and lines and used_chars + len(line) > budget_chars:
            break
        lines.append(line)
        used_chars += len(line) + 1
    return "\n".join(lines)


async def run_comparator_question(
    question: MemoryQuestion,
    *,
    memory: str,
    top_k: int,
    notes_writer: NotesWriter | None = None,
    context_budget: int | None = None,
) -> ComparatorQuestionResult:
    """Build the comparator memory for one question, retrieve top-k, score it.

    Retrieval scoring is the particle path's, at session granularity: each
    retrieved item stands on exactly one session, recall credits every
    labeled evidence session that any top-k item stands on, precision counts
    one entry per retrieved item. Abstention questions are unscoreable at the
    retrieval stage by protocol (empty evidence set ⇒ ``None`` metrics), as
    in the particle path.
    """
    notes: list[str] = []
    items: list[MemoryItem] = []
    if memory == "chunks":
        for session in question.sessions:
            for chunk in chunk_session(session):
                items.append(MemoryItem(session.session_id, chunk, session.date))
    elif memory == "notes":
        if notes_writer is None:
            raise ValueError("memory='notes' needs a NotesWriter")
        written = await notes_writer.notes_for(question.sessions)
        failed = 0
        for session, text in zip(question.sessions, written, strict=True):
            if text is None:
                failed += 1
                continue
            items.append(MemoryItem(session.session_id, text, session.date))
        if failed:
            notes.append(
                f"Question {question.question_id}: notes writer failed on {failed} of "
                f"{len(question.sessions)} session(s); those sessions are absent from the memory"
            )
    else:
        raise ValueError(f"unknown comparator memory {memory!r}")

    retrieved: list[MemoryItem] = []
    if items:
        vectors = await _embed([item.text for item in items] + [question.question])
        item_vecs, q_vec = vectors[:-1], vectors[-1]
        scores = item_vecs @ q_vec
        order = np.argsort(-scores, kind="stable")[:top_k]
        retrieved = [items[int(i)] for i in order]

    evidence = set() if question.is_abstention else set(question.answer_session_ids)
    retrieved_session_ids: list[str] = [item.session_id for item in retrieved]
    hit_sessions = set(retrieved_session_ids) & evidence
    retrieval = RetrievalQuestionResult(
        question_id=question.question_id,
        question_type=question.question_type,
        evidence_sessions=len(evidence),
        evidence_sessions_hit=len(hit_sessions),
        particles_retrieved=len(retrieved),
        recall_at_k=recall_at_k(hit_sessions, evidence),
        precision_at_k=precision_at_k(retrieved_session_ids, evidence),
        abstention=not evidence,
    )
    return ComparatorQuestionResult(
        retrieval=retrieval,
        context=_context_from_items(retrieved, memory, context_budget),
        notes=notes,
    )
