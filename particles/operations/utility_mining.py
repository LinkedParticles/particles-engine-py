# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Transcript-mining pass — the reliable utility signal for the usefulness lens.

The spine: the harvest already records *what the agent
did* — its tool calls. When a session's actions demonstrably apply a belief
(the agent ran ``git commit -s``, used a documented workaround, verified on the
pinned Python), that action is a hard, unambiguous signal that the belief was
load-bearing — **credit action, not attention** (§1/§3). This is the
"unambiguous hook-observable action telemetry" branch of the owner's
reliable-signal constraint; it needs no user reaction, so the sentiment-guessing
failure mode is structurally impossible.

**A belief is credited only when the session was shown it and an LLM judge
rules that the session's actions applied it**. Exposure (what the
session was shown, read as of its start) is :mod:`utility_exposure`'s; only
shown beliefs are candidates. Two routes nominate a candidate, both over the
distilled *tool-call* lines only (``[tool: Bash — git commit
-s]``), never the conversational prose:

- **Literal:** a shown belief that names a concrete token (a command, flag,
  path) found in the action lines. The token match only nominates it. Its
  evidence is the matching lines, at most three, each with a line of context,
  from anywhere in the session.
- **Behavioural:** a shown belief with no token hit, after the embedding
  pre-filter. Its evidence is the first and the last part of the session's
  action lines.

The judge rules per candidate between "applied" and "only touched its topic";
no event is recorded without an "applied" ruling, and the event keeps the
route that nominated it as its ``match_basis``. Every judge call, of either
route, draws on ``utility.mining.max_behavioural_calls`` per run (
discipline). A multi-session caller (the consolidation pass 5)
threads ONE shared budget through
``plan_session_mine(max_behavioural_calls=remaining)`` so the cap is genuinely
per *run*, not per session (correction, v1.74.1); a single-session
caller (the SessionEnd inline pass) leaves it ``None`` and gets the config
cap. With the judge off (``utility.mining.behavioural_matching: false``)
nothing is recorded.

A mine runs gather / decide / apply: :func:`plan_session_mine` reads
and plans, :func:`judge_sessions` makes the matcher calls, and
:func:`record_session_mine` writes. The consolidation pass plans every session
first and judges them together, so a night's sessions share one batch rather
than waiting on one each.

The mined events are recorded per session (idempotent — ``utility_store``); the
query-time ``UtilityPolicy`` turns them into a bounded projection-ranking factor.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Particle
from particles.core.status import Status
from particles.db import session_scope
from particles.embeddings import cosine_similarity
from particles.llm.registry import CompletionRequest, complete
from particles.operations.query.as_of import ensure_utc
from particles.operations.spend_estimate import PassEstimate, requests_estimate
from particles.operations.utility_exposure import (
    Exposure,
    ExposureReader,
    SessionFrame,
    session_frame,
)
from particles.store.particle_store import (
    get_active_particles_with_embeddings,
    get_particles_by_status,
)
from particles.store.utility_store import record_utility_events

log = logging.getLogger(__name__)

_T = TypeVar("_T")

# One distilled tool-call line. Prose turns are ignored — only
# actions count (credit action, not attention).
_TOOL_LINE_RE = re.compile(r"^\s*\[tool:.*$", re.MULTILINE)
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w][\w-]+)")
_PATH_RE = re.compile(r"\b([\w.-]+/[\w./-]+\.\w{1,5})\b")
# A belief phrased as a prohibition ("never prepend export PATH") must NOT be
# credited when its token *appears* in an action — that is a violation, not
# compliance. Such beliefs are routed to the behavioural tier instead.
_NEGATION_RE = re.compile(r"\b(never|don'?t|do not|avoid|not|no|without)\b", re.IGNORECASE)
# Backtick tokens too generic to be action evidence on their own.
_STOP_TOKENS = frozenset(
    {
        "true",
        "false",
        "none",
        "null",
        "python",
        "git",
        "main",
        "self",
        "str",
        "int",
        "list",
        "dict",
        "the",
    }
)

#: A behavioural candidate's evidence: the first and the last this many
#: characters of the session's action lines. Commits and other
#: closing actions come at the end, which a head-only window never showed.
_HEAD_ACTION_CHARS = 3000
_TAIL_ACTION_CHARS = 3000
#: A literal candidate's evidence: at most this many matching action lines,
#: each with this many lines of context on either side.
_LITERAL_EVIDENCE_HITS = 3
_LITERAL_EVIDENCE_CONTEXT = 1
_BEHAVIOURAL_BATCH = 15  # candidates judged per LLM call, either route

#: How a credited event came forward, kept as its ``match_basis``.
MatchBasis = Literal["literal", "behavioural"]

# The matcher's verdict shape: the reply is a JSON array of the
# followed guideline numbers, so a schema-enforcing adapter (LocalProvider
# structured output) can pin it; the digit-scraping parser below is unchanged
# and tolerant of both dialects.
_MATCHER_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {"type": "integer"},
}


@dataclass(frozen=True)
class MiningResult:
    """Disclosure counts for one mining run (logged to the hook log)."""

    #: Events credited through each route: the judge ruled "applied" on a
    #: candidate the literal (token) or the behavioural route nominated.
    literal: int
    behavioural: int
    #: Beliefs the session was shown and that are still ACTIVE: the candidate
    #: pool. Summed over sessions on a multi-session run.
    candidates: int
    #: Judge calls answered, of either route. The field keeps its name: the
    #: config cap it is measured against is ``max_behavioural_calls``.
    behavioural_calls: int
    #: Harvested entries skipped because their corpus blob is gone (an upstream
    #: fetch failure, a pruned archive). Counted rather than logged per entry so
    #: one broken source cannot bury the run's result; surfaced in the CLI line.
    skipped_missing_blob: int = 0
    #: True when the judge wanted more LLM calls than its budget allowed — the
    #: multi-session caller's truncation-disclosure signal.
    behavioural_truncated: bool = False
    #: Shown beliefs a literal token nominated, judged or not.
    literal_nominated: int = 0
    #: Explicit operator credits re-derived from the ``BELIEF_MARKED_USEFUL``
    #: event log on a rebuild. Always 0 for a single-session mine,
    #: which only produces the mined channel.
    explicit: int = 0
    #: Unmatched beliefs excluded by the behavioural relevance pre-filter
    #: (``utility.mining.behavioural_candidate_limit``) before any LLM call —
    #: disclosed so a capped candidate set is never mistaken for full coverage.
    behavioural_prefiltered: int = 0
    #: False when that pre-filter cut the candidate set in arbitrary list order
    #: rather than by action similarity, because no embedding model was
    #: available. The count above says how many were dropped; this
    #: says whether the ones kept were the relevant ones.
    behavioural_prefilter_ranked: bool = True


def extract_action_lines(transcript: str) -> list[str]:
    """Return the distilled tool-call lines of a transcript (the agent's actions)."""
    return [m.group(0).strip() for m in _TOOL_LINE_RE.finditer(transcript)]


def literal_tokens(content: str) -> set[str]:
    """Distinctive action tokens in a belief — backtick spans, flags, paths (lowercased)."""
    toks: set[str] = set()
    for m in _BACKTICK_RE.finditer(content):
        span = m.group(1).strip().lower()
        if len(span) >= 3 and span not in _STOP_TOKENS:
            toks.add(span)
    for rx in (_FLAG_RE, _PATH_RE):
        for m in rx.finditer(content):
            t = m.group(1).strip().lower()
            if len(t) >= 3 and t not in _STOP_TOKENS:
                toks.add(t)
    return toks


def _token_sequence_re(span: str) -> re.Pattern[str] | None:
    """Ordered-token matcher for a multi-token command span, or ``None`` if single-token.

    A contiguous substring test under-credits real invocations: the belief
    ``git commit -s`` must credit ``git -C /repo commit -s -F -`` and
    ``uv run git … commit -s``, which interpose flags between the tokens. So a
    multi-token span matches when its tokens appear **in order within one
    action line**, interposed arguments tolerated.

    Each token is right-bounded (``(?![\\w-])``) so a short flag like ``-s``
    does not match inside ``-short``, keeping the tier deterministic and tight.
    """
    tokens = [t for t in span.split() if t]
    if len(tokens) < 2:
        return None
    return re.compile(r".*?".join(re.escape(t) + r"(?![\w-])" for t in tokens))


def match_literal(actives: Sequence[Particle], action_lines: list[str]) -> dict[str, str]:
    """Beliefs whose literal token appears in the session's action lines → ``{pid: token}``.

    Single-token spans match as substrings; multi-token command spans match as an
    ordered token sequence within one action line (see :func:`_token_sequence_re`).
    A match nominates the belief for the judge; it does not credit it.

    Skips beliefs phrased as prohibitions (a token match there is a *violation*,
    not compliance — routed to the behavioural tier).
    """
    if not action_lines:
        return {}
    lines = [line.lower() for line in action_lines]
    hay = "\n".join(lines)
    matched: dict[str, str] = {}
    for p in actives:
        if _NEGATION_RE.search(p.content):
            continue
        for tok in literal_tokens(p.content):
            seq = _token_sequence_re(tok)
            hit = any(seq.search(line) for line in lines) if seq is not None else tok in hay
            if hit:
                matched[p.id] = tok
                break
    return matched


def literal_evidence(token: str, action_lines: Sequence[str]) -> list[str]:
    """A literal candidate's evidence: the lines its token matched, with context.

    At most :data:`_LITERAL_EVIDENCE_HITS` matching lines, spread across the
    session (the first, the last, and the middle one), each with
    :data:`_LITERAL_EVIDENCE_CONTEXT` lines either side. Overlapping windows
    merge; a gap between windows is marked ``…``. Uses the same matching rule
    as :func:`match_literal`.
    """
    seq = _token_sequence_re(token)
    hits = [
        i
        for i, line in enumerate(action_lines)
        if (seq.search(line.lower()) if seq is not None else token in line.lower())
    ]
    if not hits:
        return []
    if len(hits) > _LITERAL_EVIDENCE_HITS:
        step = (len(hits) - 1) / (_LITERAL_EVIDENCE_HITS - 1)
        hits = sorted({hits[round(k * step)] for k in range(_LITERAL_EVIDENCE_HITS)})
    keep: set[int] = set()
    for i in hits:
        lo = max(0, i - _LITERAL_EVIDENCE_CONTEXT)
        hi = min(len(action_lines), i + _LITERAL_EVIDENCE_CONTEXT + 1)
        keep.update(range(lo, hi))
    out: list[str] = []
    previous = -2
    for i in sorted(keep):
        if out and i != previous + 1:
            out.append("…")
        out.append(action_lines[i])
        previous = i
    return out


def head_and_tail(action_lines: Sequence[str]) -> str:
    """A behavioural candidate's evidence: the session's first and last action characters.

    The first :data:`_HEAD_ACTION_CHARS` and last :data:`_TAIL_ACTION_CHARS`
    characters of the joined action lines, with the omitted span counted
    between them. A session that fits is returned whole.
    """
    joined = "\n".join(action_lines)
    if len(joined) <= _HEAD_ACTION_CHARS + _TAIL_ACTION_CHARS:
        return joined
    omitted = len(joined) - _HEAD_ACTION_CHARS - _TAIL_ACTION_CHARS
    return (
        f"{joined[:_HEAD_ACTION_CHARS]}\n"
        f"[… {omitted} characters of actions omitted …]\n"
        f"{joined[-_TAIL_ACTION_CHARS:]}"
    )


async def _prefilter_behavioural_candidates(
    session: AsyncSession,
    unmatched: list[Particle],
    action_lines: list[str],
    limit: int,
) -> tuple[list[Particle], int, bool]:
    """Keep the ``limit`` unmatched beliefs most similar to the session's actions.

    The behavioural tier's candidate set is every ACTIVE belief the literal
    tier did not match — nearly the whole store — so the call budget
    would otherwise be spent on the first beliefs in list order. Ranking by
    :func:`~particles.embeddings.cosine_similarity` between the action summary
    and each belief's *stored* embedding costs one local ``encode()`` and no
    LLM calls. Returns ``(candidates, excluded_count, ranked_by_similarity)``
    — the third element is ``False`` when the cut was made in arbitrary list
    order because no encoder was available, so the caller can say
    that rather than claim a relevance ranking it did not perform.

    Degrades safely: with the filter disabled (``limit <= 0``), a small
    candidate set, or no embedding model, the input passes through (truncated
    to ``limit`` in the no-model case). Beliefs without a current-model stored
    embedding rank below every belief that has one.
    """
    if limit <= 0 or len(unmatched) <= limit:
        return unmatched, 0, True
    # Deferred import: lazy-init of the expensive embedding model (case 2).
    from particles.embeddings import get_embedding_model

    excluded = len(unmatched) - limit
    model = get_embedding_model()
    if model is None:
        # this truncation is by arbitrary list order, not similarity.
        # The count of dropped beliefs was always disclosed; which ones were
        # kept, and on what basis, was not — and the caller's log line asserted
        # "by action similarity" either way. Report the basis so it can say
        # what actually happened.
        return unmatched[:limit], excluded, False
    summary = head_and_tail(action_lines)
    action_vec = model.encode([summary], convert_to_numpy=True, normalize_embeddings=True)[0]
    vec_by_id = {p.id: vec for p, vec in await get_active_particles_with_embeddings(session)}
    ranked = sorted(
        unmatched,
        key=lambda p: cosine_similarity(action_vec, vec_by_id[p.id]) if p.id in vec_by_id else -1.0,
        reverse=True,
    )
    return ranked[:limit], excluded, True


#: The owner-decided rubric (2026-10-03) the activation gate's labels follow
#: lives in ``scripts/fixtures/use_judge_rubric.md``; rules 1 to 5 below are its
#: case rules, so the judge and the labels draw the same lines.
_MATCHER_SYSTEM = (
    "You judge whether an AI coding agent, in one session, APPLIED stated "
    "beliefs. For each numbered belief, decide whether the actions applied it: "
    "did what it prescribes, used the command or workaround it documents for "
    "the purpose it states, or relied on the fact it states to decide what to "
    "do. Actions that only touched its topic (ran, read, edited or named the "
    "thing it is about, without its content shaping what was done) did not "
    "apply it. Rules: (1) A belief saying where something lives or how code is "
    "organised is not applied by going to, reading, editing or running that "
    "thing; count it only when the actions show its specific claim was relied "
    "on. (2) A belief naming a tool or command and its purpose is applied when "
    "the actions run it for that purpose. (3) A procedural rule is applied only "
    "when the actions plainly show the rule's own step in the situation it "
    "names. (4) A belief reporting history (what happened, what changed, a "
    "count, a version) is never applied. (5) A token that matches by "
    "coincidence, the same word or path fragment used for something unrelated, "
    "never makes a belief applied. Be strict: only count a belief when the "
    "actions plainly demonstrate it, not when they merely could be consistent "
    "with it. Reply with ONLY a JSON array of the numbers of the beliefs the "
    "actions applied, e.g. [1,4]. Empty array if none. No explanation and no "
    "other text."
)

_MATCHER_QUESTION = (
    "Which belief numbers did the actions apply, rather than only touch the topic of?"
)

#: Output budget for one matcher reply. A bare array of up to
#: ``_BEHAVIOURAL_BATCH`` numbers fits in about 50 tokens, but the Anthropic
#: adapter does not enforce ``response_schema``, and a model that
#: reasons before answering ran past the old 200-token budget and was cut off
#: before the array. The headroom keeps such a reply parseable. Measured
#: 2026-09-29 on ``claude-haiku-4-5`` over ten live requests: the prompt
#: without "No explanation" drew 237 to 764 output tokens (seven of ten past
#: even 400); with it, 4 to 18. The prompt is the fix; the budget is margin.
#: Raised to 1024 for the Sonnet-class use judge: at 400, one of
#: 395 gate calls spent the whole budget before answering and two were cut off.
_MATCHER_MAX_TOKENS = 1024

#: A complete JSON array of integers (possibly empty), the only answer shape
#: :func:`_credit_matched` accepts.
_ANSWER_ARRAY = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]")


def _literal_prompt(batch: Sequence[tuple[Particle, Sequence[str]]]) -> str:
    """The matcher user turn for literal candidates, each with its own evidence lines."""
    blocks = []
    for i, (belief, evidence) in enumerate(batch):
        lines = "\n".join(f"   {line}" for line in evidence)
        blocks.append(f"{i + 1}. {belief.content}\n   Actions where it came up:\n{lines}")
    listing = "\n\n".join(blocks)
    return (
        f"Beliefs, each with the session's tool actions that mention it:\n\n{listing}\n\n"
        f"{_MATCHER_QUESTION}"
    )


def _behavioural_prompt(batch: Sequence[Particle], evidence: str) -> str:
    """The matcher user turn for behavioural candidates, sharing the head-and-tail evidence."""
    listing = "\n".join(f"{i + 1}. {p.content}" for i, p in enumerate(batch))
    return (
        f"Session actions (tool calls; the start and the end of the session):\n{evidence}\n\n"
        f"Beliefs:\n{listing}\n\n"
        f"{_MATCHER_QUESTION}"
    )


def _answered_numbers(reply: str) -> list[int] | None:
    """The guideline numbers in ``reply``'s answer array, or ``None`` if it has none.

    The answer is the **last** complete integer array in the reply, since a
    model that explains itself first puts the answer at the end. A number
    anywhere else ("guideline 3 does not apply", "12 tool calls") is prose,
    not an answer. A reply cut off before its closing bracket, or with no array
    at all, has no answer and credits nothing.
    """
    arrays = _ANSWER_ARRAY.findall(reply)
    if not arrays:
        return None
    return [int(n) for n in re.findall(r"\d+", arrays[-1])]


def _credit_matched(batch: Sequence[Particle], reply: str, matched: set[str]) -> None:
    """Add the belief ids named by ``reply``'s answer array to ``matched``."""
    numbers = _answered_numbers(reply)
    if numbers is None:
        log.debug("utility mining: matcher reply has no answer array; crediting nothing")
        return
    for n in numbers:
        idx = n - 1
        if 0 <= idx < len(batch):
            matched.add(batch[idx].id)


@dataclass(frozen=True)
class MatcherGroup:
    """One judge call: up to :data:`_BEHAVIOURAL_BATCH` candidates from one route."""

    basis: MatchBasis
    beliefs: tuple[Particle, ...]
    prompt: str

    def request(self) -> CompletionRequest:
        return CompletionRequest(prompt=self.prompt, system=_MATCHER_SYSTEM)


@dataclass(frozen=True)
class SessionMine:
    """One session's mining, decided before any LLM call (gather / decide / apply).

    Both routes are reduced to the matcher ``groups`` the session wants judged,
    literal groups first, already cut to the call budget the session was given.
    :func:`judge_sessions` answers the groups of many sessions at once and
    :func:`record_session_mine` writes the result, so a multi-session caller can
    pool every session's matcher calls into one batch and hold no
    write transaction across the wait.
    """

    session_id: str
    #: Shown beliefs a literal token nominated → the token.
    nominated: dict[str, str]
    candidates: int
    groups: list[MatcherGroup]
    truncated: bool
    prefiltered: int = 0
    prefilter_ranked: bool = True

    def requests(self) -> list[CompletionRequest]:
        """The matcher's completion requests, one per group."""
        return [group.request() for group in self.groups]


def _chunks(items: Sequence[_T]) -> list[Sequence[_T]]:
    return [
        items[start : start + _BEHAVIOURAL_BATCH]
        for start in range(0, len(items), _BEHAVIOURAL_BATCH)
    ]


async def plan_session_mine(
    session: AsyncSession,
    session_id: str,
    transcript: str,
    shown: Sequence[Particle],
    *,
    behavioural_matching: bool | None = None,
    max_behavioural_calls: int | None = None,
) -> SessionMine:
    """Nominate the session's candidates and plan their judge calls.

    ``shown`` is the candidate pool: the beliefs the session was shown that
    are still ACTIVE (:func:`shown_beliefs`). Reads only: no LLM call and no
    write. The arguments mean what they mean on :func:`mine_session`.
    """
    action_lines = extract_action_lines(transcript)
    nominated = match_literal(shown, action_lines)
    empty = SessionMine(
        session_id=session_id,
        nominated=nominated,
        candidates=len(shown),
        groups=[],
        truncated=False,
    )

    cfg = get_config().utility.mining
    run_judge = cfg.behavioural_matching if behavioural_matching is None else behavioural_matching
    if not run_judge or not action_lines or not shown:
        # a token match nominates and never credits, so with the
        # judge off nothing is recorded.
        return empty

    literal = [
        (p, literal_evidence(nominated[p.id], action_lines)) for p in shown if p.id in nominated
    ]
    groups = [
        MatcherGroup(
            basis="literal", beliefs=tuple(p for p, _ in batch), prompt=_literal_prompt(batch)
        )
        for batch in _chunks(literal)
    ]

    unmatched = [p for p in shown if p.id not in nominated]
    unmatched, prefiltered, prefilter_ranked = await _prefilter_behavioural_candidates(
        session, unmatched, action_lines, cfg.behavioural_candidate_limit
    )
    if prefiltered:
        log.info(
            "utility mining: behavioural pre-filter kept %d of %d candidate beliefs %s",
            len(unmatched),
            len(unmatched) + prefiltered,
            "by action similarity"
            if prefilter_ranked
            else "in arbitrary order — no embedding model, so the LLM budget was "
            "NOT spent on the most relevant beliefs",
        )
    if unmatched:
        evidence = head_and_tail(action_lines)
        groups.extend(
            MatcherGroup(
                basis="behavioural",
                beliefs=tuple(batch),
                prompt=_behavioural_prompt(batch, evidence),
            )
            for batch in _chunks(unmatched)
        )

    budget = cfg.max_behavioural_calls if max_behavioural_calls is None else max_behavioural_calls
    budget = max(0, budget)
    return SessionMine(
        session_id=session_id,
        nominated=nominated,
        candidates=len(shown),
        groups=groups[:budget],
        truncated=len(groups) > budget,
        prefiltered=prefiltered,
        prefilter_ranked=prefilter_ranked,
    )


async def _judge_sequential(requests: list[CompletionRequest]) -> list[str | None]:
    """Answer ``requests`` one call at a time, stopping at the first provider error."""
    replies: list[str | None] = []
    for request in requests:
        try:
            replies.append(
                await complete(
                    "use_judge",
                    request.prompt,
                    max_tokens=_MATCHER_MAX_TOKENS,
                    system=request.system,
                    response_schema=_MATCHER_RESPONSE_SCHEMA,
                )
            )
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment; any provider
            # error (CompletionError, a raw SDK 4xx/5xx like an exhausted credit
            # balance, a timeout) leaves the unanswered groups uncredited rather
            # than failing the whole mining run (no ruling, no event).
            log.info(
                "utility mining: matcher unavailable (%s); nothing credited for the rest",
                exc,
            )
            break
    return replies + [None] * (len(requests) - len(replies))


async def judge_sessions(
    mines: Sequence[SessionMine], *, latency_tolerant: bool = False
) -> list[list[str | None]]:
    """Answer every session's matcher groups, returning replies per session.

    ``latency_tolerant`` sends the groups of **all** ``mines`` as one
    ``complete_many`` submission, so N sessions wait on one batch rather than N
    in series; the replies are sliced back to each session in order.
    Otherwise the groups run as sequential calls. ``None`` marks a group with no
    usable answer: a dead request in a live batch, a job the provider could not
    run at all (every group of every session), or the groups a sequential run
    never reached after a provider error. A group without an answer credits
    nothing, exactly as a per-session mine reports it.
    """
    requests = [r for m in mines for r in m.requests()]
    if not requests:
        return [[] for _ in mines]
    replies: list[str | None]
    if latency_tolerant:
        # Deferred import: the test-mock seam (AGENTS.md § Deferred imports, case 3).
        from particles.llm import complete_many

        try:
            replies = await complete_many(
                "use_judge",
                requests,
                max_tokens=_MATCHER_MAX_TOKENS,
                response_schema=_MATCHER_RESPONSE_SCHEMA,
                latency_tolerant=True,
            )
        except Exception as exc:  # noqa: BLE001 — see _judge_sequential
            affected = sum(1 for m in mines if m.groups)
            log.info(
                "utility mining: matcher unavailable (%s); "
                "nothing credited this run for %d session(s)",
                exc,
                affected,
            )
            replies = [None] * len(requests)
    else:
        replies = await _judge_sequential(requests)
    per_session: list[list[str | None]] = []
    start = 0
    for m in mines:
        per_session.append(replies[start : start + len(m.groups)])
        start += len(m.groups)
    return per_session


async def record_session_mine(
    session: AsyncSession, mine: SessionMine, replies: Sequence[str | None]
) -> MiningResult:
    """Credit the candidates ``mine``'s answered groups ruled applied, and record them.

    Each event keeps the route that nominated it as its ``match_basis``.
    Writes, so a caller that serializes writers calls it inside
    the store's write lock. A group whose reply is ``None`` cost
    nothing usable, so it neither credits nor counts as a call.
    """
    events: dict[str, str] = {}
    calls = 0
    for group, reply in zip(mine.groups, replies, strict=True):
        if reply is None:
            continue
        calls += 1
        applied: set[str] = set()
        _credit_matched(group.beliefs, reply, applied)
        for pid in applied:
            events.setdefault(pid, group.basis)
    await record_utility_events(session, mine.session_id, events)
    return MiningResult(
        literal=sum(1 for basis in events.values() if basis == "literal"),
        behavioural=sum(1 for basis in events.values() if basis == "behavioural"),
        candidates=mine.candidates,
        behavioural_calls=calls,
        behavioural_truncated=mine.truncated,
        behavioural_prefiltered=mine.prefiltered,
        behavioural_prefilter_ranked=mine.prefilter_ranked,
        literal_nominated=len(mine.nominated),
    )


async def mine_session(
    session: AsyncSession,
    session_id: str,
    transcript: str,
    shown: Sequence[Particle],
    *,
    behavioural_matching: bool | None = None,
    max_behavioural_calls: int | None = None,
    latency_tolerant: bool = False,
) -> MiningResult:
    """Mine one session's actions into utility events for the beliefs it was shown.

    ``shown`` is the candidate pool (:func:`shown_beliefs`). Nominates
    candidates by both routes, has the judge rule on them, and records the
    ones ruled applied as utility events for ``session_id`` (idempotent).

    ``behavioural_matching`` overrides the config knob for one run. ``False``
    (the degraded consolidation pass) makes no LLM call and so
    records nothing.

    ``latency_tolerant`` lets the judge's calls go out as one
    asynchronous half-price batch. The SessionEnd inline mine leaves it
    ``False``; the consolidation pass mines many sessions and pools them through
    :func:`plan_session_mine` / :func:`judge_sessions` instead.

    ``max_behavioural_calls`` overrides the config cap for THIS invocation.
    ``0`` makes no calls, records nothing, and reports
    ``behavioural_truncated`` when judgement was wanted.
    """
    mine = await plan_session_mine(
        session,
        session_id,
        transcript,
        shown,
        behavioural_matching=behavioural_matching,
        max_behavioural_calls=max_behavioural_calls,
    )
    (replies,) = await judge_sessions([mine], latency_tolerant=latency_tolerant)
    return await record_session_mine(session, mine, replies)


async def shown_beliefs(
    session: AsyncSession,
    reader: ExposureReader,
    *,
    session_id: str,
    transcript: str,
    actives: Sequence[Particle],
    entry_tags: Sequence[str] = (),
    first_captured_at: datetime | None = None,
    started_at: datetime | None = None,
    project_key: str | None = None,
) -> tuple[list[Particle], Exposure]:
    """The session's candidate pool: what it was shown that is still ACTIVE.

    Exposure is read as of the session's start (:func:`session_frame`); the
    pool keeps only beliefs still ACTIVE now, since utility ranks the beliefs
    a reader can still be served. A session with no start time at all is read
    as of now.
    """
    frame = await session_frame(
        session,
        session_id,
        entry_tags=entry_tags,
        first_captured_at=first_captured_at,
        started_at=started_at,
        project_key=project_key,
    ) or SessionFrame(
        session_id=session_id,
        started_at=datetime.now(UTC),
        project_keys=frozenset({project_key} if project_key else ()),
    )
    exposure = await reader.exposure(session, frame, extract_action_lines(transcript))
    shown = exposure.shown
    return [p for p in actives if p.id in shown], exposure


async def mine_session_from_transcript(
    store: str,
    transcript: str,
    session_id: str,
    *,
    started_at: datetime | None = None,
    project_key: str | None = None,
) -> MiningResult:
    """Open a store session and mine ``transcript`` against what the session was shown.

    The harvest-cycle entry point: called after the session's
    harvest + extract. ``started_at`` and ``project_key`` are what the
    SessionEnd hook knows about the session; the harvested transcript's entry
    and the session's recorded exposure rows fill in the rest.
    """
    from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry

    async with session_scope(store) as session:
        actives = await get_particles_by_status(session, Status.ACTIVE)
        entry = await get_entry_by_uri(session, f"{_SESSION_URI_PREFIX}{session_id}")
        first_captured: datetime | None = None
        if entry is not None:
            captured = [
                s.captured_at for s in await list_snapshots_for_entry(session, entry.entry_id)
            ]
            first_captured = min(captured) if captured else None
        reader = await ExposureReader.load(session)
        shown, _ = await shown_beliefs(
            session,
            reader,
            session_id=session_id,
            transcript=transcript,
            actives=actives,
            entry_tags=entry.tags if entry is not None else (),
            first_captured_at=first_captured,
            started_at=started_at,
            project_key=project_key,
        )
        result = await mine_session(session, session_id, transcript, shown)
        await session.commit()
    return result


_SESSION_URI_PREFIX = "claude-code://session/"
_SESSION_URI_RE = re.compile(r"claude-code://session/(.+)$")


def session_id_from_uri(uri_r: str | None) -> str | None:
    """Extract the session id from a harvested transcript's ``uri_r``, if present."""
    if not uri_r:
        return None
    m = _SESSION_URI_RE.search(uri_r)
    return m.group(1) if m else None


@dataclass(frozen=True)
class HarvestedSession:
    """One harvested session transcript, as a mining pass reads it."""

    session_id: str
    transcript: str
    entry_tags: list[str]
    #: The transcript's first harvest: an upper bound on the session's start.
    first_captured_at: datetime
    #: Its latest harvest: what a delta pass's watermark compares.
    latest_captured_at: datetime


async def harvested_sessions(
    session: AsyncSession,
    *,
    since: datetime | None = None,
    on_read: Callable[[int, int], None] | None = None,
) -> tuple[list[HarvestedSession], int]:
    """Every harvested ``CONVERSATION`` transcript, and the count skipped for a missing blob.

    ``since`` keeps only transcripts whose latest snapshot was captured at or
    after it (a delta pass's watermark). ``on_read(read, total)`` reports
    progress. A missing blob is counted, not logged per entry, so one broken
    source cannot bury the run's result (discipline).
    """
    from particles.corpus.deposit import load_blob
    from particles.corpus.store import list_entries, list_snapshots_for_entry

    entries = await list_entries(session, limit=1_000_000, source_type="CONVERSATION")
    sessions: list[HarvestedSession] = []
    skipped = 0
    for read, entry in enumerate(entries, start=1):
        if on_read is not None:
            on_read(read, len(entries))
        snapshots = [
            s
            for s in await list_snapshots_for_entry(session, entry.entry_id)
            if s.content_hash and s.archive_path
        ]
        if not snapshots:
            continue
        latest = max(snapshots, key=lambda s: s.captured_at)
        latest_at = ensure_utc(latest.captured_at)
        if since is not None and latest_at < since:
            continue
        try:
            text = load_blob(latest.content_hash).decode("utf-8", errors="replace")
        except (OSError, FileNotFoundError):
            skipped += 1
            log.debug("utility mining: blob for entry %s missing; skipping", entry.entry_id)
            continue
        sessions.append(
            HarvestedSession(
                session_id=session_id_from_uri(entry.uri_r) or entry.entry_id,
                transcript=text,
                entry_tags=list(entry.tags),
                first_captured_at=min(ensure_utc(s.captured_at) for s in snapshots),
                latest_captured_at=latest_at,
            )
        )
    return sessions, skipped


async def plan_harvested_session(
    session: AsyncSession,
    reader: ExposureReader,
    harvested: HarvestedSession,
    actives: Sequence[Particle],
    *,
    behavioural_matching: bool | None = None,
    max_behavioural_calls: int | None = None,
) -> SessionMine:
    """Read one harvested session's exposure and plan its mine (reads only)."""
    shown, _ = await shown_beliefs(
        session,
        reader,
        session_id=harvested.session_id,
        transcript=harvested.transcript,
        actives=actives,
        entry_tags=harvested.entry_tags,
        first_captured_at=harvested.first_captured_at,
    )
    return await plan_session_mine(
        session,
        harvested.session_id,
        harvested.transcript,
        shown,
        behavioural_matching=behavioural_matching,
        max_behavioural_calls=max_behavioural_calls,
    )


def estimate_judge_calls(mines: Sequence[SessionMine]) -> PassEstimate:
    """List price of every matcher call ``mines`` would make, from the prompts' own size."""
    return requests_estimate(
        "use_judge", [r for m in mines for r in m.requests()], _MATCHER_MAX_TOKENS
    )


@dataclass(frozen=True)
class UtilityRebuildPlan:
    """A rebuild's gather: every session's planned mine, before any call or write."""

    mines: list[SessionMine]
    candidates: int
    skipped_missing_blob: int

    @property
    def calls(self) -> int:
        """Judge calls the rebuild would make."""
        return sum(len(m.groups) for m in self.mines)

    def estimate(self) -> PassEstimate:
        """The rebuild's judge calls at list price (``usage.py`` prices the run afterwards)."""
        return estimate_judge_calls(self.mines)


async def plan_store_utility(store: str) -> UtilityRebuildPlan:
    """Plan a full re-mine of every harvested session; reads only (§4).

    Each session gets the config cap, as a single-session mine does. The CLI
    reports the plan's call count and priced cost before anything is cleared.
    """
    async with session_scope(store) as session:
        actives = await get_particles_by_status(session, Status.ACTIVE)
        reader = await ExposureReader.load(session)
        sessions, skipped = await harvested_sessions(session)
        mines = [await plan_harvested_session(session, reader, h, actives) for h in sessions]
    return UtilityRebuildPlan(mines=mines, candidates=len(actives), skipped_missing_blob=skipped)


async def rebuild_store_utility(
    store: str,
    plan: UtilityRebuildPlan | None = None,
    *,
    latency_tolerant: bool = True,
) -> MiningResult:
    """Clear and re-derive every utility channel from its own system of record.

    The backfill (``particles memory rebuild-utility``), re-mining
    every harvested session: only shown beliefs are candidates
    and only an "applied" ruling credits one. With
    ``utility.mining.behavioural_matching`` off the judge never runs, so the
    mined channel is rebuilt empty.

    ``plan`` is :func:`plan_store_utility`'s result, so the caller can show its
    cost before this runs; omitted, it is planned here. The judge calls are
    sent as one batch unless ``latency_tolerant`` is ``False``. No
    write transaction is held across the wait.

    **Both** channels are rebuilt: the mined rows from the
    harvested transcripts, and the explicit operator credits replayed from the
    append-only ``BELIEF_MARKED_USEFUL`` event log. That is why the reset below
    can stay a blunt truncate — an explicit credit is never *preserved* through a
    rebuild, it is *reconstructed*, so there is no channel to special-case and no
    way for the two to drift apart. The session-start exposure record is a
    separate table this never touches.
    """
    from particles.operations.utility_feedback import rederive_explicit_credits
    from particles.store.utility_store import clear_utility_events

    if plan is None:
        plan = await plan_store_utility(store)
    replies = await judge_sessions(plan.mines, latency_tolerant=latency_tolerant)

    literal = behavioural = calls = nominated = 0
    async with session_scope(store) as session:
        await clear_utility_events(session)
        for mine, session_replies in zip(plan.mines, replies, strict=True):
            result = await record_session_mine(session, mine, session_replies)
            literal += result.literal
            behavioural += result.behavioural
            calls += result.behavioural_calls
            nominated += result.literal_nominated
        explicit = await rederive_explicit_credits(session)
        await session.commit()
    if plan.skipped_missing_blob:
        skipped = plan.skipped_missing_blob
        log.warning(
            "rebuild-utility: skipped %d harvested %s with a missing corpus blob; "
            "those sessions contribute no utility evidence (re-run with --debug to list them)",
            skipped,
            "entry" if skipped == 1 else "entries",
        )
    return MiningResult(
        literal=literal,
        behavioural=behavioural,
        candidates=plan.candidates,
        behavioural_calls=calls,
        skipped_missing_blob=plan.skipped_missing_blob,
        explicit=explicit,
        literal_nominated=nominated,
    )
