"""Transcript-mining pass — the reliable utility signal for the usefulness lens.

The spine: the harvest already records *what the agent
did* — its tool calls. When a session's actions demonstrably apply a belief
(the agent ran ``git commit -s``, used a documented workaround, verified on the
pinned Python), that action is a hard, unambiguous signal that the belief was
load-bearing — **credit action, not attention** (§1/§3). This is the
"unambiguous hook-observable action telemetry" branch of the owner's
reliable-signal constraint; it needs no user reaction, so the sentiment-guessing
failure mode is structurally impossible.

Two tiers, both over the distilled *tool-call* lines only (``[tool: Bash — git commit -s]``), never the conversational prose —
that is the "action, not attention" discipline:

- **Literal (deterministic, zero-cost, always on):** a belief that names a
  concrete token — a command, flag, path — matched against the action lines.
- **Behavioural (LLM-judged, bounded, ships in v1):** soft guidelines with no
  literal token, judged for consistency with the session's actions, capped at
  ``utility.mining.max_behavioural_calls`` per run (discipline). A
  multi-session caller (the consolidation pass 5) threads ONE shared
  budget through ``mine_session(max_behavioural_calls=remaining)`` so the cap
  is genuinely per *run*, not per session (correction, v1.74.1); a
  single-session caller (the SessionEnd inline pass) leaves it ``None`` and
  gets the config cap.

The mined events are recorded per session (idempotent — ``utility_store``); the
query-time ``UtilityPolicy`` turns them into a bounded projection-ranking factor.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import Particle
from particles.core.status import Status
from particles.db import session_scope
from particles.embeddings import cosine_similarity
from particles.llm.registry import complete
from particles.store.particle_store import (
    get_active_particles_with_embeddings,
    get_particles_by_status,
)
from particles.store.utility_store import record_utility_events

log = logging.getLogger(__name__)

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

_MAX_ACTION_CHARS = 6000  # bound the action summary fed to the behavioural LLM
_BEHAVIOURAL_BATCH = 15  # beliefs judged per LLM call

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

    literal: int
    behavioural: int
    candidates: int
    behavioural_calls: int
    #: Harvested entries skipped because their corpus blob is gone (an upstream
    #: fetch failure, a pruned archive). Counted rather than logged per entry so
    #: one broken source cannot bury the run's result; surfaced in the CLI line.
    skipped_missing_blob: int = 0
    #: True when the behavioural tier wanted more LLM calls than its budget
    #: allowed — the multi-session caller's truncation-disclosure signal.
    behavioural_truncated: bool = False
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


def match_literal(actives: list[Particle], action_lines: list[str]) -> dict[str, str]:
    """Beliefs whose literal token appears in the session's action lines → ``{pid: token}``.

    Single-token spans match as substrings; multi-token command spans match as an
    ordered token sequence within one action line (see :func:`_token_sequence_re`).

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
    summary = "\n".join(action_lines)[:_MAX_ACTION_CHARS]
    action_vec = model.encode([summary], convert_to_numpy=True, normalize_embeddings=True)[0]
    vec_by_id = {p.id: vec for p, vec in await get_active_particles_with_embeddings(session)}
    ranked = sorted(
        unmatched,
        key=lambda p: cosine_similarity(action_vec, vec_by_id[p.id]) if p.id in vec_by_id else -1.0,
        reverse=True,
    )
    return ranked[:limit], excluded, True


_MATCHER_SYSTEM = (
    "You judge whether an AI coding agent, in one session, ACTED in "
    "accordance with a stated guideline — i.e. its tool actions clearly "
    "followed or applied the guideline. Be strict: only count a guideline "
    "when the actions plainly demonstrate it, not when they merely could be "
    "consistent with it. Reply with ONLY a JSON array of the numbers of the "
    "guidelines the actions followed, e.g. [1,4]. Empty array if none."
)


def _matcher_prompt(batch: list[Particle], summary: str) -> str:
    """The behavioural-matcher user turn for one batch of beliefs."""
    listing = "\n".join(f"{i + 1}. {p.content}" for i, p in enumerate(batch))
    return (
        f"Session actions (tool calls):\n{summary}\n\n"
        f"Guidelines:\n{listing}\n\n"
        "Which guideline numbers did the actions follow?"
    )


def _credit_matched(batch: list[Particle], reply: str, matched: set[str]) -> None:
    """Add the belief ids named by ``reply``'s guideline numbers to ``matched``."""
    for n in re.findall(r"\d+", reply):
        idx = int(n) - 1
        if 0 <= idx < len(batch):
            matched.add(batch[idx].id)


async def _match_behavioural(
    unmatched: list[Particle],
    action_lines: list[str],
    max_calls: int,
    *,
    latency_tolerant: bool = False,
) -> tuple[set[str], int, bool]:
    """LLM-judge which of ``unmatched`` the session's actions followed (bounded).

    Groups beliefs into at most ``max_calls`` completions of
    ``_BEHAVIOURAL_BATCH`` beliefs each; degrades on any provider error
    (literal-only fallback). Returns ``(matched ids, calls made, truncated)`` —
    ``truncated`` is True when the budget ran out before every group was judged
    (including a zero budget with work wanting judgement).

    ``latency_tolerant`` sends the groups as one asynchronous
    half-price batch instead of sequential calls. The consolidation
    pass sets it; the SessionEnd inline mine does not, because that one runs
    while the operator's session is closing.
    """
    if not unmatched or not action_lines:
        return set(), 0, False
    if max_calls <= 0:
        # Work wanted judgement but the (shared, per-run) budget is spent.
        return set(), 0, True
    summary = "\n".join(action_lines)[:_MAX_ACTION_CHARS]
    groups = [
        unmatched[start : start + _BEHAVIOURAL_BATCH]
        for start in range(0, len(unmatched), _BEHAVIOURAL_BATCH)
    ]
    truncated = len(groups) > max_calls
    groups = groups[:max_calls]
    matched: set[str] = set()

    if latency_tolerant:
        from particles.llm import CompletionRequest, complete_many

        try:
            replies = await complete_many(
                "semantic_lint",
                [
                    CompletionRequest(
                        prompt=_matcher_prompt(group, summary), system=_MATCHER_SYSTEM
                    )
                    for group in groups
                ],
                max_tokens=200,
                response_schema=_MATCHER_RESPONSE_SCHEMA,
                latency_tolerant=True,
            )
        except Exception as exc:  # noqa: BLE001 — see the sequential arm below
            log.info(
                "utility mining: behavioural matcher unavailable (%s); literal-only this run",
                exc,
            )
            return set(), 0, truncated
        calls = 0
        for group, reply in zip(groups, replies, strict=True):
            if reply is None:
                # One dead request in an otherwise live batch. It cost nothing
                # usable, so it does not draw down the per-run call budget.
                continue
            calls += 1
            _credit_matched(group, reply, matched)
        return matched, calls, truncated

    calls = 0
    for group in groups:
        calls += 1
        try:
            reply = await complete(
                "semantic_lint",
                _matcher_prompt(group, summary),
                max_tokens=200,
                system=_MATCHER_SYSTEM,
                response_schema=_MATCHER_RESPONSE_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment; any provider
            # error (CompletionError, a raw SDK 4xx/5xx like an exhausted credit
            # balance, a timeout) degrades to literal-only rather than failing the
            # whole mining run. Literal matching has already been recorded.
            log.info(
                "utility mining: behavioural matcher unavailable (%s); literal-only this run",
                exc,
            )
            calls -= 1
            break
        _credit_matched(group, reply, matched)
    return matched, calls, truncated


async def mine_session(
    session: AsyncSession,
    session_id: str,
    transcript: str,
    actives: list[Particle],
    *,
    behavioural_matching: bool | None = None,
    max_behavioural_calls: int | None = None,
    latency_tolerant: bool = False,
) -> MiningResult:
    """Mine one session's actions into utility events for the store's beliefs.

    Runs the literal tier (always) and, when ``utility.mining.behavioural_matching``
    is set, the bounded behavioural tier over the beliefs the literal tier did
    not catch. Records the union as utility events for ``session_id`` (idempotent).

    ``behavioural_matching`` overrides the config knob for one run — the degraded consolidation pass forces the literal-only tier
    (``False``) so a structural-only cycle never attempts an LLM call.

    ``latency_tolerant`` lets the behavioural tier's calls go out as
    one asynchronous half-price batch; the consolidation pass sets it,
    the SessionEnd inline mine leaves it ``False``.

    ``max_behavioural_calls`` overrides the config cap for THIS invocation — a
    multi-session caller (the consolidation pass 5) threads its
    remaining per-run budget here so the cap accumulates across sessions
    instead of resetting per session (correction, v1.74.1). ``0``
    makes no behavioural calls (the literal tier still runs — it is LLM-free)
    and reports ``behavioural_truncated`` when judgement was wanted.
    """
    action_lines = extract_action_lines(transcript)
    literal = match_literal(actives, action_lines)

    cfg = get_config().utility.mining
    behavioural: set[str] = set()
    calls = 0
    truncated = False
    run_behavioural = (
        cfg.behavioural_matching if behavioural_matching is None else behavioural_matching
    )
    prefiltered = 0
    prefilter_ranked = True
    if run_behavioural:
        unmatched = [p for p in actives if p.id not in literal]
        if action_lines:
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
        budget = (
            cfg.max_behavioural_calls if max_behavioural_calls is None else max_behavioural_calls
        )
        behavioural, calls, truncated = await _match_behavioural(
            unmatched, action_lines, budget, latency_tolerant=latency_tolerant
        )

    events: dict[str, str] = {pid: "literal" for pid in literal}
    for pid in behavioural:
        events.setdefault(pid, "behavioural")
    await record_utility_events(session, session_id, events)
    return MiningResult(
        literal=len(literal),
        behavioural=len(behavioural),
        candidates=len(actives),
        behavioural_calls=calls,
        behavioural_truncated=truncated,
        behavioural_prefiltered=prefiltered,
        behavioural_prefilter_ranked=prefilter_ranked,
    )


async def mine_session_from_transcript(
    store: str,
    transcript: str,
    session_id: str,
) -> MiningResult:
    """Open a store session, load ACTIVE beliefs, and mine ``transcript`` for utility.

    The harvest-cycle entry point: called after the session's
    harvest + extract, against the store's *current* ACTIVE beliefs — so the
    session's actions reinforce the beliefs they applied, whether learned this
    session or long ago.
    """
    async with session_scope(store) as session:
        actives = await get_particles_by_status(session, Status.ACTIVE)
        result = await mine_session(session, session_id, transcript, actives)
        await session.commit()
    return result


_SESSION_URI_RE = re.compile(r"claude-code://session/(.+)$")


def session_id_from_uri(uri_r: str | None) -> str | None:
    """Extract the session id from a harvested transcript's ``uri_r``, if present."""
    if not uri_r:
        return None
    m = _SESSION_URI_RE.search(uri_r)
    return m.group(1) if m else None


async def rebuild_store_utility(store: str) -> MiningResult:
    """Clear and re-derive every utility channel from its own system of record.

    The backfill (``particles memory rebuild-utility``): turning the
    feature on (or changing the matchers) credits history rather than starting
    blind. Honours ``utility.mining.behavioural_matching`` — over many sessions
    the behavioural tier can be costly, so an operator may set it off in config
    for a cheap literal-only rebuild. Returns aggregate disclosure counts.

    **Both** channels are rebuilt: the mined rows from the
    harvested transcripts, and the explicit operator credits replayed from the
    append-only ``BELIEF_MARKED_USEFUL`` event log. That is why the reset below
    can stay a blunt truncate — an explicit credit is never *preserved* through a
    rebuild, it is *reconstructed*, so there is no channel to special-case and no
    way for the two to drift apart.
    """
    from particles.corpus.deposit import load_blob
    from particles.corpus.store import list_entries, list_snapshots_for_entry
    from particles.operations.utility_feedback import rederive_explicit_credits
    from particles.store.utility_store import clear_utility_events

    literal = behavioural = candidates = calls = skipped = explicit = 0
    async with session_scope(store) as session:
        await clear_utility_events(session)
        actives = await get_particles_by_status(session, Status.ACTIVE)
        candidates = len(actives)
        entries = await list_entries(session, limit=1_000_000, source_type="CONVERSATION")
        for entry in entries:
            sid = session_id_from_uri(entry.uri_r) or entry.entry_id
            snapshots = [
                s
                for s in await list_snapshots_for_entry(session, entry.entry_id)
                if s.content_hash and s.archive_path
            ]
            if not snapshots:
                continue
            latest = max(snapshots, key=lambda s: s.captured_at)
            try:
                text = load_blob(latest.content_hash).decode("utf-8", errors="replace")
            except (OSError, FileNotFoundError):
                # Aggregated, not one WARNING per entry: a broken upstream fetch
                # can strand hundreds of blobs at once, and per-item warnings bury
                # the run's actual result (discipline).
                skipped += 1
                log.debug("rebuild-utility: blob for entry %s missing; skipping", entry.entry_id)
                continue
            result = await mine_session(session, sid, text, actives)
            literal += result.literal
            behavioural += result.behavioural
            calls += result.behavioural_calls
        explicit = await rederive_explicit_credits(session)
        await session.commit()
    if skipped:
        log.warning(
            "rebuild-utility: skipped %d harvested %s with a missing corpus blob; "
            "those sessions contribute no utility evidence (re-run with --debug to list them)",
            skipped,
            "entry" if skipped == 1 else "entries",
        )
    return MiningResult(
        literal=literal,
        behavioural=behavioural,
        candidates=candidates,
        behavioural_calls=calls,
        skipped_missing_blob=skipped,
        explicit=explicit,
    )
