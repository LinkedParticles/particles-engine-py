# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Wikidata subject authority.

The Wikidata live-lookup path, migrated **verbatim** from ``subject_resolver``
into a registered :class:`SubjectAuthority`. The module-level helpers
(``_wikidata_candidates``, ``_wikidata_aliases``, ``_wikidata_link_confidence``,
``_is_prefix_expansion``, ``_continuation_aliases``) are kept at module scope so
existing tests patch them at ``particles.ingest.authorities.wikidata.*``.

Candidate selection is a config choice,
``subjects.wikidata_candidate_selection``: the top hit (the rule before
candidates were compared), the candidate whose description best matches the
claim text (see :func:`choose_candidate`), or, for an ambiguous name only, a
model's judgement among the candidates, which may be none of them (
the default; see :mod:`particles.ingest.authorities.wikidata_judge`). The
judge is shown a deeper search than the five hits every other rule reads
(``subjects.wikidata_judge_search_limit``): the one request asks for
more hits, each kept hit carries its ``rank``, and :func:`gate_hits` recovers
the first-five view, so which names reach the judge and how a name resolves
without it do not change with the depth.

``WikidataAuthority.resolve`` performs the live lookup and the QID-dedup
*read* (preserving the alias-fetch-skip optimization), but never writes:
inserts / alias-merges / cache stay in ``subject_resolver`` via the neutral
:class:`AuthorityResolution`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from particles.config import get_config
from particles.core.schema import ApplicabilityClause, ExternalRef
from particles.embeddings import cosine_similarity, get_embedding_model
from particles.http import get_capped, particles_client
from particles.ingest.authorities._shared import _RateLimiter, get_limiter
from particles.ingest.authorities.registry import AuthorityResolution
from particles.ingest.authorities.wikidata_judge import (
    Judgement,
    is_ambiguous,
    judge_candidates,
    judgeable_hits,
)
from particles.store.subject_store import find_by_external_ref
from particles.store.wikidata_cache import get_label

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


def _wikidata_limiter() -> _RateLimiter:
    return get_limiter("wikidata", get_config().subjects.wikidata_rate_limit_rps)


# ---------------------------------------------------------------------------
# Link confidence — moved verbatim from subject_resolver
# ---------------------------------------------------------------------------


#: One WARNING per process for the encoder-free link scorer. A
#: resolution pass calls the scorer once per candidate QID, so warning per call
#: would bury the message it is trying to deliver.
_unscored_warning_emitted = False


def _warn_unscored_once() -> None:
    """Disclose that link scoring is unavailable, once per process."""
    global _unscored_warning_emitted
    if _unscored_warning_emitted:
        return
    _unscored_warning_emitted = True
    log.warning(
        "No embedding model: Wikidata links cannot be scored, so every candidate "
        "attaches at the 0.5 unscoreable sentinel. The abstention cannot "
        "fire and the L-SEM-03 lint cannot flag them, so a plausible-but-wrong "
        "QID will be attached silently. Re-resolve these subjects with the model "
        "available to get real scores."
    )


def _wikidata_link_confidence(description: str, particle_content: str | None) -> float:
    """Score a Wikidata link by cosine similarity between entity description and particle content.

    Returns 1.0 when no particle content is available (conservative: display the link).
    Returns the cosine similarity score [0.0, 1.0] otherwise.

    The ``0.5`` sentinel means *could not be scored*, and the scorer deliberately
    lets it attach: the abstain floor sits strictly below it so that only
    scored-and-low links (the plausible-but-wrong mislinks) are dropped. That
    stays true here — but a bug found the missing-encoder case reaching the
    same sentinel through a `log.debug`, which made "the scorer is unavailable,
    so nothing can be abstained and nothing can be linted" indistinguishable
    from "this particular pair scored 0.5". The outcome is unchanged; the
    silence is not.
    """
    if not particle_content or not description:
        return 0.5  # uncertain but not suppressed by default threshold

    try:
        model = get_embedding_model()
        if model is None:
            _warn_unscored_once()
            return 0.5

        vecs = model.encode(
            [description, particle_content], convert_to_numpy=True, normalize_embeddings=True
        )
        # normalized cosine clamped to [0, 1] — the abstain cutoff this
        # feeds lives on that scale.
        return cosine_similarity(vecs[0], vecs[1])
    except Exception as exc:
        log.debug("Wikidata link confidence computation failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Wikidata API — moved verbatim from subject_resolver
# ---------------------------------------------------------------------------


#: The Wikidata languages whose aliases can vouch for a word-continuation label
#:. ``mul`` is Wikidata's language-neutral code, where many product
#: and brand names now live: PostgreSQL's "Postgres" alias is held only there,
#: not under ``en``.
_ALIAS_LANGUAGES = ("en", "mul")


def _is_word_continuation(query: str, label: str) -> bool:
    """True if ``label`` starts with ``query`` (any case) and goes on with a letter or digit."""
    if not query or len(label) <= len(query):
        return False
    return label.lower().startswith(query.lower()) and label[len(query)].isalnum()


def _is_prefix_expansion(query: str, label: str, aliases: Collection[str] = ()) -> bool:
    """True if ``label`` looks like an expansion of ``query`` into something else.

    Wikidata's ``wbsearchentities`` does prefix matching, so short common-name
    queries (software, projects, tools) often match the wrong entity:

    1. **Paper-title pattern** — ``query`` then a title separator
       (``:`` ``-`` ``–`` ``—``) then a subtitle. E.g.
       "FlashAttention" → "FlashAttention: Data-centric Interaction…".

    2. **Word-continuation pattern** — ``query`` is a stem; ``label`` extends it
       character-by-character into a different word. E.g. "micrograd"
       → "Microgradients of microbial oxygen consumption…".

    Either pattern adopts a wrong canonical name, so reject both. Legitimate
    longer labels keep a non-alphanumeric, non-separator break after the
    query (e.g. "OpenAI" → "OpenAI Inc.", "PyTorch" → "PyTorch (framework)").

    One word continuation is the same entity under its fuller name, and the
    entity says so: "Postgres" → "PostgreSQL", whose aliases hold
    "Postgres". A word continuation is therefore kept when ``aliases``, the
    candidate's own English and language-neutral aliases, include ``query``
    exactly. The match is case-sensitive on purpose: "Go" → "Goiás" carries
    the alias "GO" and must stay rejected. The rule is a pure function of the
    query, the label and the aliases, so the same responses always give the
    same answer (whitepaper § 3.3). The paper-title pattern takes no alias
    rescue.
    """
    if not query or not label:
        return False
    if len(label) <= len(query):
        return False
    if not label.lower().startswith(query.lower()):
        return False

    # Word continuation: the next char extends the query into a longer word,
    # unless the candidate lists the query itself as one of its names.
    if _is_word_continuation(query, label):
        return query not in aliases

    # Title separator (optionally with leading whitespace).
    rest = label[len(query) :].lstrip()
    return bool(rest) and rest[0] in (":", "-", "–", "—")


async def _continuation_aliases(qids: Sequence[str]) -> dict[str, frozenset[str]]:
    """Each QID's English and language-neutral aliases, in one ``wbgetentities`` call.

    Read only for the word-continuation candidates of a search, so a
    name with none costs no extra call. A failed read returns nothing, which
    leaves every continuation rejected: the filter's behaviour before the
    alias rescue existed.
    """
    if not qids:
        return {}
    await _wikidata_limiter().acquire()
    try:
        async with particles_client(timeout=10.0) as client:
            resp = await get_capped(
                client,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(qids),
                    "props": "aliases",
                    "languages": "|".join(_ALIAS_LANGUAGES),
                    "format": "json",
                },
            )
            resp.raise_for_status()
            entities = resp.json().get("entities", {})
            result: dict[str, frozenset[str]] = {}
            for qid in qids:
                by_language = entities.get(qid, {}).get("aliases", {})
                result[qid] = frozenset(
                    str(a["value"])
                    for language in _ALIAS_LANGUAGES
                    for a in by_language.get(language, [])
                    if a.get("value")
                )
            return result
    except Exception as exc:
        log.warning("Wikidata continuation alias fetch failed for %s: %s", qids, exc)
        return {}


def _usable_hits(
    name: str,
    hits: Sequence[dict[str, object]],
    aliases: Mapping[str, Collection[str]],
) -> list[dict[str, object]]:
    """The search hits the prefix-expansion filter keeps, in rank order (pure).

    Each kept hit is returned with its ``rank`` in the search response,
    1-based, which is what lets the ambiguity gate read only the first
    :data:`GATE_LIMIT` raw hits of a deeper search. The input
    hits are not modified.
    """
    usable: list[dict[str, object]] = []
    for rank, hit in enumerate(hits, 1):
        label = str(hit.get("label", ""))
        if _is_prefix_expansion(name, label, aliases.get(str(hit.get("id", "")), ())):
            log.info(
                "Wikidata: skipping prefix-expansion candidate %s (%r) for query %r",
                hit.get("id"),
                label,
                name,
            )
            continue
        usable.append({**hit, "rank": rank})
    return usable


#: How many search hits the ambiguity gate, ``top_hit`` and ``best_description``
#: read: the first five, as every selection has since. A deeper search
#: (``subjects.wikidata_judge_search_limit``) shows the judge more, but which
#: names reach it, and how a name resolves without it, are decided on these.
GATE_LIMIT = 5


def gate_hits(candidates: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """The usable hits among the first :data:`GATE_LIMIT` raw hits (pure).

    A hit without a ``rank`` (a recorded response from before ranks were
    stamped, or a test's) counts as within the gate. Usable hits keep rank
    order, so the result is always a prefix of ``candidates``.
    """
    gated: list[dict[str, object]] = []
    for hit in candidates:
        rank = hit.get("rank")
        if not isinstance(rank, int) or rank <= GATE_LIMIT:
            gated.append(hit)
    return gated


async def _wikidata_candidates(name: str, *, limit: int = GATE_LIMIT) -> list[dict[str, object]]:
    """Search Wikidata for an entity by name. Returns every usable hit, in rank order.

    One ``wbsearchentities`` call at ``limit`` hits; each hit carries its
    English ``description``, which is all :func:`choose_candidate` needs, so
    scoring the candidates costs no further network call. Prefix-expansion
    candidates (paper titles of the form "<query>: <subtitle>") are skipped so
    a short project name like "FlashAttention" doesn't get adopted as a
    scholarly article's full title. When a hit continues the name as a longer
    word, one batched alias read decides whether it is the same entity under a
    fuller name (see :func:`_is_prefix_expansion`). The filter and
    that alias read cover every hit the search returned, however deep. Each
    kept hit carries its ``rank`` in the response, so :func:`gate_hits` can
    recover the first-five view from a deeper search. An empty list (and thus
    a bare local Subject) when no candidate survives the filter or the search
    fails.
    """
    await _wikidata_limiter().acquire()
    try:
        async with particles_client(timeout=10.0) as client:
            resp = await get_capped(
                client,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "format": "json",
                    "limit": str(limit),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results: list[dict[str, object]] = data.get("search", [])
    except Exception as exc:
        log.warning("Wikidata search failed for %r: %s", name, exc)
        return []
    continuations = [
        str(hit.get("id", ""))
        for hit in results
        if _is_word_continuation(name, str(hit.get("label", "")))
    ]
    return _usable_hits(name, results, await _continuation_aliases(continuations))


async def _wikidata_aliases(qid: str) -> list[str]:
    """Fetch English labels and aliases for a Wikidata QID."""
    await _wikidata_limiter().acquire()
    try:
        async with particles_client(timeout=10.0) as client:
            resp = await get_capped(
                client,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "labels|aliases",
                    "languages": "en",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            entity = data.get("entities", {}).get(qid, {})
            aliases: list[str] = []
            # English label
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                aliases.append(label)
            # English aliases
            for a in entity.get("aliases", {}).get("en", []):
                v = a.get("value")
                if v and v not in aliases:
                    aliases.append(v)
            return aliases
    except Exception as exc:
        log.warning("Wikidata alias fetch failed for %s: %s", qid, exc)
        return []


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

#: The confidence a candidate takes when nothing could be scored: the
#: ``_wikidata_link_confidence`` sentinel, which attaches.
UNSCOREABLE = 0.5


@dataclass(frozen=True)
class CandidateChoice:
    """Which search candidate to use, at what confidence, and how.

    ``resolve`` is True when the candidate cleared the floor and becomes the
    Subject's identity (its label, aliases and description). False means the
    evidence is too weak to adopt it: the Subject keeps the extracted name and
    carries the candidate only as a low-confidence ref, which exporters hide
    and lint flags.
    """

    index: int
    confidence: float
    resolve: bool


def choose_candidate(
    scores: Sequence[float | None], floor: float, *, compare: bool
) -> CandidateChoice | None:
    """Pick a search candidate from its description-vs-claim scores (pure).

    ``scores[i]`` is candidate ``i``'s score in search-rank order, or ``None``
    when it has no description to score.

    ``compare=False`` is the ``top_hit`` selection: the first candidate,
    adopted whatever its score (a missing description scores
    :data:`UNSCOREABLE`), exactly the rule the resolver had before candidates
    were compared. The caller need only score that one.

    ``compare=True`` is ``best_description``. The best score wins and an exact tie
    goes to the higher-ranked candidate, so the choice is a function of the
    search response, the claim text and the encoder alone (whitepaper § 3.3).
    A candidate with no description is skipped while any candidate was scored:
    measured evidence beats the unscoreable sentinel. When none was, the top hit
    is taken at :data:`UNSCOREABLE`, which is the behaviour before candidates
    were compared.

    The best is adopted only at or above ``floor``. Below it the choice is
    still returned, with ``resolve=False``; whether even that weak ref is kept
    is the resolver's abstention floor, not this function's.
    """
    if not scores:
        return None
    if not compare:
        top = scores[0]
        return CandidateChoice(
            index=0, confidence=UNSCOREABLE if top is None else top, resolve=True
        )
    scored = [(score, i) for i, score in enumerate(scores) if score is not None]
    if not scored:
        return CandidateChoice(index=0, confidence=UNSCOREABLE, resolve=floor <= UNSCOREABLE)
    best, index = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return CandidateChoice(index=index, confidence=best, resolve=best >= floor)


def judged_choice(index: int, score: float | None) -> CandidateChoice:
    """The choice the subject-link judge's pick becomes (pure).

    The pick is adopted. Its confidence is its description score or the
    :data:`UNSCOREABLE` sentinel, whichever is higher: the judge read the
    claim and chose this candidate, which is better evidence than a cosine the
    measurement showed cannot separate right from wrong, and the sentinel is the
    value the resolver always attaches. A judged link therefore
    never falls under the abstention floor, and it is shown like any link above
    ``subjects.wikidata_link_suppress_threshold``.
    """
    confidence = UNSCOREABLE if score is None else max(score, UNSCOREABLE)
    return CandidateChoice(index=index, confidence=confidence, resolve=True)


def _score_candidates(
    candidates: Sequence[dict[str, object]], particle_content: str | None
) -> list[float | None]:
    """Each candidate's description scored against the claim text, local encoder only.

    No claim text or no encoder leaves every described candidate at the
    sentinel, so :func:`choose_candidate` falls back to the top hit exactly as
    the resolver did before it compared candidates.
    """
    return [
        _wikidata_link_confidence(desc, particle_content) if desc else None
        for desc in (str(hit.get("description") or "") for hit in candidates)
    ]


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------

# Pattern matching the old _NAMESPACE_PATTERNS wikidata row. NB: the captured id
# is the digits only (group 1) — recognize stores "123456", while the live path
# (resolve) stores the full "Q123456". This digits-vs-Qxxx asymmetry is
# pre-existing and preserved verbatim (move-not-rewrite).
_Q_PATTERN = re.compile(r"\bQ(\d{4,})\b")


class WikidataAuthority:
    """General-purpose live authority backed by the Wikidata API.

    The grandfathered broad-applicability authority (``APPLICABILITY = []`` ⇒
    applies to every domain). The § Constrained rule bars *new* unconditioned
    recognizers, not this one — Wikidata is the general fallback.
    """

    NAMESPACE = "wikidata"
    PRIORITY = 30
    LIVE = True
    DEFAULT_LINK_CONFIDENCE = 1.0
    APPLICABILITY: list[ApplicabilityClause] = []  # broad: applies to all domains

    def uri_for(self, external_id: str) -> str | None:
        # external_id is the full "Qxxx" form produced by resolve().
        return f"https://www.wikidata.org/wiki/{external_id}"

    def recognize(self, name: str) -> ExternalRef | None:
        m = _Q_PATTERN.search(name)
        if m:
            # id = digits only (parity with old _detect_namespace_pattern).
            return ExternalRef(namespace=self.NAMESPACE, id=m.group(1))
        return None

    async def resolve(
        self,
        session: AsyncSession,
        name: str,
        *,
        particle_content: str | None,
        domain: str | None,
    ) -> AuthorityResolution | None:
        cfg = get_config().subjects
        judging = cfg.wikidata_candidate_selection == "llm_judge"
        # The judge may be shown a deeper search; the gate and every
        # other selection read the first five of it, so what they decide is
        # what a five-hit search would have given them.
        limit = cfg.wikidata_judge_search_limit if judging else GATE_LIMIT
        candidates = await _wikidata_candidates(name, limit=limit)
        gated = gate_hits(candidates) if limit > GATE_LIMIT else candidates
        compare = cfg.wikidata_candidate_selection == "best_description"
        scores = _score_candidates(gated if compare else gated[:1], particle_content)
        choice = choose_candidate(scores, cfg.wikidata_link_suppress_threshold, compare=compare)
        if choice is None:
            return None
        if judging and particle_content:
            top = scores[0] if scores else None
            if is_ambiguous(len(gated), top, cfg.wikidata_link_suppress_threshold):
                # Name and disambiguation items are never what a claim names,
                # so the judge is not shown them; with nothing left,
                # its answer could only be "none".
                offered = judgeable_hits(candidates)
                positions = [
                    next(i for i, hit in enumerate(candidates) if hit is kept) for kept in offered
                ]
                judgement = (
                    await judge_candidates(session, name, particle_content, offered)
                    if offered
                    else Judgement(index=None)
                )
                # No usable answer keeps the top-hit choice already made.
                if judgement is not None and judgement.index is None:
                    log.info(
                        "Wikidata judge: none of %d candidate(s) is %r in this claim; abstaining",
                        len(offered),
                        name,
                    )
                    return AuthorityResolution(abstained=True)
                if judgement is not None and judgement.index is not None:
                    index = positions[judgement.index]
                    score = (
                        top
                        if index == 0
                        else _score_candidates([candidates[index]], particle_content)[0]
                    )
                    choice = judged_choice(index, score)

        chosen = candidates[choice.index]
        qid: str = str(chosen.get("id", ""))
        description = str(chosen.get("description", ""))
        log.debug(
            "Wikidata candidate %s (%s) of %d, confidence=%.2f, %s, for content %r",
            qid,
            description[:60],
            len(candidates),
            choice.confidence,
            "adopted" if choice.resolve else "below floor",
            (particle_content or "")[:60],
        )

        # QID-dedup read (preserves the alias-fetch-skip optimization): if the
        # QID is already stored, hand the existing subject back and let the
        # resolver own the alias-merge write.
        by_qid = await find_by_external_ref(session, self.NAMESPACE, qid)
        if by_qid:
            return AuthorityResolution(existing=by_qid)

        ref = ExternalRef(
            namespace=self.NAMESPACE,
            id=qid,
            uri=self.uri_for(qid),
            confidence=choice.confidence,
        )
        if not choice.resolve:
            # Too weak to adopt the candidate's identity: the resolver mints a
            # Subject under the extracted name carrying only this ref, at its
            # low confidence, so the link is recorded for review without the
            # entity's label and aliases replacing the name.
            return AuthorityResolution(external_ref=ref)

        aliases = await _wikidata_aliases(qid)
        if name not in aliases:
            aliases.append(name)
        canonical = aliases[0] if aliases else name
        return AuthorityResolution(
            external_ref=ref,
            canonical_name=canonical,
            aliases=aliases[1:],  # canonical is first; rest are aliases
            description=description or None,
        )

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        # external_id is the digits-only id from recognize(); reconstruct the
        # QID key for the label cache (matches the old bare-local fallback).
        qid_key = f"Q{external_id}"
        cached_label = await get_label(session, qid_key)
        if cached_label and cached_label != qid_key:
            return cached_label
        return None
