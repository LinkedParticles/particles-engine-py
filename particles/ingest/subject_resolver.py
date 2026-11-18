# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subject resolution cascade.

For each subject name extracted by the LLM, resolve to a canonical Subject:
  1. In-memory cache
  2. Local alias index — match against existing subjects in the store
  3. Recognize pass — each authority's in-name pattern, priority order
  4. Resolve pass — each applicable live authority (Wikidata, …), priority order
  5. Bare local Subject — create with extracted name; can be enriched later

The hardcoded Wikidata path and ``_NAMESPACE_PATTERNS`` regex list now live in
``particles/ingest/authorities/`` as registered ``SubjectAuthority``
plugins. This module owns the cascade and **every write** (insert /
alias-merge / cache); authorities only recognize and resolve.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles import egress
from particles.config import get_config
from particles.core.provisional import (
    NameCandidate,
    ResolutionKind,
    by_ref_choice,
    choose_local,
    mints_provisional,
)
from particles.core.schema import ExternalRef, Subject
from particles.extraction.registry import infer_domain
from particles.ingest.authorities import (
    AuthorityResolution,
    ContextualRecognizer,
    RecognizeContext,
    SubjectAuthority,
    get_authorities,
    is_applicable,
)
from particles.ingest.update_supersession import SpeakerKey, persona_names, speaker_forms
from particles.store import subject_cache
from particles.store.probe_verdict_store import (
    DeferredAnswer,
    deferring_answers,
    record_deferred,
)
from particles.store.subject_cache import CacheEntry
from particles.store.subject_provisional import (
    AcceptedVia,
    accept,
    any_provisional,
    mark_of,
    mark_provisional,
    share,
)
from particles.store.subject_store import (
    add_aliases,
    find_by_external_ref,
    find_by_name,
    find_name_candidates,
    get_subject,
    insert_subject,
    is_persona_source,
    merge_subjects_many,
    persona_alias_guard,
    split_subject,
)

log = logging.getLogger(__name__)

#: Actor on the ``SUBJECT_ALIASED`` event a recorded persona form carries.
PERSONA_FOLD_ACTOR = "persona-fold"

#: Actor on the ``SUBJECT_ALIASED`` event recorded when the cascade attaches an
#: extracted name to an existing Subject (a live-authority or dedup attach). A
#: process write, not an operator one (D1).
RESOLVER_ALIAS_ACTOR = "subject-resolver"

#: Persona forms some *other* Subject already answers to, per store — skipped
#: by :func:`_record_persona_alias`, and remembered so a split store does not
#: pay a name lookup on every mention of the speaker. Process-local: a form
#: freed later (a merge, a rename) is recorded after the next restart.
_persona_forms_held_elsewhere: set[str] = set()


@dataclass
class LiveAnswers:
    """Live-authority answers asked before the store's writer lock (D2).

    A live lookup reaches the network and, for the Wikidata judge, the model,
    so a write path that holds the writer lock asks none.
    :func:`prefetch_subjects` asks the authorities first and records each
    answer by folded name and namespace; ``resolve_subjects(answers=...)``
    replays them under the lock. An authority the prefetch did not ask for a
    name is not asked under the lock either: the replay skips it as an
    authority it could not reach, and records no negative cache entry for it.

    ``ledger`` holds the answers an authority recorded in the probe ledger
    during the prefetch (the Wikidata judge's pick). The prefetch writes none
    of them, so it opens no SQLite write transaction before the lock;
    ``resolve_subjects(answers=...)`` writes them under it (security review
    2026-10-31-1, AD6).
    """

    asked: dict[tuple[str, str], AuthorityResolution | None] = field(default_factory=dict)
    ledger: list[DeferredAnswer] = field(default_factory=list)


@dataclass
class SubjectPrefetch:
    """What :func:`prefetch_subjects` learned, with no store write."""

    answers: LiveAnswers
    known_ids: list[str]
    """The ids of the names that resolve to a Subject the store already holds,
    in name order. A name that would mint a Subject has none yet."""


def clear_cache() -> None:
    """Clear the subject-resolution cache and reset authority state.

    Used by tests (autouse fixture) and at module reload. Store-side mutations
    invalidate the cache by calling ``subject_cache.clear()`` directly — they
    must not touch resolver-internal state. ``clear_authorities`` rebuilds the
    registry (picking up config changes) and drops the per-authority rate
    limiters.
    """
    subject_cache.clear()
    _persona_forms_held_elsewhere.clear()
    from particles.ingest.authorities import clear_authorities

    clear_authorities()


def _skip_live_authorities(source_type: str | None) -> bool:
    """Whether live-ontology authorities should be skipped for this source type.

    Conversational and journal-like sources (chat-transcript harvests, personal
    journals) mint subject names that are private referents by construction —
    "the user's hamster", "Luna" — essentially none of which resolve against a
    public ontology. Skipping the live lookup for them avoids hundreds of
    fruitless, rate-limited Wikidata calls per session (the limiter otherwise
    serialises the whole process behind them). Exact-identifier authorities
    (Numista / ISBN / DOI) are recognize-only and never reach the live pass, so
    they are unaffected. Config-driven so an operator can widen or narrow the
    set.
    """
    if source_type is None:
        return False
    return source_type in get_config().subjects.skip_live_authorities_source_types


async def _attach_alias(session: AsyncSession, subject: Subject, name: str) -> Subject:
    """Record ``name`` as an alias of the existing ``subject`` it resolved to.

    Routed through :func:`~particles.store.subject_store.add_aliases`,
    so the write carries a ``SUBJECT_ALIASED`` event under
    :data:`RESOLVER_ALIAS_ACTOR` and clears the resolution cache like every
    other alias write. A name the subject already answers to returns early, so a
    repeat attach records nothing. Returns the updated Subject; the caller
    caches that one, *after* the clear.
    """
    folded = name.casefold()
    if folded in {subject.canonical_name.casefold(), *(a.casefold() for a in subject.aliases)}:
        return subject
    updated, _ = await add_aliases(session, subject.id, [name], actor=RESOLVER_ALIAS_ACTOR)
    return updated


def _persona_fold(
    name: str, source_type: str | None, source_tags: Collection[str] = ()
) -> str | None:
    """The persona Subject's name when ``name`` is the speaker of a persona source.

    ``None`` when the source is not a persona source
    (:func:`~particles.store.subject_store.is_persona_source`) or ``name`` is
    not one of ``subjects.persona_aliases`` nor the canonical name itself.
    """
    cfg = get_config().subjects
    if not is_persona_source(source_type, source_tags):
        return None
    form = name.strip().casefold()
    if form == cfg.persona_canonical_name.casefold() or form in {
        a.casefold() for a in cfg.persona_aliases
    }:
        return cfg.persona_canonical_name
    return None


def _persona_canonical(
    name: str, source_type: str | None, source_tags: Collection[str] = ()
) -> str:
    """Fold a persona source's speaker references onto one name.

    "user", "the user", "the speaker", "I" are the same person in a chat
    transcript, but each surface form minted its own Subject, which split that
    person's beliefs into groups that never reconciled against each other. The
    fold is a *resolution* rule: it changes which Subject a new claim binds to,
    never an existing binding. A persona source is a persona source type or an
    entry carrying a persona tag, such as a Claude Code memory file.
    """
    folded = _persona_fold(name, source_type, source_tags)
    return name if folded is None else folded


@dataclass(frozen=True)
class _LocalAnswer:
    """What the local-index step found for a name, and what the write would do."""

    subject: Subject
    share: bool
    provisional: bool


async def _find_local(
    session: AsyncSession,
    name: str,
    source_type: str | None,
    source_tags: Collection[str] = (),
    *,
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> _LocalAnswer | None:
    """The local name/alias lookup of a path that binds a particle to the result.

    ``name`` is already folded. Outside a persona source a persona form never
    resolves through the persona Subject's recorded aliases
    (:func:`~particles.store.subject_store.persona_alias_guard`). Which of the
    Subjects holding the name answers is :func:`choose_local`'s, by
    ``principal`` and ``kind``: a provisional Subject answers
    only a minting resolution.
    """
    guard = persona_alias_guard(name, source_type, source_tags)
    candidates = await find_name_candidates(
        session, name, skip_aliases_of=guard, stop_at_confirmed=True
    )
    if not candidates:
        return None
    choice = choose_local(
        [NameCandidate(subject.id, mark) for subject, mark in candidates],
        principal=principal,
        kind=kind,
    )
    if choice.subject_id is None:
        return None
    subject = next(s for s, _ in candidates if s.id == choice.subject_id)
    return _LocalAnswer(subject=subject, share=choice.share, provisional=choice.provisional)


def _answer_key(
    session: AsyncSession,
    name: str,
    *,
    provisional: bool,
    principal: str | None,
    kind: ResolutionKind,
) -> str:
    """The cache key an answer is stored under: scoped when it is provisional."""
    if provisional and principal is not None:
        return subject_cache.make_scoped_key(session, name, principal, kind.value)
    return subject_cache.make_key(session, name)


async def _reached_by_ref(
    session: AsyncSession,
    subject: Subject,
    *,
    principal: str | None,
    kind: ResolutionKind,
    dry: bool,
    actor: str,
    source_type: str | None,
) -> bool:
    """Apply what reaching ``subject`` by its external reference does (§4).

    A minting resolution shares another principal's scoped provisional Subject;
    any other resolution accepts a provisional one (``via: EXTERNAL_REF``). A
    dry run performs neither. Returns whether ``subject`` is still provisional
    after this, which decides the answer's cache key.
    """
    mark = await mark_of(session, subject.id)
    if mark is None:
        return False
    do_share, do_accept = by_ref_choice(
        NameCandidate(subject.id, mark), principal=principal, kind=kind
    )
    if dry:
        return not do_accept
    if do_share:
        assert principal is not None
        await share(session, subject.id, principal)
        return True
    if do_accept:
        refs = [f"{r.namespace}:{r.id}" for r in subject.external_ids]
        await accept(
            session,
            subject.id,
            via=AcceptedVia.EXTERNAL_REF,
            actor=actor,
            detail={"refs": refs, "resolution": kind.value, "source_type": source_type},
        )
        return False
    return True


async def fold_or_hold(session: AsyncSession, subject: Subject) -> Subject:
    """Fold bare-local provisional Subjects into a confirmed bare-local mint.

    ``subject`` was just minted confirmed. Each provisional Subject holding one
    of its names that carries no external reference is merged into it (a
    ``SUBJECTS_MERGED`` event, actor ``subject-resolver``, reason
    ``provisional-fold``), when ``subject`` carries none either. Otherwise the
    pair is held: both stay, and ``subjects list --provisional`` lists it.
    Returns ``subject`` as it stands after any merge.
    """
    if not await any_provisional(session):
        return subject
    holders: dict[str, Subject] = {}
    for held_name in [subject.canonical_name, *subject.aliases]:
        for candidate, mark in await find_name_candidates(session, held_name):
            if mark is not None and candidate.id != subject.id:
                holders[candidate.id] = candidate
    if not holders:
        return subject
    # The confirmed mint now holds the name for every resolution; a cached
    # provisional answer for it is stale.
    subject_cache.clear(keep_negative=True)
    if subject.external_ids:
        log.info(
            "Held %d provisional subject(s) named %r beside confirmed %s (external ref)",
            len(holders),
            subject.canonical_name,
            subject.id,
        )
        return subject
    fold = sorted(s.id for s in holders.values() if not s.external_ids)
    if not fold:
        return subject
    merged, _ = await merge_subjects_many(
        session,
        fold,
        subject.id,
        actor=RESOLVER_ALIAS_ACTOR,
        extra_payload={"reason": "provisional-fold"},
    )
    log.info("Folded %d provisional subject(s) into %s", len(fold), subject.id)
    return merged


async def find_existing_subject(
    session: AsyncSession,
    name: str,
    *,
    source_type: str | None,
    source_tags: Collection[str] = (),
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> Subject | None:
    """Resolve ``name`` to an *existing* Subject exactly as the write path would, read-only.

    The one seam for a caller that must agree with :func:`resolve_subject`
    about a name before (or without) resolving it — the §6.6 precompute is the
    reference caller. It applies the persona fold and the persona-alias scope,
    and stops where ``resolve_subject`` would start writing: no live authority,
    no mint, no alias recorded. A name resolved by two paths must be folded and
    scoped by both; routing both through here is what keeps that
    true by construction. ``principal`` and ``kind`` are the write's;
    a transition the write would perform (a share) is not performed here.
    """
    folded = _persona_canonical(name, source_type, source_tags)
    found = await _find_local(
        session, folded, source_type, source_tags, principal=principal, kind=kind
    )
    return found.subject if found is not None else None


async def load_speaker_key(session: AsyncSession) -> SpeakerKey | None:
    """The persona key every reader of the about-subject key shares.

    ``None`` when ``subjects.persona_openers`` is empty (off) or the
    store holds no persona Subject yet, in which case no stored claim can be
    keyed on it either. The persona Subject is the one named
    ``persona_canonical_name``, the Subject the fold resolves the speaker to.
    """
    cfg = get_config().subjects
    forms = speaker_forms(cfg)
    if not forms:
        return None
    persona = await find_by_name(session, cfg.persona_canonical_name)
    if persona is None:
        return None
    return SpeakerKey(subject_id=persona.id, forms=forms, names=persona_names(cfg))


async def _record_persona_alias(session: AsyncSession, subject: Subject, form: str) -> Subject:
    """Record a folded surface form as an alias of the persona Subject.

    Without it the fold leaves no trace: ``find_by_name("user")`` returns
    ``None`` on a store whose every conversational claim is about "the user",
    and each caller resolving a subject by name has to remember to fold.
    Recording the form once makes every such lookup resolve it and
    shows the fold to an operator reading ``subjects show``.

    It stays a resolution rule. No particle–Subject binding is touched, and a
    form some other Subject already answers to — a "User" split off before the
    fold existed — is skipped, so the write never changes what a name resolved
    to before it; it only answers a name that resolved to nothing. Runs on
    every fold, not only on the mint, so a store folded before this existed
    heals form by form as it keeps ingesting, inside the write lock and
    disclosed by a ``SUBJECT_ALIASED`` event — never at read time.
    """
    folded = form.casefold()
    if folded in {subject.canonical_name.casefold(), *(a.casefold() for a in subject.aliases)}:
        return subject
    held_key = subject_cache.make_key(session, folded)
    if held_key in _persona_forms_held_elsewhere:
        return subject
    holder = await find_by_name(session, form)
    if holder is not None and holder.id != subject.id:
        _persona_forms_held_elsewhere.add(held_key)
        log.info(
            "Persona form %r already names subject %s; not recorded as an alias of %r",
            form,
            holder.id,
            subject.canonical_name,
        )
        return subject
    updated, _ = await add_aliases(session, subject.id, [form], actor=PERSONA_FOLD_ACTOR)
    return updated


async def resolve_subject(
    session: AsyncSession,
    name: str,
    asserted_by: str = "general-extractor",
    particle_content: str | None = None,
    *,
    source_type: str | None = None,
    source_tags: Collection[str] = (),
    answers: LiveAnswers | None = None,
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> Subject:
    """Resolve a subject name to a canonical Subject, creating one if needed.

    ``principal`` and ``kind`` say whose resolution this is: a
    ``MINTING`` resolution acts for the agent principal ``principal``, sees its
    provisional Subjects and mints provisionally; every other kind sees
    confirmed Subjects only. ``asserted_by`` stays the attribution label.

    ``source_tags`` are the corpus entry's tags; with ``source_type`` they
    decide whether the speaker fold applies. A folded name never goes to a
    live authority: the speaker is a private referent, and Wikidata
    answered "user" with "user account" and so split one person across files.
    ``answers`` replays a :func:`prefetch_subjects` lookup instead of asking
    the live authorities; the ledger answers that prefetch kept are written
    first.
    """
    if answers is not None and answers.ledger:
        await record_deferred(session, answers.ledger)
    persona = _persona_fold(name, source_type, source_tags)
    folded = name if persona is None else persona
    subject = await _resolve(
        session,
        folded,
        asserted_by,
        particle_content,
        source_type,
        source_tags,
        live=persona is None,
        answers=answers,
        principal=principal,
        kind=kind,
    )
    assert subject is not None  # only a dry run returns None
    if folded == name:
        return subject
    recorded = await _record_persona_alias(session, subject, name.strip())
    if recorded is not subject and await mark_of(session, recorded.id) is None:
        # ``add_aliases`` cleared the resolution cache; re-seat this one.
        subject_cache.cache_set(subject_cache.make_key(session, folded), recorded)
    return recorded


# Known deviation: the live-authority loop's decisions are interleaved with I/O in
# ``_resolve``; the local-index choice was lifted into ``core.provisional``, and
# the rest is (D2).
_egress_skip_logged: set[str] = set()


def _live_authority_allowed(auth: SubjectAuthority) -> bool:
    """Whether ``network.egress`` lets a live authority reach its host.

    An authority names its destination as ``EGRESS_HOST`` (``(host, port)``);
    one that names none is left to the transport gate. Logged once per
    authority per process.
    """
    host = getattr(auth, "EGRESS_HOST", None)
    if host is None or egress.allows(*host):
        return True
    namespace = str(getattr(auth, "NAMESPACE", type(auth).__name__))
    if namespace not in _egress_skip_logged:
        _egress_skip_logged.add(namespace)
        log.info(
            "Live authority %s skipped: network.egress does not allow %s:%s; "
            "subjects resolve locally",
            namespace,
            *host,
        )
    return False


async def _resolve(
    session: AsyncSession,
    name: str,
    asserted_by: str,
    particle_content: str | None,
    source_type: str | None,
    source_tags: Collection[str] = (),
    *,
    live: bool = True,
    answers: LiveAnswers | None = None,
    dry: bool = False,
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> Subject | None:
    """The resolution cascade for an already-folded ``name``.

    ``principal`` and ``kind`` decide which provisional Subjects the local and
    by-reference steps see, and whether a mint is provisional.

    ``live=False`` skips the live-authority pass, as a conversational source
    type does for every name.

    ``dry=True`` is :func:`prefetch_subjects`' run: it asks the live
    authorities, records each answer in ``answers``, and writes nothing (no
    Subject, alias or cache entry). It returns the Subject the store already
    holds for ``name``, else ``None`` where the cascade would mint one. With
    ``dry=False`` and ``answers`` set, the recorded answers stand in for the
    live authorities, which are not asked.
    """
    cache_key = subject_cache.make_key(session, name)
    minting = mints_provisional(kind, principal)

    def key_for(provisional: bool) -> str:
        return _answer_key(session, name, provisional=provisional, principal=principal, kind=kind)

    # Check in-memory cache first (avoids repeat DB + API calls for same name).
    # A confirmed answer answers everyone; a provisional one only its own key.
    cached = subject_cache.cache_get(cache_key)
    if cached is None and minting:
        cached = subject_cache.cache_get(key_for(True))
    if isinstance(cached, CacheEntry) and cached.subject is not None:
        return cached.subject

    # Step 1: local alias index (decides which holder answers)
    local = await _find_local(
        session, name, source_type, source_tags, principal=principal, kind=kind
    )
    if local is not None:
        existing = local.subject
        if not dry:
            if local.share:
                assert principal is not None
                await share(session, existing.id, principal)
            subject_cache.cache_set(key_for(local.provisional), existing)
        log.debug("Subject resolved locally: %r → %s", name, existing.id)
        return existing

    async def by_ref(found: Subject) -> bool:
        return await _reached_by_ref(
            session,
            found,
            principal=principal,
            kind=kind,
            dry=dry,
            actor=asserted_by or RESOLVER_ALIAS_ACTOR,
            source_type=source_type,
        )

    authorities = get_authorities()  # priority-ordered

    # Step 2: recognize pass — in-name patterns, before any live lookup.
    # First (highest-priority) authority that recognises the name wins.
    recognized_ref: ExternalRef | None = None
    recognizing_auth: SubjectAuthority | None = None
    for auth in authorities:
        ref = auth.recognize(name)
        if ref is None:
            continue
        reached = await find_by_external_ref(session, ref.namespace, ref.id)
        if reached:
            still_provisional = await by_ref(reached)
            if not dry:
                subject_cache.cache_set(key_for(still_provisional), reached)
            return reached
        recognized_ref = ref
        recognizing_auth = auth
        break

    # Step 3: resolve pass — live lookups, priority order, domain-applicable.
    domain = infer_domain(source_type) if source_type else None
    # Skip the live pass entirely when either (a) the source is a conversational
    # / journal-like type whose subject names are private referents by
    # construction (see :func:`_skip_live_authorities`), or (b) a prior *real*
    # search recorded a process-global miss for this name (negative cache). Both
    # fall straight through to the Step 4 bare-local Subject; only positive
    # resolutions are store-scoped, so the negative check is unscoped by design.
    skip_live = not live or _skip_live_authorities(source_type)
    if not skip_live and subject_cache.negative_get(name):
        skip_live = True
    if not skip_live:
        searched_live = False
        all_missed = True
        for auth in authorities:
            if not auth.LIVE or not is_applicable(auth, domain):
                continue
            if not _live_authority_allowed(auth):
                # Refused by network.egress: not asked, so not a
                # miss. ``searched_live`` stays False and nothing is negative-cached.
                continue
            asked = (name, str(auth.NAMESPACE))
            if answers is not None and not dry:
                # The replay under the writer lock asks no authority.
                if asked not in answers.asked:
                    continue
                res = answers.asked[asked]
                if res is not None and res.existing is not None:
                    # Read again: another writer may have merged it away since.
                    current = await get_subject(session, res.existing.id)
                    if current is None:
                        continue
                    res = AuthorityResolution(existing=current)
            elif answers is not None and asked in answers.asked:
                # A prefetch shared by several claims asks once per name, as the
                # write loop did when the first claim's mint answered the rest.
                res = answers.asked[asked]
            elif answers is not None:
                # The prefetch writes nothing: an answer the authority records
                # in the probe ledger is kept for the apply (AD6).
                with deferring_answers(answers.ledger):
                    res = await auth.resolve(
                        session, name, particle_content=particle_content, domain=domain
                    )
                answers.asked[asked] = res
            else:
                res = await auth.resolve(
                    session, name, particle_content=particle_content, domain=domain
                )
            searched_live = True
            if res is None:
                continue
            all_missed = False
            if res.existing is not None:
                still_provisional = await by_ref(res.existing)
                if dry:
                    return res.existing
                attached = await _attach_alias(session, res.existing, name)
                subject_cache.cache_set(key_for(still_provisional), attached)
                return attached
            if res.abstained:
                # The authority judged every candidate wrong for this claim.
                # Treated as an abstention: no ref, and no negative
                # cache entry, since ``all_missed`` is already False.
                continue
            if res.unreachable:
                # The authority could not be asked. Not a miss: no
                # negative cache entry, since ``all_missed`` is already False,
                # so the next resolution of this name asks again.
                continue
            # Build a new Subject from the authority's resolution.
            assert res.external_ref is not None  # resolve() contract: existing xor new
            # resolver abstention. An external-authority candidate scored
            # below the floor is treated exactly like a declined resolution — skip
            # the attach and `continue` (try the next authority; else Step 4
            # bare-local). The floor sits strictly below the 0.5 "unscoreable"
            # sentinel (the `_wikidata_link_confidence` floor), so a link that
            # could not be scored still attaches; only scored-and-low links (the
            # plausible-but-wrong mislinks, e.g. `Particles` → "2015 studio album")
            # are dropped. The check governs every LIVE authority via the neutral
            # confidence field; exact-identifier authorities resolve at 1.0.
            abstain_floor = get_config().subjects.external_link_abstain_threshold
            if res.external_ref.confidence < abstain_floor:
                log.info(
                    "Abstaining from external link %s:%s (confidence %.2f < %.2f) for "
                    "%r; falling back to bare-local",
                    res.external_ref.namespace,
                    res.external_ref.id,
                    res.external_ref.confidence,
                    abstain_floor,
                    name,
                )
                continue
            resolved_name = res.canonical_name or name
            # Dedup guard: the Step-1 find_by_name ran against the *raw* extracted
            # name, but the authority can rewrite canonical_name (e.g. "the society
            # of mind" → "Society of Mind"). A prior run may already hold that
            # subject — under its external ref, or under the rewritten name reached
            # via a different surface form. Re-check before inserting so we attach
            # instead of creating a duplicate that shares a canonical_name.
            dup = await find_by_external_ref(
                session, res.external_ref.namespace, res.external_ref.id
            )
            dup_provisional = False
            if dup is not None:
                dup_provisional = await by_ref(dup)
            elif resolved_name.lower() != name.lower():
                local_dup = await _find_local(
                    session,
                    resolved_name,
                    source_type,
                    source_tags,
                    principal=principal,
                    kind=kind,
                )
                if local_dup is not None:
                    dup = local_dup.subject
                    dup_provisional = local_dup.provisional
                    if local_dup.share and not dry:
                        assert principal is not None
                        await share(session, dup.id, principal)
            if dry:
                return dup
            if dup is not None:
                attached = await _attach_alias(session, dup, name)
                subject_cache.cache_set(key_for(dup_provisional), attached)
                log.debug("Subject deduped on insert: %r → %s", name, attached.id)
                return attached
            subject = Subject(
                canonical_name=resolved_name,
                description=res.description,
                aliases=res.aliases,
                external_ids=[res.external_ref],
                created_at=datetime.now(UTC),
                asserted_by=asserted_by,
            )
            await insert_subject(session, subject)
            subject = await _after_mint(session, subject, principal=principal, kind=kind)
            subject_cache.cache_set(key_for(minting), subject)
            log.info(
                "Created subject from %s: %r (%s) confidence=%.2f",
                res.external_ref.namespace,
                subject.canonical_name,
                res.external_ref.id,
                res.external_ref.confidence,
            )
            return subject
        # Every applicable live authority we actually ran found nothing. Record a
        # process-global negative so the next resolution of this name — in this
        # or any concurrent store — skips the fruitless live call until the TTL
        # expires. Guard on ``searched_live`` so a domain that gates every live
        # authority out (``is_applicable`` False) is never mistaken for a real
        # miss, and skip recording after an abstention (``all_missed`` False),
        # which is content-dependent and may link on a different particle, or
        # after an unreachable authority, which answered nothing.
        if searched_live and all_missed and not dry:
            subject_cache.negative_set(name)

    if dry:
        return None

    # Step 4: bare local Subject (possibly with the recognised ref).
    # Ask the recognising authority for a better canonical name (e.g. the
    # Wikidata label cache) before falling back to the raw extracted name.
    canonical_name: str = name
    if recognized_ref is not None and recognizing_auth is not None:
        better = await recognizing_auth.canonical_name_for(session, recognized_ref.id)
        if better:
            canonical_name = better

    subject = Subject(
        canonical_name=canonical_name,
        external_ids=[recognized_ref] if recognized_ref else [],
        created_at=datetime.now(UTC),
        asserted_by=asserted_by,
    )
    await insert_subject(session, subject)
    subject = await _after_mint(session, subject, principal=principal, kind=kind)
    subject_cache.cache_set(key_for(minting), subject)
    log.info("Created bare local subject: %r", canonical_name)
    return subject


async def _after_mint(
    session: AsyncSession,
    subject: Subject,
    *,
    principal: str | None,
    kind: ResolutionKind,
) -> Subject:
    """Mark a minting resolution's mint provisional, or offer a confirmed one to the fold."""
    if mints_provisional(kind, principal):
        assert principal is not None
        await mark_provisional(session, subject.id, principal)
        return subject
    return await fold_or_hold(session, subject)


def _recognize_qualified(
    name: str, context: RecognizeContext
) -> tuple[ExternalRef, ContextualRecognizer] | None:
    """The first contextual authority's recognition of a qualified name."""
    for auth in get_authorities():
        if not isinstance(auth, ContextualRecognizer):
            continue
        ref = auth.recognize_in(name, context)
        if ref is not None:
            return ref, auth
    return None


async def find_existing_qualified(
    session: AsyncSession, name: str, context: RecognizeContext
) -> Subject | None:
    """The existing Subject a qualified name resolves to, read-only.

    The qualified twin of :func:`find_existing_subject`: it matches by the
    project-scoped external ref alone, never by the bare name, so a qualified
    ``config.py`` never binds to an unscoped Subject of the same name.
    """
    recognized = _recognize_qualified(name, context)
    if recognized is None:
        return None
    ref, _ = recognized
    return await find_by_external_ref(session, ref.namespace, ref.id)


async def resolve_qualified_subject(
    session: AsyncSession,
    name: str,
    context: RecognizeContext,
    asserted_by: str = "general-extractor",
    *,
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> Subject | None:
    """Resolve a name the gate qualified to its project-scoped Subject.

    The cascade for a qualified name is the recognize pass and nothing else:
    there is no bare-name local match (an unscoped Subject of the same name is
    a different identity), no live authority, and no bare-local fallback. A
    recognized ref finds its Subject or mints one carrying the ref and the
    authority's ``subject_class``. Returns ``None`` when no contextual
    authority recognizes the name, which the caller treats as suppression.
    A match by ref and a mint follow the ordinary cascade's provisional rules.
    """
    recognized = _recognize_qualified(name, context)
    if recognized is None:
        return None
    ref, auth = recognized
    existing = await find_by_external_ref(session, ref.namespace, ref.id)
    if existing is not None:
        await _reached_by_ref(
            session,
            existing,
            principal=principal,
            kind=kind,
            dry=False,
            actor=asserted_by,
            source_type=None,
        )
        return existing
    subject = Subject(
        canonical_name=name.strip(),
        external_ids=[ref],
        subject_class=auth.subject_class_for(context),
        created_at=datetime.now(UTC),
        asserted_by=asserted_by,
    )
    await insert_subject(session, subject)
    subject = await _after_mint(session, subject, principal=principal, kind=kind)
    log.info("Created qualified subject: %r (%s:%s)", subject.canonical_name, ref.namespace, ref.id)
    return subject


async def resolve_subjects(
    session: AsyncSession,
    names: list[str],
    asserted_by: str = "general-extractor",
    particle_content: str | None = None,
    *,
    source_type: str | None = None,
    source_tags: Collection[str] = (),
    qualified: Mapping[str, RecognizeContext] | None = None,
    answers: LiveAnswers | None = None,
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> list[str]:
    """Resolve a list of subject names and return their UUIDs.

    ``principal`` and ``kind`` are :func:`resolve_subject`'s.

    A name in ``qualified`` goes through
    :func:`resolve_qualified_subject` instead of the ordinary cascade. The
    result stays positionally aligned with ``names`` (the pipeline zips the
    two), so a qualified name no contextual authority recognizes is skipped
    only in the case the gate never produces: one with nothing left after
    normalization, the same case a blank name already is.

    ``answers`` (from :func:`prefetch_subjects`) replaces the live lookups, so
    a caller holding the store's writer lock makes no network or model call.
    """
    ids: list[str] = []
    for name in names:
        if not name.strip():
            continue
        context = (qualified or {}).get(name)
        if context is not None:
            found = await resolve_qualified_subject(
                session, name, context, asserted_by, principal=principal, kind=kind
            )
            if found is not None:
                ids.append(found.id)
            continue
        subject = await resolve_subject(
            session,
            name.strip(),
            asserted_by,
            particle_content=particle_content,
            source_type=source_type,
            source_tags=source_tags,
            answers=answers,
            principal=principal,
            kind=kind,
        )
        ids.append(subject.id)
    return ids


async def prefetch_subjects(
    session: AsyncSession,
    names: list[str],
    particle_content: str | None = None,
    *,
    source_type: str | None = None,
    source_tags: Collection[str] = (),
    answers: LiveAnswers | None = None,
    principal: str | None = None,
    kind: ResolutionKind = ResolutionKind.NON_MINTING,
) -> SubjectPrefetch:
    """Ask the live authorities for ``names`` without writing (D2).

    ``principal`` and ``kind`` are the write's, so the names it reports as
    known are the ones the write would resolve.

    The gather half of :func:`resolve_subjects` for a write path that applies
    under the store's writer lock: it runs the same cascade read-only, so every
    network lookup and judge call happens here, before the lock, and an answer
    an authority records in the probe ledger is kept in ``answers.ledger``
    rather than written. Pass the result's ``answers`` to
    :func:`resolve_subjects` under the lock. A qualified name is not
    prefetched: its recognizers make no live call; the caller leaves it out.
    ``answers`` accumulates over several calls (the extraction write phase
    prefetches every claim into one), and a name already asked is not asked
    again.
    """
    answers = LiveAnswers() if answers is None else answers
    known: list[str] = []
    for name in names:
        if not name.strip():
            continue
        persona = _persona_fold(name, source_type, source_tags)
        folded = name.strip() if persona is None else persona
        found = await _resolve(
            session,
            folded,
            "",
            particle_content,
            source_type,
            source_tags,
            live=persona is None,
            answers=answers,
            dry=True,
            principal=principal,
            kind=kind,
        )
        if found is not None:
            known.append(found.id)
    return SubjectPrefetch(answers=answers, known_ids=list(dict.fromkeys(known)))


@dataclass(frozen=True)
class SplitTargetGather:
    """What a split's target identity needs from outside the store, asked first.

    ``subjects`` holds a ``new_name``'s live authority answers; ``aliases``
    holds the Wikidata labels of a ``wikidata:`` id the store does not hold
    yet. :func:`split_subject_resolving` replays them, so a caller holding the
    store's writer lock makes no network or model call (security review
    2026-10-31-1).
    """

    subjects: SubjectPrefetch | None = None
    aliases: list[str] | None = None


async def gather_split_target(
    session: AsyncSession, *, new_name: str | None, new_external_id: str | None
) -> SplitTargetGather:
    """Ask the network for :func:`split_subject_resolving`'s target, writing nothing.

    Refuses as the split does: a malformed ``new_external_id``, and a
    ``wikidata:`` id ``network.egress`` does not let the labels be fetched for.
    """
    # Deferred: the Wikidata alias helper is authority-internal and only the
    # by-QID branch needs it; importing it lazily also keeps the by-name and
    # by-external-id paths free of the authority module's import cost.
    from particles.ingest.authorities.wikidata import _wikidata_aliases

    if new_external_id is not None:
        if ":" not in new_external_id:
            raise ValueError("new_external_id must be NAMESPACE:ID (e.g. wikidata:Q30297735)")
        ns, ext_id = new_external_id.split(":", 1)
        if ns != "wikidata" or await find_by_external_ref(session, ns, ext_id) is not None:
            return SplitTargetGather()
        # A refusal here would otherwise mint a Subject named by its raw QID
        # (the alias helper returns [] on any failure); refuse instead.
        egress.check(*egress.WIKIDATA_HOST, "subjects split (wikidata labels)")
        return SplitTargetGather(aliases=await _wikidata_aliases(ext_id))
    if new_name is not None:
        return SplitTargetGather(
            subjects=await prefetch_subjects(session, [new_name], kind=ResolutionKind.OPERATOR)
        )
    return SplitTargetGather()


async def split_subject_resolving(
    session: AsyncSession,
    *,
    source_id: str,
    particle_ids: list[str],
    new_name: str | None,
    new_external_id: str | None,
    asserted_by: str = "subjects-split",
    gathered: SplitTargetGather | None = None,
) -> tuple[Subject, list[str], list[str]]:
    """Resolve / construct the split-target Subject, then re-link particles.

    The orchestration shared by the ``subjects split`` CLI verb and the
    ``POST /subjects/{id}/split`` endpoint. Supply the target identity
    via ``new_external_id`` (authoritative — metadata pulled directly from the
    identifier) or ``new_name`` (canonicalised via :func:`resolve_subject`).
    Does **not** manage the transaction: the caller commits, or rolls back for a
    dry run.

    ``gathered`` is :func:`gather_split_target`'s answer for the same identity.
    With it the split asks no live authority and fetches nothing, so it may run
    under the store's writer lock; without it the split gathers first itself.

    Returns:
        ``(new_subject, relinked_pids, not_bound_pids)``.

    Raises:
        ValueError: ``new_external_id`` is malformed, neither identity field was
            given, or the resolver returned the source Subject (nothing to split).
    """
    if gathered is None:
        gathered = await gather_split_target(
            session, new_name=new_name, new_external_id=new_external_id
        )

    new_subject: Subject
    if new_external_id is not None:
        if ":" not in new_external_id:
            raise ValueError("new_external_id must be NAMESPACE:ID (e.g. wikidata:Q30297735)")
        ns, ext_id = new_external_id.split(":", 1)
        existing = await find_by_external_ref(session, ns, ext_id)
        if existing is not None:
            # The operator named this referent: a provisional Subject holding
            # it is confirmed.
            await _reached_by_ref(
                session,
                existing,
                principal=None,
                kind=ResolutionKind.OPERATOR,
                dry=False,
                actor=asserted_by,
                source_type=None,
            )
            new_subject = existing
        elif ns == "wikidata":
            # Labels + aliases fetched by QID in the gather.
            aliases = list(gathered.aliases or [])
            canonical = aliases[0] if aliases else ext_id
            new_subject = Subject(
                canonical_name=canonical,
                aliases=aliases[1:],
                external_ids=[
                    ExternalRef(
                        namespace="wikidata",
                        id=ext_id,
                        uri=f"https://www.wikidata.org/wiki/{ext_id}",
                        confidence=1.0,
                    )
                ],
                asserted_by=asserted_by,
            )
            await insert_subject(session, new_subject)
        else:
            # Non-Wikidata namespace not already in the store — construct a bare
            # Subject with the operator-supplied external ID. ``new_name`` (when
            # given) is the canonical name; otherwise the external ID stands in
            # so the Subject is at least addressable.
            canonical = new_name or new_external_id
            new_subject = Subject(
                canonical_name=canonical,
                external_ids=[ExternalRef(namespace=ns, id=ext_id, confidence=1.0)],
                asserted_by=asserted_by,
            )
            await insert_subject(session, new_subject)
    elif new_name is not None:
        new_subject = await resolve_subject(
            session,
            # Stripped as the gather's lookup was, so its answers replay.
            new_name.strip(),
            asserted_by=asserted_by,
            answers=gathered.subjects.answers if gathered.subjects is not None else None,
            kind=ResolutionKind.OPERATOR,
        )
    else:
        raise ValueError("provide new_name or new_external_id")

    if new_subject.id == source_id:
        raise ValueError(
            "resolver returned the source Subject — nothing to split "
            "(did you mean to merge, or supply a different new_name?)"
        )

    relinked, not_bound = await split_subject(
        session,
        source_id=source_id,
        new_subject_id=new_subject.id,
        particle_ids=particle_ids,
    )
    return new_subject, relinked, not_bound
