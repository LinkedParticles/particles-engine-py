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
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExternalRef, Subject
from particles.extraction.registry import infer_domain
from particles.ingest.authorities import (
    SubjectAuthority,
    get_authorities,
    is_applicable,
)
from particles.store import subject_cache
from particles.store.subject_cache import CacheEntry
from particles.store.subject_store import (
    add_aliases,
    find_by_external_ref,
    find_by_name,
    insert_subject,
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


def _persona_canonical(name: str, source_type: str | None) -> str:
    """Fold a conversational source's persona aliases onto one name.

    "user", "the user", "the speaker", "I" are the same person in a chat
    transcript, but each surface form minted its own Subject, which split that
    person's beliefs into groups that never reconciled against each other. The
    fold is a *resolution* rule: it changes which Subject a new claim binds to,
    never an existing binding.
    """
    cfg = get_config().subjects
    if source_type is None or source_type not in cfg.persona_source_types:
        return name
    if name.strip().casefold() in {a.casefold() for a in cfg.persona_aliases}:
        return cfg.persona_canonical_name
    return name


async def _find_local(session: AsyncSession, name: str, source_type: str | None) -> Subject | None:
    """The local name/alias lookup of a path that binds a particle to the result.

    ``name`` is already folded. Outside a persona source type a persona form
    never resolves through the persona Subject's recorded aliases
    (:func:`~particles.store.subject_store.persona_alias_guard`).
    """
    return await find_by_name(session, name, skip_aliases_of=persona_alias_guard(name, source_type))


async def find_existing_subject(
    session: AsyncSession, name: str, *, source_type: str | None
) -> Subject | None:
    """Resolve ``name`` to an *existing* Subject exactly as the write path would, read-only.

    The one seam for a caller that must agree with :func:`resolve_subject`
    about a name before (or without) resolving it — the §6.6 precompute is the
    reference caller. It applies the persona fold and the persona-alias scope,
    and stops where ``resolve_subject`` would start writing: no live authority,
    no mint, no alias recorded. A name resolved by two paths must be folded and
    scoped by both; routing both through here is what keeps that
    true by construction.
    """
    return await _find_local(session, _persona_canonical(name, source_type), source_type)


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
) -> Subject:
    """Resolve a subject name to a canonical Subject, creating one if needed."""
    folded = _persona_canonical(name, source_type)
    subject = await _resolve(session, folded, asserted_by, particle_content, source_type)
    if folded == name:
        return subject
    recorded = await _record_persona_alias(session, subject, name.strip())
    if recorded is not subject:
        # ``add_aliases`` cleared the resolution cache; re-seat this one.
        subject_cache.cache_set(subject_cache.make_key(session, folded), recorded)
    return recorded


# Known deviation: decision logic is interleaved with I/O in this function. Extract it with the
# next substantive change here (D2).
async def _resolve(
    session: AsyncSession,
    name: str,
    asserted_by: str,
    particle_content: str | None,
    source_type: str | None,
) -> Subject:
    """The resolution cascade for an already-folded ``name``."""
    cache_key = subject_cache.make_key(session, name)

    # Check in-memory cache first (avoids repeat DB + API calls for same name)
    cached = subject_cache.cache_get(cache_key)
    if isinstance(cached, CacheEntry) and cached.subject is not None:
        return cached.subject

    # Step 1: local alias index
    existing = await _find_local(session, name, source_type)
    if existing:
        subject_cache.cache_set(cache_key, existing)
        log.debug("Subject resolved locally: %r → %s", name, existing.id)
        return existing

    authorities = get_authorities()  # priority-ordered

    # Step 2: recognize pass — in-name patterns, before any live lookup.
    # First (highest-priority) authority that recognises the name wins.
    recognized_ref: ExternalRef | None = None
    recognizing_auth: SubjectAuthority | None = None
    for auth in authorities:
        ref = auth.recognize(name)
        if ref is None:
            continue
        by_ref = await find_by_external_ref(session, ref.namespace, ref.id)
        if by_ref:
            subject_cache.cache_set(cache_key, by_ref)
            return by_ref
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
    skip_live = _skip_live_authorities(source_type)
    if not skip_live and subject_cache.negative_get(name):
        skip_live = True
    if not skip_live:
        searched_live = False
        all_missed = True
        for auth in authorities:
            if not auth.LIVE or not is_applicable(auth, domain):
                continue
            searched_live = True
            res = await auth.resolve(
                session, name, particle_content=particle_content, domain=domain
            )
            if res is None:
                continue
            all_missed = False
            if res.existing is not None:
                attached = await _attach_alias(session, res.existing, name)
                subject_cache.cache_set(cache_key, attached)
                return attached
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
            if dup is None and resolved_name.lower() != name.lower():
                dup = await _find_local(session, resolved_name, source_type)
            if dup is not None:
                attached = await _attach_alias(session, dup, name)
                subject_cache.cache_set(cache_key, attached)
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
            subject_cache.cache_set(cache_key, subject)
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
        # which is content-dependent and may link on a different particle.
        if searched_live and all_missed:
            subject_cache.negative_set(name)

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
    subject_cache.cache_set(cache_key, subject)
    log.info("Created bare local subject: %r", canonical_name)
    return subject


async def resolve_subjects(
    session: AsyncSession,
    names: list[str],
    asserted_by: str = "general-extractor",
    particle_content: str | None = None,
    *,
    source_type: str | None = None,
) -> list[str]:
    """Resolve a list of subject names and return their UUIDs."""
    ids: list[str] = []
    for name in names:
        if not name.strip():
            continue
        subject = await resolve_subject(
            session,
            name.strip(),
            asserted_by,
            particle_content=particle_content,
            source_type=source_type,
        )
        ids.append(subject.id)
    return ids


async def split_subject_resolving(
    session: AsyncSession,
    *,
    source_id: str,
    particle_ids: list[str],
    new_name: str | None,
    new_external_id: str | None,
    asserted_by: str = "subjects-split",
) -> tuple[Subject, list[str], list[str]]:
    """Resolve / construct the split-target Subject, then re-link particles.

    The orchestration shared by the ``subjects split`` CLI verb and the
    ``POST /subjects/{id}/split`` endpoint. Supply the target identity
    via ``new_external_id`` (authoritative — metadata pulled directly from the
    identifier) or ``new_name`` (canonicalised via :func:`resolve_subject`).
    Does **not** manage the transaction: the caller commits, or rolls back for a
    dry run.

    Returns:
        ``(new_subject, relinked_pids, not_bound_pids)``.

    Raises:
        ValueError: ``new_external_id`` is malformed, neither identity field was
            given, or the resolver returned the source Subject (nothing to split).
    """
    # Deferred: the Wikidata alias helper is authority-internal and only the
    # by-QID branch needs it; importing it lazily also keeps the by-name and
    # by-external-id paths free of the authority module's import cost.
    from particles.ingest.authorities.wikidata import _wikidata_aliases

    new_subject: Subject
    if new_external_id is not None:
        if ":" not in new_external_id:
            raise ValueError("new_external_id must be NAMESPACE:ID (e.g. wikidata:Q30297735)")
        ns, ext_id = new_external_id.split(":", 1)
        existing = await find_by_external_ref(session, ns, ext_id)
        if existing is not None:
            new_subject = existing
        elif ns == "wikidata":
            # Fetch labels + aliases directly from Wikidata by QID.
            aliases = await _wikidata_aliases(ext_id)
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
        new_subject = await resolve_subject(session, new_name, asserted_by=asserted_by)
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
