"""Intra-pass exact-content candidate dedup for one extraction pass.

Repetitive or multi-section sources (Wikipedia timeline/response sections, FAQ
pages, changelogs) restate the *same sentence* verbatim in more than one place.
The chunked extraction path sends each section to the LLM
as a separate call and appends the per-chunk candidates, so a sentence repeated
across sections mints one ACTIVE particle per occurrence — verbatim-duplicate
beliefs that all trace to the same snapshot, asserted seconds apart. The
2026-07-11 audit dogfood confirmed this on one Wikipedia snapshot: 105 ACTIVE
particles for only 93 distinct content strings (12 exact-content duplicates).

:func:`dedupe_exact_candidates` folds those away. Within the *accumulated output
of one pass* it collapses candidates that share an identical (normalized)
content string *and* the same subject-name binding down to their first
occurrence — reasoning over accumulated per-pass state, so it is
**Engine** and lives here in ``ingest/``, adjacent to
:mod:`particles.ingest.narrative_merge` (the narrative fold it runs
beside). It mirrors that fold's shape: run *before* embedding / §6.6 / the write
loop, reassign the positional references the pipeline consumes, and be a **no-op
when nothing duplicates** so behaviour is unchanged on non-repetitive sources.

Scope guardrails (this is deliberately narrow):

* **Intra-pass, intra-snapshot, exact-content only.** Cross-*source* corroboration
  of the same sentence (co-evidence) happens across *separate* passes,
  each with its own candidate list — it never reaches this function, so it is
  preserved and merged later by the store-side machinery, not here.
* **Not carry-forward.** Carry-forward re-uses an *existing* ACTIVE
  particle across chunks via ``carry_forward_ids`` — those IDs are not in
  ``candidates`` at all, so an existing particle re-emitted by carry-forward is
  untouched. This fold only collapses two *new* candidates minted in one pass.
* **Exact content, conservative normalization.** Whitespace is collapsed and
  trailing punctuation trimmed; case and wording are left alone. Near-duplicates
  and paraphrases are *not* folded here — that is the §6.6 / co-evidential job
  , a separate concern.
* **Plain claims only.** A candidate carrying a structural role — a stance
  , a NARRATIVE container or constituent — is never dropped,
  so the positional edge-writers downstream keep every endpoint they expect. A
  stance that *targeted* a dropped duplicate is repointed to the surviving twin.
"""

from __future__ import annotations

import logging

from particles.core.duplicate_key import normalize_content as _normalize_content
from particles.core.schema import ParticleType
from particles.extraction.general import CandidateParticle

log = logging.getLogger(__name__)

# The normalization key moved to particles.core.duplicate_key so this
# intra-pass fold and the cross-pass suppression rung share one definition of
# "the same content" — two strings that dedupe within a pass must also dedupe
# across passes. Re-exported under the original private name so this module's
# call sites and tests are unchanged.
__all__ = ["dedupe_exact_candidates"]


def _is_dedupable(candidate: CandidateParticle) -> bool:
    """A plain factual claim carrying no positional structural role.

    Stances and NARRATIVE containers / constituents are
    referenced by position by the pipeline's edge-writers; never drop one, or an
    edge loses an endpoint. A verbatim-repeated sentence in a document is always a
    plain CLAIM, so this exclusion costs no real dedup coverage.
    """
    return (
        candidate.particle_type == ParticleType.CLAIM
        and candidate.stance_kind is None
        and candidate.narrative_index is None
    )


def dedupe_exact_candidates(
    candidates: list[CandidateParticle],
) -> tuple[list[CandidateParticle], list[str]]:
    """Collapse intra-pass exact-content duplicate candidates.

    Returns ``(candidates, notes)``. Two dedupable candidates (see
    :func:`_is_dedupable`) collapse to their first occurrence when they share
    both a normalized content string (:func:`_normalize_content`) and an
    identical subject-name set. The kept candidate is the earliest in list
    (= document) order; later exact duplicates are dropped. When nothing
    duplicates, the input list is returned unchanged and ``notes`` is empty — the
    no-op case for every non-repetitive source.

    Positional references the pipeline consumes are kept consistent: a surviving
    stance whose ``stance_target_index`` pointed at a dropped duplicate is
    repointed to the surviving twin, and every ``stance_target_index`` is
    rebased to the compacted list positions.
    """
    # First pass: decide which candidates to drop. ``key -> kept original index``
    # over dedupable candidates only; ``dropped index -> kept twin's index``.
    seen: dict[tuple[str, frozenset[str]], int] = {}
    drop: dict[int, int] = {}
    for i, c in enumerate(candidates):
        if not _is_dedupable(c):
            continue
        key = (_normalize_content(c.content), frozenset(c.subjects))
        twin = seen.get(key)
        if twin is None:
            seen[key] = i
        else:
            drop[i] = twin

    if not drop:
        return candidates, []

    # Build the compacted list and an old -> new position map. Dropped positions
    # resolve to their surviving twin's new position so a stance that targeted a
    # duplicate follows the merge.
    new_pos: dict[int, int] = {}
    survivors: list[CandidateParticle] = []
    for i, c in enumerate(candidates):
        if i in drop:
            continue
        new_pos[i] = len(survivors)
        survivors.append(c)
    for dropped_i, twin_i in drop.items():
        new_pos[dropped_i] = new_pos[twin_i]

    # Rebase stance targets onto the compacted positions. Guard on ``in new_pos``:
    # a multi-chunk per-chunk-relative index (already latently out of range before
    # this fold) is left untouched rather than mis-remapped.
    for c in survivors:
        if c.stance_target_index is not None and c.stance_target_index in new_pos:
            c.stance_target_index = new_pos[c.stance_target_index]

    notes = [
        f"INTRA_PASS_DEDUP: collapsed {len(drop)} exact-content duplicate "
        f"candidate(s) into their first occurrence"
    ]
    return survivors, notes
