# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Article-synthesis orchestrator.

:func:`render_article` is the single public entry point for "give me a
cited Markdown body for this subject." It chains the four other
submodules:

1. Build the standard synthesis prompt from :mod:`render`.
2. Call the LLM seam from :mod:`layer_b`.
3. Strip inline footnote definitions + run Layer A
   (citation-ID-membership + density) from :mod:`layer_a`.
4. If Layer A passes, run Layer B (semantic-alignment judge) from
   :mod:`layer_b`.
5. On Layer A failure, retry once with the strict prompt variant. On
   Layer B failure, fall back immediately by default (the Layer-B
   strict retry showed a 0% recovery rate; opt back in via
   ``config.wiki.layer_b_retry_enabled``). When attempts are
   exhausted, fall back to the deterministic structured-listing
   render (also from :mod:`render`).

Keeping this orchestration in its own module means the per-layer
modules stay independently testable (and the dependency graph stays
acyclic: orchestrate → {cache, layer_a, layer_b, render}; none of the
underlying modules import each other except :mod:`layer_b` and
:mod:`render` each pulling ``_short_id`` from :mod:`layer_a`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from particles.core.schema import LintFinding, Particle
from particles.render.article_synthesis.cache import (
    _PROMPT_VERSION,
    split_rendered_article,
)
from particles.render.article_synthesis.layer_a import (
    _short_id,
    _strip_inline_footnote_defs,
    count_uncited_paragraphs,
    validate_citations,
)
from particles.render.article_synthesis.layer_b import (
    LayerBResult,
    _call_synthesis_llm,
    layer_b_check,
)
from particles.render.article_synthesis.render import (
    _build_synthesis_prompt,
    render_structured_listing,
    render_synthesised_article,
)
from particles.render.article_synthesis.topic import SynthesisTopic
from particles.render.escape import prose_text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


class SynthesisUnavailable(Exception):
    """The LLM synthesis backend is unusable for the rest of this export run.

    Raised by :func:`render_article` when an LLM call fails with an
    account-level condition that will recur identically for every remaining
    subject — billing exhaustion, authentication failure, or quota denial —
    as opposed to a per-subject hiccup. Exporters catch it once to **abort**
    the synthesis pass (instead of hammering the API for every subject) and to
    skip persisting the article-input hash, so the affected subjects retry on
    the next export rather than being cached as a settled fallback.
    """


# Substrings that mark an LLM error as account-fatal (persistent), not a
# transient per-subject failure. Matched case-insensitively against the
# exception message; status codes 401/403 are also treated as fatal. Rate
# limits (429) are deliberately excluded — they are transient and the
# per-subject structured-listing fallback is an acceptable response.
_FATAL_LLM_MARKERS: tuple[str, ...] = (
    "credit balance",  # Anthropic billing exhaustion (HTTP 400)
    "billing",
    "insufficient_quota",  # quota exhaustion
    "authentication",  # bad / missing key
    "invalid api key",
    "invalid x-api-key",
    "permission",  # 403 permission denied
)


def _is_fatal_llm_error(exc: BaseException) -> bool:
    """Whether an LLM-call exception is fatal for the whole run (vs one subject).

    Fatal means an account/credentials/quota condition that recurs for every
    remaining subject (billing, auth, permission, quota). A transient or
    subject-specific failure returns ``False`` and keeps the existing
    fall-back-to-structured-listing behaviour.
    """
    if getattr(exc, "status_code", None) in (401, 403):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_LLM_MARKERS)


#: The ``quality_notes`` an earlier Obsidian export wrote on a shared-cache row
#: it filled from a vault note rather than from a synthesis.
_VAULT_BACKFILL_NOTE = "backfilled-from-obsidian-vault"


async def _read_back_from_a_vault(session: AsyncSession, subject_id: str, input_hash: str) -> bool:
    """Whether the cached row for this key was read back from an Obsidian vault note."""
    from sqlalchemy import select

    from particles.store.synthesis_cache_store import SynthesisCacheRow

    result = await session.execute(
        select(SynthesisCacheRow.quality_notes).where(
            SynthesisCacheRow.subject_id == subject_id,
            SynthesisCacheRow.input_hash == input_hash,
            SynthesisCacheRow.prompt_version == _PROMPT_VERSION,
        )
    )
    return result.scalar_one_or_none() == _VAULT_BACKFILL_NOTE


def _rebuild_cached_article(
    cached: str,
    *,
    subject: SynthesisTopic,
    particles: list[Particle],
    eff: dict[str, float],
    input_hash: str,
    corpus_uris: dict[str, str | None],
    min_particle_confidence: float | None,
    dropped_below_threshold: int | None,
) -> str | None:
    """A shared-cache body rebuilt around its prose from the engine's own data.

    Only the prose comes from the cache. The frontmatter, the H1 (the
    subject's name) and the References section are rendered again from the
    subject and the particles,
    as a fresh synthesis renders them, so text that reached the cache from
    somewhere other than the engine (a vault note an earlier export read back)
    is never served as the engine's title or citations. The prose is held to
    the fresh path's Layer A rule (every cited id is an input particle, and at
    least one is cited) and to its neutralization. ``None`` when the cached
    body is no synthesis or fails the rule, so the caller synthesises afresh
    (security review 2026-10-28-2, AA30).
    """
    frontmatter, _h1, prose, _references = split_rendered_article(cached)
    if frontmatter.get("synthesis") != "llm":
        return None
    prose = _strip_inline_footnote_defs(prose)
    seen, invalid = validate_citations(prose, {_short_id(p.id) for p in particles})
    if invalid or not seen:
        return None

    def _flag(key: str) -> bool | None:
        value = frontmatter.get(key)
        return value if isinstance(value, bool) else None

    def _count(key: str) -> int | None:
        value = frontmatter.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    # The time the body was synthesised, kept so a hit renders the same bytes;
    # anything that is not a timestamp is replaced by the time of this render.
    generated_at = frontmatter.get("generated_at")
    if isinstance(generated_at, str):
        try:
            generated_at = datetime.fromisoformat(generated_at)
        except ValueError:
            generated_at = None
    if not isinstance(generated_at, datetime):
        generated_at = None

    # The title is the subject's name, as the synthesis prompt asks the model
    # to write it, never the H1 the cached body carried.
    title = f"# {prose_text(subject.canonical_name)}\n\n"
    return render_synthesised_article(
        subject,
        particles,
        body=title + prose,
        cited_short_ids=seen,
        corpus_uris=corpus_uris,
        effective_confidences=eff,
        input_hash=input_hash,
        layer_a_passed=True,
        layer_b_passed=_flag("layer_b_passed"),
        layer_b_unrelated_count=_count("layer_b_unrelated_count"),
        layer_b_contradicts_count=_count("layer_b_contradicts_count"),
        min_particle_confidence=min_particle_confidence,
        dropped_below_threshold=dropped_below_threshold,
        generated_at=generated_at,
    )


async def render_article(
    *,
    subject: SynthesisTopic,
    particles: list[Particle],
    eff: dict[str, float],
    input_hash: str,
    corpus_uris: dict[str, str | None],
    max_tokens: int,
    layer_b_enabled: bool,
    lint_findings: list[LintFinding] | None = None,
    min_particle_confidence: float | None = None,
    dropped_below_threshold: int | None = None,
    session: AsyncSession | None = None,
    without_synthesis: bool = False,
    sequence_mode: bool = False,
    flowing: bool = False,
    direction: str | None = None,
    framing: str | None = None,
) -> tuple[str, bool]:
    """Try LLM synthesis with Layer-A + Layer-B retry; fall back to listing.

    ``subject`` is any :class:`SynthesisTopic` — a real
    :class:`~particles.core.schema.Subject` (the per-subject exporters /
    narrative path) or a :class:`~particles.render.article_synthesis.topic.SectionTopic`
    (documentation projection). Only its ``id`` (cache key) /
    ``canonical_name`` (title) / ``description`` are read here, so the
    Layer-A/B guard rails and the cache are unchanged by the generalization.

    Returns ``(body, used_synthesis)``. ``used_synthesis=True`` means the
    LLM produced a body that passed Layer A and (if enabled) Layer B.

    When ``without_synthesis`` is True (the ``--without-synthesis`` gate),
    the deterministic structured listing is returned
    immediately — no DB-cache lookup and no LLM call — so the export is
    reproducible and needs no API key. ``used_synthesis`` is always False.
    ``False`` means we wrote the deterministic structured-listing
    fallback because:
      * the LLM call raised (no API key, network error, model error), OR
      * the LLM invented a citation ID (or cited nothing) in both the
        standard and strict prompt attempts, OR
      * Layer B failed: at least one ``contradicts`` verdict (hard
        fail), or the ``unrelated`` fraction exceeded
        ``config.wiki.layer_b_unrelated_tolerance``
        (:func:`~particles.render.article_synthesis.layer_b._decide_layer_b_pass`).
        By default a Layer B failure falls back immediately —
        ``config.wiki.layer_b_retry_enabled`` is False; enabling it
        spends the remaining attempt on a Layer-B strict prompt.

    The two-attempt structure (standard, then strict) is the *minimum*
    cost-bounded retry for Layer A failures: an LLM that broke the
    contract once is more likely to do so again, but a temperature of
    zero plus a blunt "you broke the contract" prompt fixes it most of
    the time in practice.

    Layer B is skipped entirely when ``layer_b_enabled`` is False —
    operators trading cost for safety can disable it via
    ``config.wiki.layer_b_enabled`` and the frontmatter records the gap.

    ``sequence_mode``, ``flowing`` / ``direction`` / ``framing``
    steer the first-attempt prompt only — the citation guard rails
    (Layer A/B) and the strict-retry prompts are unchanged. ``flowing`` selects
    the heading-suppressing prose variant; ``direction`` is the per-section
    authoring brief; ``framing`` is the document-level narrative spine prepended
    as shared context. All three default to today's behaviour, so the
    per-subject exporter callers (wiki / obsidian / logseq / narrative) are
    unaffected.

    The optional ``session`` parameter wires the shared synthesis
    cache. When provided, the cache is consulted before
    any LLM call; on hit the cached rendered body is returned with
    ``used_synthesis=True`` (the cache only stores bodies that
    passed Layer A + B at write time). On miss + successful render,
    the rendered body is stored for future cross-exporter reuse.
    Pass ``session=None`` to opt out (no DB roundtrips, no caching).
    """
    # : the deterministic no-LLM gate. Short-circuit before
    # the cache lookup so a prior LLM-synthesised body is not reused either —
    # the operator asked for reproducible, key-free output, not a cache hit.
    if without_synthesis:
        body = render_structured_listing(
            subject,
            particles,
            eff,
            input_hash=input_hash,
            lint_findings=lint_findings,
            min_particle_confidence=min_particle_confidence,
            dropped_below_threshold=dropped_below_threshold,
        )
        return body, False

    from particles.store.synthesis_cache_store import (
        lookup_cached_article,
        store_cached_article,
    )

    # Shared cross-exporter cache. Bail out early on hit:
    # the stored body already passed Layer A + B at write time, so
    # used_synthesis=True is the honest answer.
    if session is not None:
        cached = await lookup_cached_article(session, subject.id, input_hash, _PROMPT_VERSION)
        if cached is not None and await _read_back_from_a_vault(session, subject.id, input_hash):
            # An earlier Obsidian export stored a vault note here. A vault
            # another party can edit holds prose the engine never wrote, so
            # the row is a miss and a fresh render replaces it (security
            # review 2026-10-30-1, AB34).
            cached = None
        if cached is not None:
            log.info(
                "Article synthesis cache hit for %r (input_hash=%s)",
                subject.canonical_name,
                input_hash[:8],
            )
            rebuilt = _rebuild_cached_article(
                cached,
                subject=subject,
                particles=particles,
                eff=eff,
                input_hash=input_hash,
                corpus_uris=corpus_uris,
                min_particle_confidence=min_particle_confidence,
                dropped_below_threshold=dropped_below_threshold,
            )
            if rebuilt is not None:
                return rebuilt, True
            log.warning(
                "Cached article for %r (input_hash=%s) is not an engine-written synthesis; "
                "synthesising afresh",
                subject.canonical_name,
                input_hash[:8],
            )

    allowed = {_short_id(p.id) for p in particles}

    # Per-attempt state: which layer failed on the *prior* attempt, plus
    # the prior Layer B misalignments (when applicable). Escalation:
    # Layer A failure on attempt N → Layer A strict prompt on attempt N+1
    # (existing behaviour). Layer B failure on attempt N → Layer B strict
    # prompt on attempt N+1, with the judge's misaligned pairs shown so
    # the LLM knows what to fix. Mixed failure → Layer A prompt (citation
    # accuracy is the more basic problem; fix it first).
    last_failure: str | None = None  # None | "layer_a" | "layer_b"
    last_layer_b_misalignments: list[dict[str, Any]] = []
    # Track the most recent Layer B verdict counts so a passing-but-not-
    # first attempt can still surface what the judge saw in the
    # frontmatter.
    last_layer_b_result: LayerBResult | None = None

    max_attempts = 2
    for attempt_idx in range(1, max_attempts + 1):
        strict = attempt_idx > 1
        # Pick the strict-prompt variant based on the prior attempt's
        # failure type. On the first attempt these are both no-ops
        # (strict=False so the misalignments arg is ignored).
        system, prompt = _build_synthesis_prompt(
            subject=subject,
            particles=particles,
            eff=eff,
            strict=strict,
            layer_b_misalignments=(
                last_layer_b_misalignments if strict and last_failure == "layer_b" else None
            ),
            sequence_mode=sequence_mode,
            flowing=flowing,
            direction=direction,
            framing=framing,
        )
        temperature = 0.4 if attempt_idx == 1 else 0.0

        try:
            llm_body = await _call_synthesis_llm(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            # An account-fatal error (billing / auth / quota) will recur for
            # every remaining subject — raise so the exporter aborts the whole
            # synthesis pass and does NOT persist this subject's input hash
            # (so it retries next run, rather than caching the fallback).
            if _is_fatal_llm_error(exc):
                raise SynthesisUnavailable(str(exc)) from exc
            log.warning(
                "Article synthesis attempt %d/%d for %r failed at LLM call: %s",
                attempt_idx,
                max_attempts,
                subject.canonical_name,
                exc,
            )
            break  # No point retrying when the seam itself is broken.

        # Strip any inline footnote definitions the LLM emitted in
        # violation of the prompt. The exporter appends the canonical
        # definitions in the References section; duplicates here would
        # silently break footnote links in Obsidian and GFM renderers.
        llm_body = _strip_inline_footnote_defs(llm_body)

        seen, invalid = validate_citations(llm_body, allowed)
        if invalid or not seen:
            # ``not seen`` is the no-citation failure mode: a body that
            # cites nothing has zero invented IDs (Layer A is vacuously
            # silent), but it also has zero citations — which defeats
            # the entire "every claim cited" promise the exporter exists
            # to enforce. Treat as a Layer A failure so retry-then-
            # fallback kicks in and the operator gets a structured
            # listing instead of an uncited prose article.
            if invalid:
                reason = f"invented IDs {sorted(invalid)}"
            else:
                reason = "zero citations (every claim must cite a particle)"
            log.warning(
                "Article synthesis attempt %d/%d for %r failed Layer A: %s",
                attempt_idx,
                max_attempts,
                subject.canonical_name,
                reason,
            )
            last_failure = "layer_a"
            continue  # retry with stricter prompt (if attempts remain)

        # Layer A density check: even when every cited ID is valid, the
        # LLM sometimes pads with unsourced general-knowledge prose. The
        # canonical failure was a CIA-agency article generated from 3
        # film-about-Central-Intelligence particles: only 1 of 7
        # paragraphs cited, the other 6 hallucinated. Reject when
        # over-quota uncited paragraphs exist; retry with strict prompt
        # then fall back to structured-listing.
        uncited_over_quota = count_uncited_paragraphs(llm_body)
        if uncited_over_quota > 0:
            log.warning(
                "Article synthesis attempt %d/%d for %r failed citation density: "
                "%d uncited paragraph(s) over quota (every paragraph after the "
                "intro must cite at least one particle)",
                attempt_idx,
                max_attempts,
                subject.canonical_name,
                uncited_over_quota,
            )
            last_failure = "layer_a"
            continue  # retry with stricter prompt (if attempts remain)

        # Layer A passed. Run Layer B unless disabled.
        layer_b_passed: bool | None = None
        if layer_b_enabled:
            last_layer_b_result = await layer_b_check(llm_body, particles)
            layer_b_passed = last_layer_b_result.passed
            if layer_b_passed is False:
                log.warning(
                    "Article synthesis attempt %d/%d for %r failed Layer B: "
                    "supports=%d unrelated=%d contradicts=%d",
                    attempt_idx,
                    max_attempts,
                    subject.canonical_name,
                    last_layer_b_result.supports_count,
                    last_layer_b_result.unrelated_count,
                    last_layer_b_result.contradicts_count,
                )
                # amendment: operator dry-runs showed the
                # Layer-B-specific retry prompt has a 0% recovery rate
                # and often regresses to zero-citation output. Default
                # to fall-back-immediately on Layer B failure; operators
                # can re-enable the long-shot retry via
                # `wiki.layer_b_retry_enabled`. Layer A retries (invented
                # IDs / zero-citation bodies) remain unconditional.
                from particles.config import get_config

                if not get_config().wiki.layer_b_retry_enabled:
                    break  # fall straight through to structured-listing fallback
                last_failure = "layer_b"
                last_layer_b_misalignments = last_layer_b_result.misalignments
                continue  # retry with stricter prompt (if attempts remain)
        rendered = render_synthesised_article(
            subject,
            particles,
            body=llm_body,
            cited_short_ids=seen,
            corpus_uris=corpus_uris,
            effective_confidences=eff,
            input_hash=input_hash,
            layer_a_passed=True,
            layer_b_passed=layer_b_passed,
            layer_b_unrelated_count=(
                last_layer_b_result.unrelated_count if last_layer_b_result is not None else None
            ),
            layer_b_contradicts_count=(
                last_layer_b_result.contradicts_count if last_layer_b_result is not None else None
            ),
            lint_findings=lint_findings,
            min_particle_confidence=min_particle_confidence,
            dropped_below_threshold=dropped_below_threshold,
        )
        # ``render_synthesised_article`` has neutralized the model's body
        # whole: a claim that steered it must not reach an exported note as
        # live markup (security review 2026-10-24-2, P31; 2026-10-25-1, Q8).
        # Populate the cross-exporter cache so the next exporter
        # asking for the same (subject, input_hash, prompt_version)
        # gets this rendered body without re-running the LLM.
        if session is not None:
            verdict = (
                f"supports={last_layer_b_result.supports_count} "
                f"unrelated={last_layer_b_result.unrelated_count} "
                f"contradicts={last_layer_b_result.contradicts_count}"
                if last_layer_b_result is not None
                else None
            )
            await store_cached_article(
                session,
                subject.id,
                input_hash,
                _PROMPT_VERSION,
                rendered,
                layer_b_verdict=verdict,
            )
        return rendered, True

    # Both attempts failed (or LLM was unavailable). Fall back to the
    # deterministic structured listing so the operator still gets
    # *something* — and so the input_hash gets persisted (preventing a
    # wasteful re-attempt on the next run without --regenerate-all).
    return (
        render_structured_listing(
            subject,
            particles,
            eff,
            input_hash=input_hash,
            lint_findings=lint_findings,
            min_particle_confidence=min_particle_confidence,
            dropped_below_threshold=dropped_below_threshold,
        ),
        False,
    )
