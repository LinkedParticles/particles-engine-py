"""synthesis splice for Obsidian per-subject notes.

The wiki exporter writes synthesised articles as their own files; the
Obsidian exporter splices the same synthesised body *into* the existing
per-subject note above the structural particle listing. The splice
machinery lives here, separate from :mod:`particles.exporters.obsidian.vault`,
so the vault renderer doesn't have to know about the LLM call seam.

State transitions returned by :func:`_splice_synthesised_article` are
documented in its docstring and fan out to the summary counters in
:func:`particles.exporters.obsidian.vault.export_vault`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from particles.core.schema import LintFinding, Particle, Subject
from particles.render.markdown import SubjectNaming

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
from particles.exporters.obsidian.format import (
    _annotate_obsidian_frontmatter,
    _insert_synthesised_prose,
    _note_has_particle_audit_trail,
    _render_particle_audit_callouts,
    _strip_per_particle_callouts,
    _to_obsidian_block_refs,
)
from particles.render.article_synthesis import (
    render_article,
    split_rendered_article,
)

log = logging.getLogger(__name__)


def _should_skip_synthesis_for_low_coverage(particle_count: int) -> bool:
    """Whether synthesis should be skipped for a low-coverage subject.

    Threshold lives in ``exporter_common.synthesis_min_particles``
    (default 3) — hoisted from ``obsidian.synthesis_min_particles`` in
    0.42.1 so the Logseq exporter shares the same gate. The
    ``OBSIDIAN_SYNTHESIS_MIN_PARTICLES`` env var still works for
    backwards compatibility.

    Pulled into its own helper so it can be unit-tested independently
    of the splice machinery — vault-level tests would have to mock the
    LLM seam, which tests/AGENTS.md explicitly carves out.
    """
    from particles.config import get_config

    return particle_count < get_config().exporter_common.synthesis_min_particles


async def _splice_synthesised_article(
    *,
    note: str,
    subject: Subject,
    particles: list[Particle],
    subject_map: dict[str, Subject],
    eligible_ids: set[str],
    eff_conf: dict[str, float],
    entry_uri_map: dict[str, str | None],
    article_hash: str,
    regenerate: bool,
    lint_findings: list[LintFinding],
    progress_prefix: str,
    session: AsyncSession,
    prior_body: str | None = None,
    naming: SubjectNaming | None = None,
) -> tuple[str, str]:
    """Splice an LLM-synthesised article body into an Obsidian note.

    Returns ``(updated_note, state)`` where ``state`` is:
      * ``"skipped"`` — particle count is below
        ``obsidian.synthesis_min_particles``; synthesis is skipped to
        avoid burning an LLM call on a subject the synthesis would only
        paraphrase. The note is written unchanged with
        ``article_synthesis: skipped-low-coverage`` in the frontmatter.
      * ``"cache_hit"`` — input_hash unchanged since prior run; the
        existing note is reused as-is and the article frontmatter
        carries the prior hash forward for next time
      * ``"synthesised"`` — the LLM produced a body that passed
        validation; it's spliced between the Obsidian H1 and the
        existing structural content, with References appended
      * ``"fallback"`` — synthesis fell back to the structured listing;
        the existing Obsidian note is written unchanged but the
        frontmatter records ``article_synthesis: structured-listing``
        so next-run cache check still works
      * ``"no_op"`` — synthesis was skipped because no LLM-side cache
        invalidation strategy is configured; written unchanged
    """
    from particles.config import get_config

    # Low-coverage short-circuit. Runs *before* the cache-hit check so
    # operators can lower the threshold and have the next export drop
    # synthesis on already-cached low-particle subjects.
    if _should_skip_synthesis_for_low_coverage(len(particles)):
        threshold = get_config().exporter_common.synthesis_min_particles
        log.info(
            "%s — skipping synthesis (%d particle%s, threshold %d)",
            progress_prefix,
            len(particles),
            "" if len(particles) == 1 else "s",
            threshold,
        )
        # If the renderer (typically _render_coin_note, which routes
        # structured particle data into frontmatter) left the body
        # without any per-particle visibility, append the standard
        # audit-trail callouts so the operator still sees the claim,
        # source URL, and related-subject wikilinks in the body. For
        # generic / pivot notes this is a no-op — they already emit
        # the audit trail at render time.
        if not _note_has_particle_audit_trail(note):
            audit_callouts = _render_particle_audit_callouts(
                particles,
                parent_subject=subject,
                subject_map=subject_map,
                eligible_ids=eligible_ids,
                eff_conf=eff_conf,
                entry_uri_map=entry_uri_map,
                naming=naming,
            )
            if audit_callouts:
                note = note.rstrip() + "\n\n---\n\n" + "\n".join(audit_callouts) + "\n"
        return _annotate_obsidian_frontmatter(
            note,
            article_input_hash=article_hash,
            article_synthesis="skipped-low-coverage",
        ), "skipped"

    if not regenerate:
        log.info("%s — article cache hit", progress_prefix)
        # backfill: when the on-disk hash matches but the
        # cross-exporter DB cache has no row for this key, populate
        # the cache from the prior note body. This handles the
        # operator who had Obsidian export artefacts on disk pre-
        # 0.41.0 — the synthesised article exists in the file but
        # never made it into the DB. Without this backfill, the first
        # cross-exporter use (wiki or Logseq) re-pays full LLM cost
        # for every subject Obsidian already synthesised.
        await _backfill_synthesis_cache_if_absent(
            session,
            subject_id=subject.id,
            input_hash=article_hash,
            prior_body=prior_body,
        )
        # Preserve the synthesised prose on a cache hit. The freshly-rendered
        # ``note`` is the *structural* template only — it has no synthesised
        # article body. The prior on-disk note (``prior_body``) does, and
        # because the input_hash matched, its structural content is current
        # too. Reuse it (the wiki exporter's cache-hit path likewise reads the
        # existing file) so the article survives a no-change re-export. The
        # caller refreshes the lint callouts on top at write time.
        # ``prior_body`` is always present on a cache hit — ``prior_hash`` was
        # read from it — but guard defensively.
        if prior_body is not None:
            return prior_body, "cache_hit"
        return _annotate_obsidian_frontmatter(
            note, article_input_hash=article_hash, article_synthesis=None
        ), "cache_hit"

    cfg = get_config().wiki
    log.info("%s — synthesising article (%d particles)…", progress_prefix, len(particles))

    body, used_synthesis = await render_article(
        subject=subject,
        particles=particles,
        eff=eff_conf,
        input_hash=article_hash,
        corpus_uris=entry_uri_map,
        max_tokens=cfg.max_tokens,
        layer_b_enabled=cfg.layer_b_enabled,
        lint_findings=lint_findings or None,
        # thread the session so render_article can
        # consult / populate the cross-exporter synthesis cache.
        session=session,
    )

    if not used_synthesis:
        log.info("%s — article fell back to structured-listing", progress_prefix)
        # We don't splice the structured-listing into the Obsidian note
        # because the Obsidian note already lists every particle in its
        # own format; doubling them up would be noise. Just record the
        # fallback in the frontmatter so next-run cache works.
        return _annotate_obsidian_frontmatter(
            note,
            article_input_hash=article_hash,
            article_synthesis="structured-listing",
        ), "fallback"

    log.info("%s — wrote article (LLM-synthesised)", progress_prefix)
    _fm, _h1, prose, references = split_rendered_article(body)

    # Drop the per-particle audit-trail callouts from the existing
    # structural note — they duplicate the synthesised References block
    # we're about to splice in. The fallback / cache-hit branches above
    # leave them in place, so notes without a synthesised References
    # section still carry the audit trail.
    note = _strip_per_particle_callouts(note)

    # Convert the portable Markdown footnote syntax that article_synthesis.py
    # emits into Obsidian-native block references, and re-render the
    # References section from the underlying particle objects so each
    # entry can show the claim, related-subject wikilinks, source URL,
    # and provenance metadata. Obsidian's own footnote parser does not
    # reliably make ``[^id]`` references clickable in Live Preview,
    # even with the GFM-compatible ``p-xxxxxxxx`` ID format; block refs
    # (``[[#^id]]`` + trailing ``^id`` marker) are the native in-note
    # jump mechanism and always work. The ``references`` value returned
    # by :func:`split_rendered_article` is discarded — the new render
    # comes from ``particles`` directly.
    prose, references = _to_obsidian_block_refs(
        prose,
        particles=particles,
        parent_subject=subject,
        subject_map=subject_map,
        eligible_ids=eligible_ids,
        eff_conf=eff_conf,
        entry_uri_map=entry_uri_map,
        naming=naming,
    )

    spliced = _insert_synthesised_prose(
        note=note,
        prose=prose,
        references=references,
    )
    return _annotate_obsidian_frontmatter(
        spliced, article_input_hash=article_hash, article_synthesis="llm"
    ), "synthesised"


# ---------------------------------------------------------------------------
# backfill — populate the shared DB synthesis cache from on-disk
# Obsidian notes when the exporter's per-note hash short-circuit fires.
# ---------------------------------------------------------------------------


async def _backfill_synthesis_cache_if_absent(
    session: AsyncSession,
    *,
    subject_id: str,
    input_hash: str,
    prior_body: str | None,
) -> None:
    """If the DB cache has no row for ``(subject_id, input_hash, prompt_version)``
    AND we have the prior note body on disk, store it.

    The case this catches: an operator whose Obsidian vault was
    synthesised before 0.41.0 (when the DB cache was introduced).
    The on-disk article exists with a matching frontmatter hash, so
    Obsidian's per-note shortcut fires and ``render_article`` is never
    called — which means the DB cache never gets populated for that
    subject through Obsidian's normal write path. Without this backfill,
    the next cross-exporter run (wiki, Logseq) sees an empty DB cache
    and re-pays the full LLM cost.

    The body stored is the FULL note text from the on-disk file, which
    is what the cross-exporter consumers expect to receive from
    ``lookup_cached_article``. The cache key uses the same
    ``_PROMPT_VERSION`` constant as live renders so a prompt-version
    bump invalidates the backfill correctly.
    """
    if prior_body is None:
        return
    from particles.render.article_synthesis.cache import _PROMPT_VERSION
    from particles.store.synthesis_cache_store import (
        lookup_cached_article,
        store_cached_article,
    )

    existing = await lookup_cached_article(session, subject_id, input_hash, _PROMPT_VERSION)
    if existing is not None:
        return
    await store_cached_article(
        session,
        subject_id,
        input_hash,
        _PROMPT_VERSION,
        prior_body,
        quality_notes="backfilled-from-obsidian-vault",
    )
    log.debug(
        "backfill: stored on-disk synthesis body for subject_id=%s (input_hash=%s)",
        subject_id,
        input_hash[:8],
    )
