"""Wiki article exporter (refactored).

The wiki exporter produces a flat directory of cited Markdown articles —
one per qualifying Subject — plus a top-level ``index.md``. It is the
demonstrable artefact of the Particles standard's "every claim cited"
promise.

the *rendering* logic (synthesis prompt, Layer A
validation, Layer B judge, retry-then-fallback, footnote formatting,
input-hash cache key) lives in
:mod:`particles.render.article_synthesis` so the Obsidian exporter,
a future Logseq exporter, or any other per-format renderer can call
the same machinery. This module is the *coordinator*: it loads
qualifying subjects from the DB, warms the trust cache, runs the lint
pre-pass, walks subjects with progress logging, and writes the
resulting article bodies to disk.

The public surface that documented (``compute_input_hash``,
``render_structured_listing``, ``validate_citations``, etc.) is
re-exported below so existing imports and tests do not need to chase
the module split.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import LintFinding, Particle, Subject
from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.status import Status
from particles.exporters.summaries import WikiSummary
from particles.extraction.polarity import is_non_asserted
from particles.render.article_synthesis import (
    SynthesisUnavailable,
    _parse_frontmatter,
    apply_lint_callouts,
    compute_input_hash,
    extract_cited_segments,
    layer_b_check,
    render_article,
    render_structured_listing,
    render_synthesised_article,
    validate_citations,
)
from particles.render.article_synthesis import (
    _parse_layer_b_verdicts as _parse_layer_b_verdicts,
)
from particles.render.markdown import (
    SubjectNaming,
    atomic_write_text,
    build_narrative_naming,
    build_subject_naming,
    disambiguation_name,
    is_within_directory,
    narrative_as_subject,
    subject_slug,
)
from particles.store.extractor_store import (
    get_cached_trust_weight,
    get_trust_weight_map,
    populate_trust_cache,
)

# Re-export the public article-synthesis surface so callers that import
# from ``particles.exporters.wiki`` (existing tests, downstream tools)
# keep working after the module split. New code should import
# directly from :mod:`particles.render.article_synthesis`.
__all__ = [
    "WikiExporter",
    "compute_input_hash",
    "extract_cited_segments",
    "layer_b_check",
    "render_article",
    "render_structured_listing",
    "render_synthesised_article",
    "validate_citations",
]

log = logging.getLogger(__name__)


# Subject → vault-path mapping lives in particles.render.markdown so
# every exporter produces identical paths for the same Subject. Local
# alias keeps call sites readable.
_subject_slug = subject_slug


# ---------------------------------------------------------------------------
# DB loading — what particles feed each subject's article
# ---------------------------------------------------------------------------


async def _load_qualifying_subjects(
    session: AsyncSession,
    *,
    min_particles: int,
    subject_filter: set[str] | None,
    min_particle_confidence: float = 0.0,
    include_non_asserted: bool = False,
    trust_map: dict[str, float] | None = None,
) -> tuple[list[tuple[Subject, list[Particle]]], dict[str, int]]:
    """Return ``(qualifying, dropped_by_subject)`` for every qualifying subject.

    ``qualifying`` is ``(subject, sorted_active_particles)`` for each
    subject whose *post-filter* particle count meets ``min_particles``.
    ``dropped_by_subject`` maps ``subject.id`` → drop count so the
    rendered article frontmatter can surface the per-subject bite wiki (the export summary sums this over the
    qualifying set).

    Loads all subjects + all ACTIVE particles + the join table in three
    queries, then groups in memory. Adequate up to corpora of tens of
    thousands of particles; if a future deployment outgrows the
    one-shot load, the natural extension is to stream per-subject — but
    the LLM-call cost will dominate well before the load cost does.

    The ``min_particle_confidence`` filter runs BEFORE the count check
    so a subject with too few *high-confidence* particles is suppressed
    rather than rendered as a thin article from low-trust evidence.
    The filter receives the in-process trust map (warmed by
    ``populate_trust_cache``); when ``min_particle_confidence`` is 0.0
    the filter is a no-op and ``trust_map`` may be omitted.
    """
    from particles.store.particle_store import get_particles_by_status
    from particles.store.subject_store import (
        list_all_subjects,
        list_particle_subject_pairs,
    )

    subjects = await list_all_subjects(session)
    if subject_filter is not None:
        subjects = [s for s in subjects if s.canonical_name in subject_filter]

    active_particles = await get_particles_by_status(session, Status.ACTIVE)
    # cap. 1: keep a document's rejected / deferred / counterfactual
    # prose out of synthesised articles unless the caller opts in.
    if not include_non_asserted:
        active_particles = [p for p in active_particles if not is_non_asserted(p.properties)]
    active_by_id: dict[str, Particle] = {p.id: p for p in active_particles}

    by_subject: dict[str, list[str]] = defaultdict(list)
    for pid, sid in await list_particle_subject_pairs(session):
        if pid in active_by_id:
            by_subject[sid].append(pid)

    # batched source-trust ranks for the quality filter,
    # loaded once over the full ACTIVE set ({} when no policy is configured).
    source_ranks: dict[str, float] = {}
    if min_particle_confidence > 0.0:
        from particles.operations.query.source_trust import load_source_trust_ranks

        source_ranks = await load_source_trust_ranks(session, active_particles)

    dropped_by_subject: dict[str, int] = {}
    out: list[tuple[Subject, list[Particle]]] = []
    for subj in subjects:
        pids = by_subject.get(subj.id, [])
        # Sort by particle id for hash stability (compute_input_hash also sorts).
        particles = sorted((active_by_id[pid] for pid in pids), key=lambda p: p.id)
        if min_particle_confidence > 0.0:
            eff_by_id = _effective_confidences(particles, trust_map or {}, source_ranks)
            filtered: list[Particle] = []
            dropped_here = 0
            for p in particles:
                if eff_by_id[p.id] < min_particle_confidence:
                    dropped_here += 1
                    continue
                filtered.append(p)
            particles = filtered
            if dropped_here:
                dropped_by_subject[subj.id] = dropped_here
        # Count check applies to the *filtered* set.
        if len(particles) < min_particles:
            continue
        out.append((subj, particles))
    return out, dropped_by_subject


def _effective_confidences(
    particles: list[Particle],
    trust_map: dict[str, float],
    source_ranks: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute trust-weighted effective confidence for each particle (Core).

    ``source_ranks`` is the query-time source-trust map
    (``{particle_id: asserted rank}``, absence = neutral 1.0) precomputed by
    the caller via ``load_source_trust_ranks`` — kept as a parameter so this
    function stays pure.

    Recency decay is intentionally skipped here: the synthesis
    prompt cares mainly about which claims to hedge, trust-weighted
    confidence is the dominant signal, and age changing which claims the
    synthesiser hedges has not been shown by a calibration pass.
    """
    result: dict[str, float] = {}
    for p in particles:
        trust = trust_map.get(p.extractor_ref.name, 1.0) if p.extractor_ref else 1.0
        # confidence.value is already calibrated at creation time when the
        # extractor carried an active calibration; no read-side
        # calibration step exists.
        result[p.id] = compute_effective_confidence(
            p.confidence.value,
            extractor_trust_weight=trust,
            source_trust_rank=(source_ranks or {}).get(p.id, 1.0),
            calibration_source=p.confidence.calibration_source,
        )
    return result


def estimate_prompt_tokens(particles: list[Particle]) -> int:
    """Cheap approximation: 4 chars ≈ 1 token over the rendered particle list.

    Used only for the ``--dry-run`` cost summary; the real LLM call uses
    Anthropic's own token counting via the SDK.
    """
    char_count = sum(len(p.content) + 64 for p in particles)  # 64 = boilerplate
    return char_count // 4


# Note: batch-loading ``uri_r`` for cited corpus entries lives in
# :func:`particles.corpus.store.get_entry_uri_map` so the Obsidian and
# Wiki exporters share one implementation. The wiki exporter passes a
# filter set (only entries actually cited by qualifying particles);
# Obsidian passes ``None`` to load every entry.


# ---------------------------------------------------------------------------
# Filesystem cache check
# ---------------------------------------------------------------------------


def _read_cached_hash(article_path: Path) -> str | None:
    """Read ``input_hash`` from an existing article's frontmatter, if any.

    The wiki exporter persists each article as its own file; the cache
    key lives in that file's YAML frontmatter so a subsequent run can
    short-circuit synthesis without consulting any out-of-band store.
    Returns ``None`` if the file doesn't exist, can't be read, has no
    frontmatter, or the frontmatter lacks an ``input_hash`` field.
    """
    if not article_path.exists():
        return None
    try:
        text = article_path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm = _parse_frontmatter(text)
    if not fm:
        return None
    val = fm.get("input_hash")
    return str(val) if isinstance(val, str) else None


# ---------------------------------------------------------------------------
# Exporter plugin
# ---------------------------------------------------------------------------


class WikiExporter:
    """Wiki-article exporter plugin (refactored).

    Options accepted by :meth:`export` (all optional):

    * ``min_particles`` (int) — override ``config.wiki.min_particles``
    * ``min_particle_confidence`` (float) — override
      ``config.exporter_common.min_particle_confidence``.
      Particles with ``effective_confidence`` below the threshold are
      dropped before the count check and the input-hash cache key.
    * ``regenerate_all`` (bool, default False) — bypass the input-hash cache
    * ``subjects`` (list[str] | None) — only render these canonical names
    * ``dry_run`` (bool, default False) — report cache hits + token estimate;
      do not write files or call the LLM
    """

    FORMAT = "wiki"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> WikiSummary:
        if output is None:
            raise ValueError("WikiExporter requires an output directory path")

        from particles.config import get_config

        cfg = get_config().wiki
        common_cfg = get_config().exporter_common
        min_particles = int(options.get("min_particles", cfg.min_particles))  # type: ignore[call-overload]
        min_particle_confidence = float(
            options.get(  # type: ignore[arg-type]
                "min_particle_confidence", common_cfg.min_particle_confidence
            )
        )
        regenerate_all = bool(options.get("regenerate_all", False))
        invalidate_stale_links = bool(options.get("invalidate_stale_links", False))
        dry_run = bool(options.get("dry_run", False))
        #: deterministic no-LLM export — every article is the
        # structured listing; no API key, no token cost, reproducible.
        without_synthesis = bool(options.get("without_synthesis", False))
        include_non_asserted = bool(options.get("include_non_asserted", False))
        raw_subjects = options.get("subjects")
        subject_filter: set[str] | None
        if raw_subjects is None:
            subject_filter = None
        elif isinstance(raw_subjects, (list, tuple, set)):
            subject_filter = {str(s) for s in raw_subjects}
        else:
            subject_filter = {str(raw_subjects)}

        output.mkdir(parents=True, exist_ok=True)

        # Warm the in-process trust cache from the DB BEFORE
        # `_load_qualifying_subjects` so the cross-exporter quality
        # filter can compute effective_confidence per
        # particle without a per-particle round-trip.
        db_trust_map = await get_trust_weight_map(session)
        populate_trust_cache(db_trust_map)

        # compute the disambiguation map from the FULL subject
        # set (collisions must be detected before the min_particles
        # filter drops members) so filenames + index links key on the
        # disambiguated display-name slug.
        from particles.store.subject_store import list_all_subjects

        all_subjects = await list_all_subjects(session)
        naming = build_subject_naming(all_subjects)

        def _slug_for(s: Subject) -> str:
            return _subject_slug(naming.display_name(s))

        # (the mechanism): one cited article per ACTIVE
        # NARRATIVE under `Narratives/`. Narratives are subject-less, so a run
        # narrowed with `--subjects` suppresses them — the operator asked for a
        # specific slice of the subject namespace, not the whole store.
        from particles.store.particle_store import get_active_narratives

        emit_narratives = cfg.emit_narrative_notes and subject_filter is None
        narrative_particles = await get_active_narratives(session) if emit_narratives else []
        if not include_non_asserted:
            narrative_particles = [
                n for n in narrative_particles if not is_non_asserted(n.properties)
            ]
        narrative_naming = build_narrative_naming(narrative_particles)  # narrative id → leaf slug

        def _narrative_rel_path(slug: str) -> str:
            """The narrative article's path relative to the output dir, no suffix."""
            return f"Narratives/{slug}"

        # stale-wikilink invalidation. Runs BEFORE the per-
        # subject cache-hit loop so any article whose ``[[X]]``
        # references a renamed subject regenerates this run rather than
        # waiting for that subject's own particles to change.
        stale_invalidated = 0
        if invalidate_stale_links and not dry_run:
            from particles.render.article_synthesis.cache import (
                invalidate_stale_link_articles,
            )
            from particles.store.synthesis_cache_store import evict_subject

            # Index links are display-name slugs; the known
            # set must contain those + the ``(disambiguation)`` page
            # names or each disambiguated page is spuriously invalidated.
            known_names: set[str] = set()
            for s in all_subjects:
                known_names.add(s.canonical_name)
                known_names.update(s.aliases)
                known_names.add(naming.display_name(s))
                known_names.add(_slug_for(s))
            for g in naming.groups:
                known_names.add(disambiguation_name(g.base_name))
                known_names.add(_subject_slug(disambiguation_name(g.base_name)))
            # the index links narrative articles as
            # `[[Narratives/<slug>]]`, and subject articles carry the same
            # target in their See-also block — valid targets, so add them or
            # every article referencing one is spuriously invalidated each run.
            for nslug in narrative_naming.values():
                known_names.add(_narrative_rel_path(nslug))
            # Narrative articles live in a subdirectory, so the scan recurses.
            invalidated_paths = invalidate_stale_link_articles(output, known_names, recursive=True)
            stale_invalidated = len(invalidated_paths)
            # evict the shared DB cache for any subject whose
            # on-disk article we just invalidated. Without this, the next
            # render would hit the DB cache and reuse the stale prose
            # rather than truly re-synthesise. Key on the path relative to
            # the output dir (not ``path.stem``) so `Narratives/<slug>.md`
            # maps back to its narrative rather than colliding with a
            # same-named subject article.
            slug_to_subject_id = {_slug_for(s): s.id for s in all_subjects}
            # A narrative article's synthesis cache is keyed by the narrative id
            # (its synthetic subject id).
            for nid, nslug in narrative_naming.items():
                slug_to_subject_id[_narrative_rel_path(nslug)] = nid
            for path in invalidated_paths:
                rel = str(path.relative_to(output).with_suffix(""))
                subject_id = slug_to_subject_id.get(rel)
                if subject_id is not None:
                    await evict_subject(session, subject_id)
            if stale_invalidated:
                log.info(
                    "invalidated %d cached article(s) with stale wikilinks",
                    stale_invalidated,
                )

        qualifying, dropped_by_subject = await _load_qualifying_subjects(
            session,
            min_particles=min_particles,
            subject_filter=subject_filter,
            min_particle_confidence=min_particle_confidence,
            include_non_asserted=include_non_asserted,
            trust_map=db_trust_map,
        )
        particles_dropped_below_threshold = sum(dropped_by_subject.values())

        # source-trust ranks for the per-article
        # confidence values handed to synthesis, loaded once over every
        # qualifying particle ({} when no policy is configured).
        from particles.operations.query.source_trust import load_source_trust_ranks

        source_ranks = await load_source_trust_ranks(
            session, [p for _, particles in qualifying for p in particles]
        )

        trust_map: dict[str, float] = {}

        cache_hits = 0
        regen_count = 0
        skipped_no_particles = 0
        articles_written: list[str] = []
        synthesis_used = 0
        synthesis_failed = 0
        synthesis_skipped = 0  # deterministic listings written under --without-synthesis
        estimated_tokens = 0
        # Set once an account-fatal LLM error (billing / auth / quota) is hit:
        # the rest of the run skips synthesis and writes no hashed fallback, so
        # those subjects retry on the next export. Cache hits are still served.
        synthesis_aborted = False

        # Collect every corpus_entry_id referenced by any particle in this
        # export so we can batch-load source URLs in one query rather than
        # one per particle. Only done in non-dry-run mode — the references
        # section is the only consumer.
        needed_entry_ids: set[str] = set()
        if not dry_run:
            for _, particles in qualifying:
                for p in particles:
                    for ref in p.provenance:
                        if ref.corpus_entry_id:
                            needed_entry_ids.add(ref.corpus_entry_id)
        from particles.corpus.store import get_entry_uri_map

        corpus_uris = await get_entry_uri_map(session, needed_entry_ids)

        # Run a lint pre-pass so per-article callouts (PROVENANCE_STALE,
        # CONTRADICTION, …) can be spliced into the body. ``semantic=False``
        # keeps it fast (no LLM-driven contradiction check) and
        # ``fix=False`` makes the export side-effect-free on the
        # particle store. Skipped in dry-run mode — operators running
        # ``--dry-run`` care about token cost, not lint coverage.
        findings_by_particle: dict[str, list[LintFinding]] = defaultdict(list)
        findings_by_subject: dict[str, list[LintFinding]] = defaultdict(list)
        if not dry_run and qualifying:
            from particles.operations.lint import run_lint

            try:
                lint_report = await run_lint(session, fix=False, semantic=False)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("Wiki export: lint pre-pass failed (%s); continuing", exc)
            else:
                for f in lint_report.findings:
                    if f.particle_id:
                        findings_by_particle[f.particle_id].append(f)
                    if f.subject_id:
                        findings_by_subject[f.subject_id].append(f)

        total = len(qualifying)
        for idx, (subject, particles) in enumerate(qualifying, start=1):
            if not particles:  # pragma: no cover — filtered upstream
                skipped_no_particles += 1
                continue

            input_hash = compute_input_hash(particles, subject)
            article_path = output / f"{_slug_for(subject)}.md"
            if not is_within_directory(output, article_path):
                # Defence-in-depth (F1): the slug derives from an untrusted,
                # LLM-extracted canonical_name. Skip any subject whose path
                # would escape the output dir — guarding the cache read below
                # as well as the write.
                log.warning(
                    "Skipping subject %r — article path escapes the output dir: %s",
                    subject.canonical_name,
                    article_path,
                )
                continue

            # Findings relevant to *this* article: any finding tagged with
            # the subject, plus any finding tagged with one of the
            # subject's particles. Dedup by id() to handle the rare case
            # where a finding hits both keys. Computed before the cache
            # check because callouts are refreshed on cache hits too.
            relevant: list[LintFinding] = list(findings_by_subject.get(subject.id, []))
            seen_ids: set[int] = {id(f) for f in relevant}
            for p in particles:
                for f in findings_by_particle.get(p.id, []):
                    if id(f) not in seen_ids:
                        relevant.append(f)
                        seen_ids.add(id(f))

            # --without-synthesis forces a deterministic re-render: skip the
            # on-disk cache so any stale or previously-LLM-synthesised article
            # is replaced by the structured listing.
            cached = (
                None if (regenerate_all or without_synthesis) else _read_cached_hash(article_path)
            )
            if cached == input_hash:
                cache_hits += 1
                # the article body is cached, but lint callouts
                # are a write-time layer. Re-apply the current findings to
                # the on-disk article so a finding's text — or a finding
                # appearing / resolving since the body was cached —
                # surfaces without regenerating prose (no LLM call).
                # Idempotent: a no-change refresh leaves the file untouched.
                if not dry_run:
                    existing = article_path.read_text(encoding="utf-8")
                    refreshed = apply_lint_callouts(existing, relevant or None)
                    if refreshed != existing:
                        atomic_write_text(article_path, refreshed)
                        log.info(
                            "[%d/%d] %s — cache hit (callouts refreshed)",
                            idx,
                            total,
                            subject.canonical_name,
                        )
                    else:
                        log.info("[%d/%d] %s — cache hit", idx, total, subject.canonical_name)
                else:
                    log.info("[%d/%d] %s — cache hit", idx, total, subject.canonical_name)
                continue

            regen_count += 1
            if not without_synthesis:
                estimated_tokens += estimate_prompt_tokens(particles)

            if dry_run:
                if without_synthesis:
                    log.info(
                        "[%d/%d] %s — would regenerate (%d particles, deterministic, no LLM)",
                        idx,
                        total,
                        subject.canonical_name,
                        len(particles),
                    )
                else:
                    log.info(
                        "[%d/%d] %s — would regenerate (%d particles, ~%d tokens)",
                        idx,
                        total,
                        subject.canonical_name,
                        len(particles),
                        estimate_prompt_tokens(particles),
                    )
                continue

            if synthesis_aborted:
                # An earlier subject hit an account-fatal LLM error; skip the
                # call and write nothing, so this subject retries next run.
                synthesis_failed += 1
                log.info(
                    "[%d/%d] %s — synthesis skipped (backend unavailable; retries next run)",
                    idx,
                    total,
                    subject.canonical_name,
                )
                continue

            log.info(
                "[%d/%d] %s — synthesising (%d particles)…",
                idx,
                total,
                subject.canonical_name,
                len(particles),
            )

            # Cache lookup for trust weights, lazily populated as we encounter
            # new extractor IDs. ``populate_trust_cache`` already warmed the
            # underlying registry; this dict is the per-export memo.
            for p in particles:
                if p.extractor_ref and p.extractor_ref.name not in trust_map:
                    name = p.extractor_ref.name
                    trust_map[name] = get_cached_trust_weight(name) or 1.0

            eff = _effective_confidences(particles, trust_map, source_ranks)
            try:
                body, used_synthesis = await render_article(
                    subject=subject,
                    particles=particles,
                    eff=eff,
                    input_hash=input_hash,
                    corpus_uris=corpus_uris,
                    max_tokens=cfg.max_tokens,
                    layer_b_enabled=cfg.layer_b_enabled,
                    lint_findings=relevant or None,
                    # wiki: surface the threshold this article
                    # was built under + the per-subject drop count in the
                    # frontmatter. Omitted (None) when the operator did not
                    # opt in to the filter — the YAML parser tolerates the
                    # absence per the wiki block of § 5.
                    min_particle_confidence=(
                        min_particle_confidence if min_particle_confidence > 0.0 else None
                    ),
                    dropped_below_threshold=dropped_by_subject.get(subject.id, 0)
                    if min_particle_confidence > 0.0
                    else None,
                    # thread the session so render_article can
                    # consult / populate the cross-exporter synthesis cache.
                    session=session,
                    without_synthesis=without_synthesis,
                )
            except SynthesisUnavailable as exc:
                # Account-fatal (billing / auth / quota): abort synthesis for
                # the rest of the run and write nothing for this subject, so it
                # retries on the next export instead of caching a fallback.
                synthesis_aborted = True
                synthesis_failed += 1
                log.warning(
                    "Article synthesis unavailable (%s) — aborting synthesis for the "
                    "remaining subjects; they retry on the next export.",
                    exc,
                )
                continue
            if used_synthesis:
                synthesis_used += 1
                log.info(
                    "[%d/%d] %s — wrote (LLM-synthesised)",
                    idx,
                    total,
                    subject.canonical_name,
                )
            elif without_synthesis:
                synthesis_skipped += 1
                log.info(
                    "[%d/%d] %s — wrote (deterministic listing; synthesis skipped)",
                    idx,
                    total,
                    subject.canonical_name,
                )
            else:
                synthesis_failed += 1
                log.info(
                    "[%d/%d] %s — wrote (structured-listing fallback)",
                    idx,
                    total,
                    subject.canonical_name,
                )
            # splice lint callouts into the freshly rendered body
            # at write time (the cached body is callout-free), so the same
            # write-time layer applies on both the fresh and cache-hit paths.
            body = apply_lint_callouts(body, relevant or None)
            atomic_write_text(article_path, body)
            articles_written.append(article_path.name)

        # one Wikipedia-style ``(disambiguation)`` page per
        # collision group whose members actually rendered as articles.
        if not dry_run and naming.has_collisions:
            subject_by_id = {s.id: s for s in all_subjects}
            qualifying_counts = {s.id: len(ps) for s, ps in qualifying}
            for group in naming.groups:
                members = [
                    subject_by_id[mid] for mid in group.member_ids if mid in qualifying_counts
                ]
                if len(members) < 2:
                    continue
                page = _render_wiki_disambiguation_page(
                    group.base_name, members, naming, qualifying_counts
                )
                page_path = output / f"{_subject_slug(disambiguation_name(group.base_name))}.md"
                if not is_within_directory(output, page_path):
                    log.warning(
                        "Skipping disambiguation page %r — path escapes the output dir: %s",
                        group.base_name,
                        page_path,
                    )
                    continue
                atomic_write_text(page_path, page)
                articles_written.append(page_path.name)

        # one cited article per ACTIVE NARRATIVE under `Narratives/`,
        # rendered by the sequence path (the mechanism).
        # NARRATIVE particles are subject-less, so these articles are purely
        # additive — no subject article loses content to them. Gated by
        # `wiki.emit_narrative_notes` and the shared constituent-count floor.
        narrative_articles = 0
        if emit_narratives and narrative_particles:
            from particles.operations.narrative import get_narrative_sequence

            gate = common_cfg.synthesis_min_particles
            narratives_dir = output / "Narratives"
            for narrative in narrative_particles:
                constituents = await get_narrative_sequence(session, narrative.id)
                if len(constituents) < gate:
                    continue
                slug = narrative_naming[narrative.id]
                rel = _narrative_rel_path(slug)
                article_path = narratives_dir / f"{slug}.md"
                if not is_within_directory(output, article_path):
                    log.warning(
                        "Skipping narrative %r — article path escapes the output dir: %s",
                        narrative.id,
                        article_path,
                    )
                    continue
                synthetic = narrative_as_subject(narrative)
                input_hash = compute_input_hash(constituents, synthetic, ordered=True)
                cached = (
                    None
                    if (regenerate_all or without_synthesis)
                    else _read_cached_hash(article_path)
                )
                if cached == input_hash:
                    cache_hits += 1
                    log.info("%s — cache hit (narrative)", rel)
                    narrative_articles += 1
                    continue

                regen_count += 1
                if not without_synthesis:
                    estimated_tokens += estimate_prompt_tokens(constituents)
                if dry_run:
                    log.info("%s — would regenerate (%d constituents)", rel, len(constituents))
                    continue
                if synthesis_aborted:
                    synthesis_failed += 1
                    continue

                eff = _effective_confidences(constituents, trust_map, source_ranks)
                try:
                    body, used_synthesis = await render_article(
                        subject=synthetic,
                        particles=constituents,
                        eff=eff,
                        input_hash=input_hash,
                        corpus_uris=corpus_uris,
                        max_tokens=cfg.max_tokens,
                        layer_b_enabled=cfg.layer_b_enabled,
                        session=session,
                        without_synthesis=without_synthesis,
                        sequence_mode=True,
                    )
                except SynthesisUnavailable as exc:
                    synthesis_aborted = True
                    synthesis_failed += 1
                    log.warning(
                        "Narrative synthesis unavailable (%s) — skipping the remaining "
                        "narrative articles; they retry on the next export.",
                        exc,
                    )
                    continue
                if used_synthesis:
                    synthesis_used += 1
                elif without_synthesis:
                    synthesis_skipped += 1
                else:
                    synthesis_failed += 1
                narratives_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_text(article_path, body)
                articles_written.append(f"{rel}.md")
                narrative_articles += 1

        if not dry_run and articles_written:
            _write_index(output, articles_written)

        summary = WikiSummary(
            dry_run=dry_run,
            qualifying_subjects=len(qualifying),
            cache_hits=cache_hits,
            articles_regenerated=regen_count,
            estimated_prompt_tokens=estimated_tokens,
            min_particle_confidence=min_particle_confidence,
            particles_dropped_below_threshold=particles_dropped_below_threshold,
            stale_link_articles_invalidated=(stale_invalidated if invalidate_stale_links else None),
            articles_written=None if dry_run else len(articles_written),
            synthesis_used=None if dry_run else synthesis_used,
            synthesis_failed=None if dry_run else synthesis_failed,
            synthesis_skipped=(None if (dry_run or not without_synthesis) else synthesis_skipped),
            narrative_articles=narrative_articles if emit_narratives else None,
        )
        log.info("wiki export summary: %s", summary.model_dump(exclude_none=True))
        return summary


def _render_wiki_disambiguation_page(
    base_name: str,
    members: list[Subject],
    naming: SubjectNaming,
    counts: dict[str, int],
) -> str:
    """Render a Wikipedia-style ``(disambiguation)`` page for a wiki collision group."""
    lines = [
        f"# {disambiguation_name(base_name)}\n",
        f'"{base_name}" refers to multiple subjects:\n',
    ]
    for s in sorted(members, key=lambda m: naming.display_name(m)):
        slug = _subject_slug(naming.display_name(s))
        qualifier = naming.qualifier_by_id.get(s.id, "")
        count = counts.get(s.id, 0)
        noun = "particle" if count == 1 else "particles"
        suffix = f" — {qualifier} ({count} {noun})" if qualifier else f" — {count} {noun}"
        lines.append(f"- [[{slug}]]{suffix}\n")
    return "".join(lines)


def _write_index(output: Path, article_filenames: list[str]) -> None:
    """Emit a top-level ``index.md`` listing every generated article."""
    lines = ["# Wiki articles\n\n"]
    for fname in sorted(article_filenames):
        title = fname.removesuffix(".md")
        lines.append(f"- [[{title}]]\n")
    atomic_write_text(output / "index.md", "".join(lines))


# Provide module-level access to the synthesis-machinery helpers that
# existing tests reach for via ``from particles.exporters.wiki import …``.
# Avoid F401 by keeping the imports referenced — the ``__all__`` above
# documents the *intentional* re-export set; the names below cover
# private symbols some tests reach for directly.
__all__ += ["_parse_frontmatter", "_parse_layer_b_verdicts"]
