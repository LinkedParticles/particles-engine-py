"""Logseq graph orchestration.

The :func:`export_vault` coroutine is the entry point that
:class:`particles.exporters.logseq.LogseqExporter.export` delegates
to. Mirrors the Obsidian orchestrator's shape (load subjects, apply
the quality-threshold filter, walk eligible subjects, write files,
optionally splice synthesis) but emits Logseq's bullet-outline
format instead of free-form Markdown.

The output directory becomes a Logseq graph root: pages live under
``<output>/pages/`` so an operator pointing at an existing graph
adds particle-derived pages without disturbing ``journals/`` or
``logseq/``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import LintFinding, Particle, ParticleType, Subject
from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.status import Status
from particles.exporters.logseq.format import (
    logseq_slug,
    render_block,
    render_narratives_block,
    render_property,
    render_subject_page,
)
from particles.exporters.summaries import LogseqSummary
from particles.extraction.polarity import is_non_asserted
from particles.render.article_synthesis import compute_input_hash
from particles.render.markdown import (
    atomic_write_text,
    build_narrative_naming,
    build_subject_naming,
    disambiguation_name,
    is_within_directory,
    narrative_as_subject,
)
from particles.store.extractor_store import (
    get_cached_trust_weight,
    get_trust_weight_map,
    populate_trust_cache,
)

log = logging.getLogger(__name__)


async def export_vault(
    session: AsyncSession,
    output_dir: Path,
    *,
    min_particles: int = 0,
    min_links: int = 1,
    with_synthesis: bool = False,
    invalidate_stale_links: bool = False,
    min_particle_confidence: float = 0.0,
    include_non_asserted: bool = False,
) -> LogseqSummary:
    """Export the particle store as a Logseq graph.

    Writes ``<output_dir>/pages/<subject_slug>.md`` for every eligible
    Subject. Subjects with fewer than ``min_particles`` post-filter
    particles or fewer than ``min_links`` graph links (in +
    out combined) are suppressed.

    With ``with_synthesis=True``, splices an LLM-synthesised prose
    article into each page above the structural outline, using the
    cross-exporter cache so the LLM cost is paid once
    across wiki / obsidian / logseq.

    ``invalidate_stale_links=True`` strips the
    ``article_input_hash::`` line from pages whose ``[[X]]``
    wikilinks reference a renamed subject and also evicts the shared
    DB cache for those subjects so the next render genuinely
    re-synthesises.

    Returns a :class:`LogseqSummary`.
    """
    from particles.corpus.store import get_entry_uri_map
    from particles.store.particle_store import get_particles_by_status
    from particles.store.subject_store import (
        list_all_subjects,
        list_particle_subject_pairs,
    )

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # compute the disambiguation map from the FULL subject set
    # up front. Page filenames, the H1 title, and the Contents index all
    # key on the disambiguated display name.
    subjects = await list_all_subjects(session)
    subject_map: dict[str, Subject] = {s.id: s for s in subjects}
    naming = build_subject_naming(subjects)

    def _slug_for(subject: Subject) -> str:
        return logseq_slug(naming.display_name(subject))

    # (the mechanism): load the ACTIVE particle set now — it
    # also feeds eff_conf and the per-subject grouping below — and pick out the
    # NARRATIVE particles up front, so the stale-link block can evict their
    # synthesis cache by narrative page name.
    from particles.config import get_config

    all_particles = await get_particles_by_status(session, Status.ACTIVE)
    # cap. 1: keep a document's rejected / deferred / counterfactual
    # prose out of the rendered graph unless the caller opts in.
    if not include_non_asserted:
        all_particles = [p for p in all_particles if not is_non_asserted(p.properties)]
    emit_narratives = with_synthesis and get_config().logseq.emit_narrative_notes
    narrative_particles = (
        [p for p in all_particles if p.particle_type == ParticleType.NARRATIVE]
        if emit_narratives
        else []
    )
    narrative_naming = build_narrative_naming(narrative_particles)  # narrative id → leaf slug

    def _narrative_page_name(slug: str) -> str:
        """The Logseq *page* name for a narrative note (``Narratives/<slug>``)."""
        return f"Narratives/{slug}"

    # --- Stale-link invalidation -----------------
    stale_link_articles_invalidated = 0
    if invalidate_stale_links and with_synthesis:
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )
        from particles.store.synthesis_cache_store import evict_subject

        # ``[[X]]`` targets are display names / display-name slugs (ADR
        # 0091); include those + the ``(disambiguation)`` page names so a
        # disambiguated graph is not spuriously invalidated each run.
        known_names: set[str] = set()
        for s in subjects:
            known_names.add(s.canonical_name)
            known_names.update(s.aliases)
            known_names.add(naming.display_name(s))
            known_names.add(_slug_for(s))
        for g in naming.groups:
            known_names.add(disambiguation_name(g.base_name))
            known_names.add(logseq_slug(disambiguation_name(g.base_name)))
        # subject pages carry `[[Narratives/<slug>]]` backlinks, so the
        # narrative page names are valid link targets — add them or every subject
        # page with a backlink is spuriously invalidated each run.
        for nslug in narrative_naming.values():
            known_names.add(_narrative_page_name(nslug))
        invalidated_paths = invalidate_stale_link_articles(
            pages_dir,
            known_names,
            hash_field="article_input_hash",
            recursive=False,
        )
        stale_link_articles_invalidated = len(invalidated_paths)
        slug_to_subject_id = {_slug_for(s): s.id for s in subjects}
        # a narrative page's synthesis cache is keyed by the narrative
        # id (its synthetic subject id); map its on-disk filename so a stale-link
        # invalidation evicts it too and the next render is genuinely fresh.
        for nid, nslug in narrative_naming.items():
            slug_to_subject_id[logseq_slug(_narrative_page_name(nslug))] = nid
        for path in invalidated_paths:
            subject_id = slug_to_subject_id.get(path.stem)
            if subject_id is not None:
                await evict_subject(session, subject_id)
        if stale_link_articles_invalidated:
            log.info(
                "invalidated %d cached page(s) with stale wikilinks",
                stale_link_articles_invalidated,
            )

    # Snapshot the synthesis-cache hashes from existing pages so the
    # synthesis splice can detect a cache hit on the prior body. The
    # 0.42.4 prune-instead-of-wipe pattern means the on-disk page is
    # already the prior body when the hash matches — no separate body
    # snapshot needed (Logseq doesn't run on-disk backfill).
    from particles.render.article_synthesis import _parse_frontmatter

    prior_synthesis_hashes: dict[str, str] = {}
    if with_synthesis:
        for existing in pages_dir.glob("*.md"):
            try:
                text = existing.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            if not fm:
                continue
            cached = fm.get("article_input_hash")
            if isinstance(cached, str):
                prior_synthesis_hashes[existing.name] = cached

    # Track every .md file this run writes so the post-write prune pass
    # can remove only files that won't be regenerated. Replaces the
    # pre-0.42.4 blanket wipe that emptied pages_dir before writing —
    # an interrupted export now leaves the graph in a consistent state.
    written_paths: set[Path] = set()

    # --- Load particles -------------------------------------------------
    # (``all_particles`` was loaded above so the stale-link block could see the
    # NARRATIVE set.)
    entry_uri_map = await get_entry_uri_map(session)

    # Bind particles to subjects via the join table — mirrors the wiki
    # orchestrator's `_load_qualifying_subjects` (the same particle can
    # belong to multiple subjects, and Particle.subject_ids isn't
    # always populated when callers use link_particle_to_subjects
    # separately).
    active_by_id: dict[str, Particle] = {p.id: p for p in all_particles}
    particles_by_subject: dict[str, list[Particle]] = defaultdict(list)
    for pid, sid in await list_particle_subject_pairs(session):
        if pid in active_by_id and sid in subject_map:
            particles_by_subject[sid].append(active_by_id[pid])

    particle_counts = {s.id: len(particles_by_subject[s.id]) for s in subjects}

    # --- Effective confidence (trust × source trust × decay) ------------
    # Mirrors the query path's per-particle computation incl.
    # source trust, so the vault's numbers match query.
    from particles.operations.query.decay_policy import load_decay_policy
    from particles.operations.query.source_info import load_source_rows
    from particles.operations.query.source_trust import load_trust_policy

    populate_trust_cache(await get_trust_weight_map(session))
    trust_policy = await load_trust_policy(session)
    decay_policy = await load_decay_policy(session)
    source_info = await load_source_rows(session, all_particles)
    eff_conf: dict[str, float] = {}
    for p in all_particles:
        extractor_id = p.extractor_ref.name if p.extractor_ref else "general-extractor"
        trust_weight = get_cached_trust_weight(extractor_id)
        pub_at, source_type, entry_id, uri_r, author_id = source_info.get(
            p.id, (None, "", None, None, None)
        )
        decay = decay_policy.recency_factor(pub_at, source_type, uri_r)
        rank = trust_policy.evaluate(entry_id, source_type, uri_r, author_id)
        eff_conf[p.id] = compute_effective_confidence(
            p.confidence.value,
            extractor_trust_weight=trust_weight,
            source_trust_rank=1.0 if rank is None else rank,
            recency_factor=decay,
            calibration_source=p.confidence.calibration_source,
        )

    # --- Quality threshold filter ----------------------------
    particles_dropped_below_threshold = 0
    if min_particle_confidence > 0.0:
        kept: list[Particle] = []
        for p in all_particles:
            if eff_conf.get(p.id, p.confidence.value) < min_particle_confidence:
                particles_dropped_below_threshold += 1
                continue
            kept.append(p)
        all_particles = kept
        kept_ids = {p.id for p in all_particles}
        particles_by_subject = defaultdict(list)
        active_by_id = {p.id: p for p in all_particles}
        for pid, sid in await list_particle_subject_pairs(session):
            if pid in kept_ids and sid in subject_map:
                particles_by_subject[sid].append(active_by_id[pid])
        particle_counts = {s.id: len(particles_by_subject[s.id]) for s in subjects}

    # precompute each emitted (gate-passing) narrative's
    # ordered constituents once — reused by the write loop — and the
    # subject→narrative membership map that drives the per-subject backlink
    # block. A subject participates in a narrative when one of the narrative's
    # constituents is about that subject.
    narrative_constituents: dict[str, list[Particle]] = {}
    subject_narratives: dict[str, list[Particle]] = defaultdict(list)
    if emit_narratives:
        from particles.operations.narrative import get_narrative_sequence

        # Derived from the join-table grouping above rather than
        # ``Particle.subject_ids``, which Logseq deliberately does not trust.
        subject_ids_by_particle: dict[str, set[str]] = defaultdict(set)
        for sid, sps in particles_by_subject.items():
            for sp in sps:
                subject_ids_by_particle[sp.id].add(sid)
        gate = get_config().exporter_common.synthesis_min_particles
        for n in narrative_particles:
            seq = await get_narrative_sequence(session, n.id)
            if len(seq) < gate:
                continue
            narrative_constituents[n.id] = seq
            for sid in {sid for c in seq for sid in subject_ids_by_particle.get(c.id, set())}:
                subject_narratives[sid].append(n)

    # --- Eligibility (min_particles + min_links) ------------------------
    # Subjects passing min_particles get a page; eligible_names is the
    # set used by the renderer's wikilink-suppression pass so phantom
    # ``[[X]]`` references collapse to bare text.
    eligible_subject_ids: set[str] = {
        s.id for s in subjects if particle_counts[s.id] >= min_particles
    }
    # Both the canonical name and the display name count as live
    # link targets: a particle that mentions ``[[Prometheus]]`` should
    # survive the dead-link suppression pass (it resolves to the
    # disambiguation page's alias), and so should ``[[Prometheus (software)]]``.
    eligible_names: set[str] = set()
    for sid in eligible_subject_ids:
        eligible_names.add(subject_map[sid].canonical_name)
        eligible_names.add(naming.display_name(subject_map[sid]))

    # Link count (in + out via the particle-mention graph). Particles
    # carrying multi-subject claims connect their subjects; particles
    # mentioning ``[[X]]`` add an inbound edge to X.
    link_counts: dict[str, int] = defaultdict(int)
    for p in all_particles:
        # Each particle's subject_ids form a clique — out-degree for
        # every pair.
        for sid in p.subject_ids:
            if sid in subject_map:
                link_counts[sid] += max(0, len(p.subject_ids) - 1)
        # Wikilinked-name → inbound edge if the target name is a known
        # subject. Cheap heuristic; the Obsidian exporter does the same.
        for name in _extract_wikilink_names(p.content):
            target_id = _subject_id_by_name(name, subject_map)
            if target_id is not None:
                link_counts[target_id] += 1

    suppressed = 0
    files_written = 0
    synthesis_used = 0
    synthesis_failed = 0
    synthesis_cache_hits = 0
    synthesis_skipped = 0
    # Set once an account-fatal LLM error (billing / auth / quota) is hit: the
    # rest of the run skips synthesis and stamps no hash, so those pages retry
    # on the next export.
    synthesis_aborted = False

    # --- Per-subject render + write -------------------------------------
    eligible_ordered = sorted(
        (s for s in subjects if s.id in eligible_subject_ids),
        key=lambda s: s.canonical_name,
    )

    for subject in eligible_ordered:
        if link_counts.get(subject.id, 0) < min_links:
            suppressed += 1
            continue

        ps = particles_by_subject[subject.id]
        # Stable order: structured (with properties) first, then by
        # confidence descending. Matches the Obsidian template.
        ps.sort(key=lambda p: (p.properties is None, -p.confidence.value))

        page = render_subject_page(
            subject,
            ps,
            eligible_ids=eligible_names,
            subject_map=subject_map,
            eff_conf=eff_conf,
            display_name=naming.display_name(subject),
        )

        # Optional synthesis splice (shared cache).
        if with_synthesis and ps and not synthesis_aborted:
            from particles.exporters.logseq.synthesis import (
                _splice_synthesised_article,
                _SynthesisOutcome,
            )
            from particles.render.article_synthesis import SynthesisUnavailable

            article_hash = compute_input_hash(ps, subject)
            prior_hash = prior_synthesis_hashes.get(f"{_slug_for(subject)}.md")
            try:
                page, outcome = await _splice_synthesised_article(
                    page=page,
                    subject=subject,
                    particles=ps,
                    eff_conf=eff_conf,
                    entry_uri_map=entry_uri_map,
                    article_hash=article_hash,
                    regenerate=prior_hash != article_hash,
                    lint_findings=[],  # LogseqExporter doesn't run the lint pre-pass yet
                    progress_prefix=f"{subject.canonical_name}",
                    session=session,
                )
            except SynthesisUnavailable as exc:
                # Account-fatal: abort synthesis for the rest of the run. `page`
                # keeps the un-stamped structural render, so it retries next run.
                synthesis_aborted = True
                synthesis_failed += 1
                log.warning(
                    "Article synthesis unavailable (%s) — aborting synthesis for the "
                    "remaining subjects; they retry on the next export.",
                    exc,
                )
            else:
                if outcome is _SynthesisOutcome.SYNTHESISED:
                    synthesis_used += 1
                elif outcome is _SynthesisOutcome.CACHE_HIT:
                    synthesis_cache_hits += 1
                elif outcome is _SynthesisOutcome.FALLBACK:
                    synthesis_failed += 1
                elif outcome is _SynthesisOutcome.SKIPPED:
                    synthesis_skipped += 1
        elif with_synthesis and ps and synthesis_aborted:
            # Synthesis already aborted earlier this run; count and move on.
            synthesis_failed += 1

        # append a `## Narratives` backlink block listing the
        # narratives this subject's claims participate in (subject → narrative).
        if emit_narratives and subject_narratives.get(subject.id):
            block = render_narratives_block(subject_narratives[subject.id], narrative_naming)
            if block:
                page = page.rstrip("\n") + "\n" + block

        page_path = pages_dir / f"{_slug_for(subject)}.md"
        if not is_within_directory(pages_dir, page_path):
            # Defence-in-depth (F1): logseq_slug already flattens "/" and
            # neutralises "..", but guard the write anyway so an untrusted,
            # LLM-extracted canonical_name can never escape pages/.
            log.warning(
                "Skipping subject %r — page path escapes pages/: %s",
                subject.canonical_name,
                page_path,
            )
            continue
        atomic_write_text(page_path, page)
        written_paths.add(page_path.resolve())
        files_written += 1

    # --- Top-level page index (Logseq's "Contents" convention) ----------
    rendered_ids: set[str] = {
        s.id for s in eligible_ordered if link_counts.get(s.id, 0) >= min_links
    }
    index_lines = ["- # Contents\n"]
    for s in eligible_ordered:
        if s.id not in rendered_ids:
            continue
        index_lines.append(f"- [[{naming.display_name(s)}]]\n")
    contents_path = pages_dir / "Contents.md"
    atomic_write_text(contents_path, "".join(index_lines))
    written_paths.add(contents_path.resolve())
    files_written += 1

    # one Wikipedia-style ``(disambiguation)`` page per collision
    # group whose members actually rendered. ``alias:: <bare name>`` lets a
    # bare ``[[Prometheus]]`` mention resolve here in Logseq.
    for group in naming.groups:
        members = [subject_map[mid] for mid in group.member_ids if mid in rendered_ids]
        if len(members) < 2:
            continue
        disamb_lines = [
            render_block(f"# {disambiguation_name(group.base_name)}"),
            render_property("alias", group.base_name, depth=0),
        ]
        for s in sorted(members, key=lambda m: naming.display_name(m)):
            disamb_lines.append(render_block(f"[[{naming.display_name(s)}]]", depth=0))
        disamb_path = pages_dir / f"{logseq_slug(disambiguation_name(group.base_name))}.md"
        if not is_within_directory(pages_dir, disamb_path):
            log.warning(
                "Skipping disambiguation page %r — path escapes pages/: %s",
                group.base_name,
                disamb_path,
            )
            continue
        atomic_write_text(disamb_path, "\n".join(disamb_lines) + "\n")
        written_paths.add(disamb_path.resolve())
        files_written += 1

    # one page per ACTIVE NARRATIVE in the `Narratives/` page
    # namespace, rendered as cited prose via the path (the
    # mechanism). NARRATIVE particles are subject-less, so these pages are
    # purely additive to the graph. Gated by `logseq.emit_narrative_notes` and
    # the shared constituent-count floor.
    narrative_notes_written = 0
    if emit_narratives and narrative_constituents:
        from particles.exporters.logseq.narrative import render_narrative_page
        from particles.render.article_synthesis import SynthesisUnavailable

        for narrative in narrative_particles:
            constituents = narrative_constituents.get(narrative.id)
            if constituents is None:
                continue  # gated out by the constituent-count floor (precomputed)
            if synthesis_aborted:
                continue  # account-fatal LLM error earlier; retry on next export
            slug = narrative_naming[narrative.id]
            article_hash = compute_input_hash(
                constituents, narrative_as_subject(narrative), ordered=True
            )
            try:
                page, _state = await render_narrative_page(
                    narrative=narrative,
                    constituents=constituents,
                    entry_uri_map=entry_uri_map,
                    eff_conf=eff_conf,
                    article_hash=article_hash,
                    session=session,
                )
            except SynthesisUnavailable as exc:
                synthesis_aborted = True
                log.warning(
                    "Narrative synthesis unavailable (%s) — skipping remaining "
                    "narrative pages; they retry on the next export.",
                    exc,
                )
                continue
            page_path = pages_dir / f"{logseq_slug(_narrative_page_name(slug))}.md"
            if not is_within_directory(pages_dir, page_path):
                log.warning(
                    "Skipping narrative page %r — path escapes pages/: %s",
                    narrative.id,
                    page_path,
                )
                continue
            atomic_write_text(page_path, page)
            written_paths.add(page_path.resolve())
            narrative_notes_written += 1
            files_written += 1

    # Post-write prune (0.42.4): remove .md files this run did not write
    # so suppressed / renamed subjects don't linger. Logseq pages are
    # flat (no subdirs), so non-recursive.
    from particles.render.markdown import prune_obsolete_markdown

    files_pruned = prune_obsolete_markdown(pages_dir, written_paths, recursive=False)
    if files_pruned:
        log.info("Pruned %d obsolete page(s) from previous export", files_pruned)

    phantoms = sum(1 for c in particle_counts.values() if c == 0)

    return LogseqSummary(
        subjects=len(subjects),
        particles=len(all_particles),
        phantoms=phantoms,
        suppressed=suppressed,
        files_written=files_written,
        particles_dropped_below_threshold=particles_dropped_below_threshold,
        synthesis_used=synthesis_used if with_synthesis else None,
        synthesis_failed=synthesis_failed if with_synthesis else None,
        synthesis_cache_hits=synthesis_cache_hits if with_synthesis else None,
        synthesis_skipped=synthesis_skipped if with_synthesis else None,
        narrative_notes=narrative_notes_written if emit_narratives else None,
        stale_link_articles_invalidated=(
            stale_link_articles_invalidated if invalidate_stale_links else None
        ),
    )


# ---------------------------------------------------------------------------
# Small helpers — link-count pass
# ---------------------------------------------------------------------------


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def _extract_wikilink_names(text: str) -> set[str]:
    return {m.group(1).strip() for m in _WIKILINK_RE.finditer(text)}


_LintFinding = LintFinding  # re-export shape for the synthesis splice type hint


def _subject_id_by_name(name: str, subject_map: dict[str, Subject]) -> str | None:
    """Reverse-lookup a Subject ID by canonical name or alias.

    Linear scan — fine for the link-count pass which is ``O(particles ×
    subjects)`` already and runs once per export.
    """
    for sid, subj in subject_map.items():
        if subj.canonical_name == name or name in subj.aliases:
            return sid
    return None
