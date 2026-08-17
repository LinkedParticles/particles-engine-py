"""Obsidian per-subject note rendering and vault assembly.

Template dispatch:
  nmo:NumismaticObject → coin template (infobox frontmatter + descriptive sections)
  nmo:Material / nmo:Denomination / nmo:Issuer / nmo:Authority /
    nmo:ObjectType / nmo:Mint → pivot template (minimal, link-target only)
  None / other → generic template (callout blocks, unchanged from pre-ADR-0046)

The :func:`export_vault` coroutine is the orchestration entry point that
:class:`particles.exporters.obsidian.exporter.ObsidianExporter.export`
delegates to. Text-shaping primitives live in
:mod:`particles.exporters.obsidian.format`; the synthesis splice
lives in :mod:`particles.exporters.obsidian.synthesis`.
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
from particles.exporters.obsidian.format import (
    _is_category,
    _provenance_label,
    _render_narratives_section,
    _render_particle_audit_callouts,
)
from particles.exporters.obsidian.narrative import (
    narrative_as_subject,
    render_narrative_note,
)
from particles.exporters.obsidian.synthesis import _splice_synthesised_article
from particles.exporters.summaries import ObsidianSummary
from particles.extraction.polarity import is_non_asserted
from particles.render.article_synthesis import (
    SynthesisUnavailable,
    _parse_frontmatter,
    apply_lint_callouts,
    compute_input_hash,
)
from particles.render.markdown import (
    DisambiguationGroup,
    SubjectNaming,
    atomic_write_text,
    build_narrative_naming,
    build_subject_naming,
    disambiguation_name,
    is_within_directory,
    sanitize_filename,
    subject_slug,
)
from particles.store.extractor_store import (
    get_cached_trust_weight,
    get_trust_weight_map,
    populate_trust_cache,
)

log = logging.getLogger(__name__)

_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)\]\]")

_STATUS_CALLOUT: dict[str, str] = {
    Status.ACTIVE.value: "success",
    Status.INCONSISTENCY.value: "danger",
    Status.PROVENANCE_STALE.value: "warning",
    Status.SUPERSEDED.value: "note",
    Status.RETRACTED.value: "failure",
}

# Nomisma property key → (frontmatter label, unit or None, is_wiki_link)
_COIN_PROPS: list[tuple[str, str, str | None, bool]] = [
    # (nmo_key, yaml_label, unit, is_link)
    ("nmo:hasIssuer", "Issuer", None, True),
    ("nmo:hasAuthority", "Period", None, True),
    ("nmo:hasObjectType", "Type", None, True),
    ("nmo:hasProductionDate", "Years", None, False),
    ("nmo:hasFaceValue", "Value", None, False),
    ("nmo:hasDenomination", "Currency", None, True),
    ("nmo:hasMaterial", "Composition", None, True),
    ("nmo:hasWeight", "Weight", "g", False),
    ("nmo:hasDiameter", "Diameter", "mm", False),
    ("nmo:hasDepth", "Thickness", "mm", False),
    ("nmo:hasShape", "Shape", None, False),
    ("nmo:hasManufacture", "Technique", None, False),
    ("nmo:hasAxis", "Orientation", None, False),
    ("nuds:demonetizationDate", "Demonetized", None, False),
    ("numista:id", "Numista ID", None, False),
]

# subject_class values that produce pivot notes
_PIVOT_CLASSES = {
    "nmo:Material",
    "nmo:Denomination",
    "nmo:Issuer",
    "nmo:Authority",
    "nmo:ObjectType",
    "nmo:Mint",
}

# class → tag suffix used in pivot frontmatter
_PIVOT_TAG: dict[str, str] = {
    "nmo:Material": "material",
    "nmo:Denomination": "currency",
    "nmo:Issuer": "issuer",
    "nmo:Authority": "period",
    "nmo:ObjectType": "type",
    "nmo:Mint": "mint",
}


def _taxonomy_tag_lines(particles: list[Particle]) -> list[str]:
    """YAML ``tags:`` entries for the union of the subject's particle taxonomy tags.

    Per the taxonomy rule (§Consequences): operator-curated tags propagate to every
    export format. The Obsidian native tag axis is the per-note frontmatter
    ``tags:`` list, so a subject note carries the union of taxonomy tags across
    all its rendered particles. Returns ``["  - <tag>", …]`` (sorted, deduped)
    or ``[]`` when no particle carries a tag.
    """
    seen: set[str] = set()
    for p in particles:
        if p.tags:
            seen.update(p.tags)
    return [f"  - {tag}" for tag in sorted(seen)]


# _sanitize and _subject_slug live in particles.render.markdown as
# `sanitize_filename` / `subject_slug` so every exporter gets the same
# Obsidian / Wiki / Logseq filename-and-nesting behaviour. These local
# aliases keep the existing call sites readable without touching every
# render function.
_sanitize = sanitize_filename
_subject_slug = subject_slug


# ---------------------------------------------------------------------------
# Coin template
# ---------------------------------------------------------------------------

# Patterns for grouping descriptive particles into sections.
# Each entry is (section_name, regex_pattern).
_DESC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("obverse", re.compile(r"^The obverse (of .+? )?(depicts|lettering)", re.I)),
    ("obverse", re.compile(r"^.+ was engraved by .+", re.I)),
    ("reverse", re.compile(r"^The reverse (of .+? )?(depicts|lettering)", re.I)),
    ("edge", re.compile(r"^.+ has a .+ edge\.", re.I)),
    ("mints", re.compile(r"^.+ was struck at .+", re.I)),
    ("catalog", re.compile(r"^.+ is catalogued as .+", re.I)),
]


def _desc_section(content: str) -> str:
    for section, pat in _DESC_PATTERNS:
        if pat.match(content):
            return section
    return "comments"


def _render_coin_note(
    subject: Subject,
    particles: list[Particle],
    subject_map: dict[str, Subject],
    eff_conf: dict[str, float] | None = None,
    eligible_ids: set[str] | None = None,
    *,
    naming: SubjectNaming | None = None,
) -> str:
    if eff_conf is None:
        eff_conf = {}
    if eligible_ids is None:
        eligible_ids = set(subject_map)
    display = naming.display_name(subject) if naming is not None else subject.canonical_name
    structured = [p for p in particles if p.properties]
    descriptive = [p for p in particles if not p.properties]

    # Best structured particle: highest confidence
    best = max(structured, key=lambda p: p.confidence.value) if structured else None
    props: dict[str, object] = (best.properties or {}) if best else {}

    lines: list[str] = []

    # --- YAML frontmatter ---
    lines.append("---")
    lines.append("tags:")
    lines.append("  - particles/coin")
    for ref in subject.external_ids:
        lines.append(f"  - {ref.namespace}-{ref.id}")
    lines.extend(_taxonomy_tag_lines(particles))
    lines.append('Instance of: "[[Coin]]"')

    for nmo_key, yaml_label, unit, is_link in _COIN_PROPS:
        val = props.get(nmo_key)
        if val is None:
            continue
        if is_link and isinstance(val, str):
            lines.append(f'{yaml_label}: "[[{val}]]"')
        elif unit:
            lines.append(f"{yaml_label}: {val} {unit}")
        else:
            lines.append(f"{yaml_label}: {val}")

    # Catalog references (list → comma-separated string)
    refs = props.get("nuds:references")
    if isinstance(refs, list) and refs:
        lines.append(f'References: "{", ".join(refs)}"')

    # Numista URL
    numista_url = props.get("numista:url")
    if isinstance(numista_url, str):
        lines.append(f"Numista URL: {numista_url}")
    else:
        # Fallback: construct from id
        nid = props.get("numista:id")
        if isinstance(nid, str) and nid.isdigit():
            lines.append(f"Numista URL: https://en.numista.com/catalogue/pieces{nid}.html")

    lines.append("---")
    lines.append("")
    lines.append(f"# {display}")
    lines.append("")

    # Conflict callouts
    conflicts = _find_property_conflicts(structured)
    for nmo_key, competing in sorted(conflicts.items()):
        yaml_label = next((label for k, label, _, _ in _COIN_PROPS if k == nmo_key), nmo_key)
        lines.append(f"> [!warning] Conflicting {yaml_label}")
        for p, val in competing:
            src = _provenance_label(p)
            conf = eff_conf.get(p.id, p.confidence.value)
            lines.append(f"> **{val}** — {src} (confidence {conf:.2f})")
        lines.append("")

    # --- Descriptive body sections ---
    sections: dict[str, list[str]] = defaultdict(list)
    for p in descriptive:
        section = _desc_section(p.content)
        if section != "catalog":  # catalog refs already in frontmatter
            sections[section].append(p.content)

    _section_order = ["obverse", "reverse", "edge", "mints", "comments"]
    _section_titles = {
        "obverse": "Obverse",
        "reverse": "Reverse",
        "edge": "Edge",
        "mints": "Mints",
        "comments": "Comments",
    }

    for section in _section_order:
        items = sections.get(section)
        if not items:
            continue
        lines.append(f"### {_section_titles[section]}")
        lines.append("")
        for item in items:
            lines.append(item)
            lines.append("")

    return "\n".join(lines)


def _find_property_conflicts(
    structured: list[Particle],
) -> dict[str, list[tuple[Particle, object]]]:
    """Return property keys where >1 distinct value exists across structured particles."""
    by_key: dict[str, list[tuple[Particle, object]]] = defaultdict(list)
    for p in structured:
        if not p.properties:
            continue
        for key, val in p.properties.items():
            by_key[key].append((p, val))

    conflicts: dict[str, list[tuple[Particle, object]]] = {}
    for key, entries in by_key.items():
        unique_vals = {str(val) for _, val in entries}
        if len(unique_vals) > 1:
            # Sort by confidence descending so the winner is first
            conflicts[key] = sorted(entries, key=lambda x: -x[0].confidence.value)
    return conflicts


# ---------------------------------------------------------------------------
# Pivot template
# ---------------------------------------------------------------------------


def _render_pivot_note(
    subject: Subject,
    particles: list[Particle],
    subject_map: dict[str, Subject] | None = None,
    eff_conf: dict[str, float] | None = None,
    eligible_ids: set[str] | None = None,
    entry_uri_map: dict[str, str | None] | None = None,
    *,
    naming: SubjectNaming | None = None,
) -> str:
    if eff_conf is None:
        eff_conf = {}
    if subject_map is None:
        subject_map = {subject.id: subject}
    if eligible_ids is None:
        eligible_ids = set(subject_map)
    display = naming.display_name(subject) if naming is not None else subject.canonical_name
    tag = _PIVOT_TAG.get(subject.subject_class or "", "subject")
    lines: list[str] = []
    lines.append("---")
    lines.append("tags:")
    lines.append(f"  - particles/{tag}")
    for ref in subject.external_ids:
        lines.append(f"  - {ref.namespace}-{ref.id}")
    lines.extend(_taxonomy_tag_lines(particles))
    lines.append("---")
    lines.append("")
    lines.append(f"# {display}")
    lines.append("")

    # Per-particle audit trail in Format C — same shape the generic
    # template and the synthesised References section emit, so the
    # operator's vault stays visually consistent across templates.
    lines.extend(
        _render_particle_audit_callouts(
            particles,
            parent_subject=subject,
            subject_map=subject_map,
            eligible_ids=eligible_ids,
            eff_conf=eff_conf,
            entry_uri_map=entry_uri_map,
            naming=naming,
        )
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generic template (unchanged from pre-ADR-0046)
# ---------------------------------------------------------------------------


def _render_subject_note(
    subject: Subject,
    particles: list[Particle],
    subject_map: dict[str, Subject],
    eff_conf: dict[str, float] | None = None,
    eligible_ids: set[str] | None = None,
    entry_uri_map: dict[str, str | None] | None = None,
    *,
    naming: SubjectNaming | None = None,
) -> str:
    if eff_conf is None:
        eff_conf = {}
    if eligible_ids is None:
        eligible_ids = set(subject_map)
    display = naming.display_name(subject) if naming is not None else subject.canonical_name
    lines: list[str] = []

    from particles.config import get_config

    suppress_threshold = get_config().subjects.wikidata_link_suppress_threshold

    # Only include external refs whose confidence meets the threshold in frontmatter tags
    trusted_refs = [r for r in subject.external_ids if r.confidence >= suppress_threshold]
    unverified_refs = [r for r in subject.external_ids if r.confidence < suppress_threshold]

    ext_tags = [f"{r.namespace}-{r.id}" for r in trusted_refs]
    lines += ["---"]
    lines += ["tags:"]
    lines += ["  - particles/subject"]
    for t in ext_tags:
        lines += [f"  - {t}"]
    lines += _taxonomy_tag_lines(particles)
    lines += [f"particle_count: {len(particles)}"]
    lines += [f"phantom: {str(len(particles) == 0).lower()}"]
    lines += ["---"]
    lines += [""]

    lines += [f"# {display}"]
    lines += [""]

    # Only show the description as an authoritative definition when it comes from
    # a trusted source. If all Wikidata links are unverified, the description is
    # likely from the wrong entity and should not be presented as settled fact.
    has_trusted_wikidata = any(r.namespace == "wikidata" for r in trusted_refs)
    all_wikidata_unverified = (
        any(r.namespace == "wikidata" for r in unverified_refs) and not has_trusted_wikidata
    )
    if subject.description and not all_wikidata_unverified:
        lines += [f"_{subject.description}_", ""]

    if subject.aliases:
        lines += [f"**Aliases:** {', '.join(subject.aliases)}", ""]

    for ref in trusted_refs:
        label = f"{ref.namespace}:{ref.id}"
        if ref.uri:
            lines += [f"**{ref.namespace.title()}:** [{ref.id}]({ref.uri})"]
        else:
            lines += [f"**{ref.namespace.title()}:** `{label}`"]
    if trusted_refs:
        lines += [""]

    for ref in unverified_refs:
        label = f"{ref.namespace}:{ref.id}"
        sid = subject.id[:8]
        name = subject.canonical_name
        desc_line = (
            [f"> Wikidata description: _{subject.description}_"]
            if subject.description and ref.namespace == "wikidata"
            else []
        )
        lines += [
            f"> [!warning] Unverified {ref.namespace.title()} link",
            f"> Candidate: `{label}` (confidence {ref.confidence:.2f})"
            " — context mismatch detected.",
            *desc_line,
            f"> **To confirm this link is correct:** `particles subjects confirm {sid} {label}`",
            f"> **If the link is wrong, remove it:** `particles subjects unlink {sid} {label}`",
            f"> **To inspect:** `particles subjects show {sid}`",
            f"> **To find a better match:** `particles subjects search {name}`",
            "",
        ]

    if not particles:
        lines += [
            "> [!warning] Phantom subject",
            "> This subject has no ACTIVE claim particles.",
            "> Consider depositing a source about this entity or merging with another subject.",
            "",
        ]
        return "\n".join(lines)

    lines += [f"**Particles:** {len(particles)}", ""]
    lines += ["---", ""]

    lines.extend(
        _render_particle_audit_callouts(
            particles,
            parent_subject=subject,
            subject_map=subject_map,
            eligible_ids=eligible_ids,
            eff_conf=eff_conf,
            entry_uri_map=entry_uri_map,
            naming=naming,
        )
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def _render_index(
    subjects: list[Subject],
    particle_counts: dict[str, int],
    naming: SubjectNaming | None = None,
) -> str:
    lines = [
        "---",
        "tags: [particles/index]",
        "---",
        "# Particles — Subject Index",
        "",
        f"**Total subjects:** {len(subjects)}  ",
        f"**Phantom subjects:** {sum(1 for s in subjects if particle_counts.get(s.id, 0) == 0)}  ",
        f"**Total particles:** {sum(particle_counts.values())}",
        "",
        "---",
        "",
        "| Subject | Class | Particles | External IDs |",
        "|---|---|---|---|",
    ]
    for s in subjects:
        count = particle_counts.get(s.id, 0)
        phantom = " *(phantom)*" if count == 0 else ""
        cls = s.subject_class or ""
        ext = ", ".join(f"`{r.namespace}:{r.id}`" for r in s.external_ids)
        name = naming.display_name(s) if naming is not None else s.canonical_name
        lines.append(f"| {name}{phantom} | {cls} | {count} | {ext} |")

    return "\n".join(lines)


def _render_disambiguation_note(
    group: DisambiguationGroup,
    members: list[Subject],
    naming: SubjectNaming,
    particle_counts: dict[str, int],
) -> str:
    """Render a Wikipedia-style ``(disambiguation)`` note for a collision group.

    Lists every same-named subject that rendered, linking to each
    qualified note. ``aliases: [<bare name>]`` lets a bare
    ``[[Prometheus]]`` resolve here in Obsidian.
    """
    lines = [
        "---",
        "tags:",
        "  - particles/disambiguation",
        "aliases:",
        f"  - {group.base_name}",
        "---",
        "",
        f"# {disambiguation_name(group.base_name)}",
        "",
        f'"{group.base_name}" refers to multiple subjects:',
        "",
    ]
    for s in sorted(members, key=lambda m: naming.display_name(m)):
        slug = subject_slug(naming.display_name(s))
        qualifier = naming.qualifier_by_id.get(s.id, "")
        count = particle_counts.get(s.id, 0)
        noun = "particle" if count == 1 else "particles"
        suffix = f" — {qualifier} ({count} {noun})" if qualifier else f" — {count} {noun}"
        lines.append(f"- [[{slug}]]{suffix}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


async def export_vault(
    session: AsyncSession,
    output_dir: Path,
    min_particles: int = 0,
    min_links: int = 1,
    with_synthesis: bool = False,
    min_particle_confidence: float = 0.0,
    invalidate_stale_links: bool = False,
    include_non_asserted: bool = False,
) -> ObsidianSummary:
    """Export the particle store as an Obsidian vault.

    Creates one .md file per Subject plus a _index.md overview.
    Only subjects with at least min_particles particles and min_links graph
    links (outgoing + incoming combined) are written.
    Returns an :class:`ObsidianSummary`.

    When ``with_synthesis`` is True, each per-subject note
    additionally carries the LLM-synthesised prose article above its
    existing structural particle listing. The synthesis machinery lives
    in :mod:`particles.render.article_synthesis` and is shared with
    the WikiExporter; per-subject input-hash caching means an operator
    who runs both exports against the same store pays LLM cost once
    per Subject.

    ``min_particle_confidence``: drop particles whose
    ``effective_confidence`` falls below this threshold from every
    rendered note — coin-property tables, pivot-note callouts, generic
    subject-note audit trails, and the ``--with-synthesis`` splice all
    consume the post-filter list. Default 0.0 keeps every existing
    invocation backwards-compatible. The ``min_particles`` count check
    runs against the filtered set per the cross-exporter contract, so
    a subject with too few high-confidence particles is suppressed
    rather than rendered as a thin note.
    """
    from particles.corpus.store import get_entry_uri_map
    from particles.store.particle_store import get_particles_by_status
    from particles.store.subject_store import list_all_subjects

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the full subject set once and compute the
    # disambiguation map up front: note filenames, wikilink targets, and
    # the stale-link / known-names sets below all key on the
    # disambiguated display-name slug, so the map must exist before any
    # of them.
    subjects = await list_all_subjects(session)
    subject_map: dict[str, Subject] = {s.id: s for s in subjects}
    naming = build_subject_naming(subjects)

    def _slug_for(subject: Subject) -> str:
        return _subject_slug(naming.display_name(subject))

    # load the ACTIVE particle set now (it also feeds eff_conf + the
    # per-subject grouping below) and pick out the NARRATIVE particles up front,
    # so the stale-link block can evict their synthesis cache by narrative slug.
    from particles.config import get_config

    all_particles = await get_particles_by_status(session, Status.ACTIVE)
    # cap. 1: keep a document's rejected / deferred / counterfactual
    # prose out of the rendered vault unless the caller opts in.
    if not include_non_asserted:
        all_particles = [p for p in all_particles if not is_non_asserted(p.properties)]
    emit_narratives = with_synthesis and get_config().obsidian.emit_narrative_notes
    narrative_particles = (
        [p for p in all_particles if p.particle_type == ParticleType.NARRATIVE]
        if emit_narratives
        else []
    )
    narrative_naming = build_narrative_naming(narrative_particles)  # narrative id → leaf slug

    # precompute each emitted (gate-passing) narrative's ordered
    # constituents once — reused by the write loop — and the subject→narrative
    # membership map that drives the per-subject backlink section. A subject
    # participates in a narrative when one of the narrative's constituents is
    # about that subject.
    narrative_constituents: dict[str, list[Particle]] = {}
    subject_narratives: dict[str, list[Particle]] = defaultdict(list)
    if emit_narratives:
        from particles.operations.narrative import get_narrative_sequence

        gate = get_config().exporter_common.synthesis_min_particles
        for n in narrative_particles:
            seq = await get_narrative_sequence(session, n.id)
            if len(seq) < gate:
                continue
            narrative_constituents[n.id] = seq
            for sid in {sid for c in seq for sid in c.subject_ids}:
                subject_narratives[sid].append(n)

    # stale-wikilink invalidation. Runs BEFORE the synthesis
    # cache snapshot below so any note whose ``[[X]]`` references a
    # renamed subject loses its ``article_input_hash`` and regenerates
    # this run.
    stale_link_articles_invalidated = 0
    if invalidate_stale_links and with_synthesis:
        from particles.render.article_synthesis.cache import (
            invalidate_stale_link_articles,
        )
        from particles.store.synthesis_cache_store import evict_subject

        # Targets of emitted ``[[X]]`` links are display-name slugs (ADR
        # 0091), so the known set must contain those — plus the
        # ``(disambiguation)`` note names — or every disambiguated note
        # would be spuriously invalidated each run.
        known_names: set[str] = set()
        for s in subjects:
            known_names.add(s.canonical_name)
            known_names.update(s.aliases)
            known_names.add(naming.display_name(s))
            known_names.add(_slug_for(s))
        for g in naming.groups:
            known_names.add(disambiguation_name(g.base_name))
            known_names.add(_subject_slug(disambiguation_name(g.base_name)))
        # subject notes carry `[[Narratives/<slug>]]` backlinks, so the
        # narrative-note paths are valid link targets — add them or every subject
        # note with a backlink is spuriously invalidated each run.
        for nslug in narrative_naming.values():
            known_names.add(f"Narratives/{nslug}")
        invalidated_paths = invalidate_stale_link_articles(
            output_dir,
            known_names,
            hash_field="article_input_hash",
            recursive=True,
        )
        stale_link_articles_invalidated = len(invalidated_paths)
        # also drop the shared DB cache entries for any
        # subject whose on-disk note we just invalidated. Notes live in
        # nested subdirs (e.g. ``github.com/foo.md``); key on the path
        # relative to output_dir without its suffix so nested slugs match
        # (``path.stem`` alone would drop the ``github.com/`` directory).
        slug_to_subject_id = {_slug_for(s): s.id for s in subjects}
        # a narrative note's synthesis cache is keyed by the narrative
        # id (its synthetic subject id); map its `Narratives/<slug>` path so a
        # stale-link invalidation evicts it too and the next render is fresh.
        for nid, nslug in narrative_naming.items():
            slug_to_subject_id[f"Narratives/{nslug}"] = nid
        for path in invalidated_paths:
            rel = str(path.relative_to(output_dir).with_suffix(""))
            subject_id = slug_to_subject_id.get(rel)
            if subject_id is not None:
                await evict_subject(session, subject_id)
        if stale_link_articles_invalidated:
            log.info(
                "invalidated %d cached article(s) with stale wikilinks",
                stale_link_articles_invalidated,
            )

    # Snapshot every existing per-subject note's article_input_hash AND
    # full body so the synthesis splice can detect a cache-hit on the
    # prior body and backfill the DB cache from on-disk
    # content when the hash matches but the DB row is absent. The dict
    # maps note-filename → (input_hash, full_note_text).
    prior_synthesis_bodies: dict[str, tuple[str, str]] = {}
    if with_synthesis:
        for existing in output_dir.rglob("*.md"):
            try:
                text = existing.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            if not fm:
                continue
            cached = fm.get("article_input_hash")
            if isinstance(cached, str):
                # Relative to output_dir so nested github.com/foo.md works.
                prior_synthesis_bodies[str(existing.relative_to(output_dir))] = (cached, text)

    # Track every .md file this run intends to leave on disk so the
    # post-write prune pass below can remove only files that won't be
    # regenerated. Replaces the pre-0.42.4 blanket wipe that deleted
    # every .md *before* writing — robust against suppressed subjects
    # but destructive on interrupt and a constant source of Obsidian
    # file-watcher churn even when nothing changed semantically.
    written_paths: set[Path] = set()

    # entry_id → uri_r — used to render `> **Source:** [...](url)` on each particle
    entry_uri_map = await get_entry_uri_map(session)

    particles_by_subject: dict[str, list[Particle]] = defaultdict(list)
    for p in all_particles:
        for sid in p.subject_ids:
            if sid in subject_map:
                particles_by_subject[sid].append(p)

    particle_counts = {s.id: len(particles_by_subject[s.id]) for s in subjects}

    # Build effective confidence map: extractor trust weight + content age
    # decay + source trust — the same per-particle
    # computation the query path renders, so the vault's numbers match query.
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

    # Cross-exporter quality threshold. Drop particles below
    # the threshold from every downstream view: coin-property tables,
    # pivot-note audit trails, generic-note audit trails, and the
    # ``--with-synthesis`` splice (which reads the same dicts). The
    # filter runs after ``eff_conf`` is computed so each dropped
    # particle is judged by the effective number an operator's trust
    # policy actually produces.
    particles_dropped_below_threshold = 0
    if min_particle_confidence > 0.0:
        kept_particles: list[Particle] = []
        for p in all_particles:
            if eff_conf.get(p.id, p.confidence.value) < min_particle_confidence:
                particles_dropped_below_threshold += 1
                continue
            kept_particles.append(p)
        all_particles = kept_particles
        # Rebuild the per-subject map from the filtered list so every
        # rendered note (and the synthesis splice) sees only the
        # particles that cleared the threshold.
        particles_by_subject = defaultdict(list)
        for p in all_particles:
            for sid in p.subject_ids:
                if sid in subject_map:
                    particles_by_subject[sid].append(p)
        # Recompute counts so the eligibility set (which gates rendering
        # and the ``min_particles`` check) reflects the filtered set per
        # § 1.
        particle_counts = {s.id: len(particles_by_subject[s.id]) for s in subjects}

    files_written = 0

    # Write the synthetic "Coin" meta-node so Instance of: [[Coin]] resolves
    coin_meta = output_dir / "Coin.md"
    atomic_write_text(
        coin_meta,
        "---\ntags:\n  - particles/meta\n---\n# Coin\n\n"
        "Meta-node: all coin subjects link here via `Instance of`.\n",
    )
    written_paths.add(coin_meta.resolve())
    files_written += 1

    # Subjects eligible for export (pass min_particles). Used to suppress wikilinks
    # to subjects that will never have a file — prevents Obsidian phantom nodes.
    eligible_ids: set[str] = {
        s.id for s in subjects if not _is_category(s) and particle_counts[s.id] >= min_particles
    }

    # --- Pass 1: render all eligible notes into memory ---
    rendered: dict[str, str] = {}  # subject.id → note text
    for subject in subjects:
        if subject.id not in eligible_ids:
            continue
        ps = particles_by_subject[subject.id]
        ps.sort(key=lambda p: (p.properties is None, -p.confidence.value))
        cls = subject.subject_class
        if cls == "nmo:NumismaticObject":
            rendered[subject.id] = _render_coin_note(
                subject, ps, subject_map, eff_conf, eligible_ids, naming=naming
            )
        elif cls in _PIVOT_CLASSES:
            rendered[subject.id] = _render_pivot_note(
                subject, ps, subject_map, eff_conf, eligible_ids, entry_uri_map, naming=naming
            )
        else:
            rendered[subject.id] = _render_subject_note(
                subject, ps, subject_map, eff_conf, eligible_ids, entry_uri_map, naming=naming
            )
        # append a `## Narratives` backlink section listing the
        # narratives this subject's claims participate in (subject → narrative).
        if emit_narratives and subject_narratives.get(subject.id):
            section = _render_narratives_section(subject_narratives[subject.id], narrative_naming)
            if section:
                rendered[subject.id] = rendered[subject.id].rstrip() + "\n\n" + section

    # Build case-insensitive map: link target → number of distinct notes that reference it
    incoming_counts: dict[str, int] = {"coin": 1}
    for note_text in rendered.values():
        targets_in_note = {m.group(1).strip().lower() for m in _WIKI_LINK_RE.finditer(note_text)}
        for t in targets_in_note:
            incoming_counts[t] = incoming_counts.get(t, 0) + 1

    # If --with-synthesis is enabled, run a lint pre-pass once (same as
    # the wiki exporter) so per-article callouts can be spliced into the
    # synthesised body. semantic=False keeps it cheap; fix=False keeps
    # the export side-effect-free on the particle store.
    findings_by_particle: dict[str, list[LintFinding]] = defaultdict(list)
    findings_by_subject: dict[str, list[LintFinding]] = defaultdict(list)
    if with_synthesis and rendered:
        from particles.operations.lint import run_lint

        try:
            lint_report = await run_lint(session, fix=False, semantic=False)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Obsidian export: lint pre-pass failed (%s); continuing", exc)
        else:
            for f in lint_report.findings:
                if f.particle_id:
                    findings_by_particle[f.particle_id].append(f)
                if f.subject_id:
                    findings_by_subject[f.subject_id].append(f)

    # --- Pass 2: write subjects, filtering by min_links ---
    suppressed = 0
    synthesis_used = 0
    synthesis_failed = 0
    synthesis_cache_hits = 0
    synthesis_skipped = 0
    # Set once an account-fatal LLM error (billing / auth / quota) is hit:
    # the rest of the run skips synthesis (no point hammering the API) and
    # leaves those notes un-hashed so they retry on the next export.
    synthesis_aborted = False
    total_eligible = sum(1 for s in subjects if s.id in rendered)
    eligible_idx = 0
    for subject in subjects:
        if subject.id not in rendered:
            continue
        eligible_idx += 1
        note = rendered[subject.id]
        if min_links > 0:
            outgoing = len(_WIKI_LINK_RE.findall(note))
            incoming = incoming_counts.get(_slug_for(subject).lower(), 0)
            if outgoing + incoming < min_links:
                log.debug(
                    "Suppressing subject (links=%d): %s",
                    outgoing + incoming,
                    subject.canonical_name,
                )
                suppressed += 1
                continue

        if with_synthesis:
            slug = _slug_for(subject)
            ps_for_subject = particles_by_subject[subject.id]
            article_hash = compute_input_hash(ps_for_subject, subject)
            prior_entry = prior_synthesis_bodies.get(f"{slug}.md")
            prior_hash = prior_entry[0] if prior_entry is not None else None
            prior_body = prior_entry[1] if prior_entry is not None else None
            subject_findings = findings_by_subject.get(subject.id, []) + [
                f for p in ps_for_subject for f in findings_by_particle.get(p.id, [])
            ]

            if synthesis_aborted:
                # A prior subject hit an account-fatal LLM error; skip synthesis
                # and leave the structural note un-hashed so it retries next run.
                synthesis_failed += 1
            else:
                try:
                    note, syn_state = await _splice_synthesised_article(
                        note=note,
                        subject=subject,
                        particles=ps_for_subject,
                        subject_map=subject_map,
                        eligible_ids=eligible_ids,
                        eff_conf=eff_conf,
                        entry_uri_map=entry_uri_map,
                        article_hash=article_hash,
                        regenerate=prior_hash != article_hash,
                        lint_findings=subject_findings,
                        progress_prefix=(
                            f"[{eligible_idx}/{total_eligible}] {subject.canonical_name}"
                        ),
                        session=session,
                        prior_body=prior_body,
                        naming=naming,
                    )
                except SynthesisUnavailable as exc:
                    synthesis_aborted = True
                    synthesis_failed += 1
                    log.warning(
                        "Article synthesis unavailable (%s) — aborting synthesis for the "
                        "remaining %d subject(s); they keep their structural notes and "
                        "retry on the next export.",
                        exc,
                        total_eligible - eligible_idx,
                    )
                    # `note` stays the structural render (no article_input_hash),
                    # so the next run regenerates this subject.
                else:
                    if syn_state == "cache_hit":
                        synthesis_cache_hits += 1
                    elif syn_state == "synthesised":
                        synthesis_used += 1
                    elif syn_state == "fallback":
                        synthesis_failed += 1
                    elif syn_state == "skipped":
                        synthesis_skipped += 1
                    # else "no_op" — synthesis was attempted but the helper
                    # produced nothing usable; the unchanged Obsidian note is
                    # written below as if --with-synthesis were off.

            # lint callouts are a write-time layer, not cached
            # content. Apply the *current* findings to the note on every
            # path (fresh, cache-hit, fallback) so a finding's text — or a
            # finding appearing / resolving since the body was cached —
            # always reflects current lint state. Idempotent: strips any
            # prior (fenced or legacy) callout block before re-splicing.
            note = apply_lint_callouts(note, subject_findings)

        path = output_dir / f"{_slug_for(subject)}.md"
        if not is_within_directory(output_dir, path):
            # Defence-in-depth (F1): the slug derives from an untrusted,
            # LLM-extracted canonical_name. subject_slug already neutralises
            # traversal; this second gate skips any write that would still
            # escape the vault rather than clobber a file outside it.
            log.warning(
                "Skipping subject %r — export path escapes the vault: %s",
                subject.canonical_name,
                path,
            )
            continue
        atomic_write_text(path, note)
        written_paths.add(path.resolve())
        files_written += 1
        log.debug("Wrote %s (%s)", path.name, subject.subject_class or "generic")

    if suppressed:
        log.info("Suppressed %d isolated subjects (no links in or out)", suppressed)

    # one Wikipedia-style ``(disambiguation)`` note per collision
    # group, listing every same-named entity that actually rendered. The
    # ``aliases: [<bare name>]`` lets a bare ``[[Prometheus]]`` resolve to
    # this note in Obsidian.
    for group in naming.groups:
        members = [subject_map[mid] for mid in group.member_ids if mid in rendered]
        if len(members) < 2:
            continue  # fewer than two actually rendered — no ambiguity on disk
        disamb_note = _render_disambiguation_note(group, members, naming, particle_counts)
        disamb_path = output_dir / f"{_subject_slug(disambiguation_name(group.base_name))}.md"
        if not is_within_directory(output_dir, disamb_path):
            log.warning(
                "Skipping disambiguation note %r — export path escapes the vault: %s",
                group.base_name,
                disamb_path,
            )
            continue
        atomic_write_text(disamb_path, disamb_note)
        written_paths.add(disamb_path.resolve())
        files_written += 1

    # one note per ACTIVE NARRATIVE under Narratives/, rendered as
    # cited prose via the path. NARRATIVE particles are subject-less,
    # so these notes are purely additive to the vault. Gated by
    # emit_narratives (config) and a constituent-count floor.
    narrative_notes_written = 0
    if emit_narratives and narrative_constituents:
        narratives_dir = output_dir / "Narratives"
        narratives_dir.mkdir(parents=True, exist_ok=True)
        for narrative in narrative_particles:
            constituents = narrative_constituents.get(narrative.id)
            if constituents is None:
                continue  # gated out by the constituent-count floor (precomputed)
            slug = narrative_naming[narrative.id]
            article_hash = compute_input_hash(
                constituents, narrative_as_subject(narrative), ordered=True
            )
            prior = prior_synthesis_bodies.get(f"Narratives/{slug}.md")
            prior_hash = prior[0] if prior is not None else None
            prior_body = prior[1] if prior is not None else None
            if prior_hash == article_hash and prior_body is not None:
                note = prior_body  # unchanged since last export — reuse verbatim
            elif synthesis_aborted:
                continue  # account-fatal LLM error earlier; retry on next export
            else:
                try:
                    note, _state = await render_narrative_note(
                        narrative=narrative,
                        constituents=constituents,
                        subject_map=subject_map,
                        eligible_ids=eligible_ids,
                        eff_conf=eff_conf,
                        entry_uri_map=entry_uri_map,
                        naming=naming,
                        article_hash=article_hash,
                        session=session,
                    )
                except SynthesisUnavailable as exc:
                    synthesis_aborted = True
                    log.warning(
                        "Narrative synthesis unavailable (%s) — skipping remaining "
                        "narrative notes; they retry on the next export.",
                        exc,
                    )
                    continue
            path = narratives_dir / f"{slug}.md"
            if not is_within_directory(output_dir, path):
                log.warning(
                    "Skipping narrative note %r — export path escapes the vault: %s",
                    narrative.id,
                    path,
                )
                continue
            atomic_write_text(path, note)
            written_paths.add(path.resolve())
            narrative_notes_written += 1
            files_written += 1

    index_content = _render_index(subjects, particle_counts, naming)
    index_path = output_dir / "_index.md"
    atomic_write_text(index_path, index_content)
    written_paths.add(index_path.resolve())
    files_written += 1

    # Post-write prune (0.42.4): remove only .md files that this run did
    # NOT write. Replaces the pre-write blanket wipe — an interrupted
    # export now leaves the vault in a consistent state (every file is
    # either freshly written or unchanged from before) instead of empty.
    # Files outside the target set are subjects suppressed by min_links
    # / min_particles / threshold filters or renamed since the last run.
    from particles.render.markdown import prune_obsolete_markdown

    files_pruned = prune_obsolete_markdown(output_dir, written_paths, recursive=True)
    if files_pruned:
        log.info("Pruned %d obsolete .md file(s) from previous export", files_pruned)

    phantoms = sum(1 for c in particle_counts.values() if c == 0)
    return ObsidianSummary(
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
