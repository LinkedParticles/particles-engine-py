"""Logseq bullet-outline rendering primitives.

Logseq's signature on-disk format: every line is a bullet (``- ``)
with two-space indentation per level, and every block can carry
inline ``key:: value`` properties on subsequent indented lines.
Particle IDs become ``id:: <particle_id>`` so anywhere
``((<particle_id>))`` appears in the vault, Logseq's linked-
references view threads the citation.

This module is pure rendering — no DB access, no LLM calls. The
orchestrator in :mod:`particles.exporters.logseq.vault` walks
subjects and stitches blocks together via these primitives.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, TypeVar

from particles.render.markdown import sanitize_filename

if TYPE_CHECKING:
    from particles.core.schema import Particle, Subject

# Logseq's hierarchical-page separator. The canonical name
# ``Material/Aluminium`` becomes ``Material___Aluminium.md`` on
# disk; in the vault UI Logseq displays it as a parent-child
# hierarchy.
_LOGSEQ_HIERARCHY_SEPARATOR = "___"

# Logseq displays underscores as spaces in the page title; the
# filesystem name uses underscores so the renderer round-trips
# cleanly.
_SPACE_REPLACEMENT = "_"

# Two-space indent per nesting level — matches Logseq's parse
# behaviour. Anything else risks Logseq treating a re-indented
# block as a new block (losing the block-reference graph).
_INDENT = "  "


def logseq_slug(canonical_name: str) -> str:
    """Map a canonical subject name to its Logseq page filename (no extension).

    Conventions (per Logseq's documented filename mapping):

    * Spaces → underscores (``5 Pfennigs`` → ``5_Pfennigs``)
    * ``/`` → ``___`` (triple underscore) for hierarchical pages
      (``Material/Aluminium`` → ``Material___Aluminium``)
    * Filesystem-invalid characters dashed via the shared sanitiser

    Reddit / GitHub conventions from the Obsidian slug helper are NOT
    applied — Logseq doesn't use nested-directory layouts; the whole
    vault is flat under ``pages/``.
    """
    # Map ``/`` → ``___`` BEFORE the invalid-char sanitiser; the
    # shared sanitiser treats ``/`` as filesystem-invalid and would
    # otherwise dash it out, defeating the hierarchy mapping. Spaces
    # → underscores happens last because the sanitiser preserves
    # them.
    name = canonical_name.replace("/", _LOGSEQ_HIERARCHY_SEPARATOR)
    name = sanitize_filename(name)
    name = name.replace(" ", _SPACE_REPLACEMENT)
    return name


_T = TypeVar("_T")


def _flag_first(items: Iterable[_T]) -> Iterator[tuple[bool, _T]]:
    """Yield ``(is_first, item)`` so a caller can act only on the first element."""
    first = True
    for item in items:
        yield first, item
        first = False


def render_inline_tags(tags: list[str] | None) -> str:
    """Render a particle's taxonomy tags as inline Logseq ``#tag`` tokens.

    Returns a leading-space-prefixed string (e.g. ``" #ml/optimizers #cold-war"``)
    suitable for appending to a block's content line, or ``""`` when the
    particle carries no tags. A tag path's ``/`` is preserved — Logseq reads
    it as tag hierarchy. Tags containing spaces are wrapped as ``#[[tag]]``
    because a bare ``#`` token terminates at the first whitespace.
    """
    if not tags:
        return ""
    rendered = [f"#[[{t}]]" if " " in t else f"#{t}" for t in tags]
    return " " + " ".join(rendered)


def render_block(content: str, *, depth: int = 0) -> str:
    """Render a single block: ``- <content>`` at the requested depth.

    The content is emitted verbatim — no escaping. The orchestrator
    is responsible for not feeding raw newlines into a block (split
    multi-line content into multiple blocks instead).
    """
    return f"{_INDENT * depth}- {content}"


def render_property(key: str, value: str, *, depth: int = 0) -> str:
    """Render a ``key:: value`` property line at the requested depth.

    Per Logseq's grammar, properties live on indented continuation
    lines of their parent block (depth + 1 conceptually). Callers pass
    the *parent block's* depth and we add the one-level indent.
    """
    return f"{_INDENT * (depth + 1)}{key}:: {value}"


def render_subject_page(
    subject: Subject,
    particles: list[Particle],
    *,
    eligible_ids: set[str],
    subject_map: dict[str, Subject],
    eff_conf: dict[str, float],
    display_name: str | None = None,
) -> str:
    """Render one Subject's Logseq page as a bullet outline.

    Structure:

    * H1 block: ``# <canonical_name>`` (page title — Logseq treats
      the first ``#`` heading as the page header).
    * Subject metadata properties (subject_class, aliases,
      external_ids) on the H1 block.
    * ``## Properties`` parent block — one child block per
      structured-particle entry, each with ``id:: <particle_id>``,
      ``confidence:: <float>``, and the property as
      ``<key>:: <value>``.
    * ``## Description`` parent block — one child block per
      descriptive-particle entry (each with ``id::`` +
      ``confidence::``).

    Wikilinks to subjects not in ``eligible_ids`` (suppressed
    upstream by the export filters) get replaced with their bare
    name so Logseq doesn't render a dead page link.
    """
    lines: list[str] = []

    # H1 block + subject-level properties as inline ``key:: value``
    # properties on the H1's continuation indent. ``display_name`` carries
    # the disambiguation qualifier when the canonical name
    # collides with another subject's.
    lines.append(render_block(f"# {display_name or subject.canonical_name}"))
    if subject.subject_class:
        lines.append(render_property("type", subject.subject_class, depth=0))
    for alias in subject.aliases:
        lines.append(render_property("alias", alias, depth=0))
    for ref in subject.external_ids:
        lines.append(render_property("external-id", f"{ref.namespace}:{ref.id}", depth=0))

    structured = [p for p in particles if p.properties]
    descriptive = [p for p in particles if not p.properties]

    if structured:
        lines.append(render_block("## Properties"))
        for p in structured:
            # Taxonomy tags ride inline on the particle's first
            # property block so Logseq's tag-search indexes them once per
            # particle rather than once per property key.
            for first, (key, value) in _flag_first((p.properties or {}).items()):
                rendered_value = _format_property_value(value, eligible_ids=eligible_ids)
                tag_suffix = render_inline_tags(p.tags) if first else ""
                lines.append(render_block(f"{key}:: {rendered_value}{tag_suffix}", depth=1))
                lines.append(render_property("id", p.id, depth=1))
                lines.append(
                    render_property(
                        "confidence", f"{eff_conf.get(p.id, p.confidence.value):.2f}", depth=1
                    )
                )

    if descriptive:
        lines.append(render_block("## Description"))
        for p in descriptive:
            # The particle content might contain raw newlines — collapse
            # to spaces so the block stays a single Logseq line.
            content = _suppress_dead_wikilinks(p.content, eligible_ids=eligible_ids)
            content = content.replace("\n", " ").strip()
            lines.append(render_block(f"{content}{render_inline_tags(p.tags)}", depth=1))
            lines.append(render_property("id", p.id, depth=1))
            lines.append(
                render_property(
                    "confidence", f"{eff_conf.get(p.id, p.confidence.value):.2f}", depth=1
                )
            )

    return "\n".join(lines) + "\n"


def render_narratives_block(narratives: list[Particle], narrative_naming: dict[str, str]) -> str:
    """A ``## Narratives`` backlink block for a per-subject page.

    Lists the narratives whose constituents include this subject's claims as
    ``[label]([[Narratives/<slug>]])`` links — Logseq's aliased-page-link
    syntax, since Logseq has no ``[[page|label]]`` pipe form. The link target
    is the same ``Narratives/`` page namespace the narrative pages
    live in, so subject → narrative navigation works in the graph view.
    Narratives absent from ``narrative_naming`` (gated out, hence not emitted)
    are skipped so no link dangles. Returns ``""`` when there is nothing to
    link.
    """
    links: list[str] = []
    for n in sorted(narratives, key=lambda p: p.content.lower()):
        slug = narrative_naming.get(n.id)
        if slug is None:
            continue
        # Labels are sentences; keep them on one line — Logseq treats a
        # newline inside a block as a block boundary.
        label = " ".join(n.content.split())
        links.append(f"{_INDENT}- [{label}]([[Narratives/{slug}]])")
    if not links:
        return ""
    return "- ## Narratives\n" + "\n".join(links) + "\n"


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def _suppress_dead_wikilinks(text: str, *, eligible_ids: set[str]) -> str:
    """Replace ``[[X]]`` with bare ``X`` when ``X`` isn't a known subject name.

    Phantom-link suppression mirrors the Obsidian exporter's
    behaviour — particles that mention a subject upstream-filtered
    out should not render as a clickable dead link. ``eligible_ids``
    here is the set of *canonical names* (not subject IDs) that
    survived the filter pass; the orchestrator builds it once and
    passes it through.
    """

    def _replace(m: re.Match[str]) -> str:
        target = m.group(1).strip()
        if target in eligible_ids:
            return m.group(0)
        return target

    return _WIKILINK_RE.sub(_replace, text)


def _format_property_value(value: object, *, eligible_ids: set[str]) -> str:
    """Render a structured-property value for the bullet block.

    Strings get the dead-wikilink-suppression pass; lists are
    comma-joined (with each element passed through). Other types
    are stringified. The result MUST be a single line — Logseq's
    block parser treats newlines as block boundaries.
    """
    if isinstance(value, list):
        return ", ".join(_format_property_value(v, eligible_ids=eligible_ids) for v in value)
    if isinstance(value, str):
        return _suppress_dead_wikilinks(value, eligible_ids=eligible_ids).replace("\n", " ")
    return str(value)
