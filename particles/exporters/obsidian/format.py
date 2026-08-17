"""Obsidian-specific Markdown text shaping helpers.

These helpers operate on already-rendered Markdown strings or compose
small Obsidian-flavoured fragments (``[[wikilinks]]``, ``> [!info]``
callouts, ``^p-…`` block-reference markers, ``[[#^id|ⁿ]]`` jumps).
They're separated from :mod:`particles.exporters.obsidian.vault` so the
heavy-weight note renderers don't have to share a file with the regex /
string-surgery primitives they call.

Nothing here is generic enough to live in
:mod:`particles.render.markdown` — that module is the
exporter-agnostic callout renderer used by the lint/query operations,
and these helpers all assume Obsidian-specific syntax.
"""

from __future__ import annotations

import re

from particles.core.schema import Particle, Subject
from particles.render.markdown import SubjectNaming

# Sequential citation markers shown in the body. Index 0 is a dummy so
# the table can be indexed by 1-based ordinal directly.
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

_OBS_FOOTNOTE_REF_RE = re.compile(r"\[\^p-([0-9a-fA-F]{8})\]")


# Per-particle audit-trail callout emitted by _render_subject_note and
# _render_pivot_note as Format-C entries. Each entry is a numbered
# heading followed by an info callout whose body holds the Related /
# Source / Extractor lines and a trailing ``^p-{hex}`` block marker.
# Detection is keyed on the callout's ``[!info] p-{hex} · confidence``
# title — lint findings and subject-level info callouts don't carry
# the ``p-{hex}`` token so they're left alone. Pattern:
#   ### N. {claim text}
#
#   > [!info] p-{short_id} · confidence X.XX
#   > **Related:** [[Subject]] …
#   > **Source:** [...](...)
#   > **Extractor:** name version on YYYY-MM-DD ^p-{short_id}
_PARTICLE_AUDIT_CALLOUT_RE = re.compile(
    r"^### \d+\.[^\n]*\n"  # numbered heading
    r"\n*"  # blank line(s) between heading and callout
    r"^> \[!info\] p-[0-9a-fA-F]{8}[^\n]*\n"  # info-callout title
    r"(?:^>.*\n)*"  # continuation lines (Related / Source / Extractor / ^p-… marker)
    r"\n*",  # consume trailing blank lines so back-to-back entries strip cleanly
    re.MULTILINE,
)


# Detector used on the skip-synthesis path: does the pre-rendered note
# already contain a Format-C per-particle audit trail? Generic and
# pivot templates emit them; the coin template doesn't (structured data
# goes to frontmatter, descriptive particles to sections). If the
# marker is absent we append a fresh audit trail so low-coverage coin
# subjects still show their particle in the body.
_AUDIT_CALLOUT_MARKER_RE = re.compile(
    r"^> \[!info\] p-[0-9a-fA-F]{8}\b",
    re.MULTILINE,
)


# Whitespace within a claim string (newlines + multi-space runs) is
# collapsed to single spaces before the claim is rendered into the
# reference heading. Heading lines can't span newlines and look bad
# with double-spaces.
_WHITESPACE_RUN_FOR_HEADING = re.compile(r"\s+")


# Pivot-class subjects (Material, Denomination, Issuer, …) are filtered
# out of "Related" wikilink lists alongside categories — we don't want
# every coin's References block to wikilink to the same handful of
# generic pivots. Duplicated from vault._PIVOT_CLASSES so this module
# doesn't import upward from vault; the canonical definition stays in
# vault.py because that's where note dispatch lives.
_PIVOT_CLASSES_FOR_RELATED = {
    "nmo:Material",
    "nmo:Denomination",
    "nmo:Issuer",
    "nmo:Authority",
    "nmo:ObjectType",
    "nmo:Mint",
}


def _wiki(name: str) -> str:
    return f"[[{name}]]"


def _is_category(subject: Subject) -> bool:
    return subject.canonical_name.lower().startswith(("category:", "category "))


def _provenance_label(p: Particle) -> str:
    """Best-effort short label for a particle's provenance source."""
    for ref in p.provenance:
        if ref.corpus_entry_id:
            return ref.corpus_entry_id[:8]
    return p.asserted_by


def _source_link_line(p: Particle, entry_uri_map: dict[str, str | None] | None) -> str | None:
    """Render a ``> **Source:** [label](url)`` line, or None when no URL is known."""
    if not entry_uri_map:
        return None
    for ref in p.provenance:
        if ref.corpus_entry_id and entry_uri_map.get(ref.corpus_entry_id):
            uri = entry_uri_map[ref.corpus_entry_id]
            assert uri is not None  # narrowed by the truthiness check above
            label = re.sub(r"^https?://", "", uri)
            if len(label) > 70:
                label = label[:69] + "…"
            return f"> **Source:** [{label}]({uri})"
    return None


def _to_superscript(n: int) -> str:
    """Render a positive integer as Unicode superscript digits.

    Used as the alias text in ``[[#^id|ⁿ]]`` so the body renders
    citations as compact, footnote-like markers rather than full
    block-reference text.
    """
    return "".join(_SUPERSCRIPT_DIGITS[int(d)] for d in str(n))


def _claim_for_heading(content: str) -> str:
    """Sanitise a particle's content for use as an H3 heading.

    The heading line cannot contain newlines, and a leading ``#`` would
    confuse the Markdown parser into making the heading a different
    level. Collapse internal whitespace and strip leading ``#``-runs.
    """
    cleaned = _WHITESPACE_RUN_FOR_HEADING.sub(" ", content).strip()
    return cleaned.lstrip("#").strip() or "(empty claim)"


def _render_obsidian_reference_entry(
    *,
    ordinal: int,
    particle: Particle,
    parent_subject: Subject,
    subject_map: dict[str, Subject],
    eligible_ids: set[str],
    eff_conf: dict[str, float],
    entry_uri_map: dict[str, str | None],
    naming: SubjectNaming | None = None,
) -> list[str]:
    """Render one cited particle as a Format-C References entry.

    Format::

        ### {ordinal}. {claim content}

        > [!info] p-{short_id} · confidence {eff_conf}
        > **Related:** [[Subject A]], [[Subject B]]
        > **Source:** [domain/path](full-url)
        > **Extractor:** {name} {version} on {YYYY-MM-DD} ^p-{short_id}

    "Related" excludes the parent subject (no self-link), categories,
    and pivot-class subjects (Material, Denomination, Issuer, Authority,
    ObjectType, Mint) — same filter as the Obsidian generic template.
    The Source and Related lines are skipped when no data is available.
    The trailing ``^p-{short_id}`` is the Obsidian block marker; it
    makes the whole callout the jump target for ``[[#^p-{short_id}]]``
    in the article body.
    """
    from particles.render.markdown import subject_slug

    sid = particle.id[:8]
    conf = eff_conf.get(particle.id, particle.confidence.value)
    lines: list[str] = [
        f"### {ordinal}. {_claim_for_heading(particle.content)}",
        "",
        f"> [!info] p-{sid} · confidence {conf:.2f}",
    ]

    # Related subjects — wikilinks to the *other* subjects this particle
    # touches. Mirrors the filter in _render_subject_note so the link
    # graph is consistent across the Obsidian note's audit trail and
    # the synthesised article's References section.
    others = [
        subject_map[other_id]
        for other_id in particle.subject_ids
        if other_id != parent_subject.id
        and other_id in subject_map
        and other_id in eligible_ids
        and not _is_category(subject_map[other_id])
        and subject_map[other_id].subject_class not in _PIVOT_CLASSES_FOR_RELATED
    ]
    if others:
        links = ", ".join(
            _wiki(subject_slug(naming.display_name(s) if naming is not None else s.canonical_name))
            for s in others
        )
        lines.append(f"> **Related:** {links}")

    # Source link — first corpus entry with a known URI.
    source_line = _source_link_line(particle, entry_uri_map)
    if source_line:
        lines.append(source_line)

    # Extractor + timestamp + block marker. The marker rides on the
    # last line of the callout so Obsidian treats the whole blockquote
    # as the link target; the literal ``^p-…`` text is hidden in
    # Reading View.
    ts = particle.asserted_at.date().isoformat() if particle.asserted_at else "?"
    if particle.extractor_ref:
        ext = f"{particle.extractor_ref.name} {particle.extractor_ref.version}"
        lines.append(f"> **Extractor:** {ext} on {ts} ^p-{sid}")
    else:
        # No extractor_ref → a direct assertion. Attribute it to the
        # asserting principal rather than a phantom "Extractor: ?".
        lines.append(f"> **Asserted by:** {particle.asserted_by or '?'} on {ts} ^p-{sid}")

    return lines


def _render_particle_audit_callouts(
    particles: list[Particle],
    *,
    parent_subject: Subject,
    subject_map: dict[str, Subject],
    eligible_ids: set[str],
    eff_conf: dict[str, float],
    entry_uri_map: dict[str, str | None] | None = None,
    naming: SubjectNaming | None = None,
) -> list[str]:
    """Render the per-particle audit trail in Format-C style.

    Each particle becomes one ``### N. {claim}`` heading + one
    ``> [!info] p-{hex} · confidence X.XX`` callout with the Related
    wikilinks + Source + Extractor lines and a trailing ``^p-{hex}``
    block marker. This is the same shape that
    :func:`_render_obsidian_reference_entry` produces for the
    synthesised References section — using one renderer keeps the
    operator's vault visually consistent regardless of whether
    synthesis succeeded, fell back, or was skipped for the subject.

    On the synthesis-success path, :func:`_strip_per_particle_callouts`
    removes these audit entries before the synthesised References
    (also Format C, but limited to cited particles) is appended, so
    the article doesn't show two sets of numbered references.

    Ordinals run 1..N over the ``particles`` list in the order given;
    the caller is responsible for stable ordering (the exporter sorts
    by ``(properties is None, -confidence)`` for coin notes, declaration
    order otherwise).

    Lives in :mod:`particles.exporters.obsidian.format` rather than
    ``vault`` so the synthesis splicer can use it without creating
    a ``vault <-> synthesis`` import cycle.
    """
    lines: list[str] = []
    for ordinal, p in enumerate(particles, start=1):
        lines.extend(
            _render_obsidian_reference_entry(
                ordinal=ordinal,
                particle=p,
                parent_subject=parent_subject,
                subject_map=subject_map,
                eligible_ids=eligible_ids,
                eff_conf=eff_conf,
                entry_uri_map=entry_uri_map or {},
                naming=naming,
            )
        )
        lines.append("")
    return lines


def _render_narratives_section(narratives: list[Particle], narrative_naming: dict[str, str]) -> str:
    """a ``## Narratives`` backlink section for a per-subject note.

    Lists the narratives whose constituents include this subject's claims as
    ``[[Narratives/<slug>|<label>]]`` wikilinks (the note paths),
    giving subject → narrative navigation. Narratives absent from
    ``narrative_naming`` (not emitted) are skipped so no link dangles. Returns
    ``""`` when there is nothing to link.
    """
    links: list[str] = []
    for n in sorted(narratives, key=lambda p: p.content.lower()):
        slug = narrative_naming.get(n.id)
        if slug is None:
            continue
        links.append(f"- [[Narratives/{slug}|{n.content}]]")
    if not links:
        return ""
    return "## Narratives\n\n" + "\n".join(links) + "\n"


def _to_obsidian_block_refs(
    prose: str,
    *,
    particles: list[Particle],
    parent_subject: Subject,
    subject_map: dict[str, Subject],
    eligible_ids: set[str],
    eff_conf: dict[str, float],
    entry_uri_map: dict[str, str | None],
    naming: SubjectNaming | None = None,
) -> tuple[str, str]:
    """Convert Markdown-footnote syntax to Obsidian block references.

    ``article_synthesis.py`` emits portable GFM/Pandoc footnotes so the
    wiki exporter's output is renderer-agnostic. Obsidian's own footnote
    parser, however, does not reliably produce clickable references in
    Live Preview; the user-facing failure is that ``[^p-xxxxxxxx]`` in
    the body appears as literal bracketed text instead of a jumpable
    superscript link. This function rewrites both halves of the article
    for Obsidian-native navigation:

      * **Body**: ``[^p-xxxxxxxx]`` → ``[[#^p-xxxxxxxx|ⁿ]]`` where ``n``
        is a 1-based sequential ordinal (first-appearance order). Repeat
        citations of the same particle reuse the same ordinal. The
        superscript alias keeps the inline rendering compact.
      * **References**: the original deterministic References block
        (just URLs + metadata) is discarded entirely and re-rendered
        from the underlying particle objects per :func:`_render_obsidian_reference_entry`
        so each entry can carry the claim content, Related wikilinks,
        source, confidence, and an Obsidian-native ``^p-xxxxxxxx``
        block marker.

    Returns ``(new_prose, new_references)``. ``new_references`` includes
    its own ``## References`` heading. If no citations appear in the
    prose, ``new_references`` is the empty string.
    """
    # Assign ordinals in order of first appearance in the prose. Repeat
    # citations of the same particle reuse the same ordinal so the body
    # shows ``¹ … ² … ¹`` rather than three different numbers.
    ordinal_map: dict[str, int] = {}
    for match in _OBS_FOOTNOTE_REF_RE.finditer(prose):
        sid = match.group(1)
        if sid not in ordinal_map:
            ordinal_map[sid] = len(ordinal_map) + 1

    def _ref_sub(match: re.Match[str]) -> str:
        sid = match.group(1)
        ordinal = ordinal_map.get(sid, len(ordinal_map) + 1)
        return f"[[#^p-{sid}|{_to_superscript(ordinal)}]]"

    new_prose = _OBS_FOOTNOTE_REF_RE.sub(_ref_sub, prose)

    if not ordinal_map:
        return new_prose, ""

    # Re-render the References section from the particle objects so each
    # entry can show the claim, Related wikilinks, source, etc. The
    # body's first-appearance ordering drives the numbering.
    particle_by_short = {p.id[:8]: p for p in particles}
    ref_lines: list[str] = ["## References", ""]
    for sid, ordinal in sorted(ordinal_map.items(), key=lambda kv: kv[1]):
        particle = particle_by_short.get(sid)
        if particle is None:
            # The body cited a particle that isn't in our input set —
            # should be impossible because Layer A already rejected
            # invented IDs, but tolerate it rather than KeyError.
            continue
        ref_lines.extend(
            _render_obsidian_reference_entry(
                ordinal=ordinal,
                particle=particle,
                parent_subject=parent_subject,
                subject_map=subject_map,
                eligible_ids=eligible_ids,
                eff_conf=eff_conf,
                entry_uri_map=entry_uri_map,
                naming=naming,
            )
        )
        ref_lines.append("")  # blank line between entries

    new_references = "\n".join(ref_lines).rstrip() + "\n"
    return new_prose, new_references


def _strip_per_particle_callouts(text: str) -> str:
    """Drop the per-particle audit-trail Format-C entries from a note body.

    When ``--with-synthesis`` is active, the synthesised article's
    References section lists every *cited* particle with the same
    Format-C entry shape (numbered heading + info callout + block
    marker — see :func:`_render_obsidian_reference_entry`). The
    structural audit trail emitted by ``_render_subject_note`` and
    ``_render_pivot_note`` shows *every* particle in the same shape,
    so leaving both in produces two numbered-references blocks.

    Strip the structural audit trail before splicing the synthesised
    References so the article has exactly one. The orphan ``---``
    separator that ``_render_subject_note`` writes above its audit
    trail is collapsed too, so the surviving section doesn't end
    with a hanging rule.

    This helper is only called on the synthesis-success path of
    :func:`_splice_synthesised_article`. The fallback / skip /
    cache-hit paths keep the audit trail because *it is the
    references* for those articles.
    """
    text = _PARTICLE_AUDIT_CALLOUT_RE.sub("", text)
    # Orphan `---` separator: was above the now-stripped audit trail
    # and is now followed only by a Markdown heading (typically the
    # synthesised ``## References`` that gets appended later) or
    # end-of-string. Collapse to a single blank line.
    text = re.sub(r"\n---[ \t]*\n+(?=\n*## |\Z)", "\n\n", text)
    # Collapse any blank-line runs the previous steps may have left.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _note_has_particle_audit_trail(note: str) -> bool:
    """True when the note's body already contains Format-C audit-trail entries."""
    return bool(_AUDIT_CALLOUT_MARKER_RE.search(note))


def _annotate_obsidian_frontmatter(
    note: str, *, article_input_hash: str, article_synthesis: str | None
) -> str:
    """Add or update ``article_input_hash`` + ``article_synthesis`` fields.

    The Obsidian note already has its own YAML frontmatter. We splice
    two extra lines in before the closing ``---`` so the next export's
    cache lookup can read ``article_input_hash`` without re-parsing the
    article body. ``article_synthesis`` records which path produced the
    article (``llm`` / ``structured-listing`` / None for cache hit).
    Both fields are no-ops if the note has no frontmatter.
    """
    m = re.match(r"\A---\n(.*?\n)---\n", note, flags=re.DOTALL)
    if not m:
        return note
    body = m.group(1)
    extras = f"article_input_hash: {article_input_hash}\n"
    if article_synthesis is not None:
        extras += f"article_synthesis: {article_synthesis}\n"
    return f"---\n{body}{extras}---\n" + note[m.end() :]


def _insert_synthesised_prose(*, note: str, prose: str, references: str) -> str:
    """Insert the synthesised prose between the Obsidian H1 and the rest.

    The Obsidian note structure is::

        ---
        frontmatter
        ---

        # Subject Title
        existing structural content (callouts, lists, etc.)

    With synthesis spliced in it becomes::

        ---
        frontmatter (now also carries article_input_hash + article_synthesis)
        ---

        # Subject Title

        {synthesised prose body}

        ## Source particles
        existing structural content

        ## References
        {synthesised References section}

    The ``## Source particles`` heading is inserted to label the
    existing Obsidian content for what it is — the per-particle
    audit trail — and to visually separate it from the synthesised
    prose above. If no H1 is found in the Obsidian note (unusual but
    not impossible), the synthesised body is prepended to the note
    body and a single H1 is inserted from the subject canonical name.
    """
    # Split off the frontmatter so we can manipulate the body without
    # touching the YAML block.
    fm_match = re.match(r"\A(---\n.*?\n---\n)", note, flags=re.DOTALL)
    if fm_match:
        head = fm_match.group(1)
        body = note[fm_match.end() :]
    else:
        head = ""
        body = note

    h1_match = re.search(r"^# .*\n", body, flags=re.MULTILINE)
    if not h1_match:
        # No H1 — splice prose at the top of the body.
        new_body = f"\n{prose}\n\n## Source particles\n\n{body.lstrip()}"
    else:
        h1_end = h1_match.end()
        existing_h1 = body[:h1_end]
        rest = body[h1_end:].lstrip()
        new_body = f"{existing_h1}\n{prose}\n\n## Source particles\n\n{rest}"

    if references:
        # Append references at the very end with a blank-line separator.
        new_body = new_body.rstrip() + f"\n\n{references}\n"

    return head + new_body
