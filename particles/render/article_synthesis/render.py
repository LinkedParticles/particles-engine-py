"""Article rendering — frontmatter, structured-listing, synthesised-article.

This is the largest submodule of the package because it owns three
related but distinct render paths:

* ``_render_frontmatter`` — the YAML frontmatter every article opens
  with.
* ``render_structured_listing`` — the deterministic LLM-less fallback:
  one bullet per particle plus a References section. Used when the
  operator has no API key, when synthesis fails Layer A / Layer B in
  both attempts, or when the operator explicitly asks for the
  deterministic path. Doubles as the "minimum viable article" for any
  exporter that wants citation-grade output without paying for an LLM
  call.
* ``render_synthesised_article`` — wraps an LLM-produced body in
  frontmatter + a deterministic References section. The body is
  callout-free: lint callouts are applied at write time by
  ``apply_lint_callouts``, not baked into the cached body.
* ``apply_lint_callouts`` / ``strip_lint_callouts`` — the write-time
  lint-callout layer. Exporters call these so a finding's
  text — or a finding appearing / resolving since the body was cached —
  surfaces on the next export without regenerating cached prose.

Also home to the synthesis prompt constants
(``_SYNTHESIS_PROMPT_STANDARD`` / ``_SYNTHESIS_PROMPT_STRICT`` /
``_SYNTHESIS_PROMPT_STRICT_LAYER_B``) and the helpers that interpolate
particles + misalignments into them. The orchestrator
(:mod:`orchestrate`) calls :func:`_build_synthesis_prompt` to produce
each attempt's user message.

This module is filesystem-blind — it returns Markdown strings; the
exporter writes them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import yaml

from particles.core.schema import Particle
from particles.render.article_synthesis.cache import _FRONTMATTER_RE
from particles.render.article_synthesis.layer_a import _short_id
from particles.render.article_synthesis.topic import SynthesisTopic

# Frontmatter tag every generated article carries. Used by
# Obsidian property indexing and by future tooling that wants to find the
# subset of vault files that originated from this exporter.
_ARTICLE_TAG = "particles/article"


# Lint finding types that translate to inline article-level callouts.
# Findings whose type is *not* in this map are recorded only in the
# frontmatter ``contradictions`` count when applicable; they are not
# surfaced as prose-disturbing callouts.
_LINT_CALLOUT_TYPES: dict[str, str] = {
    "CONTRADICTION": "danger",  # opposing claims from different sources
    "RETRACTION_CASCADE": "warning",
    "STALENESS": "warning",  # PROVENANCE_STALE
    "LOW_COVERAGE_SUBJECT": "info",
    "WIKIDATA_LINK_MISMATCH": "warning",
}


def _taxonomy_tags(particles: list[Particle]) -> list[str]:
    """Sorted, deduped union of taxonomy tags across an article's particles."""
    seen: set[str] = set()
    for p in particles:
        if p.tags:
            seen.update(p.tags)
    return sorted(seen)


def _count_contradictions(findings: list[Any] | None) -> int:
    """Number of CONTRADICTION findings in the list (None-safe)."""
    if not findings:
        return 0
    return sum(1 for f in findings if getattr(f, "finding_type", "") == "CONTRADICTION")


def _render_lint_callouts(findings: list[Any]) -> str:
    """Render a stack of `> [!type]` callouts for the article-relevant findings.

    Findings whose type is not in ``_LINT_CALLOUT_TYPES`` are skipped so
    the article does not get cluttered with INFO-level chatter that the
    operator would scan past. CONTRADICTION findings render first (most
    actionable), then warnings, then info.
    """
    by_severity: dict[str, list[str]] = {"danger": [], "warning": [], "info": []}
    for f in findings:
        ft = getattr(f, "finding_type", "")
        callout = _LINT_CALLOUT_TYPES.get(ft)
        if callout is None:
            continue
        detail = str(getattr(f, "detail", "")).strip() or ft
        action = getattr(f, "recommended_action", None)
        body = detail if not action else f"{detail}\n> {action}"
        by_severity[callout].append(
            f"> [!{callout}] {ft}\n> {body.replace(chr(10), chr(10) + '> ')}\n"
        )
    out: list[str] = []
    for sev in ("danger", "warning", "info"):
        out.extend(by_severity[sev])
    return "\n".join(out)


def _render_frontmatter(
    *,
    particle_count: int,
    input_hash: str,
    sources: int,
    contradictions: int,
    synthesis: str,
    layer_a_passed: bool | None,
    layer_b_passed: bool | None,
    layer_b_unrelated_count: int | None = None,
    layer_b_contradicts_count: int | None = None,
    min_particle_confidence: float | None = None,
    dropped_below_threshold: int | None = None,
    taxonomy_tags: list[str] | None = None,
) -> str:
    """Emit the YAML frontmatter every article opens with.

    The ``synthesis`` field records which path produced the body:
    ``"structured-listing"`` (no LLM, deterministic) or ``"llm"`` (synthesis
    prompt). Validation-layer fields are tri-state: ``true`` / ``false`` /
    ``null`` (the latter when the layer was not applicable for this run).

    ``layer_b_unrelated_count`` / ``layer_b_contradicts_count``:
    when present, surface the trichotomy verdict counts the judge saw.
    Omitted (None) when the layer wasn't run or wasn't applicable; an
    operator auditing the judge's distribution per article can grep
    these fields across the vault to inform tolerance tuning.

    ``min_particle_confidence`` / ``dropped_below_threshold``:
    when present, record the cross-exporter quality threshold the article
    was built under and how many particles were dropped because they fell
    below it. Both fields are optional wiki: when the
    operator did not opt in to the filter, the frontmatter parser must
    tolerate their absence.

    ``taxonomy_tags``: operator-curated taxonomy tags carried by
    the article's particles are appended to the ``tags`` list after the
    ``particles/article`` marker so they propagate to the wiki export the
    same way they do for Obsidian / Logseq.
    """
    fm: dict[str, Any] = {
        "particle_count": particle_count,
        "generated_at": datetime.now(UTC).isoformat(),
        "input_hash": input_hash,
        "sources": sources,
        "contradictions": contradictions,
        "synthesis": synthesis,
        "layer_a_passed": layer_a_passed,
        "layer_b_passed": layer_b_passed,
        "tags": [_ARTICLE_TAG, *(taxonomy_tags or [])],
    }
    if layer_b_unrelated_count is not None:
        fm["layer_b_unrelated_count"] = layer_b_unrelated_count
    if layer_b_contradicts_count is not None:
        fm["layer_b_contradicts_count"] = layer_b_contradicts_count
    if min_particle_confidence is not None:
        fm["min_particle_confidence"] = min_particle_confidence
    if dropped_below_threshold is not None:
        fm["dropped_below_threshold"] = dropped_below_threshold
    body = yaml.safe_dump(fm, sort_keys=True, default_flow_style=False).strip()
    return f"---\n{body}\n---\n\n"


# ---------------------------------------------------------------------------
# Structured-listing render — the synthesis-failure fallback
# ---------------------------------------------------------------------------


def render_structured_listing(
    subject: SynthesisTopic,
    particles: list[Particle],
    effective_confidences: dict[str, float],
    *,
    input_hash: str,
    lint_findings: list[Any] | None = None,
    min_particle_confidence: float | None = None,
    dropped_below_threshold: int | None = None,
) -> str:
    """Render a deterministic per-subject article from the particle list.

    No LLM. Each particle becomes one bullet with its content, effective
    confidence, and the ``[^p-xxxxxxxx]`` footnote reference. The
    references section at the end lists each particle's source URL and
    extractor identity. This doubles as the synthesis-failure fallback:
    when ``render_article`` cannot get a Layer-A-clean LLM body, it
    emits this rendering so the operator still gets a fully-cited
    article.

    The fallback is intentionally usable on its own. An operator who has
    no ANTHROPIC_API_KEY, or who is offline, still gets a per-subject
    digest with full citations.

    ``lint_findings``: if provided, each finding's ``detail`` is
    surfaced as a ``> [!warning]`` callout at the top of the article and
    the ``contradictions`` frontmatter field is set to the count of
    ``CONTRADICTION`` findings.
    """
    # Count unique source corpus entries — surfaces in the frontmatter.
    sources = {
        ref.corpus_entry_id for p in particles for ref in p.provenance if ref.corpus_entry_id
    }
    contradictions = _count_contradictions(lint_findings)

    head = _render_frontmatter(
        particle_count=len(particles),
        input_hash=input_hash,
        sources=len(sources),
        contradictions=contradictions,
        synthesis="structured-listing",
        layer_a_passed=None,  # not applicable — no LLM output to check
        layer_b_passed=None,
        min_particle_confidence=min_particle_confidence,
        dropped_below_threshold=dropped_below_threshold,
        taxonomy_tags=_taxonomy_tags(particles),
    )

    lines: list[str] = [head, f"# {subject.canonical_name}\n"]
    if subject.description:
        lines.append(f"\n_{subject.description}_\n")

    # lint callouts are applied fresh at write time, not baked
    # into the cached listing. ``lint_findings`` still drives the
    # ``contradictions`` frontmatter count above.

    lines.append(
        "\n> [!note] Structured-listing render\n"
        "> This article was rendered without LLM synthesis — each ACTIVE\n"
        "> particle is listed verbatim with its citation. This is the\n"
        "> fallback path; the LLM-synthesised body was either unavailable\n"
        "> (no API key) or failed citation validation.\n"
    )

    lines.append("\n## Claims\n\n")
    for p in particles:
        eff = effective_confidences.get(p.id, p.confidence.value)
        lines.append(f"- {p.content} [^p-{_short_id(p.id)}] _(effective confidence {eff:.2f})_\n")

    lines.append("\n## References\n\n")
    for p in particles:
        # First SOURCE-type provenance entry is the corpus entry the claim
        # originated from; later REVISIT entries are downstream snapshots.
        first_source = next((r for r in p.provenance if r.corpus_entry_id is not None), None)
        # Extractor-produced claims are attributed to the extractor; a direct
        # assertion (no extractor_ref) to its asserting principal —
        # never a phantom "extractor: ?".
        attribution = (
            f"extractor: {p.extractor_ref.name} {p.extractor_ref.version}"
            if p.extractor_ref
            else f"asserted by: {p.asserted_by or '?'}"
        )
        ts = p.asserted_at.date().isoformat() if p.asserted_at else "?"
        corpus_id = first_source.corpus_entry_id if first_source else "?"
        sid = _short_id(p.id)
        # Anchor heading + footnote definition. The heading lets readers
        # navigate the References section as a per-particle TOC and
        # enables cross-article wikilinks of the form
        # ``[[Subject#p-abc12345]]``. The footnote definition is what
        # the body's ``[^p-abc12345]`` syntax resolves to in
        # markdown-footnote-aware renderers.
        lines.append(
            f"### p-{sid}\n\n"
            f"[^p-{sid}]: corpus entry `{corpus_id}` "
            f"(confidence: {p.confidence.value:.2f}, {attribution}, "
            f"asserted {ts})\n\n"
        )

    return "".join(lines)


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

# Two prompt variants. The *standard* prompt is used on the first attempt;
# the *strict* prompt is used on the retry after a Layer-A failure. The
# only operational difference is the strict prompt's far blunter framing
# of the citation constraint — making the LLM more conservative about
# inventing IDs after it has already been caught doing so.

_SYNTHESIS_PROMPT_STANDARD = """\
{framing}You are writing one encyclopedic article about a single subject, based ONLY
on the structured particle list below. Every claim in your article must be
cited back to a particle with a Markdown footnote in the form `[^p-xxxxxxxx]`
where `xxxxxxxx` is the eight-character particle ID prefix shown on each
particle below. Do not invent claims; do not paraphrase beyond the source
particles' content; do not use any external knowledge.
{direction}
Modality handling — render each particle in the voice of its modality:
- FALSIFIABLE: an observer-independent fact. State it as fact (per the
  confidence handling below).
- EXPERIENTIAL: the author's inner state / lived experience. Render it as
  experience ("the author felt …", "found it …"), never as a hedged fact.
- EVALUATIVE: the author's opinion or value judgement. Render it as opinion
  ("in the author's view …", "found X tedious"), not as objective fact.
- CONSTITUTIVE: a definition or rule the source establishes. State it as such.

Confidence handling (FALSIFIABLE / CONSTITUTIVE only — for EXPERIENTIAL and
EVALUATIVE, effective_confidence reflects how clearly the feeling/opinion is
expressed, not how likely it is true, so NEVER apply a truth-hedge to them):
- Claims with effective_confidence below 0.5 MUST be hedged
  ("according to one source", "one report claims", "is reportedly").
- Claims at or above 0.9 may be stated declaratively.
- Claims between 0.5 and 0.9 should be stated declaratively with a
  citation, but without an emphatic adverb.

Cross-references:
- When a claim mentions another subject's name, render it as an
  Obsidian-style `[[Subject Name]]` wikilink so the resulting vault is
  graph-navigable.

Style:
- Encyclopedic tone (third person, neutral, factual).
- One-paragraph introduction, then short topical sections with `## `
  headings. No more than 600 words.
- Do NOT write a References section — that is appended deterministically
  by the exporter after your output.
- Emit footnote *references* only — i.e. `[^p-xxxxxxxx]` inline. Do NOT
  emit any footnote *definition* lines of the form `[^p-xxxxxxxx]: …`.
  The exporter appends a single canonical definition per cited particle
  in the References section. Emitting your own definition produces a
  duplicate, and markdown footnote parsers silently drop the link.

Output: the article body in Markdown, starting with the `# {title}` H1
heading. No frontmatter; no preamble before the H1; no closing remarks
after the last paragraph.

----
SUBJECT: {subject}
PARTICLE LIST (id_prefix, modality, effective_confidence, content):
{particles}
----"""


# Flowing-prose first-attempt variant. Mirrors
# ``_SYNTHESIS_PROMPT_STANDARD`` exactly — same placeholders, same modality /
# confidence / cross-reference / citation rules — except the Style block's
# "short topical sections with ``## `` headings" instruction is replaced with
# "one continuous passage of connected prose under the section title — no
# sub-headings". Selected on the first attempt when ``flowing`` is set; strict
# retries reuse the citation-focused prompts (heading suppression is a
# first-attempt presentation concern).
_SYNTHESIS_PROMPT_FLOWING = """\
{framing}You are writing one encyclopedic article about a single subject, based ONLY
on the structured particle list below. Every claim in your article must be
cited back to a particle with a Markdown footnote in the form `[^p-xxxxxxxx]`
where `xxxxxxxx` is the eight-character particle ID prefix shown on each
particle below. Do not invent claims; do not paraphrase beyond the source
particles' content; do not use any external knowledge.
{direction}
Modality handling — render each particle in the voice of its modality:
- FALSIFIABLE: an observer-independent fact. State it as fact (per the
  confidence handling below).
- EXPERIENTIAL: the author's inner state / lived experience. Render it as
  experience ("the author felt …", "found it …"), never as a hedged fact.
- EVALUATIVE: the author's opinion or value judgement. Render it as opinion
  ("in the author's view …", "found X tedious"), not as objective fact.
- CONSTITUTIVE: a definition or rule the source establishes. State it as such.

Confidence handling (FALSIFIABLE / CONSTITUTIVE only — for EXPERIENTIAL and
EVALUATIVE, effective_confidence reflects how clearly the feeling/opinion is
expressed, not how likely it is true, so NEVER apply a truth-hedge to them):
- Claims with effective_confidence below 0.5 MUST be hedged
  ("according to one source", "one report claims", "is reportedly").
- Claims at or above 0.9 may be stated declaratively.
- Claims between 0.5 and 0.9 should be stated declaratively with a
  citation, but without an emphatic adverb.

Cross-references:
- When a claim mentions another subject's name, render it as an
  Obsidian-style `[[Subject Name]]` wikilink so the resulting vault is
  graph-navigable.

Style:
- Encyclopedic tone (third person, neutral, factual).
- One continuous passage of connected prose under the section title — no
  sub-headings, no bullet lists. No more than 600 words.
- Do NOT write a References section — that is appended deterministically
  by the exporter after your output.
- Emit footnote *references* only — i.e. `[^p-xxxxxxxx]` inline. Do NOT
  emit any footnote *definition* lines of the form `[^p-xxxxxxxx]: …`.
  The exporter appends a single canonical definition per cited particle
  in the References section. Emitting your own definition produces a
  duplicate, and markdown footnote parsers silently drop the link.

Output: the article body in Markdown, starting with the `# {title}` H1
heading. No frontmatter; no preamble before the H1; no closing remarks
after the last paragraph.

----
SUBJECT: {subject}
PARTICLE LIST (id_prefix, modality, effective_confidence, content):
{particles}
----"""


_SYNTHESIS_PROMPT_STRICT = """\
PRIOR ATTEMPT WAS REJECTED. One or more of the following went wrong:

  1. You invented citation IDs that do not appear in the particle list.
  2. You wrote paragraphs that contain no citation — i.e. claims
     unbacked by any particle ("padding with general knowledge").
  3. You cited a particle but the claim's meaning was not derivable
     from that particle's content.

Rewrite the article strictly under these rules:

* **Every paragraph in the body must contain at least one
  `[^p-xxxxxxxx]` citation.** No paragraph of unsourced prose, no
  matter how plausible. A one-paragraph intro that paraphrases a
  cited claim's content is acceptable; an intro that draws on your
  general knowledge is not.
* **Every `[^p-xxxxxxxx]` must match a prefix listed below.** Do not
  truncate, transpose, or invent characters.
* **Use only the particles below.** If you cannot back a claim with a
  listed particle, omit the claim. Shorter is better than padded.
  A 3-sentence article citing 3 particles is preferable to a
  6-paragraph essay where only one paragraph is cited.

Emit footnote *references* only (`[^p-xxxxxxxx]` inline). Do NOT emit any
`[^p-xxxxxxxx]: …` *definition* lines — the exporter appends those.

Style: short encyclopedic article, third person, under 400 words.
Start with `# {title}`. No frontmatter; no closing References section.

----
SUBJECT: {subject}
ALLOWED CITATION PREFIXES: {allowed_ids}
PARTICLE LIST (id_prefix, modality, effective_confidence, content):
{particles}
----"""


_SYNTHESIS_PROMPT_STRICT_LAYER_B = """\
PRIOR ATTEMPT FAILED SEMANTIC-ALIGNMENT VALIDATION.

A judge reviewed your article and ruled that the following
(claim, citation) pairs were misaligned — the cited particle either
contradicts the claim, or the particle isn't a necessary input to the
claim and the citation looks ornamental.

**Fix priority — try these in order; drop claims only as a last resort:**

  1. REWRITE the claim so it's directly supported by the cited
     particle's content. Most misalignments fix this way: the
     particle does support *some* claim, just not the one as written.
  2. REPLACE the citation with a different particle from the list
     that actually supports the claim as written.
  3. DROP the claim ONLY if no particle in the list supports any
     version of it. This is a last resort — almost every misalignment
     above can be repaired by option 1 or 2.

**The retried article must still be cited.** An article that drops
so many claims it ends up with zero `[^p-xxxxxxxx]` citations is a
failure, not a fix. Every content paragraph must cite at least one
particle. If you genuinely cannot back enough claims to fill an
article, write a shorter article — 2-3 sentences citing 2-3 particles
is preferable to several paragraphs of uncited prose.

Do NOT cite a particle that contradicts the claim. Do NOT cite a
particle whose content is unrelated to the claim's meaning.

Emit footnote *references* only (`[^p-xxxxxxxx]` inline). Do NOT emit any
`[^p-xxxxxxxx]: …` *definition* lines — the exporter appends those.

Style: short encyclopedic article, third person, under 400 words.
Start with `# {title}`. No frontmatter; no closing References section.

MISALIGNED PAIRS FROM PRIOR ATTEMPT:
{misalignments}

----
SUBJECT: {subject}
ALLOWED CITATION PREFIXES: {allowed_ids}
PARTICLE LIST (id_prefix, modality, effective_confidence, content):
{particles}
----"""


_SYNTHESIS_PROMPT_NARRATIVE = """\
You are rendering ONE narrative as connected prose, based ONLY on the ordered
particle list below. The particles are the narrative's constituents IN ORDER —
each is one claim, and together they form a single connected account (a journal
entry, a memory, a sequence of events). Tell it as one flowing piece of prose
that FOLLOWS THE GIVEN ORDER from the first particle to the last. Do not reorder,
do not invent claims, do not add external knowledge, do not paraphrase beyond the
particles' content.

Every claim in your prose must be cited back to the particle it came from with a
Markdown footnote `[^p-xxxxxxxx]`, where `xxxxxxxx` is the eight-character ID
prefix shown on each particle below.

Modality handling — a journal is mostly feelings and opinions; render each
particle in the voice of its modality, NOT as a uniform list of facts:
- EXPERIENTIAL: the author's inner state / lived experience — render as
  experience ("the author felt …", "found it …"), never as a hedged fact.
- EVALUATIVE: the author's opinion or value judgement — render as opinion
  ("in the author's view …", "found X tedious"), not as objective fact.
- FALSIFIABLE: an observer-independent fact (dates, places, events) — state it
  as fact (per the confidence handling below).
- CONSTITUTIVE: a rule or definition the entry establishes — state it as such.

Confidence handling (FALSIFIABLE / CONSTITUTIVE only — for EXPERIENTIAL and
EVALUATIVE, effective_confidence is how clearly the feeling/opinion is
expressed, not how likely it is true, so NEVER truth-hedge them):
- Claims with effective_confidence below 0.5 MUST be hedged ("reportedly",
  "according to one account").
- Claims at or above 0.9 may be stated declaratively.
- Claims between 0.5 and 0.9 should be stated declaratively with a citation but
  without an emphatic adverb.

Cross-references:
- When a claim mentions a named entity, render it as an Obsidian-style
  `[[Name]]` wikilink so the vault stays graph-navigable.

Style:
- Connected narrative prose preserving the order and arc of the particles — not
  a bulleted list, not an encyclopedic article.
- Faithful to the particles' point of view (they are already reified, e.g.
  "The author …"); keep that voice.
- One flowing account; short paragraphs are fine. No more than 600 words.
- Do NOT write a References section — the exporter appends it.
- Emit footnote *references* only (`[^p-xxxxxxxx]` inline). Do NOT emit footnote
  *definition* lines (`[^p-xxxxxxxx]: …`).

Output: the narrative body in Markdown, starting with the `# {title}` H1 heading.
No frontmatter; no preamble before the H1; no closing remarks.

----
NARRATIVE: {subject}
PARTICLE LIST IN ORDER (id_prefix, modality, effective_confidence, content):
{particles}
----"""


def _format_particles_for_prompt(particles: list[Particle], eff: dict[str, float]) -> str:
    """One ``id_prefix | modality | conf | content`` line per particle."""
    lines = []
    for p in particles:
        conf = eff.get(p.id, p.confidence.value)
        line = f"  {_short_id(p.id)} | {p.assertion_modality.value} | {conf:.2f} | {p.content}"
        lines.append(line)
    return "\n".join(lines)


def _format_misalignments_for_prompt(misalignments: list[dict[str, Any]]) -> str:
    """One-per-line summary of judge-flagged pairs for the Layer-B retry prompt.

    Each entry shows the judge's verdict (``unrelated`` / ``contradicts``)
    and reason so the LLM knows what to fix on the next attempt.
    """
    if not misalignments:
        return "  (none)"
    lines: list[str] = []
    for m in misalignments:
        verdict = str(m.get("verdict", "unrelated"))
        reason = str(m.get("reason", "")).strip() or "(no reason given)"
        # ``id`` is the integer pair index assigned by extract_cited_segments;
        # the LLM uses it to correlate with its own output.
        pair_id = m.get("id", "?")
        lines.append(f"  pair {pair_id} [{verdict}]: {reason}")
    return "\n".join(lines)


def _framing_block(framing: str | None) -> str:
    """The document-spine prefix interpolated into a first-attempt prompt.

    Empty string when absent (byte-identical to the pre-0163 prompt); when
    present, the framing text plus a blank-line separator so it reads as a
    standalone document-context block ahead of the task instruction.
    """
    if not framing:
        return ""
    return f"{framing.strip()}\n\n"


def _direction_block(direction: str | None) -> str:
    """The per-section authoring brief interpolated into a first-attempt prompt.

    Empty string when absent (byte-identical to the pre-0163 prompt); when
    present, wrapped in leading/trailing newlines so it forms its own
    blank-line-separated block between the citation rules and the modality
    handling.
    """
    if not direction:
        return ""
    return f"\n{direction.strip()}\n"


def _build_synthesis_prompt(
    *,
    subject: SynthesisTopic,
    particles: list[Particle],
    eff: dict[str, float],
    strict: bool,
    layer_b_misalignments: list[dict[str, Any]] | None = None,
    sequence_mode: bool = False,
    flowing: bool = False,
    direction: str | None = None,
    framing: str | None = None,
) -> str:
    """Render the user-message string for one subject's synthesis call.

    ``strict=False`` returns the standard first-attempt prompt.
    ``strict=True`` returns the retry prompt. When ``layer_b_misalignments``
    is provided (i.e. the prior failure was Layer B, not Layer A), the
    Layer-B-specific strict prompt is used instead of the generic strict
    prompt.

    ``sequence_mode=True`` selects the narrative prompt on the
    first attempt: the particles are rendered as one ordered prose narrative
    rather than an encyclopedic article. Strict retries reuse the generic
    citation-focused prompts (order-awareness is a first-attempt concern;
    the retry's job is to fix citations).

    ``flowing=True`` selects the flowing-prose first-attempt
    prompt: one continuous passage under the section title, with sub-heading
    invention suppressed. ``direction`` (the section's authoring brief) and ``framing`` (the document spine) are interpolated
    into the standard / flowing first-attempt templates when present; both
    absent leaves the prompt byte-identical to the pre-0163 prompt. As with
    ``sequence_mode``, all three are first-attempt-only — strict retries reuse
    the citation-focused prompts, whose job is to fix citations.
    """
    if strict and layer_b_misalignments is not None:
        template = _SYNTHESIS_PROMPT_STRICT_LAYER_B
        particle_block = _format_particles_for_prompt(particles, eff)
        return template.format(
            title=subject.canonical_name,
            subject=subject.canonical_name,
            allowed_ids=", ".join(_short_id(p.id) for p in particles),
            particles=particle_block,
            misalignments=_format_misalignments_for_prompt(layer_b_misalignments),
        )

    if sequence_mode and not strict:
        particle_block = _format_particles_for_prompt(particles, eff)
        return _SYNTHESIS_PROMPT_NARRATIVE.format(
            title=subject.canonical_name,
            subject=subject.canonical_name,
            allowed_ids=", ".join(_short_id(p.id) for p in particles),
            particles=particle_block,
        )

    if flowing and not strict:
        particle_block = _format_particles_for_prompt(particles, eff)
        return _SYNTHESIS_PROMPT_FLOWING.format(
            title=subject.canonical_name,
            subject=subject.canonical_name,
            allowed_ids=", ".join(_short_id(p.id) for p in particles),
            particles=particle_block,
            framing=_framing_block(framing),
            direction=_direction_block(direction),
        )

    template = _SYNTHESIS_PROMPT_STRICT if strict else _SYNTHESIS_PROMPT_STANDARD
    particle_block = _format_particles_for_prompt(particles, eff)
    # The strict templates carry no ``{framing}`` / ``{direction}`` placeholders
    # (heading suppression and authoring briefs are first-attempt concerns); only
    # the standard first-attempt prompt interpolates them. Passing the extra
    # kwargs to ``str.format`` on the strict template is harmless — unused format
    # kwargs are ignored — so the single call below serves both.
    return template.format(
        title=subject.canonical_name,
        subject=subject.canonical_name,
        allowed_ids=", ".join(_short_id(p.id) for p in particles),
        particles=particle_block,
        framing=_framing_block(framing),
        direction=_direction_block(direction),
    )


# ---------------------------------------------------------------------------
# Synthesised-article renderer (frontmatter + LLM body + References)
# ---------------------------------------------------------------------------


def render_synthesised_article(
    subject: SynthesisTopic,
    particles: list[Particle],
    *,
    body: str,
    cited_short_ids: set[str],
    corpus_uris: dict[str, str | None],
    effective_confidences: dict[str, float],
    input_hash: str,
    layer_a_passed: bool,
    layer_b_passed: bool | None,
    layer_b_unrelated_count: int | None = None,
    layer_b_contradicts_count: int | None = None,
    lint_findings: list[Any] | None = None,
    min_particle_confidence: float | None = None,
    dropped_below_threshold: int | None = None,
) -> str:
    """Wrap the LLM body in frontmatter + a deterministic References section.

    The References section is appended outside the LLM's output so the
    citation footnote *bodies* — source URL, confidence, extractor
    identity, extraction timestamp — cannot be hallucinated.
    Only particles the LLM actually cited get a footnote entry; this keeps
    the article tight when the synthesis only used a subset of the input.

    ``lint_findings``: article-relevant findings are surfaced as ``>
    [!type]`` callouts inserted between the H1 and the synthesised body;
    the frontmatter ``contradictions`` count is derived from them.
    """
    sources = {
        ref.corpus_entry_id for p in particles for ref in p.provenance if ref.corpus_entry_id
    }
    contradictions = _count_contradictions(lint_findings)
    head = _render_frontmatter(
        particle_count=len(particles),
        input_hash=input_hash,
        sources=len(sources),
        contradictions=contradictions,
        synthesis="llm",
        layer_a_passed=layer_a_passed,
        layer_b_passed=layer_b_passed,
        layer_b_unrelated_count=layer_b_unrelated_count,
        layer_b_contradicts_count=layer_b_contradicts_count,
        min_particle_confidence=min_particle_confidence,
        dropped_below_threshold=dropped_below_threshold,
        taxonomy_tags=_taxonomy_tags(particles),
    )

    # lint callouts are NOT spliced into the cached body here.
    # They are a write-time presentation layer applied fresh on every
    # export (see ``apply_lint_callouts``), because the cache key omits
    # lint findings — baking them in froze stale warning text into cache
    # hits. ``lint_findings`` still feeds the ``contradictions`` count
    # above (refreshed at write time when callouts are applied).

    # Only render footnote bodies for particles the LLM actually cited.
    cited_lower = {sid.lower() for sid in cited_short_ids}
    refs: list[str] = []
    for p in particles:
        if _short_id(p.id).lower() not in cited_lower:
            continue
        first_source = next((r for r in p.provenance if r.corpus_entry_id is not None), None)
        uri = corpus_uris.get(first_source.corpus_entry_id) if first_source else None
        eff = effective_confidences.get(p.id, p.confidence.value)
        # See the Claims-section note: attribute direct assertions (no
        # extractor_ref) to the asserting principal, not "extractor: ?".
        attribution = (
            f"extractor: {p.extractor_ref.name} {p.extractor_ref.version}"
            if p.extractor_ref
            else f"asserted by: {p.asserted_by or '?'}"
        )
        ts = p.asserted_at.date().isoformat() if p.asserted_at else "?"
        uri_str = uri if uri else "(no public URI)"
        sid = _short_id(p.id)
        # Anchor heading + footnote definition. The ``### p-{sid}``
        # heading lets readers navigate the References section as a
        # per-particle TOC, and (more importantly) lets *other* wiki
        # articles cite the same particle with Obsidian's
        # ``[[Subject Title#p-{sid}]]`` wikilink syntax. The footnote
        # definition is what the body's ``[^p-{sid}]`` resolves to in
        # markdown-footnote-aware renderers.
        refs.append(
            f"### p-{sid}\n\n"
            f"[^p-{sid}]: {uri_str}\n"
            f"  (effective confidence: {eff:.2f}, "
            f"{attribution}, extracted {ts})\n\n"
        )

    parts = [head, body.rstrip(), "\n\n## References\n\n"]
    parts.extend(refs)
    return "".join(parts)


# ---------------------------------------------------------------------------
# write-time lint-callout layer
# ---------------------------------------------------------------------------
#
# Lint callouts are applied to a rendered article / note *at write time*,
# not baked into the cached body. The cache key (compute_input_hash +
# _PROMPT_VERSION) omits lint findings on purpose — lint state churns and
# must not trigger LLM regenerations — so freezing callout text into the
# cached body left stale warnings on every cache hit. ``apply_lint_callouts``
# is idempotent: it strips any previously-applied callout block and
# re-splices the current findings, so re-running an export with unchanged
# findings is a no-op on the file.
#
# The block is identified *structurally* — a run of ``> [!sev] FINDING_TYPE``
# callouts whose FINDING_TYPE is a known lint type — so no sentinel marker is
# needed in the text. (originally fenced the block in an HTML comment
# ``<!-- particles:lint-callouts -->``; Obsidian's Live Preview renders HTML
# comments as visible grey text, so the fence is no longer emitted. It is
# still stripped on sight to clean up notes written by 0.61.x.)

# A run of one or more lint callouts: a ``> [!sev] FINDING_TYPE`` header plus
# its ``>`` continuation lines, blank-line separated. Scoped to the known
# finding-type tokens so an operator's own callout — or the structural
# ``> [!warning] Unverified Wikidata link`` / ``> [!note] Structured-listing
# render`` banners — is never matched. This is both what ``apply_lint_callouts``
# emits and what it strips, which is what makes the round-trip idempotent.
_LINT_CALLOUT_BLOCK_RE = re.compile(
    r"(?:^> \[!(?:danger|warning|info)\] "
    r"(?:" + "|".join(re.escape(ft) for ft in _LINT_CALLOUT_TYPES) + r")\b"
    r".*\n(?:^>.*\n)*\n?)+",
    re.MULTILINE,
)

# Legacy HTML-comment fence emitted by 0.61.x (v1). Stripped on sight
# to clean up already-exported notes; never emitted now. DOTALL so it spans
# the callouts it once wrapped.
_LEGACY_FENCE_RE = re.compile(
    r"\n*<!-- particles:lint-callouts -->.*?<!-- /particles:lint-callouts -->\n*",
    re.DOTALL,
)


def strip_lint_callouts(text: str) -> str:
    """Remove any previously-applied lint-callout block.

    Removes both the structural callout run this module emits and the legacy
    HTML-comment fence emitted by 0.61.x. Idempotent and otherwise
    shape-preserving: collapses the 3+ newline runs a strip can create back
    to a blank-line separator so a strip → re-splice round-trip is stable.
    """
    text = _LEGACY_FENCE_RE.sub("\n\n", text)
    text = _LINT_CALLOUT_BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _refresh_contradictions_count(text: str, findings: list[Any] | None) -> str:
    """Update the ``contradictions:`` frontmatter field to the current count.

    Only the wiki article carries this field (it writes the whole rendered
    article, frontmatter included); the Obsidian / Logseq note frontmatter
    has no such field, so this is a no-op there. Scoped to the frontmatter
    block so it never rewrites a `contradictions:` that appears in body prose.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text
    count = _count_contradictions(findings)
    fm_block = m.group(0)
    new_block, n = re.subn(r"(?m)^contradictions: \d+$", f"contradictions: {count}", fm_block)
    if n == 0:
        return text
    return new_block + text[m.end() :]


def apply_lint_callouts(text: str, findings: list[Any] | None) -> str:
    """(Re)apply the current lint callouts to a rendered article or note.

    Strips any prior callout block, refreshes the frontmatter
    ``contradictions`` count (when present), then splices the callout block
    immediately after the first H1 so the warnings are the first thing a
    reader sees. With no article-relevant findings the result is simply the
    stripped text — which is how a resolved finding's stale callout is removed
    on the next export. Idempotent.
    """
    text = _refresh_contradictions_count(text, findings)
    stripped = strip_lint_callouts(text)
    callouts = _render_lint_callouts(findings) if findings else ""
    if not callouts:
        return stripped
    h1 = re.search(r"^# .*\n", stripped, flags=re.MULTILINE)
    if h1 is None:
        # No H1 (defensive): place the block after the frontmatter, else top.
        fm = _FRONTMATTER_RE.match(stripped)
        pos = fm.end() if fm else 0
        return f"{stripped[:pos]}{callouts}\n\n{stripped[pos:].lstrip(chr(10))}"
    end = h1.end()
    return f"{stripped[:end]}\n{callouts}\n\n{stripped[end:].lstrip(chr(10))}"
