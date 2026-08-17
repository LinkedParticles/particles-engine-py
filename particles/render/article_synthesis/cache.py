"""Article cache key + rendered-article decomposition.

Three operator-facing concerns live here:

* ``compute_input_hash`` — the cache key the exporter uses to decide
  whether a previously-rendered article is still current. SHA-256 over
  (particles × subject identity × prompt version) — any drift in those
  inputs invalidates the cache and the next export regenerates the
  article without the operator needing ``--regenerate-all``.
* ``split_rendered_article`` / ``_parse_frontmatter`` — the inverse:
  given a previously-rendered article file on disk, decompose it into
  frontmatter + H1 + prose + references so the exporter can read the
  cached ``input_hash`` and/or splice the pieces into a larger note.
* ``invalidate_stale_link_articles`` — the cross-subject staleness fix
  . Scans every cached article for ``[[X]]`` wikilinks
  whose X has been renamed away from any current canonical name +
  alias; strips ``input_hash`` from those frontmatter blocks so the
  next export regenerates them.

This module is dependency-free within the package — all three pieces
are deliberately reachable without pulling in layer_a / layer_b /
render so exporters that only need cache bookkeeping (Obsidian's
``vault.py`` read-the-frontmatter path) don't have to import the
prompt constants.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from particles.core.schema import Particle, Subject
from particles.render.markdown import subject_slug

log = logging.getLogger(__name__)

# Bumped whenever the synthesis prompt, post-process rules, or any
# exporter-side article-rendering transform changes such that re-running
# would produce a different output from the *same* particle set. Mixed
# into ``compute_input_hash`` so any prior cached article is treated as
# stale and regenerated on next run, without the operator having to
# remember ``--regenerate-all``. Increment when editing the prompt
# constants in ``render.py``, the post-process strip rules, the citation
# format spec, or a downstream exporter's splice transform (e.g. the
# Obsidian block-ref conversion).
# 12: synthesis prompts are modality-aware — the particle block
#    carries assertion_modality and the prompt renders EXPERIENTIAL/EVALUATIVE
#    content distinctly, with confidence-hedging scoped to truth-apt particles.
_PROMPT_VERSION = "12"


def compute_input_hash(
    particles: list[Particle], subject: Subject | None = None, *, ordered: bool = False
) -> str:
    """SHA-256 over (particles × subject identity × prompt version).

    Particles contribute their (id, status, confidence_value) tuple.
    Adding or removing a particle changes the hash; so does a status
    transition (ACTIVE → SUPERSEDED) and a confidence update. We
    deliberately do NOT include particle ``content`` (immutable, would
    only inflate the hash without catching any real change).

    ``ordered=True`` (narrative synthesis) makes the **order** of
    ``particles`` part of the hash: the triples are hashed in the given
    sequence (each prefixed by its position) instead of sorted, and an
    ``ordered=1`` marker keeps an ordered hash distinct from the unordered one.
    A re-ordered ``SEQUENCE_IN`` chain then invalidates the cached narrative.
    The default (``ordered=False``) is byte-for-byte the per-subject behaviour.

    Subject identity — when provided — contributes ``canonical_name``,
    ``description``, sorted ``aliases``, sorted ``external_ids``, and
    ``subject_class``. Any of those changing alters the article's
    inputs (subject is interpolated into the synthesis prompt; aliases
    and external_ids drive Obsidian frontmatter tags; class selects
    the template). Without the subject in the hash, operator-driven
    fixes like ``particles subjects unlink`` / ``confirm`` / ``alias``
    / ``merge`` don't invalidate the cache, so the next export reuses
    the stale article. ``subject`` is optional for backwards
    compatibility with callers that hash particle-only state (tests
    and operations that don't render per-subject articles).

    ``_PROMPT_VERSION`` is mixed in so a synthesis-prompt edit
    invalidates every cached article on next run.
    """
    hasher = hashlib.sha256()
    raw_triples = [(p.id, p.status.value, round(p.confidence.value, 4)) for p in particles]
    if ordered:
        hasher.update(b"ordered=1\x00")
        triples = list(enumerate(raw_triples))
    else:
        triples = list(enumerate(sorted(raw_triples)))
    for position, (pid, status_str, conf) in triples:
        if ordered:
            hasher.update(f"pos={position}".encode())
            hasher.update(b"\x00")
        hasher.update(pid.encode())
        hasher.update(b"\x00")
        hasher.update(status_str.encode())
        hasher.update(b"\x00")
        hasher.update(f"{conf:.4f}".encode())
        hasher.update(b"\x00")
    if subject is not None:
        hasher.update(b"subject=")
        hasher.update(subject.canonical_name.encode())
        hasher.update(b"\x00")
        hasher.update((subject.description or "").encode())
        hasher.update(b"\x00")
        hasher.update((subject.subject_class or "").encode())
        hasher.update(b"\x00")
        for alias in sorted(subject.aliases):
            hasher.update(b"alias=")
            hasher.update(alias.encode())
            hasher.update(b"\x00")
        for ref in sorted(subject.external_ids, key=lambda r: (r.namespace, r.id)):
            hasher.update(b"ref=")
            hasher.update(f"{ref.namespace}:{ref.id}".encode())
            hasher.update(b"\x00")
    hasher.update(b"prompt_version=")
    hasher.update(_PROMPT_VERSION.encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Frontmatter parsing + rendered-article decomposition
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, Any] | None:
    """Return the YAML frontmatter as a dict, or None if absent / malformed."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def split_rendered_article(rendered: str) -> tuple[dict[str, Any], str, str, str]:
    """Decompose a rendered article into (frontmatter, h1_line, prose, references).

    Used by exporters (Obsidian's ``--with-synthesis``, future Logseq, …)
    that need to splice the article into a *larger* note that already
    has its own frontmatter and structure. The wiki exporter writes
    the rendered article whole; this function is for everyone else.

    Returns a 4-tuple:

    * ``frontmatter`` — the parsed YAML frontmatter dict (empty dict
      if the article has no frontmatter, though
      :func:`render_synthesised_article` / :func:`render_structured_listing`
      always emit one)
    * ``h1_line`` — the ``# Subject Title`` line, **including** the
      leading ``#`` and trailing newline. Empty string if no H1 was
      present (defensive — both renderers always emit one).
    * ``prose`` — everything between the H1 and the ``## References``
      heading, with leading/trailing whitespace stripped
    * ``references`` — the ``## References`` heading and everything
      after it, with leading/trailing whitespace stripped. Empty string
      if the article has no References section.

    The four pieces always re-concatenate (with appropriate separators)
    into a string equivalent to the input modulo whitespace.
    """
    fm: dict[str, Any] = {}
    body = rendered
    m = _FRONTMATTER_RE.match(rendered)
    if m:
        try:
            parsed = yaml.safe_load(m.group(1))
            if isinstance(parsed, dict):
                fm = parsed
        except yaml.YAMLError:
            pass
        body = rendered[m.end() :]

    # Find the H1. Both renderers emit a single `# {title}\n` near the
    # top; we take the *first* `#` heading.
    h1_match = re.search(r"^# .*\n", body, flags=re.MULTILINE)
    h1_line = h1_match.group(0) if h1_match else ""
    after_h1 = body[h1_match.end() :] if h1_match else body

    # Split on the References heading.
    refs_idx = after_h1.find("\n## References")
    if refs_idx == -1:
        prose = after_h1.strip()
        references = ""
    else:
        prose = after_h1[:refs_idx].strip()
        # +1 to skip the leading newline so the heading starts the string
        references = after_h1[refs_idx + 1 :].strip()

    return fm, h1_line, prose, references


# ---------------------------------------------------------------------------
# Cross-subject staleness
# ---------------------------------------------------------------------------


# Captures the page name from Obsidian / wiki-style wikilinks:
#   [[Foo]]                    → "Foo"
#   [[Foo|alias text]]         → "Foo"
#   [[Foo#heading]]            → "Foo"
#   [[Foo#heading|alias text]] → "Foo"
# The page name is everything up to the first ``|`` (alias separator)
# or ``#`` (heading anchor) or ``]``.
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def _extract_wikilink_targets(body: str) -> set[str]:
    """Return the set of page names referenced by ``[[X]]`` wikilinks in body."""
    return {m.group(1).strip() for m in _WIKILINK_RE.finditer(body)}


def _strip_input_hash_from_frontmatter(
    article_text: str, hash_field: str = "input_hash"
) -> str | None:
    """Return ``article_text`` with ``hash_field`` removed from its YAML
    frontmatter. Returns None if the article has no parseable frontmatter
    or the field isn't present (nothing to do).

    ``hash_field`` differs per exporter: the wiki exporter writes
    ``input_hash``; the Obsidian exporter writes ``article_input_hash``
    (the prefix disambiguates from the operator's own frontmatter inside
    a vault note that ALSO holds non-synthesis content).

    Preserves every other frontmatter field, the body content, and the
    surrounding ``---`` fences. The stripped result re-renders cleanly
    via ``yaml.safe_dump`` so the field order may shift but the data
    survives.
    """
    m = _FRONTMATTER_RE.match(article_text)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict) or hash_field not in fm:
        return None
    del fm[hash_field]
    new_fm_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip("\n")
    body_start = m.end()
    return f"---\n{new_fm_text}\n---\n{article_text[body_start:]}"


def invalidate_stale_link_articles(
    output_dir: Path,
    known_names: set[str],
    *,
    hash_field: str = "input_hash",
    recursive: bool = False,
) -> list[Path]:
    """Strip the article-cache hash from any cached article whose ``[[X]]``
    wikilinks reference a page name not in ``known_names``.

    Scans every ``*.md`` file in ``output_dir`` — non-recursive for the
    flat-per-subject wiki layout; ``recursive=True`` walks subdirectories
    for Obsidian's nested vault layout. For each file:

    1. Extract the wikilink targets.
    2. If every target is in ``known_names``, leave the file alone.
    3. Otherwise the article links to a name that no longer exists as a
       canonical or alias — strip ``hash_field`` from frontmatter so the
       next export regenerates the article.

    The file's body and non-``hash_field`` frontmatter fields are
    preserved (the operator inspecting the vault between invalidation
    and regeneration sees the prior content, not an empty file).

    Returns the list of invalidated file paths. Callers that also want
    to evict the shared synthesis cache map each path back
    to its subject via the exporter's slug helper and call
    ``evict_subject(session, subject_id)`` for each. Returning paths
    rather than a bare count lets that DB-side eviction stay opt-in
    per exporter without re-walking the directory.
    """
    if not output_dir.exists():
        return []
    paths = output_dir.rglob("*.md") if recursive else output_dir.glob("*.md")
    invalidated: list[Path] = []
    for article_path in sorted(paths):
        try:
            text = article_path.read_text(encoding="utf-8")
        except OSError:
            continue
        targets = _extract_wikilink_targets(text)
        if not targets:
            continue
        stale = targets - known_names
        if not stale:
            continue
        stripped = _strip_input_hash_from_frontmatter(text, hash_field=hash_field)
        if stripped is None:
            # No frontmatter / hash_field absent — nothing to strip.
            # The next export will treat the file as a fresh write
            # anyway.
            continue
        try:
            article_path.write_text(stripped, encoding="utf-8")
        except OSError as exc:  # pragma: no cover — defensive
            log.warning("Couldn't invalidate %s: %s", article_path, exc)
            continue
        log.info(
            "Invalidated %s — stale wikilinks: %s",
            article_path.name,
            ", ".join(sorted(stale)),
        )
        invalidated.append(article_path)
    return invalidated


def find_unresolved_wikilinks(
    output_dir: Path,
    *,
    recursive: bool = False,
) -> dict[Path, list[str]]:
    """Return, per article, the ``[[X]]`` wikilink targets that resolve to no
    page within ``output_dir`` — the one-shot cross-reference check.

    The read-only complement of :func:`invalidate_stale_link_articles`: that
    strips the cache hash of articles linking to a *renamed* subject against an
    externally supplied ``known_names`` set; this is self-contained — it asks
    "does the export, taken on its own, leave any synthesised ``[[Subject]]``
    cross-reference pointing nowhere?" The wiki and Obsidian exporters emit the
    same ``[[display name]]`` wikilinks (the synthesis prompt instructs it), so
    one resolver serves both. Pass ``recursive=True`` for Obsidian's nested
    vault layout; leave it off for the flat wiki directory.

    A target ``X`` resolves when any of these hold:

    * ``X`` matches a written file's basename — a bare ``[[X]]`` resolves to a
      note named ``X`` regardless of folder (Obsidian / wiki behaviour); or
    * ``subject_slug(X)`` matches a written file's path relative to
      ``output_dir`` (``.md`` dropped) — every per-subject exporter names its
      files ``subject_slug(display_name)``, so ``[[display name]]`` round-trips
      (including the ``reddit.com/`` / ``github.com/`` nested slugs); or
    * ``X`` (or ``subject_slug(X)``) appears in some note's frontmatter
      ``aliases`` — the disambiguation alias that lets
      ``[[Prometheus]]`` resolve to the ``Prometheus (software)`` note.

    Files with no wikilinks are omitted; an empty dict means every
    cross-reference in the export resolves.
    """
    if not output_dir.exists():
        return {}

    paths = sorted(output_dir.rglob("*.md") if recursive else output_dir.glob("*.md"))

    basenames: set[str] = set()
    rel_slugs: set[str] = set()
    aliases: set[str] = set()
    texts: dict[Path, str] = {}
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover — defensive
            continue
        texts[p] = text
        basenames.add(p.stem)
        rel_slugs.add(p.relative_to(output_dir).with_suffix("").as_posix())
        fm = _parse_frontmatter(text)
        if fm:
            raw_aliases = fm.get("aliases")
            if isinstance(raw_aliases, list):
                aliases.update(str(a) for a in raw_aliases)
            elif isinstance(raw_aliases, str):
                aliases.add(raw_aliases)

    def _resolves(target: str) -> bool:
        slug = subject_slug(target)
        return (
            target in basenames
            or slug in rel_slugs
            or Path(slug).name in basenames
            or target in aliases
            or slug in aliases
        )

    unresolved: dict[Path, list[str]] = {}
    for p, text in texts.items():
        missing = sorted(t for t in _extract_wikilink_targets(text) if not _resolves(t))
        if missing:
            unresolved[p] = missing
    return unresolved
