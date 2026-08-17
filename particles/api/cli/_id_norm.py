"""Particle-ID normalisation for CLI verbs (0.43.1).

The rendered output (Obsidian / wiki articles, lint findings) refers to
particles by their *display form* ``p-XXXXXXXX``. The Subject store and
the ``particles`` table store the bare UUID, so a CLI verb's prefix-LIKE
match against ``ParticleRow.id`` won't match the display form. This
module strips the operator-visible prefixes so the operator can paste
exactly what they see in their vault / lint output and have the CLI work.

Two prefixes are recognised — both are display-only conventions:

- ``p-`` — the rendered form in Markdown footnotes (``[^p-de005b0e]``)
  and Obsidian / wiki / Logseq exporters' frontmatter and callouts.
- ``p:`` — the CLI shorthand documented in older ``links add`` help text
  (``particles links add p:abc p:def``); kept recognisable so existing
  scripts / muscle memory don't break.

The helper is deliberately defensive: it strips only known display
prefixes and leaves any other input untouched (so a future ``q-…`` or
``s-…`` namespace wouldn't be silently mangled).
"""

from __future__ import annotations


def normalise_particle_id(raw: str) -> str:
    """Strip a leading ``p-`` or ``p:`` display prefix from a particle ID.

    Returns the input unchanged if no recognised prefix is present.
    Whitespace is also stripped so a paste-with-trailing-newline works.
    """
    s = raw.strip()
    for prefix in ("p-", "p:"):
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s
