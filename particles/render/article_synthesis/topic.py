"""Synthesis topic — the subject of one synthesis run.

``render_article`` was subject-scoped: it wrote one cited article *about a
Subject*. Documentation projection needs the same engine to write a
doc *section* whose candidate particles span many subjects. The only thing the
synthesis path needs from "what is this about" is a title to write under, a
stable id for the cache key, and an optional descriptive subtitle — so this
module names that minimal contract as :class:`SynthesisTopic` and ships a
concrete :class:`SectionTopic` for the doc-projection case.

:class:`particles.core.schema.Subject` satisfies :class:`SynthesisTopic`
structurally (it already exposes ``id`` / ``canonical_name`` / ``description``),
so every existing subject-scoped caller keeps working unchanged — the "thin
shim" the ADR names is structural conformance, not an adapter object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SynthesisTopic(Protocol):
    """Minimal identity the synthesis engine needs for one run.

    The three members are declared read-only (as ``@property``) so any object
    exposing an ``id`` / ``canonical_name`` / ``description`` attribute of the
    right type satisfies the protocol — notably
    :class:`particles.core.schema.Subject` and :class:`SectionTopic`.
    """

    @property
    def id(self) -> str:
        """Stable identity, used as the synthesis-cache key."""

    @property
    def canonical_name(self) -> str:
        """The title the article / section is written under (prompt + H1)."""

    @property
    def description(self) -> str | None:
        """Optional one-line subtitle rendered under the H1."""


@dataclass(frozen=True)
class SectionTopic:
    """A documentation-projection section's identity for synthesis.

    ``id`` is a manifest-local section key (e.g. ``readme:overview``), never a
    real Subject id, so a projected section's synthesis can never collide with a
    per-subject article in the shared cache. Projection opts out of that cache
    anyway (it calls ``render_article(session=None)``), so the id is only a
    within-run identity and a deterministic label.
    """

    id: str
    canonical_name: str
    description: str | None = None
