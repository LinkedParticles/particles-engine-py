# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The artifact authority: project-scoped identity for gated names.

The non-entity gate classifies filenames, reference codes,
snake_case identifiers and CLI command strings. Where the gate *qualifies* such
a name, because the source's project is known, this authority gives it an
identity scoped by that project: ``artifact:<namespace key>/<normalized name>``.
The same file in two repositories is two Subjects, and the second mention of a
record in one project finds the first one's Subject by that ref.

It is a :class:`~particles.ingest.authorities.registry.ContextualRecognizer`
only. Its plain ``recognize`` answers nothing, so an unqualified name never
reaches it, and it has no live lookup. The resolver owns every write.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from particles.core.schema import ApplicabilityClause, ExternalRef

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from particles.ingest.authorities.registry import AuthorityResolution, RecognizeContext

#: The ``ExternalRef.namespace`` every artifact identity carries.
ARTIFACT_NAMESPACE = "artifact"

#: Gate class → the ``subject_class`` a qualified Subject carries.
ARTIFACT_SUBJECT_CLASSES: dict[str, str] = {
    "reference_code": "artifact:record",
    "filename": "artifact:file",
    "snake_case": "artifact:symbol",
    "cli_command": "artifact:command",
}

_CODE_SEPARATORS = re.compile(r"[ ._/\-]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_artifact_name(name: str, token_class: str) -> str:
    """The identity-bearing form of a qualified name, per gate class.

    A reference code folds case and separators, so ``RFC 2119``, ``RFC-2119``
    and ``rfc 2119`` are one record. A CLI command collapses whitespace. A
    filename drops a leading ``./`` and is otherwise kept as spelled: a bare
    filename and a path are different Subjects by design, and an
    absolute path keeps its prefix, since the Engine cannot map a project key
    back to a directory. A snake_case identifier is kept as spelled.
    """
    stripped = name.strip().strip("`").strip()
    match token_class:
        case "reference_code":
            return _CODE_SEPARATORS.sub("-", stripped).strip("-").casefold()
        case "cli_command":
            return _WHITESPACE.sub(" ", stripped)
        case "filename":
            return stripped[2:] if stripped.startswith("./") else stripped
        case _:
            return stripped


class ArtifactAuthority:
    """Recognizes qualified gate names as project-scoped artifacts."""

    NAMESPACE = ARTIFACT_NAMESPACE
    PRIORITY = 5
    LIVE = False
    DEFAULT_LINK_CONFIDENCE = 1.0
    APPLICABILITY: list[ApplicabilityClause] = []

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> ExternalRef | None:
        # Never from a name alone: an unscoped filename is exactly the
        # cross-repository merge the gate exists to prevent.
        return None

    def recognize_in(self, name: str, context: RecognizeContext) -> ExternalRef | None:
        if context.token_class not in ARTIFACT_SUBJECT_CLASSES or not context.namespace_key:
            return None
        normalized = normalize_artifact_name(name, context.token_class)
        if not normalized:
            return None
        return ExternalRef(
            namespace=ARTIFACT_NAMESPACE,
            id=f"{context.namespace_key}/{normalized}",
            confidence=self.DEFAULT_LINK_CONFIDENCE,
        )

    def subject_class_for(self, context: RecognizeContext) -> str | None:
        return ARTIFACT_SUBJECT_CLASSES.get(context.token_class)

    async def resolve(
        self,
        session: AsyncSession,
        name: str,
        *,
        particle_content: str | None,
        domain: str | None,
    ) -> AuthorityResolution | None:
        return None

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        return None
