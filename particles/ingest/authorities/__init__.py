# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subject authority registry.

Public surface: the :class:`SubjectAuthority` Protocol and its additive
:class:`ContextualRecognizer` form, the
:class:`AuthorityResolution` and :class:`RecognizeContext` types, the
registry accessors, and the built-in authority classes.
"""

from __future__ import annotations

from particles.ingest.authorities._shared import PatternAuthority
from particles.ingest.authorities.artifact import ArtifactAuthority
from particles.ingest.authorities.registry import (
    AuthorityResolution,
    ContextualRecognizer,
    RecognizeContext,
    SubjectAuthority,
    clear_authorities,
    get_authorities,
    is_applicable,
)
from particles.ingest.authorities.wikidata import WikidataAuthority

__all__ = [
    "ArtifactAuthority",
    "AuthorityResolution",
    "ContextualRecognizer",
    "PatternAuthority",
    "RecognizeContext",
    "SubjectAuthority",
    "WikidataAuthority",
    "clear_authorities",
    "get_authorities",
    "is_applicable",
]
