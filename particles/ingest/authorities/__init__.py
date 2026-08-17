"""Subject authority registry.

Public surface: the :class:`SubjectAuthority` Protocol, the
:class:`AuthorityResolution` result type, the registry accessors, and the
built-in authority classes.
"""

from __future__ import annotations

from particles.ingest.authorities._shared import PatternAuthority
from particles.ingest.authorities.registry import (
    AuthorityResolution,
    SubjectAuthority,
    clear_authorities,
    get_authorities,
    is_applicable,
)
from particles.ingest.authorities.wikidata import WikidataAuthority

__all__ = [
    "AuthorityResolution",
    "PatternAuthority",
    "SubjectAuthority",
    "WikidataAuthority",
    "clear_authorities",
    "get_authorities",
    "is_applicable",
]
