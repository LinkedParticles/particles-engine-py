"""§9.3 Query operation package — semantic retrieval + NL response.

Public surface (stable):
    query, load_source_info

Internal helpers re-exported for tests and tooling that still reference the
pre-split module paths (``particles.operations.query.<name>``):
    _generate_response                 (./respond.py)
    _embed,                            (./rank.py)
        _collapse_co_evidential_top_k,
        _first_source_key
    _find_coverage_gaps,               (./gaps.py)
        _find_subject_coverage_gaps

See ``main.query`` for the entry-point orchestration; the heavy lifting
lives in the sibling submodules.
"""

from __future__ import annotations

from .gaps import _find_coverage_gaps, _find_subject_coverage_gaps
from .main import query, query_federated, retrieve_ranked
from .rank import _collapse_co_evidential_top_k, _embed, _first_source_key
from .respond import _generate_response
from .source_info import load_source_info

__all__ = [
    # Public
    "query",
    "query_federated",
    "retrieve_ranked",
    "load_source_info",
    # Internal helpers (re-exported for tests / external callers that pin
    # the pre-split import paths). New code should import from the submodule
    # directly.
    "_generate_response",
    "_embed",
    "_collapse_co_evidential_top_k",
    "_first_source_key",
    "_find_coverage_gaps",
    "_find_subject_coverage_gaps",
]
