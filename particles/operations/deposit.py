"""§9.1 Deposit operation — re-export.

The Deposit implementation lives in ``particles.corpus.deposit`` because it
operates directly on the corpus ORM (CorpusEntry/Snapshot rows + the
content-addressed blob store) and is depended on by extraction plugins that
re-deposit derived sources. This module exists so that all six §9 Core
operations are importable from one place:

  - §9.1 Deposit   ← here (re-export)
  - §9.2 Extract   — particles.operations.extract (re-export)
  - §9.3 Query     — particles.operations.query
  - §9.4 Lint      — particles.operations.lint
  - §9.5 Reindex   — particles.operations.reindex
  - §9.6 Review    — particles.operations.review

New code in ``particles.api`` (CLI + FastAPI) should prefer
``from particles.operations.deposit import deposit_file`` to keep the
spec-to-package mapping uniform. The ``particles.corpus.deposit`` module
remains the implementation home; other low-level helpers there
(``save_blob``, ``sha256``, ``write_entry_and_snapshot``, ``load_blob``) are
not part of the §9.1 surface and should still be imported from
``particles.corpus.deposit`` directly.
"""

from __future__ import annotations

from particles.corpus.deposit import (
    deposit_file,
    deposit_project,
    deposit_text,
    deposit_text_versioned,
    deposit_url,
    deposit_vault,
    deposit_web_clipper,
    split_file_by_date,
)

__all__ = [
    "deposit_file",
    "deposit_project",
    "deposit_text",
    "deposit_text_versioned",
    "deposit_url",
    "deposit_vault",
    "deposit_web_clipper",
    "split_file_by_date",
]
