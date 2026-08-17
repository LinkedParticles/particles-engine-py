"""§9.2 Extract operation — re-export.

The Extract orchestration is implemented in ``particles.ingest.pipeline``
because it drives the plugin registry that lives there. This module exists so
that all six §9 Core operations are importable from one place:

  - §9.1 Deposit   — particles.operations.deposit (re-export)
  - §9.2 Extract   ← here (re-export)
  - §9.3 Query     — particles.operations.query
  - §9.4 Lint      — particles.operations.lint
  - §9.5 Reindex   — particles.operations.reindex
  - §9.6 Review    — particles.operations.review

New code should prefer ``from particles.operations.extract import extract_snapshot``
to keep the spec-to-package mapping uniform. The pipeline module remains the
implementation home; importing from either path resolves to the same function.
"""

from __future__ import annotations

from particles.ingest.pipeline import extract_snapshot

__all__ = ["extract_snapshot"]
