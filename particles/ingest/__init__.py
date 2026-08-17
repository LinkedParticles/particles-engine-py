"""Engine-side ingest: reconcile candidate particles against accumulated state.

The Engine layer's ingest half. Where ``particles.extraction``
produces store-free *candidate* particles (Client), ``particles.ingest`` turns
those candidates into reconciled, persisted particles by reasoning over the
graph:

- :mod:`particles.ingest.pipeline` — the ``extract_snapshot`` orchestration,
  ``reconcile_and_insert``, and §6.6 conflict resolution against store state.
- :mod:`particles.ingest.subject_resolver` — matching a candidate's subject
  name to an existing canonical Subject via the store and live authorities.
- :mod:`particles.ingest.authorities` — the Subject Authority registry
  : live external-ID lookup (Wikidata, …), graph-aware.

These modules import the Engine freely (store, corpus, db). They are
deliberately *not* importable from the Client layer — the ``import-linter``
contract fails any Client → ``particles.ingest`` import.
"""

from __future__ import annotations
