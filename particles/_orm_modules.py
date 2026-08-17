"""Single source of truth for ORM-module registration.

Every module that declares ``Mapped[...]`` columns on
:class:`particles.db.Base` must be imported here so the table appears on
``Base.metadata``. Three call sites depend on a fully-populated metadata:

* ``alembic/env.py`` — autogenerate sees only registered tables.
* ``tests/conftest.py::cli_db`` — ``Base.metadata.create_all`` skips
  unregistered tables.
* ``particles/api/cli/db.py::_db_init_force`` — the scrap-and-re-extract
  upgrade path iterates ``Base.metadata.sorted_tables`` to clear every
  non-corpus table; unregistered tables would survive a ``--force``
  rebuild and silently leak stale schema_version state.

Adding a new ORM module: append one ``import`` line below. The other three
call sites need no change.
"""

from __future__ import annotations

import particles.corpus.follow_edges  # noqa: F401
import particles.corpus.store  # noqa: F401
import particles.store.curation_snapshot_store  # noqa: F401
import particles.store.event_store  # noqa: F401
import particles.store.extractor_store  # noqa: F401
import particles.store.lens_store  # noqa: F401
import particles.store.particle_store  # noqa: F401
import particles.store.relation_store  # noqa: F401
import particles.store.subject_store  # noqa: F401
import particles.store.synthesis_cache_store  # noqa: F401
import particles.store.taxonomy_store  # noqa: F401
import particles.store.trust_store  # noqa: F401
import particles.store.url_mention_store  # noqa: F401
import particles.store.utility_store  # noqa: F401
import particles.store.wikidata_cache  # noqa: F401
