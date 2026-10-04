# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Can a second writer write the store right now?

The cycle runs its passes on one long-lived session. A pass that
flushes a write and then awaits an LLM call keeps SQLite's write lock for the
whole call, so a SessionEnd harvest or an operator verb landing mid-cycle waits
out ``busy_timeout`` and fails with ``database is locked``. Tests call
:func:`store_accepts_a_writer` from inside a stubbed LLM seam to catch exactly
that: it opens its own connection, as another process would, and asks for the
write lock without waiting.
"""

from __future__ import annotations

import sqlite3

from particles.config import get_config, sqlite_file_path


def store_accepts_a_writer() -> bool:
    """True when another connection can take the store's write lock at once."""
    path = sqlite_file_path(get_config().storage.database_url)
    assert path is not None, "the probe needs a file-backed store (file_db_session)"
    con = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        return False
    finally:
        con.close()
