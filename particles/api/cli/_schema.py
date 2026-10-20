# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A store whose migrations lag the running install, and the command that catches it up.

The hooks, the MCP server and scheduled consolidation never migrate the store
they open; only ``particles db init`` does. An install advanced to a release
that ships a new migration therefore keeps running against the old schema
until someone migrates it, and fails on the first table or column the
migration adds. ``hook doctor`` reports it and ``memory consolidate`` refuses
to start on it, both with the same command.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from particles.config import get_config, validate_config
from particles.db import DEFAULT_STORE, SchemaState, StoreHandle


def this_install_executable() -> str:
    """The ``particles`` executable of the running install, as an absolute path when known."""
    argv0 = Path(sys.argv[0])
    if argv0.name == "particles" and argv0.is_absolute():
        return str(argv0)
    sibling = Path(sys.executable).parent / "particles"
    return str(sibling) if sibling.exists() else "particles"


def migrate_command(store: StoreHandle) -> str:
    """The command that migrates ``store`` to this install's head, runnable from any directory.

    ``db init`` migrates the default store, so the store's own DSN rides in
    ``DATABASE_URL``, which overrides config.yaml. ``PARTICLES_CONFIG`` is
    carried when this process has one, so the rest of the config matches.
    """
    storage = get_config().storage
    dsn = storage.database_url if store == DEFAULT_STORE else storage.stores[store]
    parts = ["env", f"DATABASE_URL={dsn}"]
    config = os.environ.get("PARTICLES_CONFIG")
    if not config:
        found, _ = validate_config()
        config = str(found.resolve()) if found is not None else None
    if config:
        parts.append(f"PARTICLES_CONFIG={config}")
    parts += [this_install_executable(), "db", "init"]
    return " ".join(shlex.quote(p) for p in parts)


def behind_summary(state: SchemaState) -> str:
    """One sentence naming where the store is and what this install expects."""
    import particles

    count = len(state.pending)
    noun = "migration" if count == 1 else "migrations"
    return (
        f"the store is at migration {', '.join(state.stored)}, {count} {noun} behind "
        f"this install (Particles {particles.__version__}, head {state.pending[-1]})"
    )
