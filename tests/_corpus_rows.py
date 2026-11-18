# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Corpus rows as an in-process deposit writes them."""

from __future__ import annotations

from particles.core.schema import CorpusEntry
from particles.corpus.origin import stamp_new_entry
from particles.corpus.store import CorpusEntryRow


def local_entry_row(entry: CorpusEntry) -> CorpusEntryRow:
    """``entry`` as a row, a ``file://`` one stamped as this host's file.

    A test that adds rows by hand stands in for a local deposit, which stamps
    the origin; an unstamped ``file://`` row is read by no host.
    """
    row = CorpusEntryRow.from_model(entry)
    stamp_new_entry(row)
    return row


#: A client host id for the harvester keys tests mint. Never
#: this test process's own, which ``keys create`` refuses.
CLIENT_HOST_ID = "0c1e7a5e-0000-4000-8000-00000000c11e"
