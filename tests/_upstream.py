"""Marker for assertions that only hold in the development upstream's checkout.

Most of this suite tests the SDK and runs anywhere. A few tests instead assert
something about *this repository*: the presence of files that are not part of
any published distribution (design notes, internal planning documents), or the
byte-for-byte content of a golden file pinned to prose that a published copy
carries in edited form. Those assertions are true of the tree they were written
in and say nothing about the code, so outside it they skip rather than fail — a
red check on correct code teaches contributors to ignore the suite.

The marker is the `docs/ADR/` decision-record corpus, which the export never
carries into any published tree (the corpus is in no export allowlist). An
earlier version keyed off the `scripts/` directory, but that broke once the
engine repo began shipping `scripts/projection_drift.py` for its projection
drift gate — a published engine tree then has a `scripts/` dir, so the marker
read as upstream there and these tests ran (and failed) instead of skipping.
The ADR corpus has no such exception: it is definitionally private.

Use it as sparingly as it is used today. A test that *can* be made portable —
by shipping the data it reads, or by asserting on behaviour instead of on
bytes — should be, and belongs to whichever distribution owns that data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: True in the development upstream's own checkout, false in a published tree.
IS_UPSTREAM = (REPO_ROOT / "docs" / "ADR").is_dir()

upstream_only = pytest.mark.skipif(
    not IS_UPSTREAM,
    reason="asserts on repository content that is not part of a published distribution",
)
