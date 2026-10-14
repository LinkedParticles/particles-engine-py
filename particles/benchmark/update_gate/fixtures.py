# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Locate the case notes and the registered pair sets.

The fixtures stay where the rest of the repository reads them,
``tests/fixtures/trip_vs_move/`` and ``tests/fixtures/update_probe/``. A built
wheel force-includes the files these suites read at
``particles/_artifacts/benchmarks/`` (``[tool.hatch.build.targets.wheel.force-include]``
in ``pyproject.toml``), so a ``pip install`` carries them. Prefer the packaged
copy; fall back to the source tree, as the schema artifacts do.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: The trip-versus-move notes, in deposit order (the move case reads all three).
MOVE_NOTES = ("session-04.md", "session-11.md", "session-12-moved.md")
#: The trip-alone variant: a week away, and no move.
TRIP_ALONE_NOTES = ("session-04.md", "session-11.md")

#: The four registered pair sets the update-gate verb replays, by set name.
PAIR_SETS = {
    "gate": "gate_pairs.json",
    "gate-heldout": "gate_pairs_heldout.json",
    "past": "past_value_pairs.json",
    "past-marker": "past_marker_heldout.json",
}
#: Sets whose bar lets one positive miss (registered as held-out).
HELDOUT_SETS = frozenset({"gate-heldout", "past-marker"})

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _resolve(kind: str, name: str, *, package_root: Path | None = None) -> Path:
    root = package_root or _PACKAGE_ROOT
    packaged = root / "_artifacts" / "benchmarks" / kind / name
    if packaged.exists():
        return packaged
    return root.parent / "tests" / "fixtures" / kind / name


def case_note(name: str, *, package_root: Path | None = None) -> Path:
    """The path of one trip-versus-move note, packaged copy first."""
    return _resolve("trip_vs_move", name, package_root=package_root)


def pair_set(set_name: str, *, package_root: Path | None = None) -> Path:
    """The path of one registered pair set (``gate``, ``past-marker``, …)."""
    return _resolve("update_probe", PAIR_SETS[set_name], package_root=package_root)


def sha256(path: Path) -> str:
    """Hex digest of a fixture file's bytes, recorded on every report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
