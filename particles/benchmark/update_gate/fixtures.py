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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

#: The mid-trip case's notes, in deposit order (its move variant reads all three).
MOVE_NOTES = ("session-04.md", "session-11.md", "session-12-moved.md")
#: The mid-trip case's trip-alone variant: a week away, and no move.
TRIP_ALONE_NOTES = ("session-04.md", "session-11.md")


@dataclass(frozen=True)
class Case:
    """One trip-versus-move case: its notes per variant, and the roles they play.

    The checks select claims by the note they came from: the **home** note
    states the residence, the **trip** note the week away, the **move** note
    the move. In the reader's case one note is both home and trip.
    """

    name: str
    label: str
    #: Variant (``move``, ``trip-alone``) → notes in deposit order.
    variants: dict[str, tuple[str, ...]]
    home: str
    trip: str
    move: str
    #: The move note's own date (its heading's).
    move_date: date
    #: Whether a trip claim lapsed on its own validity by the move counts as
    #: no longer current (the reader's case only: there the week ends three
    #: weeks before the move, and in the mid-trip case it cannot).
    lapse_counts: bool

    @property
    def notes(self) -> tuple[str, ...]:
        """Every note the case reads, in deposit order."""
        return self.variants["move"]


#: The cases ``particles benchmark trip-vs-move --case`` runs.
CASES: dict[str, Case] = {
    "reader": Case(
        name="reader",
        label="A week away, then a move a month later",
        variants={
            "move": ("reader-01.md", "reader-02-moved.md"),
            "trip-alone": ("reader-01.md",),
        },
        home="reader-01.md",
        trip="reader-01.md",
        move="reader-02-moved.md",
        move_date=date(2026, 10, 1),
        lapse_counts=True,
    ),
    "mid-trip": Case(
        name="mid-trip",
        label="Change of plans mid-trip",
        variants={"move": MOVE_NOTES, "trip-alone": TRIP_ALONE_NOTES},
        home="session-04.md",
        trip="session-11.md",
        move="session-12-moved.md",
        move_date=date(2026, 10, 9),
        lapse_counts=False,
    ),
}
#: The verb's default case: the reader's own.
DEFAULT_CASE = "reader"
#: The case a report without a ``case`` field measured (every report before).
LEGACY_CASE = "mid-trip"

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
