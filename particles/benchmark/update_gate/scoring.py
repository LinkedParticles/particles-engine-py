# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Score the registered pair sets under three gates.

Each sample of a pair is the contradiction probe's verdict and the settled
update verdict, and three gates are scored from the same calls:

* ``production`` retires the older claim on a settled ``CHANGES``, plus a
  contradiction YES when the older claim carries a past-value marker. It is
  what 1.187.0 ships, and the records of the update rule call it ``fallback``.
* ``changes-only`` retires on a settled ``CHANGES`` alone (``proposed`` in
  those records).
* ``before-1.187`` retires on a contradiction YES and a settled ``CHANGES``
  (``shipped`` in those records).

The registered bars apply to ``production``. Each set's result carries two
judge-free readings: ``bar`` (strict, as registered) and ``regression`` (no
control that ``production`` retires is kept by ``before-1.187`` on the same
samples). Pure: a saved report re-scores with no call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from particles.core.past_value import records_past_value

PRODUCTION = "production"
CHANGES_ONLY = "changes-only"
BEFORE_1_187 = "before-1.187"
GATES = (PRODUCTION, CHANGES_ONLY, BEFORE_1_187)

#: The names the update rule's records use for the same gates.
ADR_0313_NAMES = {PRODUCTION: "fallback", CHANGES_ONLY: "proposed", BEFORE_1_187: "shipped"}

#: The class printed and never barred: the trip case's own pairs, as registered.
#: Rows labelled ``report`` (the past-marker set's ``past-unmarked``) are never
#: barred either.
CASE_CLASS = "case"


def gate_retires(sample: Mapping[str, Any], gate: str, older: str = "") -> bool:
    """Whether one sample retires the older claim under ``gate``."""
    changes = sample.get("slot") == "changes"
    if gate == PRODUCTION:
        gate = BEFORE_1_187 if records_past_value(older) else CHANGES_ONLY
    if gate == CHANGES_ONLY:
        return changes
    if gate == BEFORE_1_187:
        return changes and sample.get("contradiction") is True
    raise ValueError(f"unknown gate {gate!r}")


def retirements(row: Mapping[str, Any], gate: str) -> int:
    """How many of a row's samples retire its older claim under ``gate``."""
    older = str(row.get("older", ""))
    return sum(gate_retires(s, gate, older) for s in row["samples"])


def is_barred(row: Mapping[str, Any]) -> bool:
    """Whether a row counts toward the bars (not ``report``, not an unbarred class)."""
    return row.get("class") != CASE_CLASS and row.get("expected") != "report"


def gate_result(rows: Sequence[Mapping[str, Any]], gate: str, *, heldout: bool) -> dict[str, Any]:
    """One gate's tally over a set: positives retired, control retirements, ``ok``.

    A positive passes when the gate retired it in at least two samples; a
    held-out set lets one positive miss, a development set none. A control
    fails on any retirement in any sample.
    """
    barred = [r for r in rows if is_barred(r)]
    positives = [r for r in barred if r["expected"] == "changes"]
    controls = [r for r in barred if r["expected"] == "not_changes"]
    hit = [r for r in positives if retirements(r, gate) >= 2]
    false = [(str(r["id"]), retirements(r, gate)) for r in controls if retirements(r, gate)]
    control_samples = sum(len(r["samples"]) for r in controls)
    need = len(positives) - (1 if heldout else 0)
    return {
        "positives_retired": len(hit),
        "positives": len(positives),
        "positive_misses": [str(r["id"]) for r in positives if r not in hit],
        "control_retirements": sum(n for _, n in false),
        "control_samples": control_samples,
        "control_rows_retired": [i for i, _ in false],
        "ok": len(hit) >= need and not false,
    }


def set_bars(rows: Sequence[Mapping[str, Any]], *, heldout: bool) -> dict[str, Any]:
    """Every gate's tally over one set, the two readings, and the unbarred rows.

    ``bar`` is the registered bar on ``production``. ``regression`` holds when
    every control sample ``production`` retires is one ``before-1.187`` also
    retires: the gate the fix shipped adds no false retirement to the one it
    replaced.
    """
    gates = {g: gate_result(rows, g, heldout=heldout) for g in GATES}
    controls = [r for r in rows if is_barred(r) and r["expected"] == "not_changes"]
    added = [
        str(r["id"])
        for r in controls
        if any(
            gate_retires(s, PRODUCTION, str(r.get("older", "")))
            and not gate_retires(s, BEFORE_1_187, str(r.get("older", "")))
            for s in r["samples"]
        )
    ]
    reported = {
        str(r["id"]): {
            "class": r.get("class"),
            "expected": r.get("expected"),
            **{g: retirements(r, g) for g in GATES},
            "of": len(r["samples"]),
        }
        for r in rows
        if not is_barred(r)
    }
    return {
        "gates": gates,
        "bar": gates[PRODUCTION]["ok"],
        "regression": not added,
        "controls_added_by_production": added,
        "reported": reported,
    }
