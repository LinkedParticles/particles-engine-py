# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Print the restatement judge's ``SAME`` verdicts as a labelling sheet.

Read-only. The restatement match records each duplicate-judge verdict in the
probe ledger under the ``duplicate`` kind, keyed on the two claims' §6.10
content hashes. This joins every ``SAME`` (``verdict = 1``) row back to the
claims that carry those hashes and prints both texts, one pair per row, with an
empty ``label`` column for the owner to fill with ``same``, ``changed`` or
``different``. During the judge's admission period both claims are stored (the
pass still mints the new one), so every recorded pair resolves.

The labelled sheet is what admits the judge: at least 100 labelled ``SAME``
verdicts whose 95 % Wilson lower bound on precision is at least 0.95
(§ Acceptance). ``--summary`` prints that bound for a labelled file.

    python -m particles.operations.restatement_sheet > restatements.tsv
    python -m particles.operations.restatement_sheet --db /path/to/particles.db
    python -m particles.operations.restatement_sheet --summary restatements.tsv

The sheet holds claim text from the store it reads: keep it out of the
repository.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_DB = Path("particles.db")
_COLUMNS = ("verdict_id", "recorded_at", "earlier", "later", "label")


def wilson_lower(correct: int, total: int, z: float = 1.96) -> float:
    """The Wilson score interval's lower bound for ``correct`` of ``total``."""
    if total == 0:
        return 0.0
    p = correct / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - margin) / (1 + z * z / total)


def sheet_rows(conn: sqlite3.Connection) -> list[tuple[str, ...]]:
    """One row per recorded ``SAME`` verdict whose two claims are both stored."""
    by_hash: dict[str, str] = {}
    for content, key in conn.execute(
        "SELECT content, content_norm_hash FROM particles WHERE content_norm_hash IS NOT NULL"
    ):
        by_hash.setdefault(key, content)
    rows: list[tuple[str, ...]] = []
    for row_id, recorded_at, hash_a, hash_b in conn.execute(
        "SELECT id, recorded_at, hash_a, hash_b FROM probe_verdicts "
        "WHERE probe_kind = 'duplicate' AND verdict = 1 ORDER BY id"
    ):
        a, b = by_hash.get(hash_a), by_hash.get(hash_b)
        if a is None or b is None:
            continue
        rows.append((str(row_id), str(recorded_at), a, b, ""))
    return rows


def summarise(labels: Iterable[str]) -> tuple[int, int, float]:
    """``(labelled, correct, Wilson lower bound)`` over a sheet's label column."""
    filled = [label.strip().lower() for label in labels if label.strip()]
    correct = sum(1 for label in filled if label == "same")
    return len(filled), correct, wilson_lower(correct, len(filled))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite store to read")
    parser.add_argument("--summary", type=Path, help="a labelled sheet to summarise instead")
    args = parser.parse_args(argv)

    if args.summary is not None:
        with args.summary.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            labelled, correct, bound = summarise(row.get("label") or "" for row in reader)
        verdict = "admitted" if labelled >= 100 and bound >= 0.95 else "not yet admitted"
        sys.stdout.write(
            f"{labelled} labelled, {correct} same; Wilson 95 % lower bound {bound:.3f}\n{verdict}\n"
        )
        return 0

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = sheet_rows(conn)
    finally:
        conn.close()
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(_COLUMNS)
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
