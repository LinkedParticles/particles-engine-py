# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The trip-versus-move case's judge-free checks (item 5).

The checks read only a store's rows and events, never a model's opinion of
them, and they were fixed before any run. They are versioned as the rot
scorer's lexicons are: a change adds a version to :data:`CHECKS` and never
edits a shipped one, so a saved report re-scores under the version it ran.

Version 1 is ``scripts/trip_vs_move_e2e.py``'s ``evaluate()``, unchanged.

Move case (sessions 4, 11 and 12-moved, then consolidation):

* (i) no ACTIVE claim that is not a re-anchor restatement names Delhi
  together with a residence or flight word, and every ACTIVE restatement that
  names a flight carries an ``as of`` ISO date and a past-tense ``was`` or
  ``were``;
* (ii) no restatement is kept back for contradicting the claim that retired
  its anchor, and every restatement the re-anchor pass wrote is ACTIVE;
* (iii) no ACTIVE session-11 claim that is not a restatement matches
  ``for a week`` or ``work trip``: the trip is retired.

Trip alone (sessions 4 and 11): an ACTIVE session-4 claim names Delhi with
live, lives or home, so a week away is not read as a move.

Version 2 defines both cases, the reader's and the mid-trip
one, with one structure over the case's home, trip and move notes
(:class:`~particles.benchmark.update_gate.fixtures.Case`). Trip alone needs
both halves: (a) the home stays in Delhi and (b) the trip is current. The
move needs (i) no current Delhi residence or flight, (ii) version 1's check,
(iii) the week not read as current, and (iv) the move kept. Version 1 has no
reader case.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from particles.benchmark.update_gate.fixtures import CASES, LEGACY_CASE, Case

#: The checks version a new report runs under.
CHECKS_VERSION = 2

#: The contributor identity a re-anchor restatement carries (``core.reanchor``).
REANCHOR_ACTOR = "consolidation:reanchor"
#: The operator events the checks read; the report saves exactly these.
CHECKED_EVENTS = ("UPDATE_SUPERSESSION", "DEPENDENT_CLAIM_EXAMINED")

RESIDENCE_OR_FLIGHT = (
    "live",
    "lives",
    "home",
    "flat",
    "fly",
    "flies",
    "flying",
    "flight",
    "back to",
)
FLIGHT = ("fly", "flies", "flying", "flight")
RESIDENCE = ("live", "lives", "home")
#: Store rows the printed table shows, by the words that place them in the case.
CASE_WORDS = ("mumbai", "delhi", "fly", "flight", "trip", "home", "bandra", "lajpat", "sandeep")


#: Version 2's residence words, and with them version 1's flight words.
RESIDENCE_V2 = ("live", "lives", "living", "home", "flat", "reside", "resides", "residence")
RESIDENCE_OR_FLIGHT_V2 = (*RESIDENCE_V2, "fly", "flies", "flying", "flight", "back to")
#: Version 2's trip words: a claim that matches one reads the week as a stay.
TRIP_WORDS = re.compile(
    r"\bfor a week\b|\bweek-long\b|\btrip\b|\bvisit\b|\bvisiting\b|\btemporary\b|\btemporarily\b",
    re.IGNORECASE,
)
#: Version 2's checks that name the claims satisfying them, not offending ones.
SUPPORTING = frozenset({"residence_claims", "a_home_in_delhi", "b_trip_current", "iv_move_kept"})


def _has_word(text: str, words: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)


def _from(row: Mapping[str, Any], session: str) -> bool:
    return any(str(source).endswith(session) for source in row["sources"])


def evaluate_v1(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    trip_alone: bool,
    case: Case | None = None,
) -> dict[str, Any]:
    """Score one run's store under checks version 1.

    ``rows`` are the store's particles as ``{content, status, sources,
    restatement}`` mappings, ``sources`` the corpus-entry URIs (or note names)
    of the SOURCE refs and ``restatement`` whether the re-anchor pass wrote
    the claim. ``events`` are ``(event_type, payload)`` in log order. Returns
    each check's offending claims and ``pass``.
    """
    active = [r for r in rows if r["status"] == "ACTIVE"]
    if trip_alone:
        hits = [
            r["content"]
            for r in active
            if _from(r, "session-04.md")
            and "delhi" in r["content"].lower()
            and _has_word(r["content"], RESIDENCE)
        ]
        return {"residence_claims": hits, "pass": bool(hits)}

    delhi = [
        r["content"]
        for r in active
        if not r["restatement"]
        and "delhi" in r["content"].lower()
        and _has_word(r["content"], RESIDENCE_OR_FLIGHT)
    ]
    shape = [
        r["content"]
        for r in active
        if r["restatement"]
        and _has_word(r["content"], FLIGHT)
        and not (
            re.search(r"\bas of \d{4}-\d{2}-\d{2}\b", r["content"], re.IGNORECASE)
            and _has_word(r["content"], ("was", "were"))
        )
    ]
    winners = {
        p["retired_id"]: p.get("winner_id")
        for kind, p in events
        if kind == "UPDATE_SUPERSESSION" and p.get("retired_id")
    }
    held = [
        p.get("restatement")
        for kind, p in events
        if kind == "DEPENDENT_CLAIM_EXAMINED"
        and p.get("outcome") == "unrestated"
        and str(p.get("why", "")).startswith("the restatement contradicts")
        and p.get("other_id") is not None
        and p.get("other_id") == winners.get(p.get("retired_id"))
    ]
    not_active = [r["content"] for r in rows if r["restatement"] and r["status"] != "ACTIVE"]
    trip = [
        r["content"]
        for r in active
        if not r["restatement"]
        and _from(r, "session-11.md")
        and re.search(r"\bfor a week\b|\bwork trip\b", r["content"], re.IGNORECASE)
    ]
    return {
        "i_delhi_residence_or_flight": delhi,
        "i_restatement_shape": shape,
        "ii_held_by_winner": held,
        "ii_restatements_not_active": not_active,
        "iii_trip_active": trip,
        "pass": not (delhi or shape or held or not_active or trip),
    }


def _held_by_winner(events: Sequence[tuple[str, Mapping[str, Any]]]) -> list[Any]:
    """Restatements kept back for contradicting the claim that retired their anchor."""
    winners = {
        p["retired_id"]: p.get("winner_id")
        for kind, p in events
        if kind == "UPDATE_SUPERSESSION" and p.get("retired_id")
    }
    return [
        p.get("restatement")
        for kind, p in events
        if kind == "DEPENDENT_CLAIM_EXAMINED"
        and p.get("outcome") == "unrestated"
        and str(p.get("why", "")).startswith("the restatement contradicts")
        and p.get("other_id") is not None
        and p.get("other_id") == winners.get(p.get("retired_id"))
    ]


def _dated(row: Mapping[str, Any]) -> bool:
    """A dated restatement: the re-anchor pass wrote it, with an ISO ``as of`` and was/were."""
    text = str(row["content"])
    return bool(
        row["restatement"]
        and re.search(r"\bas of \d{4}-\d{2}-\d{2}\b", text, re.IGNORECASE)
        and _has_word(text, ("was", "were"))
    )


def _lapsed(row: Mapping[str, Any], by: date) -> bool:
    """Whether the claim's own ``valid_until`` falls on or before ``by``."""
    until = row.get("valid_until")
    if until is None:
        return False
    if isinstance(until, str):
        until = datetime.fromisoformat(until)
    return bool(until.date() <= by)


def evaluate_v2(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    trip_alone: bool,
    case: Case | None = None,
) -> dict[str, Any]:
    """Score one run's store under checks version 2.

    ``case`` names the home, trip and move notes; the mid-trip case when
    omitted. A ``missing`` list names each check that wanted a claim and
    found none.
    """
    case = case or CASES[LEGACY_CASE]
    active = [r for r in rows if r["status"] == "ACTIVE"]
    if trip_alone:
        home = [
            r["content"]
            for r in active
            if _from(r, case.home)
            and "delhi" in r["content"].lower()
            and _has_word(r["content"], RESIDENCE_V2)
        ]
        current = [
            r["content"]
            for r in active
            if not r["restatement"] and _from(r, case.trip) and "mumbai" in r["content"].lower()
        ]
        missing = [
            name
            for name, found in (("a_home_in_delhi", home), ("b_trip_current", current))
            if not found
        ]
        return {
            "a_home_in_delhi": home,
            "b_trip_current": current,
            "missing": missing,
            "pass": not missing,
        }

    def delhi_home(r: Mapping[str, Any]) -> bool:
        return "delhi" in r["content"].lower() and _has_word(r["content"], RESIDENCE_OR_FLIGHT_V2)

    delhi = [r["content"] for r in active if not r["restatement"] and delhi_home(r)]
    shape = [r["content"] for r in active if r["restatement"] and delhi_home(r) and not _dated(r)]
    held = _held_by_winner(events)
    not_active = [r["content"] for r in rows if r["restatement"] and r["status"] != "ACTIVE"]
    trip = [
        r["content"]
        for r in active
        if _from(r, case.trip)
        and TRIP_WORDS.search(r["content"])
        and not _dated(r)
        and not (case.lapse_counts and _lapsed(r, case.move_date))
    ]
    moved = [
        r["content"] for r in active if _from(r, case.move) and "mumbai" in r["content"].lower()
    ]
    missing = [] if moved else ["iv_move_kept"]
    return {
        "i_delhi_residence_or_flight": delhi,
        "i_restatement_shape": shape,
        "ii_held_by_winner": held,
        "ii_restatements_not_active": not_active,
        "iii_trip_current": trip,
        "iv_move_kept": moved,
        "missing": missing,
        "pass": not (delhi or shape or held or not_active or trip or missing),
    }


Checks = Callable[..., dict[str, Any]]

_V1_FIELDS = frozenset({"content", "status", "sources", "restatement"})

#: Every shipped checks version, and per case the row fields it reads. A
#: version that needs a field an older report did not save names it here, so
#: re-scoring that report under it is refused rather than guessed; a case a
#: version does not name is one it does not define.
CHECKS: dict[int, tuple[Checks, dict[str, frozenset[str]]]] = {
    1: (evaluate_v1, {"mid-trip": _V1_FIELDS}),
    2: (evaluate_v2, {"mid-trip": _V1_FIELDS, "reader": _V1_FIELDS | {"valid_until"}}),
}


class NotRescorable(ValueError):
    """A saved run lacks a field the requested checks version reads, or its case."""


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    trip_alone: bool,
    case: str,
    version: int = CHECKS_VERSION,
) -> dict[str, Any]:
    """Score one run's rows and events for ``case`` under checks ``version``.

    Raises:
        KeyError: If ``version`` is not a shipped checks version.
        NotRescorable: If that version does not define ``case``, or a row
            lacks a field it reads.
    """
    checks, fields_by_case = CHECKS[version]
    if case not in fields_by_case:
        raise NotRescorable(f"checks version {version} does not define the {case} case")
    fields = fields_by_case[case]
    for row in rows:
        missing = fields - set(row)
        if missing:
            raise NotRescorable(
                f"checks version {version} reads {sorted(missing)}, which this run did not save"
            )
    return checks(rows, events, trip_alone=trip_alone, case=CASES[case])


def note_name(uri: str) -> str:
    """The fixture note a corpus-entry URI names (its last path segment)."""
    return PurePosixPath(uri.split("?", 1)[0]).name


def read_store(db: Path) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Everything the checks read, from a finished run's SQLite store, read-only.

    Every particle row (id, content, status, status reason, the restatement
    flag, the note names of its SOURCE refs and its world-time fields) and the
    checked events in log order, with their full payloads. The report saves both, which is what lets
    a saved report re-score.
    """
    uri = f"file:{db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        uris = dict(conn.execute("select entry_id, uri_r from corpus_entries"))
        rows: list[dict[str, Any]] = []
        for (
            pid,
            content,
            status,
            reason,
            provenance,
            contributors,
            valid_from,
            valid_until,
            valid_kind,
        ) in conn.execute(
            "select id, content, status, status_reason, provenance_json, "
            "contributors_json, valid_from, valid_until, valid_kind "
            "from particles order by content, id"
        ):
            refs = json.loads(provenance or "[]")
            people = json.loads(contributors or "null") or []
            rows.append(
                {
                    "id": pid,
                    "content": content,
                    "status": status,
                    "reason": reason,
                    "sources": [
                        note_name(uris.get(r.get("corpus_entry_id"), "") or "")
                        for r in refs
                        if r.get("type") == "SOURCE"
                    ],
                    "restatement": any(c.get("id") == REANCHOR_ACTOR for c in people),
                    "valid_from": valid_from,
                    "valid_until": valid_until,
                    "valid_kind": valid_kind,
                }
            )
        marks = ",".join("?" for _ in CHECKED_EVENTS)
        events = [
            (kind, json.loads(payload or "{}"))
            for kind, payload in conn.execute(
                "select event_type, payload from operator_events "
                f"where event_type in ({marks}) order by rowid",
                CHECKED_EVENTS,
            )
        ]
    return rows, events


def case_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows the printed table shows: those naming a word of the case."""
    return [r for r in rows if any(w in str(r["content"]).lower() for w in CASE_WORDS)]
