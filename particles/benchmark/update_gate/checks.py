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
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

#: The checks version a new report runs under.
CHECKS_VERSION = 1

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


def _has_word(text: str, words: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)


def _from(row: Mapping[str, Any], session: str) -> bool:
    return any(str(source).endswith(session) for source in row["sources"])


def evaluate_v1(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    trip_alone: bool,
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


Checks = Callable[..., dict[str, Any]]

#: Every shipped checks version, and the row and event fields each one reads.
#: A version that needs a field an older report did not save names it here, so
#: re-scoring that report under it is refused rather than guessed.
CHECKS: dict[int, tuple[Checks, frozenset[str]]] = {
    1: (evaluate_v1, frozenset({"content", "status", "sources", "restatement"})),
}


class NotRescorable(ValueError):
    """A saved run lacks a field the requested checks version reads."""


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    trip_alone: bool,
    version: int = CHECKS_VERSION,
) -> dict[str, Any]:
    """Score one run's rows and events under checks ``version``.

    Raises:
        KeyError: If ``version`` is not a shipped checks version.
        NotRescorable: If a row lacks a field that version reads.
    """
    checks, fields = CHECKS[version]
    for row in rows:
        missing = fields - set(row)
        if missing:
            raise NotRescorable(
                f"checks version {version} reads {sorted(missing)}, which this run did not save"
            )
    return checks(rows, events, trip_alone=trip_alone)


def note_name(uri: str) -> str:
    """The fixture note a corpus-entry URI names (its last path segment)."""
    return PurePosixPath(uri.split("?", 1)[0]).name


def read_store(db: Path) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Everything the checks read, from a finished run's SQLite store, read-only.

    Every particle row (id, content, status, status reason, the restatement
    flag and the note names of its SOURCE refs) and the checked events in log
    order, with their full payloads. The report saves both, which is what lets
    a saved report re-score.
    """
    uri = f"file:{db}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        uris = dict(conn.execute("select entry_id, uri_r from corpus_entries"))
        rows: list[dict[str, Any]] = []
        for pid, content, status, reason, provenance, contributors in conn.execute(
            "select id, content, status, status_reason, provenance_json, "
            "contributors_json from particles order by content, id"
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
