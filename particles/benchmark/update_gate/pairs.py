# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Replay the registered pair sets through the production probes.

Each sample of a pair asks the two production calls rung 2.5 depends on: the
contradiction probe as the sweep asks it (newer claim first, the attribution
pre-filter included) and the update probe with its second reading
(``read_update_slot``). The update probe is asked whatever the
contradiction probe answered, so every gate in :mod:`.scoring` scores from the
same calls.

The replay builds no store, so it runs in process. The two purposes it asks,
``semantic_lint`` and ``verification``, are routed to the chosen model through
:func:`~particles.llm.override_providers`; config is never written.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

from particles.benchmark.update_gate.fixtures import HELDOUT_SETS, PAIR_SETS, pair_set, sha256
from particles.benchmark.update_gate.schema import (
    FixtureFile,
    PairRow,
    SetResult,
    Spend,
    UpdateGateReport,
)
from particles.benchmark.update_gate.scoring import set_bars
from particles.config import ProviderSelection
from particles.egress import current_policy
from particles.llm import override_providers, provider_for_selection, selection_endpoint

#: The purposes one sample asks.
PROBE_PURPOSES = ("semantic_lint", "verification")
#: Calls one sample can make: the contradiction probe, the update probe, and
#: the second reading a first ``CHANGES`` gets (an upper bound).
CALLS_PER_SAMPLE = 3


class ReplayRefused(RuntimeError):
    """The replay cannot start: the chosen model's endpoint is not allowed."""


def load_rows(set_name: str) -> list[dict[str, Any]]:
    """The registered rows of one pair set."""
    rows: list[dict[str, Any]] = json.loads(pair_set(set_name).read_text())["rows"]
    return rows


def fixture_digests(sets: Sequence[str]) -> dict[str, str]:
    """The SHA-256 of each set's file, as the report records it."""
    return {PAIR_SETS[s]: sha256(pair_set(s)) for s in sets}


def check_egress(selection: ProviderSelection) -> None:
    """Refuse before any call when ``network.egress`` does not allow the model's endpoint.

    An override is never checked by the registry, so the replay checks the
    selection it routes itself.

    Raises:
        ReplayRefused: If the posture refuses the endpoint.
    """
    policy = current_policy()
    if not policy.refusing:
        return
    host, port, provider = selection_endpoint(selection)
    if not policy.allows(host, port):
        raise ReplayRefused(
            f"network.egress ({policy.posture}) does not allow {provider} at "
            f"{host or '<unknown>'}{f':{port}' if port else ''}; nothing was asked"
        )


@contextlib.contextmanager
def pinned_probes(selection: ProviderSelection) -> Iterator[None]:
    """Route both probe purposes to ``selection`` for the block, config untouched."""
    provider = provider_for_selection(selection, context="--model")
    with override_providers(dict.fromkeys(PROBE_PURPOSES, provider)):  # type: ignore[arg-type]
        yield


async def sample(older: str, newer: str) -> dict[str, Any]:
    """One sample: the contradiction probe's verdict and the settled update verdict.

    ``contradiction`` is ``None`` when the probe could not complete, and
    ``slot`` is ``None`` when the update probe gave no usable verdict.
    """
    # Deferred import: the ingest pipeline is a heavy import the pure halves
    # of this package (and their tests) never need.
    from particles.ingest.pipeline import _has_contradiction_signal, read_update_slot

    contradiction = await _has_contradiction_signal(newer, older)
    reading = await read_update_slot(older, newer)
    return {
        "contradiction": contradiction,
        "slot": None if reading.verdict is None else reading.verdict.value,
    }


async def replay(
    sets: Sequence[str],
    *,
    samples: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Ask every row of every set ``samples`` times; the rows gain a ``samples`` list.

    The caller opens :func:`pinned_probes` (or a scripted override) around it.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for name in sets:
        rows = []
        for row in load_rows(name):
            got = [await sample(row["older"], row["newer"]) for _ in range(samples)]
            rows.append({**row, "samples": got})
            if progress is not None:
                progress(f"{name} {row['id']}: " + " ".join(_mark(s) for s in got))
        out[name] = rows
    return out


def _mark(s: dict[str, Any]) -> str:
    c = s.get("contradiction")
    return f"{'Y' if c else ('?' if c is None else 'n')}/{s.get('slot')}"


def call_bound(sets: Sequence[str], samples: int) -> int:
    """The most calls a replay of ``sets`` can make (rows × samples × readings)."""
    return sum(len(load_rows(s)) for s in sets) * samples * CALLS_PER_SAMPLE


def build_report(
    rows_by_set: dict[str, list[dict[str, Any]]],
    *,
    model: str,
    samples: int,
    spend: Spend,
    notes: Sequence[str] = (),
) -> UpdateGateReport:
    """The update-gate report of record from replayed rows."""
    from particles import __version__

    return UpdateGateReport(
        sdk_version=__version__,
        run_at=datetime.now(UTC),
        model=model,
        samples_per_pair=samples,
        fixture=[
            FixtureFile(name=n, sha256=d) for n, d in fixture_digests(list(rows_by_set)).items()
        ],
        sets={
            name: SetResult(
                rows=[PairRow.model_validate(r) for r in rows],
                bars=set_bars(rows, heldout=name in HELDOUT_SETS),
            )
            for name, rows in rows_by_set.items()
        },
        spend=spend,
        notes=list(notes),
    )


def rescore(report: UpdateGateReport) -> UpdateGateReport:
    """Re-score a saved report's samples under the current marker and bars; no call."""
    sets = {}
    for name, result in report.sets.items():
        rows = [r.model_dump(by_alias=True) for r in result.rows]
        sets[name] = SetResult(rows=result.rows, bars=set_bars(rows, heldout=name in HELDOUT_SETS))
    return report.model_copy(
        update={
            "sets": sets,
            "notes": [*report.notes, "Re-scored from the saved samples with no LLM call."],
        }
    )
