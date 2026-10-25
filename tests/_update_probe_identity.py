# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The update probe's identity replay: the suites under scripted completions.

The ported update-probe ladder is accepted on zero differences from the
parent commit's ``read_update_slot``, pair for pair, on the trip-versus-move
and update-gate suites. A live model is not deterministic and its
reading is not what the port changes, so the replay fixes the completions: each
call is answered from a seeded palette keyed on the seed, the purpose, the
budget and the two claims, covering every reply class the probe parses (a
same-slot ``CHANGES`` or ``FIXED``, a ``NO``, an off-protocol ``YES``, a reply
cut to nothing, a failed call). Every pair of the four update-gate pair sets
and every update pair of the trip-versus-move scripted table is replayed under
each seed, with and without a retry budget larger than the probe's own, and
with the breaker closed, open from the start, or tripping after the first call
an input sends. The scripted seam behaves as the real one does: while the
breaker is open no call is sent and the reply is ``None``.

``collect()`` returns, per input, the settled verdict, the deciding purpose,
and the exact sequence of calls made. The golden file
``tests/fixtures/rung_ladder/update_probe_parent.json`` was written by
``python -m tests._update_probe_identity`` before the port, on the commit
the ladder was built on; ``tests/test_rung_ladder.py`` replays it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from particles.benchmark.update_gate.scripted import unfence_claim

GOLDEN = Path(__file__).parent / "fixtures" / "rung_ladder" / "update_probe_parent.json"
_PAIR_SETS = ("gate_pairs", "gate_pairs_heldout", "past_value_pairs", "past_marker_heldout")
_CLAIMS = re.compile(r"Claim A:\s*(?P<a>.*?)\n\s*\nClaim B:\s*(?P<b>.*)\Z", re.DOTALL)

#: The reply palette. ``None`` is a failed call; ``""`` a reply cut to nothing.
PALETTE: tuple[str | None, ...] = (
    "REASON: one slot.\nSLOT: CHANGES\nVERDICT: YES",
    "REASON: one slot.\nSLOT: FIXED\nVERDICT: YES",
    "REASON: two attributes.\nSLOT: NONE\nVERDICT: NO",
    "REASON: one slot.\nVERDICT: YES",
    "",
    None,
)
#: ``(seed, retry budget larger than the probe's own, breaker)``. The breaker
#: is ``closed``, ``open`` from the start, or ``opens`` after the first call
#: an input sends, the case where it trips between the two readings.
SCENARIOS: tuple[tuple[int, bool, str], ...] = (
    (0, False, "closed"),
    (1, False, "closed"),
    (2, True, "closed"),
    (3, True, "closed"),
    (4, True, "opens"),
    (5, False, "open"),
    (6, False, "opens"),
)


def inputs() -> list[tuple[str, str, str]]:
    """``(input id, earlier, later)`` for every pair the replay asks."""
    root = Path(__file__).parent / "fixtures" / "update_probe"
    out: list[tuple[str, str, str]] = []
    for name in _PAIR_SETS:
        rows = json.loads((root / f"{name}.json").read_text())["rows"]
        out += [(f"{name}:{r['id']}", r["older"], r["newer"]) for r in rows]
    table = Path(__file__).parents[1] / "particles/benchmark/update_gate/scripted_table.json"
    for i, row in enumerate(json.loads(table.read_text())["update"]):
        out.append((f"trip-vs-move:{i}", row["a"], row["b"]))
    return out


def _reply(seed: int, purpose: str, max_tokens: int, prompt: str) -> str | None:
    m = _CLAIMS.search(prompt)
    # The claims are fenced since the update probe's F3 hardening; the reply is
    # keyed on the claim texts, so the golden file is unchanged by the fence.
    tail = f"{unfence_claim(m.group('a'))}\x1f{unfence_claim(m.group('b'))}" if m else prompt
    digest = hashlib.sha256(f"{seed}\x1f{purpose}\x1f{max_tokens}\x1f{tail}".encode()).digest()
    return PALETTE[digest[0] % len(PALETTE)]


class ScriptedSeam:
    """The breaker seam as the real one behaves: no call is sent while it is open."""

    def __init__(self, seed: int, breaker: str) -> None:
        self.seed = seed
        self.breaker = breaker
        self.open = breaker == "open"
        self.calls: list[list[Any]] = []

    def reset(self) -> None:
        """Start one input: no calls yet, the breaker as the scenario starts it."""
        self.calls = []
        self.open = self.breaker == "open"

    def circuit_open(self) -> bool:
        """``llm_circuit_open`` as the scenario has it now."""
        return self.open

    async def call(
        self,
        prompt: str,
        *,
        max_tokens: int,
        purpose: str,
        empty_reply_as_text: bool = False,
        **_: Any,
    ) -> str | None:
        """``_llm_call``: ``None`` and no call while open, else the scripted reply."""
        if self.open:
            return None
        self.calls.append([purpose, max_tokens])
        if self.breaker == "opens":
            self.open = True
        return _reply(self.seed, purpose, max_tokens, prompt)


async def collect() -> dict[str, Any]:
    """Every input under every scenario: verdict, deciding purpose, calls sent."""
    from unittest import mock

    from particles.config import get_config, reset_config
    from particles.ingest import pipeline
    from particles.llm.breaker import PROBE_MAX_TOKENS

    results: dict[str, Any] = {}
    for seed, larger_retry, breaker in SCENARIOS:
        reset_config()
        if larger_retry:
            get_config().audit.verify_retry_max_tokens = 2 * PROBE_MAX_TOKENS
        seam = ScriptedSeam(seed, breaker)
        with (
            mock.patch.object(pipeline, "_llm_call", seam.call),
            mock.patch.object(pipeline, "llm_circuit_open", seam.circuit_open),
            mock.patch.object(pipeline, "record_unusable_reply", lambda *a, **k: None),
        ):
            for key, earlier, later in inputs():
                seam.reset()
                reading = await pipeline.read_update_slot(earlier, later)
                results[f"{seed}:{key}"] = {
                    "verdict": None if reading.verdict is None else reading.verdict.value,
                    "purpose": reading.purpose,
                    "calls": seam.calls,
                }
    reset_config()
    return results


def main() -> None:
    """Write the golden file from the code as it is."""
    out = asyncio.run(collect())
    GOLDEN.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(out)} readings written to {GOLDEN}", file=sys.stderr)


if __name__ == "__main__":
    main()
