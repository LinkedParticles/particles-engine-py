# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-rot benchmark — currency, supersession, and source-trust leakage.

The fifth measurement package, on the system-benchmark mould. Where
every other harness scores a *settled* corpus at one instant, this one scores
the **same facts across simulated change**: a deterministic, seed-addressed
world (:mod:`.generator`) is deposited in time order into a scratch store,
and at day checkpoints every attribute is probed through the query op's
selection half. Three families are scored (:mod:`.scoring`) — currency
(``recall_current@k``, ``current_first``), supersession
(``stale_over_current``, ``stale_retained@k``), and poison
(``poison_leak@k`` over three untrusted channels) — with no aggregate score.

Three perception arms (:mod:`.oracle`, :mod:`.runner`) separate the Engine's
decision logic from its perception: ``oracle`` (scripted, zero LLM calls),
``probe`` (live §6.6 probe only), and ``live`` (the product). Report-only:
a run never touches a user store.
"""

from __future__ import annotations

from particles.benchmark.rot.generator import (
    GENERATOR_VERSION,
    SLOTS,
    check_value_invariants,
    generate_world,
    render_session,
    slot_state,
)
from particles.benchmark.rot.oracle import (
    OracleExtractor,
    OracleProbeProvider,
    RefusingProvider,
    oracle_claims,
    oracle_contradiction,
)
from particles.benchmark.rot.render import render_report
from particles.benchmark.rot.rescore import RescoreError, rescore_report
from particles.benchmark.rot.runner import (
    ARMS,
    RotArmError,
    RotRunEstimate,
    estimate_rot_run,
    render_estimate,
    run_rot_benchmark,
)
from particles.benchmark.rot.schema import (
    HitClass,
    ProbeResult,
    Rate,
    RotBenchmarkReport,
    RotMetrics,
    RotWorld,
    SlotState,
)
from particles.benchmark.rot.scoring import (
    SCORER_VERSION,
    classify,
    floor_sweep,
    metrics_for,
)

__all__ = [
    "ARMS",
    "GENERATOR_VERSION",
    "SCORER_VERSION",
    "SLOTS",
    "HitClass",
    "OracleExtractor",
    "OracleProbeProvider",
    "ProbeResult",
    "Rate",
    "RefusingProvider",
    "RescoreError",
    "RotArmError",
    "RotBenchmarkReport",
    "RotMetrics",
    "RotRunEstimate",
    "RotWorld",
    "SlotState",
    "check_value_invariants",
    "classify",
    "estimate_rot_run",
    "floor_sweep",
    "generate_world",
    "metrics_for",
    "oracle_claims",
    "oracle_contradiction",
    "render_estimate",
    "render_report",
    "render_session",
    "rescore_report",
    "run_rot_benchmark",
    "slot_state",
]
