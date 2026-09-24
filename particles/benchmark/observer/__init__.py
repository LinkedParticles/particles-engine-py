# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The two-project observer fixture (gate B).

Measures how often a belief vanishes from its own project under the
mechanisms — cross-project supersession, the generation cascade, and the lens's
own ``active_elsewhere`` cost — on a seeded two-project world driven through the
real ingest pipeline with scripted perception. Report-only, zero LLM calls.
CLI: ``particles benchmark observer``.
"""

from particles.benchmark.observer.generator import (
    generate_world,
    lines_on_day,
    render_memory_file,
)
from particles.benchmark.observer.render import render_report
from particles.benchmark.observer.runner import (
    ObserverFixtureError,
    metrics_for,
    run_observer_fixture,
)
from particles.benchmark.observer.schema import (
    GENERATOR_VERSION,
    LineKind,
    LineObservation,
    ObserverArm,
    ObserverMetrics,
    ObserverReport,
    ObserverWorld,
    VanishCause,
    WorldResult,
    WriteCensus,
)

__all__ = [
    "GENERATOR_VERSION",
    "LineKind",
    "LineObservation",
    "ObserverArm",
    "ObserverFixtureError",
    "ObserverMetrics",
    "ObserverReport",
    "ObserverWorld",
    "VanishCause",
    "WorldResult",
    "WriteCensus",
    "generate_world",
    "lines_on_day",
    "metrics_for",
    "render_memory_file",
    "render_report",
    "run_observer_fixture",
]
