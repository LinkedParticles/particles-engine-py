# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The Particles memory through the door arm, end to end, for free.

The child process runs isolated, with scripted perception and the
real local encoder, so it makes no model call and needs no key; it is in the
integration tier because it loads the real encoder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.benchmark.rot.door_runner import baseline_plans, particles_plan, run_door_benchmark
from particles.benchmark.rot.manifest import ArmsManifest

pytestmark = pytest.mark.integration


def test_scripted_particles_door_quick_world(tmp_path: Path) -> None:
    plan = particles_plan(
        role="variant", variant_note="scripted perception, not the product", scripted=True
    )
    report = run_door_benchmark(
        [plan, *baseline_plans()[1:]],
        tier="quick",
        seeds=[42],
        work_dir=tmp_path,
        manifest=ArmsManifest(),
    )
    particles = report.memories[0]
    build = particles.builds[0]
    assert build.notes == [] or all("failed" not in n for n in build.notes)
    assert build.write_methods.get("import_session", 0) > 0
    assert build.write_methods.get("import_page", 0) == 1
    assert build.usd == 0.0
    assert particles.metrics.current_in_door.value == 1.0
    assert particles.metrics.poison_leak.value == 0.0
    assert build.probes[0].door_text.lstrip().startswith("{")
