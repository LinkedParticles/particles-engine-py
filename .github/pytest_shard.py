# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run one shard of the test suite: ``pytest -p pytest_shard --shard I/N``.

CI splits the suite across N runners and reports one ``pytest`` check that
passes only when every shard passed. Each shard collects the whole suite and
keeps the whole test modules assigned to it, so module and class fixtures are
set up once, as in an unsharded run. Modules are dealt largest first to the
lightest shard so far, weighed by the size of their source file: on this
suite that tracks run time more closely than the number of tests does. The
modules dealt are those holding a test that every other deselection (``-m``,
``-k``) kept, so the N shards together run exactly the tests one unsharded
run would, each once. The assignment depends only on the checkout and the
collected node ids, which every shard sees alike.

Standard library and pytest only: CI loads it from ``.github`` on
``PYTHONPATH``, with nothing installed.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("shard").addoption(
        "--shard",
        default=None,
        metavar="I/N",
        help="Run shard I of N (1-based) of the selected tests, by whole module.",
    )


def parse_shard(spec: str) -> tuple[int, int]:
    """``"I/N"`` as ``(I, N)``; a usage error unless ``1 <= I <= N``."""
    index, sep, total = spec.partition("/")
    if not sep or not index.isdigit() or not total.isdigit() or not 1 <= int(index) <= int(total):
        raise pytest.UsageError(f"--shard takes I/N with 1 <= I <= N, not {spec!r}")
    return int(index), int(total)


def assign(modules: dict[str, int], total: int) -> dict[str, int]:
    """Each module's shard (0-based), given its weight: heaviest first, onto the lightest."""
    loads = [0] * total
    owner: dict[str, int] = {}
    for module in sorted(modules, key=lambda m: (-modules[m], m)):
        shard = min(range(total), key=lambda s: (loads[s], s))
        owner[module] = shard
        loads[shard] += modules[module]
    return owner


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    spec = config.getoption("--shard")
    if not spec:
        return
    index, total = parse_shard(spec)
    weights: dict[str, int] = {}
    for item in items:
        module = item.nodeid.split("::", 1)[0]
        if module not in weights:
            weights[module] = item.path.stat().st_size
    owner = assign(weights, total)
    keep = [i for i in items if owner[i.nodeid.split("::", 1)[0]] == index - 1]
    dropped = [i for i in items if owner[i.nodeid.split("::", 1)[0]] != index - 1]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = keep
