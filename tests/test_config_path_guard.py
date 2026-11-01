# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""No code builds a config or store path except the loader.

The loader reads ``PARTICLES_CONFIG`` or the user file and never the working
directory. Code that built ``Path.cwd() / "config.yaml"`` itself (the
installer did, until) would read or write a file a cloned repository
controls, and the same holds for a ``particles.db`` or ``corpus_blobs`` path
built outside the config. So every such construction outside
``particles/config.py`` fails here, in the shape of the egress
chokepoint test.

The scan is over string constants in the syntax tree, not over spellings, so
a join, ``joinpath``, ``os.path.join``, an f-string and a DSN all count
(security review 2026-10-24-2, P46): a constant that is one of the names, or
ends in a path separator and one of them, is a construction.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "particles"

#: A string constant naming a config file, the store or the blob directory.
_PATH_NAME = re.compile(r"(?:^|[/\\])(?:config\.ya?ml|particles\.db|corpus_blobs)$")

#: Calls whose first argument is an instrumentation name, never a path
#: (``metrics.get_meter("particles.db")``).
_NAME_CALLS = frozenset({"get_meter", "get_tracer", "getLogger", "get_logger"})

#: The named exemptions, and the one criterion for adding to the list: the file
#: is written by the code that pins it with ``PARTICLES_CONFIG``, in a directory
#: that code created, and nothing reads it except through that pin. A new entry
#: is a reviewed edit to this list, never a pattern.
EXEMPT = {
    "config.py",  # the loader itself
    # Writes its pinned config into the run's work_dir and hands it to each
    # child process through PARTICLES_CONFIG.
    "benchmark/query_latency/runner.py",
    # Writes one into each case's run_dir, beside a private HOME, pinned the
    # same way, with its store beside it pinned by DATABASE_URL.
    "benchmark/update_gate/case.py",
}


def config_path_constructions(source: str) -> list[int]:
    """The 1-based line numbers of ``source`` that build a config or store path."""
    tree = ast.parse(textwrap.dedent(source))
    names: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in _NAME_CALLS:
                names.add(id(node.args[0]))
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in names
            and _PATH_NAME.search(node.value.strip())
        }
    )


def test_only_the_loader_and_the_named_exemptions_build_a_config_path() -> None:
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in EXEMPT:
            continue
        for line in config_path_constructions(path.read_text(encoding="utf-8")):
            offenders.append(f"particles/{rel}:{line}")
    assert offenders == [], (
        "config or store path built outside the loader; "
        "ask particles.config for the path instead: " + ", ".join(offenders)
    )


def test_every_exemption_still_builds_one() -> None:
    """An exemption that no longer needs its entry is a dead entry; remove it.

    The two published distributions each carry part of the package, so an
    exempt file a tree does not carry is skipped rather than failed.
    """
    for rel in EXEMPT:
        if not (PACKAGE / rel).exists():
            continue
        source = (PACKAGE / rel).read_text(encoding="utf-8")
        assert config_path_constructions(source), f"particles/{rel} no longer needs its exemption"


def test_the_installer_shape_is_caught() -> None:
    assert config_path_constructions('    return Path.cwd() / "config.yaml"\n') == [1]
    assert config_path_constructions("cfg = Path('./config.yaml')\n") == [1]
    assert config_path_constructions('"""Edit config.yaml by hand."""\n') == []


@pytest.mark.parametrize(
    "line",
    [
        'path = Path.cwd().joinpath("config.yaml")',
        'path = os.path.join(os.getcwd(), "config.yaml")',
        'path = Path(f"{Path.cwd()}/config.yaml")',
        'path = Path.cwd() / "config.yml"',
        'db = Path.cwd() / "particles.db"',
        'blobs = Path("corpus_blobs")',
        'url = "sqlite+aiosqlite:///./particles.db"',
    ],
)
def test_every_spelling_of_a_config_or_store_path_is_caught(line: str) -> None:
    """Security review 2026-10-24-2, P46 (K7): not only the two literal spellings."""
    assert config_path_constructions(line + "\n") == [1]


def test_a_mention_or_an_instrumentation_name_is_not_a_path() -> None:
    assert config_path_constructions('msg = "whatever particles.db the directory holds"\n') == []
    assert config_path_constructions('_meter = metrics.get_meter("particles.db")\n') == []
