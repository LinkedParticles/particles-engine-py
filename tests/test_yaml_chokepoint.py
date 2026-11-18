# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Third-party YAML is read through ``particles.core.bounded_yaml``.

Plain ``yaml.safe_load`` resolves an alias to the object its anchor built, so a
few lines of anchors stand for a structure exponentially larger than the text,
and the cost arrives when a caller walks or serialises the result. Every read
of third-party YAML (deposit frontmatter, a web capture's header, an
interchange YAML-LD member, a memory file's header) therefore goes through the
bounded loader. This test makes that a checked rule rather than a review habit,
the way ``tests/test_egress_chokepoint.py`` does for outbound connections: an
AST walk of ``particles/`` finds every reference to a PyYAML load function and
fails on any outside the allowlist below. Adding a plain load is a deliberate
edit here, with a one-line reason saying why its input is trusted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import particles
from tests import _source_tree

PKG_ROOT = Path(particles.__file__).parent

# Every PyYAML entry point that builds Python objects from text, and the
# loader classes a caller could pass to ``yaml.load`` or drive by hand.
_LOAD_NAMES = frozenset(
    {
        "safe_load",
        "safe_load_all",
        "load",
        "load_all",
        "full_load",
        "full_load_all",
        "unsafe_load",
        "unsafe_load_all",
        "SafeLoader",
        "CSafeLoader",
        "Loader",
        "CLoader",
        "FullLoader",
        "CFullLoader",
        "UnsafeLoader",
        "CUnsafeLoader",
        "BaseLoader",
        "CBaseLoader",
    }
)

# The bounded loader itself subclasses ``yaml.SafeLoader``.
_EXEMPT_MODULES = frozenset({"core/bounded_yaml.py"})

# (module path under particles/, enclosing function) -> why its input is trusted.
ALLOWED: dict[tuple[str, str], str] = {
    ("config.py", "_load_config_recorded"): (
        "the operator's own config.yaml, at PARTICLES_CONFIG or ~/.particles"
    ),
    ("config.py", "_names_a_particles_section"): (
        "NOT trusted (a nearby config.yaml, maybe a cloned repo's), but only decides a "
        "warning: a capped read, every exception caught, only top-level keys hashed; "
        "config cannot import core.bounded_yaml (core imports config)"
    ),
    ("api/cli/init.py", "_note_explicit_store_scope"): (
        "the operator's own config.yaml, read to print an upgrade note"
    ),
    ("api/cli/_claude_code.py", "_pinned_config_tree"): (
        "the config.yaml the installed hooks pin, which the operator wrote"
    ),
    ("api/cli/_claude_code.py", "_parse_yaml_mapping"): (
        "the operator's own config.yaml, re-parsed to verify the installer's edit"
    ),
    ("api/cli/_claude_code.py", "_add_enabled_store_item"): (
        "one flow-list value from the operator's own config.yaml"
    ),
    ("api/cli/_claude_code.py", "disable_memory_store_text"): (
        "one flow-list value from the operator's own config.yaml"
    ),
    ("benchmark/loader.py", "_load"): "a benchmark suite file the operator runs",
    ("benchmark/resolution.py", "load_resolution_gold"): (
        "a resolution gold set the operator runs"
    ),
    ("benchmark/modality/loader.py", "load_modality_suite"): (
        "a modality benchmark suite the operator runs"
    ),
    ("benchmark/validity/loader.py", "load_validity_suite"): (
        "a validity benchmark suite the operator runs"
    ),
    ("benchmark/polarity/loader.py", "load_polarity_suite"): (
        "a polarity benchmark suite the operator runs"
    ),
    ("conformance/profile.py", "load_profile"): (
        "the bundled Conformance Profile, or one the operator names"
    ),
    ("conformance/fixtures.py", "_load_one"): (
        "a conformance fixture's manifest from a fixture tree the operator names"
    ),
    ("conformance/fixtures.py", "_register_in_manifest"): (
        "the fixture corpus MANIFEST.yaml the conformance recorder maintains"
    ),
}


def _yaml_load_sites(tree: ast.Module) -> list[tuple[str, int]]:
    """``(enclosing function, line)`` of each reference to a PyYAML load name."""
    modules: set[str] = set()
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "yaml":
                    modules.add(alias.asname or "yaml")
        elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
            for alias in node.names:
                if alias.name in _LOAD_NAMES or alias.name == "*":
                    bare.add(alias.asname or alias.name)

    sites: list[tuple[str, int]] = []

    def visit(node: ast.AST, scope: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        hit = (
            isinstance(node, ast.Attribute)
            and node.attr in _LOAD_NAMES
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
        ) or (isinstance(node, ast.ImportFrom) and node.module == "yaml" and bool(bare))
        if hit:
            sites.append((scope, getattr(node, "lineno", 0)))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "<module>")
    return sites


def _all_sites() -> dict[tuple[str, str], list[int]]:
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(PKG_ROOT).as_posix()
        if rel in _EXEMPT_MODULES:
            continue
        for scope, line in _yaml_load_sites(_source_tree.parse(path.read_text())):
            found.setdefault((rel, scope), []).append(line)
    return found


def test_no_plain_yaml_load_outside_the_allowlist() -> None:
    unexpected = {key: lines for key, lines in _all_sites().items() if key not in ALLOWED}
    assert not unexpected, (
        "plain PyYAML load outside the allowlist; read third-party YAML through "
        "particles.core.bounded_yaml.load (or load_sequence), or, if the input is "
        "the operator's own or bundled, add the site to ALLOWED with its reason: "
        + ", ".join(
            f"particles/{rel}:{lines} in {scope}()" for (rel, scope), lines in unexpected.items()
        )
    )


def test_every_allowlist_entry_still_names_a_site() -> None:
    stale = sorted(set(ALLOWED) - set(_all_sites()))
    assert not stale, f"allowlist entries with no plain YAML load left; remove them: {stale}"


def test_the_walk_sees_every_spelling() -> None:
    source = (
        "import yaml\n"
        "import yaml as y\n"
        "from yaml import safe_load as s\n"
        "def a(t):\n    return yaml.safe_load(t)\n"
        "def b(t):\n    return y.full_load_all(t)\n"
        "def c(t):\n    return yaml.load(t, Loader=yaml.SafeLoader)\n"
        "def d(t):\n    return yaml.safe_dump(t)\n"
    )
    scopes = [scope for scope, _line in _yaml_load_sites(ast.parse(source))]
    assert scopes.count("<module>") == 1  # the from-import
    assert scopes.count("a") == 1
    assert scopes.count("b") == 1
    assert scopes.count("c") == 2  # yaml.load and yaml.SafeLoader
    assert "d" not in scopes
