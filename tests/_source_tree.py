# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""One parse of each source text, shared by the static guards that scan ``particles/``.

Several guard tests walk the AST of every module in the package (the egress
chokepoint, the row-write and currency-reader ratchets, the append-only
writers). Parsing the package costs about half of each such test, so the trees
are shared here, keyed on the source text itself: a file edited between two
reads (a test that scans a temporary tree it rewrites) is parsed again, never
served a stale tree.

The trees of the whole package hold about 220 MB, so they are not kept for the
session: the autouse ``release_parsed_sources`` fixture in ``tests/conftest.py``
clears the cache when each test module finishes. Callers must not mutate a
returned tree.
"""

from __future__ import annotations

import ast

_trees: dict[str, ast.Module] = {}


def parse(source: str) -> ast.Module:
    """``ast.parse(source)``, parsed once per distinct text until :func:`clear`."""
    tree = _trees.get(source)
    if tree is None:
        tree = _trees[source] = ast.parse(source)
    return tree


def clear() -> None:
    """Drop every cached tree."""
    _trees.clear()
