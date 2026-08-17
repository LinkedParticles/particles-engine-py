"""Layer-boundary regression tests.

The *authoritative* gate is the three ``import-linter`` contracts in
``pyproject.toml`` (run by pre-commit and CI via ``lint-imports``). This module
mirrors them inside the pytest suite so the guards are covered by ``pytest``
too, and — via the ``*_is_not_vacuous`` tests — proves each check fails on the
shape it is meant to catch rather than passing trivially.

The three contracts, each guarding a different layering regression:

1. ``forbidden`` — the Client substrate (the genuinely store-free surface: what
   a downstream needs to produce a candidate particle without a graph) must never import the Engine. Since it asserts
   ``particles.extraction`` and ``particles.conformance`` wholesale — a new
   domain extractor is covered automatically; its importer lives in ``ingest``,
   never in ``extraction``. Keep ``CLIENT_SUBSTRATE`` / ``ENGINE`` below and the
   ``source_modules`` / ``forbidden_modules`` in ``pyproject.toml`` in lockstep.
2. ``layers`` — the macro tier ordering *Surface > Engine > Client*; no lower
   tier may import a higher one (``SURFACE_TIER`` / ``ENGINE_TIER`` /
   ``CLIENT_TIER``).
3. ``acyclic_siblings`` — no *new* import cycle between the top-level
   subpackages of ``particles``. The cycles that exist today are pinned in
   ``SANCTIONED_CYCLE_SEAMS`` (mirroring the contract's ``ignore_imports``);
   removing exactly those edges must leave the subpackage graph acyclic.
"""

from __future__ import annotations

import grimp
import pytest

# Client substrate — must never import the Engine (directly or transitively).
CLIENT_SUBSTRATE = [
    "particles.core",
    "particles.config",
    "particles.secrets",
    "particles.embeddings",
    "particles.http",
    "particles.url_safety",
    "particles.llm",
    "particles.interchange.codec",
    "particles.interchange.jsonl",
    "particles.extraction",
    "particles.conformance",
    "particles.render.markdown",
]

# Engine — stateful / graph-aware. Engine may import Client freely; the reverse
# is forbidden.
ENGINE = [
    "particles.store",
    "particles.corpus",
    "particles.db",
    "particles.operations",
    "particles.ingest",
    "particles.interchange.store",
    "particles.render.article_synthesis",
    "particles.exporters",
    "particles.observability",
    "particles.mcp",
    "particles.api",
    "particles.benchmark",
    "particles.sql_safety",
    "particles._orm_modules",
]

# Macro tiers for the ``layers`` contract (Surface > Engine > Client). A lower
# tier may never import a higher one; intra-tier imports are allowed.
SURFACE_TIER = ["particles.api", "particles.mcp"]
ENGINE_TIER = [
    "particles.store",
    "particles.corpus",
    "particles.db",
    "particles.operations",
    "particles.ingest",
    "particles.interchange",
    "particles.exporters",
    "particles.render.article_synthesis",
    "particles.observability",
    "particles.benchmark",
    "particles.sql_safety",
    "particles._orm_modules",
]
CLIENT_TIER = [
    "particles.core",
    "particles.config",
    "particles.secrets",
    "particles.embeddings",
    "particles.http",
    "particles.url_safety",
    "particles.url_canonical",
    "particles.llm",
    "particles.conformance",
    "particles.extraction",
    "particles.render.markdown",
]

# Sanctioned subpackage-cycle seams (mirrors ``ignore_imports`` on the
# ``acyclic_siblings`` contract). Removing exactly these edges makes the
# top-level subpackage graph acyclic. The first four are permanent deferred
# cycle-break seams (AGENTS.md § Deferred imports case 1); the last one is the
# top-level back-edge the corpus↔ingest cycle-fix task removes — guarded by
# ``direct_import_exists`` to keep this test green both before and after that
# task lands. The operations↔exporters cycle is fully broken (the digest/inbox
# markdown edges and the narrative_synthesis article-synthesis edge both moved
# to the Client-layer ``particles.render``), so no operations→exporters seam
# remains.
SANCTIONED_CYCLE_SEAMS = [
    ("particles.api.cli.mcp", "particles.mcp"),
    ("particles.api.cli.memory", "particles.mcp.memory_compat"),
    ("particles.api.cli.memory", "particles.mcp.memory_compat.server"),
    ("particles.corpus.fetch", "particles.store.particle_store"),
    ("particles.corpus.deposit", "particles.store.url_mention_store"),
    ("particles.ingest.pipeline", "particles.operations.version_guard"),
    ("particles.corpus.deposit", "particles.ingest.importers.registry"),
]


@pytest.fixture(scope="module")
def graph() -> grimp.ImportGraph:
    # grimp builds the graph by static analysis (no module execution), so this
    # is fast and free of import side effects.
    return grimp.build_graph("particles")


def _fresh_graph() -> grimp.ImportGraph:
    # Tests that mutate the graph (remove_import) build their own so they do not
    # contaminate the module-scoped ``graph`` fixture shared by the others.
    return grimp.build_graph("particles")


def test_client_substrate_never_imports_engine(graph: grimp.ImportGraph) -> None:
    violations: list[str] = []
    for client in CLIENT_SUBSTRATE:
        for engine in ENGINE:
            if graph.chain_exists(importer=client, imported=engine, as_packages=True):
                violations.append(f"{client} -> {engine}")
    assert not violations, "Client→Engine imports forbidden: " + ", ".join(violations)


def test_no_upward_tier_imports(graph: grimp.ImportGraph) -> None:
    # Mirrors the ``layers`` contract: Engine and Client may not import Surface,
    # and Client may not import Engine (directly or transitively).
    violations: list[str] = []
    for lower in ENGINE_TIER + CLIENT_TIER:
        for higher in SURFACE_TIER:
            if graph.chain_exists(importer=lower, imported=higher, as_packages=True):
                violations.append(f"{lower} -> {higher}")
    for lower in CLIENT_TIER:
        for higher in ENGINE_TIER:
            if graph.chain_exists(importer=lower, imported=higher, as_packages=True):
                violations.append(f"{lower} -> {higher}")
    assert not violations, "Upward tier imports forbidden: " + ", ".join(violations)


def test_subpackages_acyclic_modulo_sanctioned_seams() -> None:
    # Mirrors the ``acyclic_siblings`` contract: with the sanctioned seams
    # removed, the top-level subpackage graph must be a DAG.
    graph = _fresh_graph()
    for importer, imported in SANCTIONED_CYCLE_SEAMS:
        if graph.direct_import_exists(importer=importer, imported=imported):
            graph.remove_import(importer=importer, imported=imported)
    breakers = graph.nominate_cycle_breakers("particles")
    assert not breakers, "Unexpected subpackage import cycle(s); offending edges: " + ", ".join(
        f"{imp} -> {imported}" for imp, imported in sorted(breakers)
    )


def test_detection_mechanism_sees_engine_edges(graph: grimp.ImportGraph) -> None:
    # Guard coverage: confirm chain_exists actually detects the Client/Engine
    # *shape* of dependency, so the assertions above cannot pass vacuously. These
    # are real Engine→Engine (and Engine→Client) edges in the allowed direction.
    assert graph.chain_exists(
        importer="particles.ingest", imported="particles.store", as_packages=True
    ), "expected ingest to import the store"
    assert graph.chain_exists(
        importer="particles.operations", imported="particles.store", as_packages=True
    ), "expected operations to import the store"
    # ...and a real higher→lower tier edge, so test_no_upward_tier_imports is
    # exercising a live detector rather than passing trivially.
    assert graph.chain_exists(
        importer="particles.api", imported="particles.operations", as_packages=True
    ), "expected the Surface tier (api) to import the Engine tier (operations)"


def test_acyclic_check_is_not_vacuous() -> None:
    # Without removing the sanctioned seams, the subpackage graph DOES contain
    # cycles — proving nominate_cycle_breakers is a live detector.
    breakers = _fresh_graph().nominate_cycle_breakers("particles")
    assert breakers, "expected real subpackage cycles before removing sanctioned seams"
