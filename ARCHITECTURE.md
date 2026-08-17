# Architecture — linkedparticles

This is the **Engine layer**: everything that holds or reasons over accumulated
state. It depends on `linkedparticles-core` (the store-free Client layer) and
adds the store, reconciliation, query, lint, exporters, and the surfaces.

The Client → Engine dependency is one-way and enforced: an import-linter
contract fails CI if the Client layer ever imports the Engine.

## What lives here

| Area | Responsibility |
|---|---|
| `corpus/` | Append-only source corpus; deposit; lazy re-fetch |
| `ingest/` | Reconciliation: the extract pipeline, subject resolution, subject-authority and importer registries |
| `store/` | The particle store, subject store, and source-trust store |
| `operations/` | Query, lint, review, reindex, curation, projection, audit, consolidation |
| `exporters/`, `render.article_synthesis` | One-way projections of the store; cited-prose synthesis |
| `interchange.store` | Store-aware, round-trippable import/export |
| `api/`, `mcp/`, `integrations/` | The surfaces — FastAPI + CLI, the read-only MCP server, integrations |
| `db.py`, `_orm_modules.py` | Per-store async engine registry and ORM metadata |

## Data flow

`deposit` → `extract` → `query` → `lint` → `review` → `reindex`. Each operation
is the single entry point the surfaces drive; the subject store is a knowledge
graph in which single-subject particles are properties and multi-subject
particles are edges.

## Surfaces are modes, not the definition

The Engine is a library. The FastAPI server is one surface over it; so are the
CLI, the read-only MCP server, and the resident daemon (`engine serve`). The
in-process backend and the CLI flows are first-class and do not depend on the
server running.

## One import package, two distributions

`linkedparticles` and `linkedparticles-core` both ship modules under the
`particles` import package: you write `from particles.core.schema import
Particle` and `from particles.store.particle_store import ParticleRow` without
caring which wheel each came from. Exactly one distribution ships each file —
`linkedparticles-core` owns `particles/__init__.py`, `particles/py.typed`, and
the `__init__.py` of the two packages the layers split (`particles/render/`,
`particles/interchange/`), because it is the dependency and so is always
present. Each of those calls `pkgutil.extend_path`, which is what lets this
distribution's modules resolve from a different directory.

The two are **version-locked**: `linkedparticles` depends on
`linkedparticles-core==<the same version>`. They are one source tree cut along
an architectural line, and engine modules import client modules by exact
symbol, so a version skew is a broken install rather than a compatibility
question. Publishing one means publishing both.

One consequence is worth knowing before it surprises you. **mypy does not
follow `extend_path`.** In an ordinary install — both distributions in the same
`site-packages`, which is what `pip install linkedparticles` produces — type
checking resolves everything, including across the two wheels. But when they
land in *different* `sys.path` roots (`pip install --target`, `--user` site, a
Lambda layer, or an editable install of both repositories side by side),
imports still work at runtime while mypy reports `import-not-found` for the
other distribution's modules. Nothing is wrong with the code. The alternative —
making `particles` a native namespace package — would have deleted
`particles.__version__`, deleted the `particles.interchange` re-export surface,
and forced the `py.typed` marker into every subpackage, so the type-checking
degradation in an uncommon layout was the cheaper cost.

## Deployment

`deploy/` holds the container image and chart. The served web UI under
`clients/web-ui/` is an engine-served surface and rides this repository. The
service is a single writer — that constraint is load-bearing, not incidental.

For the normative definitions behind the invariants named here, see the
technical specification in
[`particles-standard`](https://github.com/LinkedParticles/particles-standard).
