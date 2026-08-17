# Operator guide

You're running Particles long-term. This guide covers configuration,
hygiene, tuning, and the things that go wrong.

| Page | When to read |
|---|---|
| [Configuration](configuration.md) | `config.yaml` structure, env vars, precedence |
| [Running in a container](container-deployment.md) | The OCI image, compose, the Helm chart, resident daemon mode, the `/data` volume |
| [Lint and review](lint-and-review.md) | `PROVENANCE_STALE`, `INCONSISTENCY`, the review → trust workflow |
| [Co-evidential curation](co-evidential.md) | Finding and linking duplicate claims with `links suggest` |
| [Citation-signal deposits](citation-signals.md) | Surfacing frequently-cited undeposited primary sources with `corpus links suggest` |
| [Auditing](auditing.md) | The operator event log, `corpus retract`, who-changed-what |
| [Structured claims](structured-claims.md) | The derived S-P-O annotation beside a claim: what it is for, `particles structure`, coverage |
| [Store consolidation](store-consolidation.md) | Merging two divergent local stores into one canonical archive (multi-machine) |
| [Schema migration](schema-migration.md) | `SCHEMA_VERSION` major bumps, `db init --force`, what survives |
| [Tuning](tuning.md) | Trust weights, calibration, trust lenses, `min_particle_confidence`, age decay |
| [Troubleshooting](troubleshooting.md) | Database locked, tables not found, fetch failures, follow-edges |

For first-time setup → first query, see the
[user guide → getting started](../user-guide/getting-started.md).
For writing a new extractor or exporter, see the
[plugin-author guide](../plugin-author-guide/index.md).
