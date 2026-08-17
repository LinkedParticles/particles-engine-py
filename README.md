# linkedparticles

> **Particles is shared memory for humans and AI agents.** Each particle is one
> claim, plus what you need to judge it: who said it, where, when, and how
> confident they were. Facts, opinions, and memories are all claims, recorded
> the same way as particles. Particles are not edited or deleted. Particles are
> superseded, retracted, or disputed in the open. How much to trust it is a
> perspective applied at query time, never baked into the record.

The **Engine layer** of the Particles reference implementation — the
state-holding SDK and its surfaces. This is the package you install to actually
run a Particles knowledge store. The core loop:

- **deposit** source material into an append-only corpus;
- **extract** claim-granularity particles with subject resolution and
  confidence/provenance metadata;
- **query** with effective-confidence ranking and subject filtering;
- **lint** for contradictions and staleness;
- **review** inconsistencies into a reusable source-trust policy.

When your agent is wrong, you can see exactly why, and fix it at the source.

The Engine is a library first. The FastAPI server, the CLI, the read-only MCP
server, and the resident daemon are all *surfaces* over it.

## Install

```bash
pip install linkedparticles
```

This depends on `linkedparticles-core` (the store-free Client layer) — it is
pulled in automatically.

## The three repositories

| Repo | What it is |
|---|---|
| [`particles-standard`](https://github.com/LinkedParticles/particles-standard) | The standard: whitepaper, technical specification, normative schema + SHACL artifacts, conformance fixtures |
| [`particles-core-py`](https://github.com/LinkedParticles/particles-core-py) | The Python Client layer (`linkedparticles-core`) |
| [`particles-engine-py`](https://github.com/LinkedParticles/particles-engine-py) | **This repo** — the Python Engine layer + surfaces (`linkedparticles`) |

## Deployment

Container and chart artifacts live under [`deploy/`](deploy/). The engine ships
as a single-writer service; see the operator guide under `docs/operator-guide/`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
Contributions are accepted under a Developer Certificate of Origin sign-off —
there is no CLA.
