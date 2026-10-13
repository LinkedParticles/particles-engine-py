# Running with no egress

Running the LLM locally does not, by itself, keep data on the machine. Particles
reaches the network in other places too: subject resolution looks names up on
Wikidata, a URL deposit fetches the page, the sentence encoder downloads its
model from Hugging Face, and any LLM purpose still routed to a hosted provider
sends its prompt there. The `network.egress` setting governs all of them with
one switch.

## The three postures

```yaml
network:
  egress: allowlist        # open | none | allowlist
  allow:
    - host: localhost
      port: 11434          # a model server on this machine
    - host: 192.168.1.20
      port: 8000           # one on the LAN
```

| Posture | What it does |
|---|---|
| `open` | No check. The default, and the behaviour of every earlier release. |
| `none` | Refuses every outbound connection and name lookup the engine makes, loopback included. |
| `allowlist` | Refuses everything except the destinations listed in `allow`. |

Each `allow` entry names a `host` and, optionally, a `port` (omitted means any
port). A host is an IP literal (`127.0.0.1`), a CIDR block (`10.0.0.0/24`), or
an exact host name (`models.lan`). Wildcards are not supported, and a redirect
is a new destination that must be listed in its own right.

**Local is named, never inferred.** An endpoint counts as local only when it
matches an entry. `localhost` and `127.0.0.1` are different entries, and a
private address is not treated as local unless it is listed, because a private
address can belong to a gateway or a cloud endpoint that forwards off the
machine. The compiled-in `local` LLM provider points at
`http://localhost:11434/v1`, so an allowlist for it names `localhost`, not
`127.0.0.1`.

Allowing a host by name trusts your resolver for that name. To avoid relying on
DNS, list IP literals instead.

## Check what a posture switches off

Run either report before relying on a posture. Both read the same posture, so
they cannot disagree, and neither makes a network call.

```bash
particles config validate
```

```text
network: egress=allowlist · allowed: localhost:11434
  ✓ llm extraction, semantic_lint, query_response, … (local) → localhost:11434: allowed
  ⚠ llm use_judge (anthropic) → api.anthropic.com:443: refused; these purposes are unavailable
  ⚠ authority wikidata → www.wikidata.org:443: refused; subjects resolve locally
  ✓ encoder all-MiniLM-L6-v2 → huggingface.co:443: cached, loads offline
```

A refusal that leaves the install unusable fails the command: a refused remote
engine (`engine.base_url`) or a refused store host. Every other refusal is a
warning. `particles hook doctor` prints the same posture as one `egress:` line,
flagging the purposes the Claude Code hooks depend on.

## What each refused path does

A refused call never leaves the machine, is never cached, and is never counted
as a miss, so opening the posture later restores full behaviour.

| Path | Under refusal |
|---|---|
| URL deposit and the domain importers | Fails before any lookup, naming the host. Deposit a saved copy of the page with `particles deposit <file>` instead. The HTTP API answers 403 with `"error": "egress_refused"`. |
| Links a deposited post points to | Skipped and logged; the post itself is deposited. |
| `particles corpus refresh` of a web entry | Reported as refused, counted apart from errors. |
| The inbox | The line is marked `# Refused (network.egress)`, distinct from `# Failed`. |
| Subject resolution | Wikidata is not asked; subjects resolve locally. A Subject created this way stays local after the posture opens, because nothing re-links it automatically. |
| Wikidata entity labels | Left as their identifiers for this extraction, with a quality note, and not cached. Re-extract after the posture opens to restore them. |
| `particles subjects fix-labels`, `particles export notion`, the memory benchmark's dataset download | Refused up front, before any partial work. |
| An LLM purpose whose endpoint is refused | Unavailable, never rerouted to another provider. Extraction leaves snapshots PENDING without counting an attempt; lint and consolidation report their LLM checks as skipped with reason `egress`; a query returns its retrieval listing instead of an answer. |
| The contradiction check, when `semantic_lint` is refused | Background extraction reads an unchecked pair as compatible, so contradiction detection is off for the store, and both reports say so. Agent-side extraction holds such pairs for review instead. |
| The sentence encoder | Loads from the local cache only, with no request to the Hub. A cold cache disables semantic features; warm it under `open`, or copy a populated `HF_HOME` onto the machine. |
| OpenTelemetry export (OTLP) | No exporter is installed for a refused endpoint. |

Under a refusing posture every extraction routed through the registry waits,
including the deterministic extractors that never call a model. Route
`extraction` to an allowed endpoint to extract.

## How it is enforced

The check sits at the transports, so a code path cannot forget it: the shared
fetch client, the subprocess pin that `curl` and `git` connect through, the
clients for LLM endpoints and the remote engine, the encoder load, and the
telemetry exporter. Under `none` and `allowlist` a process-wide backstop also
refuses any other connection or name lookup the policy does not admit.

The setting is an in-process control. For a guarantee that holds even against a
path Particles does not know about, also run it without network access at the
operating-system level, for example in a container attached only to an
internal network that reaches the model server and nothing else (see
[Running in a container](container-deployment.md)).
