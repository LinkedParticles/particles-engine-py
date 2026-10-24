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
path Particles does not know about, also enforce it at the operating system, as
the next section describes.

## Enforce it at the operating system as well

A rule at the operating system holds against a connection Particles never sees:
a library that opens its own socket, a plugin, a dependency added later. Use it
beside `network.egress`, not instead of it. The in-process posture makes each
refused path degrade cleanly and report itself; the system rule stops whatever
the posture does not cover. Keep the two naming the same destinations. A
destination in `network.allow` that the system blocks fails as a connection
error rather than as a clean refusal, and a destination the system
admits but the posture omits is refused in-process anyway.

Every recipe below admits the two model servers of this posture and nothing
else:

```yaml
network:
  egress: allowlist
  allow:
    - host: localhost
      port: 11434          # a model server on this machine
    - host: 192.168.1.20
      port: 8000           # one on the LAN
```

### In a container (Docker, any host)

With no model server outside the container, give it no network at all:

```bash
docker run --network none <image> …
```

The container then has a loopback interface and nothing else, and any outbound
connection fails with `Network is unreachable`. A container started with
`--network none` is its own network namespace; on a Linux host,
`sudo unshare --net -- <command>` gives a single command the same isolation
without a container (its loopback starts down, so not even `localhost` answers).

When the model server runs in a container too, attach both to an internal
network, which reaches its own members and nothing beyond them:

```bash
docker network create --internal particles-net
docker run -d --name model --network particles-net <model-server-image>
docker run --network particles-net <image> …
```

The engine reaches the server at `model` (name it in `network.allow`, with its
port), and every other destination is unreachable. An internal network cannot
reach a server on the LAN; for that, use the host rule below.
[Running in a container](container-deployment.md) covers the engine image
itself.

### On a Linux host: a rule for the Particles account

Run Particles under an account of its own (here `particles`), and add an
nftables table that rejects that account's traffic except over loopback and to
the LAN server:

```text
table inet particles_egress {
	chain output {
		type filter hook output priority filter; policy accept;
		meta skuid "particles" oif "lo" accept
		meta skuid "particles" ip daddr 192.168.1.20 tcp dport 8000 accept
		meta skuid "particles" counter reject
	}
}
```

```bash
sudo nft -f particles-egress.nft
```

The rule matches the sending account, so other accounts on the machine are
untouched, and it covers every process the `particles` account runs. A local
caching resolver is reached over loopback and still forwards the names it is
asked about; the rule stops the connection, not that lookup.

### On macOS: a pf anchor

Run Particles under an account of its own (here `_particles`), and write the
anchor to `/etc/pf.anchors/particles`:

```text
pass out quick on lo0 all user _particles
pass out quick proto tcp to 192.168.1.20 port 8000 user _particles
block return out quick all user _particles
```

Reference it from `/etc/pf.conf`:

```text
anchor "particles"
load anchor "particles" from "/etc/pf.anchors/particles"
```

Then load the ruleset with `sudo pfctl -f /etc/pf.conf` and enable pf with
`sudo pfctl -E`. pf matches an account only on TCP and UDP sockets. Check the
files with `pfctl -n -f /etc/pf.conf`, which parses them without loading
anything; the account must exist for the parse to pass.

### The Claude Code hooks

The hooks run as the account that runs Claude Code, so a per-account rule
would cut that account off entirely. To put the hooks behind a system rule,
run the engine as a service under its own account and point the hooks at it
([Claude Code hooks against a remote engine](remote-engine.md#claude-code-hooks-against-a-remote-engine));
the rule then applies to the engine, where the model calls and lookups are
made.

## Check both halves

`particles config validate` and `particles hook doctor` report the in-process
half. They read the configuration and cannot see a firewall rule. Under the
posture above, `particles hook doctor` prints:

```text
  egress:         allowlist · refused: llm extraction, semantic_lint, query_response, synthesis, benchmark, benchmark_answer, abstraction, verification, subject_resolution, use_judge (anthropic), authority wikidata
                  ⚠ extraction refused (api.anthropic.com:443): background extraction of harvested transcripts is unavailable
                  ⚠ use_judge refused (api.anthropic.com:443): SessionEnd utility mining is unavailable
```

A purpose listed as refused there has an endpoint outside the allowlist; route
it to an allowed one, or accept that it is unavailable. To check the system
half, connect as the Particles account to one destination the rule admits and
one it does not:

```bash
sudo -u particles python3 -c "import socket; socket.create_connection(('192.168.1.20', 8000), 3); print('model server reachable')"
```

```bash
sudo -u particles python3 -c "import socket; socket.create_connection(('1.1.1.1', 443), 3)"
```

The first prints its line. The second must fail: with the nftables rule above,
it ends in `ConnectionRefusedError`.
