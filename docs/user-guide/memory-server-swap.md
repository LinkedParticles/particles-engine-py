# Swapping in for the reference memory server

If you already run Anthropic's knowledge-graph memory server
([`@modelcontextprotocol/server-memory`](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)),
you can point your agent at a Particles store by editing one entry in your MCP
client config. Nothing else changes: same nine tools, same arguments, same
responses, same system prompt.

What you gain is underneath the protocol — every observation becomes a
[claim with a source](concepts.md), deletes become retractions with an
[audit trail](../operator-guide/auditing.md), and the store is readable by
every other Particles surface.

This is the drop-in route, chosen when you already run the reference server
and want to change nothing else. If you use Claude Code and would rather have
memory pushed and harvested on session lifecycle events than exposed as tools
the agent must call, see [Claude Code memory](claude-code.md) instead.

The swap starts you from an **empty store**. To carry your existing graph over,
see [Bringing your existing graph with you](#bringing-your-existing-graph-with-you)
below.

## The swap

Replace the reference server's entry:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

with the façade:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/particles-engine-py",
               "particles", "memory", "serve"],
      "env": {
        "DATABASE_URL": "sqlite+aiosqlite:////path/to/memory.db",
        "PARTICLES_BLOB_DIR": "/path/to/blobs",
        "PARTICLES_CONFIG": "/path/to/config.yaml"
      }
    }
  }
}
```

In VS Code the key is `servers` rather than `mcpServers`; everything else is
identical.

!!! warning "Pin the paths for any unattended process"
    An MCP server is launched by your client and inherits almost nothing.
    `storage.blob_dir` defaults to the **relative** `./corpus_blobs`, so an
    unpinned server started from the wrong directory will silently split a
    store from its content. Always set `DATABASE_URL` and `PARTICLES_BLOB_DIR`
    explicitly.

## Enabling writes

Writes are default-deny. Until you opt a store in, the nine tools
are all still listed — a reference client must not discover a short tool list —
but the six writing tools refuse with a message naming the key to change.

```yaml
# config.yaml
mcp:
  write:
    enabled_stores: ["default"]
```

Create the store first:

```bash
particles db init
```

## Setting it up from scratch

```bash
mkdir -p ~/particles-memory/blobs
cat > ~/particles-memory/config.yaml <<'YAML'
mcp:
  write:
    enabled_stores: ["default"]
YAML
export PARTICLES_CONFIG=~/particles-memory/config.yaml
export DATABASE_URL="sqlite+aiosqlite:///$HOME/particles-memory/memory.db"
export PARTICLES_BLOB_DIR=~/particles-memory/blobs
particles db init
```

A write-enabled store automatically reconciles in `multi` mode, so a
contradiction between two claims is surfaced rather than silently resolved.

## What your agent sees

Exactly what it saw before. Sending the reference README's own example payload
through `create_entities` and then `read_graph` returns:

```json
{
  "entities": [
    {"name": "Anthropic", "entityType": "organization", "observations": []},
    {"name": "John_Smith", "entityType": "person",
     "observations": ["Speaks fluent Spanish"]}
  ],
  "relations": [
    {"from": "John_Smith", "to": "Anthropic", "relationType": "works_at"}
  ]
}
```

The `memory://knowledge-graph` resource and its update subscriptions work too,
so a client that reads the graph as a resource rather than a tool is unaffected.

## Three things that behave differently

These are deliberate, and each is disclosed in the tool description your agent
reads — not just in this page.

**Deletes retract instead of destroying.** A client that deletes and then reads
does not see the item, exactly as before. Underneath, the record survives.
After deleting `John_Smith` in the example above, the store still holds the
observation:

```
$ sqlite3 memory.db "select substr(content,1,40), status, asserted_by
                     from particles where content like '%Spanish%';"
Speaks fluent Spanish|RETRACTED|mcp:memory-compat
```

…along with the operator events (`PARTICLE_ASSERTED` ×3,
`PARTICLE_RETRACTED` ×2) and the verbatim deposits of every payload your agent
sent. This is the whole point of the swap: a confused turn that deletes a
memory no longer destroys the evidence.

**`read_graph` is capped.** The reference dumps the entire graph. On a store of
any real size that is megabytes of JSON pushed into your model's context, so
the façade caps entities and per-entity observations
(`mcp.memory_compat.read_graph_max_entities` and
`…_max_observations_per_entity`). Truncation is never silent: the JSON stays in
the first content block, exactly where a client expects it, and a plain-text
notice is *appended* as a second block.

**Writes are attributed and trust-weighted.** Everything the façade writes is
stamped `mcp:memory-compat` and enters
[effective confidence](concepts.md#confidence) *below* operator-asserted
content, so your agent's memories can never outrank your own. How far below is
yours to set: see
[Operator guide → extractor trust weight](../operator-guide/tuning.md#extractor-trust-weight).

## What is not different

`search_nodes` is still case-insensitive **substring** matching over entity
names, types, and observations — not semantic search. That is what the
reference does, and changing it would change result sets your agent did not ask
to change. Semantic recall is available behind
`mcp.memory_compat.semantic_augmentation`, off by default — and always
available on the native surface, via the `query` tool
([Querying → MCP server](querying.md#mcp-server)).

## Inspecting the surface

```bash
particles memory tools
```

prints the exact tool contract — names, titles, input and output schemas, and
annotations — which is also pinned as a test golden, so a parity regression
fails the build rather than reaching your agent.

## Bringing your existing graph with you

Swapping the server does not move your data. One command does:

```bash
particles import mcp-memory ~/.mcp/memory.jsonl
particles extract --all-pending
```

Your entities become Subjects, observations become claims about them, and
relations become edges — written in the **same encoding** the swapped-in server
reads, so `read_graph`, `search_nodes`, and `open_nodes` return your graph
immediately. The translation is structural: no model is called and nothing goes
over the network, because the records are already one claim each.

Running it twice is safe. The export dedups by content hash, your entities
re-attach to the Subjects they already have, and identical claims are caught
before they can be stored twice. Exporting again later from the old server
brings across only what is new.

### Your migrated memories will look less confident. That is deliberate.

They are second-hand: this store never saw the claim made, cannot check it, and
did not calibrate any number attached to it. So every migrated belief gets one
flat, low confidence (`migration.import_confidence`, default `0.35`) and is
labelled `IMPORTED` — which is also what lets you tell **what you brought with
you** apart from **what your agent has learned since**.

Where a source store keeps its own scores, they are preserved as tags and never
become confidence values. A confidence is fixed when a belief is created and
multiplies through every ranking, so importing another system's number would
quietly re-rank your whole store on a scale that was never yours.

To trust an import more, say so as policy rather than editing the floor:

```bash
particles trust set --source-type MCP_MEMORY_EXPORT --rank 0.8
```

That is revisable and recorded; the import floor is frozen into each belief at
creation and cannot be changed afterwards.

### What the import records, and what it refuses to

Each migrated belief points back at the export file itself — deposited verbatim,
hashed, and cited down to the line the record came from — plus who ran the
import. Nothing is attributed to the server you migrated from: Particles never
fetched it and cannot re-verify it, so inventing a source there would be exactly
the fabricated provenance the store exists to prevent.

Two consequences worth knowing:

- **Malformed lines are skipped and reported**, never silently dropped. The
  original bytes stay in the corpus, so a later fix can re-read them without
  asking you to export again.
- **An entity with no observations does not survive.** The reference server can
  hold a name it has never recorded anything about; a Particles store has no
  belief to attach it to. The count is reported when you extract.

## Configuration reference

All knobs live under `mcp.memory_compat`; see `config.yaml.sample` for the
commented block and [Configuration](../operator-guide/configuration.md) for the
loader's precedence rules.
