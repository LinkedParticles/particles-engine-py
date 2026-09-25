# Swapping in for the reference memory server

If you already run Anthropic's knowledge-graph memory server
([`@modelcontextprotocol/server-memory`](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)),
you can point your agent at a Particles store by editing one entry in your MCP
client config. Nothing else changes: same nine tools, same arguments, same
responses, same system prompt.

What you gain is underneath the protocol: every observation becomes a
[claim with a source](concepts.md), deletes become retractions with an
[audit trail](../operator-guide/auditing.md), and the store is readable by
every other Particles surface.

This is the drop-in route, chosen when you already run the reference server
and want to change nothing else. If you use Claude Code and would rather have
memory pushed and harvested on session lifecycle events than exposed as tools
the agent must call, see [Claude Code memory](claude-code.md) instead. For the
native tool surface, and where to put the config for each harness, see
[Connecting your coding agent](coding-agents.md).

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
are all still listed (a reference client must not discover a short tool list),
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

## What behaves differently

These are deliberate, and each is disclosed in the tool description your agent
reads, not just in this page.

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

**Entity names match case-insensitively.** The reference compares names
exactly, so `Alice` and `alice` are two entities there. Here they are one:
creating `alice` when `Alice` exists is skipped like any other duplicate, and
`open_nodes`, `add_observations`, and the delete tools find the entity under
either spelling. Reads always return the spelling it was created with.

**`read_graph` returns entities in name order,** not the order they were
created in. A store has no single creation order to preserve once other
Particles surfaces write to it too.

Two smaller differences are invisible to a tool call. The server reports the
Particles package version as its `serverInfo.version`, not the reference's.
Where the reference queues mutations behind one lock, here each tool call is
its own database transaction.

## What is not different

The façade tracks the reference as it moves, and the behaviours the reference
gained in September 2026 are matched exactly. `create_relations` refuses a
relation whose `from` or `to` entity does not exist (`Entity with name X not
found`), for the whole batch and before writing anything, so create the
entities first. The delete tools no longer claim a deletion that did not
happen: when part of a request matched nothing, the message reads `Deleted 1 of
3 entities. Not found: …` instead of the plain success sentence. `search_nodes`
also caps a query at 2,048 characters, as the reference does.

`search_nodes` is still case-insensitive **substring** matching over entity
names, types, and observations, not semantic search. That is what the
reference does, and changing it would change result sets your agent did not ask
to change. Semantic recall is available behind
`mcp.memory_compat.semantic_augmentation`, off by default, and always
available on the native surface, via the `query` tool
([Querying → MCP server](querying.md#mcp-server)).

## Inspecting the surface

```bash
particles memory tools
```

prints the exact tool contract (names, titles, input and output schemas, and
annotations), which is also pinned as a test golden, so a parity regression
fails the build rather than reaching your agent.

## Bringing your existing graph with you

Swapping the server does not move your data. One command does, and it has a
rehearsal. **Run the rehearsal first**, because this is your real graph:

```bash
particles import mcp-memory ~/.mcp/memory.jsonl --dry-run
```

The dry run reads the export and runs the same translation the import runs,
then prints what it would produce and stops. It opens no store and deposits
nothing, so it is safe to run against a store you care about, and it works
before `particles db init`. For the reference README's example graph, plus one
entity nothing else mentions and one line that is not JSON, it prints:

```
Dry run: nothing was deposited or written. Report for /home/you/.mcp/memory.jsonl

In the export:
        3  entities
        1  observations
        1  relations

The import would produce (at most; an existing store re-attaches and dedups):
        2  Subjects
        2  particles  (1 about one Subject, 1 linking several)
           confidence 0.35, calibration source IMPORTED

Entities with no observations: 2. They produce no particle.
        1  would NOT survive the import: Old_Project
        1  survive as a relation endpoint, with their entity type

Dropped from the particles, preserved in the export:
  - Line 5: not valid JSON (Expecting value); skipped.
  - 1 entity/entities carry no observations and appear in no relation, and will not migrate: 'Old_Project'. A Particles store holds a name only through something believed about it.

Sample (2 of 2):
  [line 1 observation 0] John_Smith: Speaks fluent Spanish
  [line 4] John_Smith + Anthropic: John_Smith works_at Anthropic

Next: run again without --dry-run, then `particles extract --all-pending`.
```

Read three things in it before going further:

- **The counts should match what you expect your graph to hold.** Entities,
  observations, and relations are counted in the old server's own terms, then
  in this store's.
- **"Would NOT survive" names every entity you would lose.** See
  [what the import refuses to record](#what-the-import-records-and-what-it-refuses-to)
  below for why; if a name on that list matters, give it an observation in the
  old server and export again, or recreate it after the swap.
- **"Dropped" is everything the translation cannot place**: malformed lines,
  values that are not text, and fields the reference format does not define.
  None of it is destroyed. It stays in the export file, which the import keeps.

The numbers are what the export *contributes*. The dry run does not look at
your store, so on one that already holds part of this graph the real import
writes fewer records than reported, never more. Add `--json` for a
machine-readable report, and `--sample N` to see more of the translated
records. The command exits non-zero when nothing in the file would become a
particle, which usually means it is not a `memory.jsonl` at all.

When the report looks right, run the import itself:

```bash
particles import mcp-memory ~/.mcp/memory.jsonl
particles extract --all-pending
```

Your entities become Subjects, observations become claims about them, and
relations become edges, written in the **same encoding** the swapped-in server
reads, so `read_graph`, `search_nodes`, and `open_nodes` return your graph
immediately. The translation is structural: no model is called and nothing goes
over the network, because the records are already one claim each.

!!! warning "One thing does not come across: an entity nothing is recorded about"
    The reference server will keep an entity with an empty `observations` list.
    A Particles store holds a name only through something believed about it, so
    an entity with **no observations and no relations** is not migrated, and
    `read_graph` will return that many fewer nodes than it did before the swap.

    You are told which ones. `particles import mcp-memory` lists them by name
    as it runs, and `particles extract` repeats the count.

    An observation-less entity that some relation points at **does** come
    across, with its `entityType`: the relation is something believed about it.
    In practice that is most of them, since the usual reason an entity has no
    observations is that it was created only to be related to.

Running it twice is safe. The export dedups by content hash, your entities
re-attach to the Subjects they already have, and identical claims are caught
before they can be stored twice. Exporting again later from the old server
brings across only what is new.

### Your migrated memories will look less confident. That is deliberate.

They are second-hand: this store never saw the claim made, cannot check it, and
did not calibrate any number attached to it. Every migrated belief therefore
gets one flat, low confidence (`migration.import_confidence`, default `0.35`) and is
labelled `IMPORTED`, which is also what lets you tell **what you brought with
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

Each migrated belief points back at the export file itself (deposited verbatim,
hashed, and cited down to the line the record came from) plus who ran the
import. Nothing is attributed to the server you migrated from: Particles never
fetched it and cannot re-verify it, so inventing a source there would be exactly
the fabricated provenance the store exists to prevent.

Two consequences worth knowing:

- **Malformed lines are skipped and reported**, never silently dropped. The
  original bytes stay in the corpus, so a later fix can re-read them without
  asking you to export again. The same goes for a field the reference format
  does not define (a fork's timestamp, say): it is counted and left in the
  file, never guessed into a field that means something else.
- **An entity with no observations and no relations is not migrated**, as the
  warning above says. This is a decision, not an oversight. Creating the bare
  name anyway would hand you a Subject that `particles lint` flags as a
  `PHANTOM_SUBJECT` on the very next command and that `particles subjects gc`
  exists to delete, so the importer would be manufacturing the defect the
  store's own tools then ask you to clean up.

  If you want one of them back, say so yourself after the swap: ask your agent
  to call `create_entities` with that name and an empty `observations` list.
  The swapped-in server accepts it, exactly as the reference does. Lint will
  still mention it until something is recorded about it, but it is there
  because you put it there.

## Configuration reference

All knobs live under `mcp.memory_compat`; see `config.yaml.sample` for the
commented block and [Configuration](../operator-guide/configuration.md) for the
loader's precedence rules.
