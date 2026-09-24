<!--
SPDX-FileCopyrightText: 2026 The Particles authors

SPDX-License-Identifier: Apache-2.0
-->

# Connecting your coding agent

The Particles MCP server is **harness-agnostic**. `particles mcp serve` speaks
plain stdio MCP, so anything that can launch an MCP server can read a store:
Claude Code, Claude Desktop, Codex, Cursor, Windsurf, Zed, OpenCode, and
whatever ships next. Nothing about it is Claude-specific.

What *is* harness-specific is the part above the protocol: whether your store's
standing context is **pushed into the session** without anyone asking for it,
and whether what happened in the session is **harvested back** without the
agent choosing to save it. Those two behaviours ride the harness's own session
lifecycle hooks, and harnesses differ. So this page says plainly, per harness,
what you get.

## What each harness gets

Three capabilities, in increasing order of what the harness has to provide:

- **Read tools**: `query`, `lint`, `graph_view`, and the accessors, as tools
  the agent calls. Needs only MCP. Works everywhere.
- **Recall at session start**: the ranked digest of standing beliefs is *in
  the context window* before the first prompt, rather than behind a tool the
  agent may not think to call. Needs either a hook that can inject context, or
  a file the harness already loads (which is where the projection comes in).
- **Automatic harvest**: the session is deposited into the corpus when it
  ends, with no cooperation from the agent. Needs a session-end hook that
  hands over the transcript.

| Harness | Read tools | Recall at session start | Automatic harvest |
|---|---|---|---|
| **Claude Code** | yes | **shipped** (`particles init claude-code`) | **shipped** (`particles init claude-code`) |
| **Codex** | yes | possible: `SessionStart` injects stdout as context | possible: `SessionEnd` passes `transcript_path` |
| **Cursor** | yes | possible: `sessionStart` returns `additional_context` | possible: `sessionEnd` fires on conversation end |
| **OpenCode** | yes | via the `AGENTS.md` projection | possible: plugin `session.idle` event |
| **Windsurf** | yes | via the `AGENTS.md` projection | partial: `post_cascade_response_with_transcript` is per-response, not per-session |
| **Zed** | yes | via the `AGENTS.md` projection | no agent-session hook documented |
| **Claude Desktop** | yes | the digest resource, if your client reads MCP resources | no hook surface |
| **Any other MCP client** | yes | depends on the client | depends on the client |

Read "**shipped**" as *one command and it works*; "possible" as *the harness
exposes everything needed and Particles does not yet ship the installer*. The
hook scripts are yours to write, and
[the manual route](#harvesting-by-hand) works meanwhile. Only the Claude Code
row is turnkey today.

## Wire up the MCP server

Every block below launches the same server. `uvx --from linkedparticles` needs
no prior install; `--from` is required because the distribution is
`linkedparticles` and the console script it installs is `particles`. If you
already have the package installed, `"command": "particles"` with
`"args": ["mcp", "serve"]` is equivalent.

!!! warning "Pin `DATABASE_URL` and `PARTICLES_BLOB_DIR`, always"
    An MCP server is launched by your client as a child process and inherits
    almost nothing: not your shell, not your working directory, not your
    exports. `storage.blob_dir` defaults to the **relative** `./corpus_blobs`,
    so an unpinned server started from the wrong directory silently splits a
    store from its content. Set both explicitly, with absolute paths, in every
    config below. `PARTICLES_CONFIG` is worth pinning for the same reason
    ([Configuration](../operator-guide/configuration.md) covers what lives in
    it).

`ANTHROPIC_API_KEY` is needed only by the tools that call a model: `query`
composes a cited answer and the semantic half of `lint` reasons over claim
pairs. The pure accessors (`particle_show`, `particles_list`, `subjects_*`,
`events_*`, `graph_view`, `quality_report`) work without it.

### Claude Code

```bash
claude mcp add --scope user particles \
  --env DATABASE_URL=sqlite+aiosqlite:////Users/you/particles/memory.db \
  --env PARTICLES_BLOB_DIR=/Users/you/particles/blobs \
  --env PARTICLES_CONFIG=/Users/you/particles/config.yaml \
  -- uvx --from linkedparticles particles mcp serve
```

The `--` separates Claude Code's own options from the command that runs the
server. `--scope project` writes a `.mcp.json` at the project root instead, in
the `mcpServers` shape below; check that one in if you want the whole team on
the same store. See
[Claude Code → MCP](https://code.claude.com/docs/en/mcp).

**One store, many projects.** Append `--project-observer cwd` to the server
command and the server is bound to the project of the directory it was started
in: its `query`, `particles_list`, `particle_search` and `graph_view` return
global beliefs plus that project's (passing `all_projects: true` reads the
whole store for one call), and what the agent writes is attributed to that
project. Without the flag the server is store-wide, as before. It is a launch
flag rather than configuration because `config.yaml` is shared by every client
on the machine. See
[Claude Code → Choose what a session sees](claude-code.md#choose-what-a-session-sees).

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows. Settings →
Developer → Edit Config opens it either way. Restart the app afterwards.

```json
{
  "mcpServers": {
    "particles": {
      "command": "uvx",
      "args": ["--from", "linkedparticles", "particles", "mcp", "serve"],
      "env": {
        "DATABASE_URL": "sqlite+aiosqlite:////Users/you/particles/memory.db",
        "PARTICLES_BLOB_DIR": "/Users/you/particles/blobs",
        "PARTICLES_CONFIG": "/Users/you/particles/config.yaml"
      }
    }
  }
}
```

See
[Connect to local MCP servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

### Codex

`~/.codex/config.toml`, or `.codex/config.toml` for one project. Note the
snake_case table name: `mcp_servers`, not `mcpServers`.

```toml
[mcp_servers.particles]
command = "uvx"
args = ["--from", "linkedparticles", "particles", "mcp", "serve"]

[mcp_servers.particles.env]
DATABASE_URL = "sqlite+aiosqlite:////Users/you/particles/memory.db"
PARTICLES_BLOB_DIR = "/Users/you/particles/blobs"
PARTICLES_CONFIG = "/Users/you/particles/config.yaml"
```

Or from the CLI:

```bash
codex mcp add particles \
  --env DATABASE_URL=sqlite+aiosqlite:////Users/you/particles/memory.db \
  --env PARTICLES_BLOB_DIR=/Users/you/particles/blobs \
  -- uvx --from linkedparticles particles mcp serve
```

See [Codex → MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

### Cursor

`~/.cursor/mcp.json` globally, or `.cursor/mcp.json` in the project root.

```json
{
  "mcpServers": {
    "particles": {
      "command": "uvx",
      "args": ["--from", "linkedparticles", "particles", "mcp", "serve"],
      "env": {
        "DATABASE_URL": "sqlite+aiosqlite:////Users/you/particles/memory.db",
        "PARTICLES_BLOB_DIR": "/Users/you/particles/blobs",
        "PARTICLES_CONFIG": "/Users/you/particles/config.yaml"
      }
    }
  }
}
```

See [Cursor → MCP](https://cursor.com/docs/context/mcp).

### Windsurf

`~/.codeium/windsurf/mcp_config.json`.

```json
{
  "mcpServers": {
    "particles": {
      "command": "uvx",
      "args": ["--from", "linkedparticles", "particles", "mcp", "serve"],
      "env": {
        "DATABASE_URL": "sqlite+aiosqlite:////Users/you/particles/memory.db",
        "PARTICLES_BLOB_DIR": "/Users/you/particles/blobs",
        "PARTICLES_CONFIG": "/Users/you/particles/config.yaml"
      }
    }
  }
}
```

See [Windsurf → MCP](https://docs.windsurf.com/windsurf/cascade/mcp).

### Zed

`~/.config/zed/settings.json` (or `$XDG_CONFIG_HOME/zed/settings.json`);
`.zed/settings.json` scopes it to one project. The key is `context_servers`,
not `mcpServers`.

```json
{
  "context_servers": {
    "particles": {
      "command": "uvx",
      "args": ["--from", "linkedparticles", "particles", "mcp", "serve"],
      "env": {
        "DATABASE_URL": "sqlite+aiosqlite:////Users/you/particles/memory.db",
        "PARTICLES_BLOB_DIR": "/Users/you/particles/blobs",
        "PARTICLES_CONFIG": "/Users/you/particles/config.yaml"
      }
    }
  }
}
```

See [Zed → MCP](https://zed.dev/docs/ai/mcp).

### OpenCode

`opencode.json` in the project, or `~/.config/opencode/opencode.json`. The key
is `mcp`, the command is a single array, and environment variables go under
`environment`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "particles": {
      "type": "local",
      "command": ["uvx", "--from", "linkedparticles", "particles", "mcp", "serve"],
      "enabled": true,
      "environment": {
        "DATABASE_URL": "sqlite+aiosqlite:////Users/you/particles/memory.db",
        "PARTICLES_BLOB_DIR": "/Users/you/particles/blobs",
        "PARTICLES_CONFIG": "/Users/you/particles/config.yaml"
      }
    }
  }
}
```

See [OpenCode → MCP servers](https://opencode.ai/docs/mcp-servers/).

### Anything else

The server is ordinary stdio MCP. Whatever shape your client's config takes,
these are the four facts it needs:

| | |
|---|---|
| transport | stdio |
| command | `uvx` |
| args | `--from linkedparticles particles mcp serve` |
| env | `DATABASE_URL`, `PARTICLES_BLOB_DIR`, `PARTICLES_CONFIG`, `ANTHROPIC_API_KEY` |

`particles mcp tools` prints the exact tool contract (names, descriptions, and
input schemas) without spawning a transport, which is the quickest way to
confirm a client is seeing what you expect. `particles mcp resources` does the
same for the digest resource.

## What the agent can do once it is connected

The read tools are listed in [Querying → MCP server](querying.md#mcp-server):
`query`, `particle_show`, `particles_list`, `particle_search`, `subjects_*`,
[`lint`](../operator-guide/lint-and-review.md) and `quality_report`,
`list_corpus_entries`, `list_taxonomies`,
[`links_suggest`](../operator-guide/co-evidential.md),
[`corpus_links_suggest`](../operator-guide/citation-signals.md),
[`events_list` / `event_show`](../operator-guide/auditing.md), and
[`graph_view`](graph-view.md#served-get-graph-and-the-mcp-graph_view-tool).

These are the same tools in every harness. They are read-only: they cannot
deposit, extract, or change a particle's status.

### Enabling writes

Writes are default-deny. A store must be named explicitly before
the assert / supersede / retract / tag / link / deposit tools are registered at
all:

```yaml
# config.yaml
mcp:
  write:
    enabled_stores: ["default"]
```

Everything an agent writes is attributed and enters
[effective confidence](concepts.md#confidence) below operator-asserted
content, so it can never outrank you.

## Recall without asking

A tool the agent must decide to call is not memory. Two ways to get the
store's standing context into the session whether or not the agent reaches for
it:

**The digest resource.** The server publishes `particles://digest/<store>`:
one line per ACTIVE belief, ranked by effective confidence, contested beliefs
flagged. Clients that support MCP *resources* can attach it; clients that only
support tools cannot see it. Resource support varies by harness and changes
often, so check yours rather than assuming.

**The projection: works in any harness that reads a file.** Every harness in
the table above loads an instructions file at session start (`AGENTS.md` for
Codex, Cursor, Windsurf, Zed and OpenCode; `CLAUDE.md` for Claude Code).
Render the digest into a sentinel-delimited region of that file and the store's
standing context is in the prompt with no protocol support required at all:

```yaml
# ~/particles/memory.yaml
name: memory-index
sections:
  - title: "Memory index"
    query: null            # rank purely by effective confidence
    top_k: 60
    min_confidence: 0.30
    render: bullets        # deterministic ranked bullets, never LLM prose
max_lines: 120
max_bytes: 16384
```

```bash
particles project ~/particles/memory.yaml AGENTS.md \
  --splice memory-index --without-synthesis
```

`--splice` writes *between* the region's sentinels and preserves everything
outside them, so the rest of your `AGENTS.md` is untouched. The file must
already carry the `<!-- BEGIN PROJECTED: memory-index -->` /
`<!-- END PROJECTED: memory-index -->` pair. `--without-synthesis` keeps the
render deterministic: no model call, no API key, byte-stable for a given store.
Run it from whatever your harness already runs: a git hook, a `make` target,
a cron entry, or the harness's own hook if it has one.

This is exactly the mechanism the Claude Code integration uses for `MEMORY.md`;
[Claude Code memory → the MEMORY.md projection](claude-code.md#the-memorymd-projection)
describes the full contract, including the fold-and-archive behaviour and what
happens if you edit inside the region.

## Automatic harvest

Harvest is *harvest, don't ask*: the session is deposited when it ends,
whether or not the agent thought anything was worth remembering. That needs a
session-end hook which hands over the transcript, and it is the one capability
that genuinely differs across harnesses.

**Claude Code: one command.**

```bash
particles init claude-code
```

installs `SessionStart` / `SessionEnd` hooks, picks or creates the memory
store, seeds the projected region, and runs a first-run audit of the memory you
already have. [Claude Code memory](claude-code.md) is the whole story: what
gets deposited, what is redacted, and how to turn any of it off. One default
to know first: a single store serves every project on the machine, so the
audit reads all of them and the digest is shared across them; see
[One store serves every project](claude-code.md#one-store-serves-every-project).

**Codex and Cursor: the hooks exist; the installer does not.** Both now expose
the full pair. Codex's `SessionStart` adds its stdout to the session's context
and its `SessionEnd` receives `transcript_path`, configured in
`~/.codex/hooks.json` or `~/.codex/config.toml`
([Codex → Hooks](https://learn.chatgpt.com/docs/hooks)). Cursor's `sessionStart`
returns an `additional_context` field and its `sessionEnd` fires when a
conversation ends, configured in `~/.cursor/hooks.json` or
`.cursor/hooks.json` ([Cursor → Hooks](https://cursor.com/docs/agent/hooks)).
Particles ships no `particles init` for either yet, so the hook scripts are
yours to write today; `particles deposit` and `particles extract` are the
verbs they would call.

**OpenCode: a plugin event, not a hook file.** Plugins receive session
lifecycle events including `session.idle`, which fires when a session completes
([OpenCode → Plugins](https://opencode.ai/docs/plugins/)). Same position as
Codex and Cursor: buildable, not built.

**Windsurf: per-response, not per-session.** Cascade's
`post_cascade_response_with_transcript` hands over a transcript after each
response rather than at the end of a conversation, and no Cascade hook can add
context to the conversation
([Windsurf → Cascade Hooks](https://docs.windsurf.com/windsurf/cascade/hooks)).
Harvest is reachable from there; start-of-session recall is not, so use the
projection for that half.

**Zed and Claude Desktop: no agent-session hook.** Zed documents task hooks
but no agent lifecycle event; Claude Desktop has no hook surface. Use the MCP
tools plus, for Zed, the `AGENTS.md` projection, and harvest by hand.

### Harvesting by hand

Wherever nothing automatic is available, the harvest is two verbs over
whatever your harness leaves on disk:

```bash
particles deposit ~/path/to/notes.md          # archive the source, verbatim
particles extract --all-pending               # turn pending snapshots into beliefs
```

`particles deposit --text "…"` records a single note without a temp file, and
`-` reads from stdin. Everything downstream (ranking, contradiction detection,
staleness, the digest) is identical no matter how the material arrived. What
the Claude Code hooks add is not a different pipeline; it is that nobody has to
remember to run it.

## Already running the reference memory server?

If your agent is configured against
[`@modelcontextprotocol/server-memory`](https://github.com/modelcontextprotocol/servers/tree/main/src/memory),
the shortest path in is to change one entry in the config blocks above:
`particles memory serve` presents the same nine tools with the same arguments
and the same responses, so nothing else about your setup changes, and it works
in any MCP client, with no hooks involved.

See [Swapping in for the reference memory server](memory-server-swap.md),
which also covers bringing your existing graph across.

## Related

- [Querying](querying.md): what the ranking actually does, and every filter
  the `query` tool accepts.
- [Claude Code memory](claude-code.md): the shipped end-to-end integration.
- [Using from LangChain](integrations.md): the Python-side adapter, for when
  you are building the agent rather than configuring one.
- [Operator guide → remote engine](../operator-guide/remote-engine.md): when
  the store lives on another machine and several harnesses share it.
