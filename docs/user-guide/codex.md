# Codex memory integration

The macOS Codex integration captures session testimony and recalls project
context on new sessions. It can use a local Particles store or a same-host
engine shared with another harness such as Claude Code. A local installation
provisions or reuses a consolidation schedule. An engine installation relies
on the engine's resident scheduler.

Run `particles init codex --dry-run --json` to inspect the plan. Select a
configured local store with `--store` and an absolute Particles config with
`--config`. `--command` selects an absolute Particles executable. Apply with
`particles init codex`, then review the two user hooks in Codex's hook UI.
The installer never changes Codex's trust state. Until that review and usable
scheduled credentials are established, the result is “configured, learning
blocked”. A successful configuration write does not prove a learning run.

The installer merges user `hooks.json` and a Particles MCP block in
`config.toml`, and installs the shipped onboarding guidance under owned
`skills/particles-*` directories. It preserves unrelated hooks, TOML comments,
skills and instructions. An existing conflicting Particles registration,
inline hook, plugin hook or edited owned file requires operator resolution.
`particles init codex --remove` removes unchanged owned resources and preserves
the store, captured corpus, credentials and any reused consolidation job.

## Connect Codex to a same-host engine

Set `engine.base_url` in the absolute configuration file you pass to
`particles init codex --config /absolute/path/to/config.yaml`. The first
release supports an engine on the same machine at a loopback address.
A loopback URL can also point to an SSH forward, which sends content
off-machine. Forwarded capture is outside this release's support boundary.

Run `particles init codex --dry-run --json` to see the engine, store,
credential paths and key creation commands. Mint a reader key for recall
and an agent key for MCP on the engine host. For capture, enable
`codex.harvest.allow_remote` and mint a separate Codex capture key. Pass the
absolute checked files through `--recall-token-file`, `--mcp-token-file`
and, when capture is enabled, `--capture-token-file`. The installer
checks their roles before it installs private copies. Review the hooks
in Codex after applying the plan. If you disable remote capture on an
existing installation, revoke its retained capture credential on the engine.

Re-run the installer to switch an existing local installation to the engine.
The plan shows whether it will retire an owned local job or preserve a reused
job. The previous corpus and pending hints stay with their original
destination. With consent, the new capture worker can read retained rollouts
during catch-up. Changing the endpoint requires reinstalling and checking
the keys at the new destination. The engine masks capture text again and
computes the message body hashes from the masked text, while the client
attests to `raw_hash` and the prefix proofs from its local rollout.

Use `particles hook doctor --harness codex` to check the engine, installed
credentials, capture hints and engine consolidation status. Project recall
requires the engine to confirm that it applied project scope. If the engine
store needs rescoping, run that operation on the engine host.

## Capture and recall

SessionEnd writes a small private retry hint and starts a detached capture
worker. Every SessionStart independently scans retained sessions for missed
capture, including on resume. Recall injects complete lines within
`codex.digest_max_bytes`, default 9,000 bytes, on startup, clear, compact and
fork. Resume skips another recall injection. Unresolved capture failures or blocked
scheduled credentials still produce a separate warning on every start,
including resume and sessions with no digest. `particles hook codex-dismiss
--acknowledge` acknowledges the current warning signature; a new failure or
increased count warns again. Capture does not call a model.

Rollouts are resolved by validated session identity inside the configured
Codex home's sessions and archive directories. The event's transcript path
is never a file-opening authority. Unknown complete envelopes quarantine the
whole session. Recognized user and assistant testimony is redacted and
deposited as an append-only conversation; tools, harness instructions and
compaction summaries are excluded. Forks retain their verified ancestry and
capture only the child's delta. A changed captured prefix is quarantined.
A recognized session with no testimony records an empty snapshot and capture
manifest; later testimony appends normally without a model call for the empty input.

Codex and Claude derive the same project key, including linked worktrees.
`codex.observer_scope` independently selects project or store recall; project
is the default. An unresolved project injects nothing under project scope.
A new empty local store is initialized for project scope. An existing unscoped
store requires the explicit `particles memory rescope` workflow; project
recall remains blocked until then, and installation changes no existing
project attribution. Different repository roots that produce the same key share one scope across
both harnesses. Doctor discloses that limitation. A Codex home moved or
restored to a new path is refused; operator rebind is deferred.

Codex session URIs identify harness provenance, not a web publisher. Domain
trust defaults do not score them; explicit URL-pattern policies still apply.
Session update lineage requires a shared project key, including on stores
that have not been rescoped. Different harness namespaces stay separate.

Delivery remains unknown on every Codex version. The integration writes no
exposure rows and supplies no utility credit or rank-lift signal. An exact
output fingerprint is stored before injection solely to quarantine a future
mislabelled echo; it is not a delivery receipt.

## Learning and diagnosis

For a local store, a daily 03:30 user LaunchAgent with login catch-up invokes the bounded
consolidation run. An existing job for the same physical database is reused,
including jobs created for Claude or by hand. An ambiguous destination,
disabled job or duplicate schedule requires operator repair. The installer
never takes ownership of a reused job. The schedule can incur the configured
provider's bounded extraction costs.

Scheduled credentials use the explicit `consolidation.credential_profile`
context through `particles.secrets`, with a macOS Keychain fallback bound to
the store, provider and endpoint. Interactive installation offers enrolment;
noninteractive installation never prompts. No key is written into a job,
hook, TOML, YAML, argument or installer report. Missing or inaccessible keys
block paid passes while deposited sources remain pending. Ordinary hooks,
MCP and library calls retain their environment-only secret lookup. Scheduled
native clients are confined to that run and closed when it ends.
`consolidation.keyless_extraction_providers` explicitly lists credential-free
routes at literal `127.0.0.1` or `::1` addresses (default: `local`);
`localhost` is not treated as proof of loopback. Remove a provider from that list if its
local gateway requires authentication. Reusing an existing schedule preserves
its configuration and credential profile; doctor verifies that job's own route.
Removal retains enrolled credentials. Delete an unused Particles extraction
item explicitly in Keychain Access after checking whether other jobs use it.

Use `particles hook doctor --harness codex` and
`particles hook log --harness codex` for ownership, pins, version evidence,
observer policies, schedule, credential and capture diagnostics. Scheduled
consolidation logs also disclose quarantined-session counts even when the
run is not due. The disclosure explicitly identifies when every recent
completed capture attempt was quarantined.

There is no minimum rollout-retention guarantee. Verify capture before
permanently deleting a Codex session. Uncaptured turns can be lost; a crash
before a retry hint followed by deletion can leave no trace of that loss.
Retry hints retain metadata, not a second transcript copy. The operator can
acknowledge an unresolved hint with `particles hook codex-dismiss --session
SESSION_UUID`; acknowledgement does not recover testimony or delete corpus
content. Extraction progress is the ordinary corpus snapshot hash, not a
hint, cursor or worker exit code.
