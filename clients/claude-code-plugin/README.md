# Particles curation pane for Claude Code

A Claude Code plugin that puts the Particles curation queue in a pane beside
the conversation, and lists a conflict when the agent's Particles `query` uses
a belief it contests. The pane renders the shared curation core
(`clients/curation-core`), the same one the web UI renders, so a card reads
and behaves the same in both.

Installing, configuring and using it are described in the user guide's
Claude Code page, in the section on the curation pane.

## How it is built

- `src/register.tsx` is the hooks module's source. It draws the core's views
  with Claude Code's elements, shows each form an action needs in place of the
  card's buttons, and talks to the configured engine through `$.http.fetch`.
  It holds no curation logic.
- `src/connection.ts` decides where the pane may connect: the configured
  engine only, and https for one that is not on this machine.
- `src/contested.ts` reads contested beliefs out of a `query` result. An id is
  used only if it is a canonical lowercase UUID.
- `hooks/register.js` is the bundle Claude Code loads, built from `src/` and
  the core by `esbuild.config.mjs` and committed, so the plugin installs with
  no build step. The build also writes `test/contract.gen.ts` from the core's
  contract.
- `.claude-plugin/marketplace.json` makes this folder its own marketplace.
- `.claude-plugin/plugin.json` carries the release's version: the version-bump
  PR sets it with `pyproject.toml`'s, and a pull request never edits it. Claude
  Code caches an installed plugin by its version, so after each release
  `claude plugin update particles-curation@particles-local` picks up the new
  bundle; then open a new session.

## Commands

```sh
npm ci --ignore-scripts
node node_modules/@anthropic-ai/claude-code/install.cjs   # links the pinned release's binary
npm run build                                             # bundle and contract copy
node esbuild.config.mjs --check                           # fail if either is stale
./node_modules/.bin/claude plugin validate .
./node_modules/.bin/claude plugin test .
```

The tests run under the Claude Code release pinned in `package.json` and need
no login. They draw every card in the core's contract on the desktop and
terminal surfaces, press every action, answer its forms as the contract's
scripted operator does, and check the requests against the contract. A hook
on `http.fetch` stands in for the engine.

The hooks the plugin registers, and the calls it makes, are what
`claude plugin validate` lists: `session.start`, `session.attach`,
`command.run`, `tool.call` and `ui.render`, and calls to `$.http.fetch`,
`$.ui` and `$.command.register`. It makes no model call, starts no process
and reads no file.
