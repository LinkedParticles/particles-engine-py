# Particles curation core

Everything about curating a Particles store except drawing it: what each
curation card shows and in what order, which actions it offers, the steps
and forms behind each action, the engine request each one sends, and every
string a card shows. Each client renders what this package returns. The web
UI (`clients/web-ui`) and the Claude Code pane (`clients/claude-code-plugin`)
both do, so a change to curation is made here once.

The package has no DOM and no dependencies at run time. Its output is plain,
serialisable data: a `CardView` for a card, `ActionView`s for its actions, and
a `Form` for anything an action needs from the operator.

## Using it

```ts
import { CurationClient, cardView, perform } from "../curation-core/src/index.ts";

const client = new CurationClient({
  transport: async (req) => { /* send req.method, req.path, req.body; return { status, body } */ },
  facts: () => ({ hasToken: true, baseUrl: "http://127.0.0.1:8000" }),
  depositedBy: "my-client",
});

const queue = await client.curation();
const view = cardView(queue.cards[0], { readOnly: false });
// Draw `view`; on a press:
const result = await perform(view.actions[0].gesture, queue.cards[0], {
  client,
  reviewerId: "me",
  ask: async (form) => { /* draw the form; return its values, or null to cancel */ },
});
```

A client imports the TypeScript sources directly and bundles them. A client
that cannot import TypeScript checks itself against the fixtures instead.

## Fixtures

- `fixtures/cards.json`: synthetic example cards, at least one of every card
  kind. They hold no real store data.
- `fixtures/golden.json`: what the web UI drew, asked and sent for each card
  before this package existed, recorded by `clients/web-ui/test/golden/capture.mjs`.
  The core's tests check that it reproduces every form, request and string.
- `fixtures/contract.json`: the core's own output for each card: its view,
  and for each action the forms, the scripted answers, the requests and the
  result. It is the contract another client checks itself against.

## Commands

```sh
npm ci --ignore-scripts
npm run typecheck   # regenerates src/openapi.d.ts from artifacts/openapi.json
npm test            # golden and contract checks
npm run fixtures    # rewrite fixtures/contract.json after a deliberate change
```

A deliberate change to what a card shows, offers or sends updates the core,
regenerates `contract.json`, and rebuilds every client bundle in the same
commit. The golden file changes only when the web UI's behaviour is meant to
change, by running the capture again.
