# Particles design system

The visual and verbal system for every Particles surface: the public website
— the mkdocs front door — the web UI
(`clients/web-ui`), the graph exporter's HTML artifact, the Markdown / Obsidian
callouts, and any future client. It exists so that Claude Design (or a human)
can skin those surfaces from one source and land on the same product.

**Status:** v0.1 — authored 2026-08-16 as the input to a Claude Design project.
Adopted by the **web UI**, the **docs site** and the **graph exporter**
(§ Adoption, all three steps done). The Obsidian plugin stays theme-native.

## What is in this directory

| File | Role |
|---|---|
| `tokens.css` | **Source of truth.** Colour, type, space, radius, elevation, motion, focus — as CSS custom properties. Light default, dark via `[data-theme="dark"]` and `prefers-color-scheme`. Semantic layer (status / badge / severity / confidence) points at primitives. |
| `tokens.json` | The same tokens in W3C design-token JSON for tooling (Figma / Style Dictionary / Claude Design import). Regenerate by hand when `tokens.css` changes; keep them equal. |
| `components.css` | Component classes (`p-` prefixed) built **only** on the tokens — buttons, chips, subject pills, confidence bar, particle row, cards, banners, callouts, forms, top nav, site header, sheet, hero, footer, graph legend. |
| `logo-mark.svg`, `logo-tile.svg` | The mark on transparent (core dot = `currentColor`) and the dark app tile / favicon. |
| `cards/*.html` | Preview-card fragments; line 1 is the `@dsCard` marker Claude Design indexes. |
| `build.py`, `preview.css` | Renders `cards/` → `previews/*.html` (self-contained, light + dark side by side) and `index.html` (everything on one page). stdlib only: `uv run python design/build.py`. `preview.css` is the card chrome (pane labels, swatch grid) — not part of the system. |
| `previews/`, `index.html` | Generated. Do not edit; edit `cards/`, `tokens.css`, `components.css` and rebuild. |

## Principles

1. **Colour is epistemic, not decorative.** One accent (blue) means "the engine is
   pointing here" — links, the primary action, the retrieval hit. Green, amber
   and red are reserved for the *state of a particle* and never used to
   decorate a heading, a CTA or a marketing block. A green "Get started" button
   is a bug.
2. **Confidence is opacity and a bar; status is colour and stroke.** These are
   two different axes and the system never lets one borrow the other's channel.
   No colour ramp for confidence; no fading for status.
3. **The vocabulary is fixed.** Five statuses (`ACTIVE`, `SUPERSEDED`,
   `RETRACTED`, `PROVENANCE_STALE`, `INCONSISTENCY`) rendered in UPPER_SNAKE,
   never paraphrased. *Contested* is a badge, not a status. Two glyphs only:
   `⚠` contested, `☒` retracted. No emoji anywhere in product surfaces.
4. **Say the tradeoff.** Copy pairs each benefit with what it costs. Voice is
   plain, precise, a little dry; humour only where the docs already have it.
5. **Reconcile, don't reinvent.** Every hex value here already existed in the
   codebase (web UI dark palette; graph exporter light palette; shared chip
   colours). This system removes the third and fourth palettes rather than
   adding a fifth.
6. **Light and dark are equal citizens.** Web UI ships dark-first, docs are
   slate-only, the exporter pins light. The token file makes all three the same
   system with a flip.

## Brand

- **Name.** *Particles* is the product and the standard. *LinkedParticles* is
  the legal / distribution identity — the GitHub org, the PyPI dists
  `linkedparticles` / `linkedparticles-core`, the host `linkedparticles.org`
  . Never "Particls", "LinkedParticles SDK", or a domain suffix.
- **Mark.** A Subject ring with three particles in orbit — blue (retrieval
  hit), green (ACTIVE), amber (contested) — around a core claim in
  `currentColor`. Same geometry as the existing app icon; the tile variant is
  the icon. Minimum 16 px; clear space one dot-radius; the orbit dots are the
  semantic palette and are never recoloured.
- **Wordmark.** "Particles" in the sans at 700, tracking −0.015em, set beside
  the mark at cap-height.
- **Lead line.** *A git-like ledger for what an AI system believes.* Supporting
  lines (one per surface): "Particles is shared memory for humans and AI
  agents." · "AI memory you can audit and trust." · "Version control for
  beliefs." · "When your agent is wrong, you can see exactly why, and fix it
  at the source."

## Colour

Primitives (light → dark):

| Token | Light | Dark | Role |
|---|---|---|---|
| `--p-page` | `#f6f8fb` | `#0b1020` | page ground |
| `--p-surface` | `#ffffff` | `#141b2e` | cards, sheets, panels |
| `--p-surface-2` | `#eef2f7` | `#1b2540` | inputs, chips, secondary buttons |
| `--p-surface-3` | `#e3e9f2` | `#243052` | hover, bar tracks |
| `--p-border` | `#d8dfe9` | `#2a3552` | hairlines |
| `--p-text` | `#1c2733` | `#e6edf3` | ink |
| `--p-text-muted` | `#6b7a8c` | `#9aa4b8` | meta |
| `--p-text-faint` | `#94a1b2` | `#6d7890` | placeholder |
| `--p-accent` | `#2563eb` | `#6ea8fe` | links, primary, retrieval hit |
| `--p-green` | `#15803d` | `#7ee787` | ACTIVE, success |
| `--p-amber` | `#b45309` | `#f0883e` | PROVENANCE_STALE, contested, warning |
| `--p-red` | `#b91c1c` | `#ff7b72` | INCONSISTENCY, error, destructive |
| `--p-slate` | `#64748b` | `#64748b` | SUPERSEDED, neutral chip |
| `--p-slate-deep` | `#475569` | `#475569` | RETRACTED |

Each hue has a `-soft` tint for banner backgrounds. `--p-on-accent` is white on
light and `#07101f` on dark — the vivid hues (blue / green / amber / red) are
deep on light and pastel on dark, so anything filled with one takes
`--p-on-accent` as its ink. The two slates are the same in both themes and
take `--p-on-status` (white).

Semantic tokens never carry a hex: `--p-status-*`, `--p-badge-contested`,
`--p-badge-hit`, `--p-sev-*`, `--p-conf-track/fill`.

Contrast: all text-on-surface pairs ≥ 4.5:1 in both themes; chips ≥ 3:1 (they
are 12 px bold and always accompanied by the status word).

## Type

- Sans (UI + prose): `Inter` → `ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto`.
- Mono (ids, hashes, `pip install`): `JetBrains Mono` → `ui-monospace, "SF Mono", Menlo`.
- Display headlines share the sans (`--p-font-display`); swap that token to
  explore a serif for the marketing hero.
- Scale: 12 xs · 13 sm · 15 UI base · 16 prose · 18 lg · 20 xl · 24 h1 · 30 docs h1 · 38/48 display.
- Weights: 400 body, 500 nav/labels, 600 headings + buttons, 700 wordmark only.
- Headings track −0.015em; the eyebrow is 12 px 600 uppercase +0.02em.
- Meta lines use `tabular-nums`. Prose measure 72ch.

Webfonts are optional: every surface must look right on the fallback stack
(the graph exporter's artifact and the Obsidian plugin cannot load fonts).

## Space, shape, elevation, motion

- Space: 4 px base — 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64 · 96.
- Radius: xs 4 chips · sm 6 code/rows · md 10 buttons/inputs/tabs · lg 16 cards · xl 20 sheets · pill subjects.
- Borders: 1 px `--p-border`. The only 2 px border is the retrieval-hit ring.
- Elevation: `shadow-1` resting card, `shadow-2` sheet / top of stack. Cards on `--p-surface` sit on `--p-page`.
- Motion: 120 / 180 / 280 ms, `cubic-bezier(.2,0,0,1)`; zero under `prefers-reduced-motion`. Press feedback is `scale(.97)`.
- Layout: 560 px single-column app; 1100 px graph + marketing sections; 72ch prose. Below 460 px the top nav drops its labels — it never wraps.
- Focus: 3 px accent ring at 35 %.

## Epistemic encodings (the part that is ours)

Shared verbatim by the web UI, the graph exporter, and the Markdown renderer,
and kept in lockstep:

| Channel | Meaning |
|---|---|
| Opacity | effective confidence — "decay is literal fading" (`.p-fade` with `--conf`) |
| Bar (5 px, accent) | effective confidence, with the number in the meta line |
| Solid / dashed / dotted border | ACTIVE / SUPERSEDED-or-RETRACTED ghost / PROVENANCE_STALE |
| 2 px accent ring, bold blue edge | retrieval hit |
| Filled chip colour | stored status (five values) |
| Amber ⚠ chip or inline | contested (read-time badge) |
| ☒ before the claim | retracted tombstone |
| Outlined chip | lint severity ERROR / WARNING / INFO |
| Pill | subject (an entity) |
| Node size / shade | evidence mass / best-supported claim (`hsl(214 45% L)` from `--p-graph-node-l-weak` to `-strong`, inverts per theme) |

Callouts map 1:1 to `_STATUS_CALLOUT` / `_SEVERITY_CALLOUT` in
`particles/render/markdown.py`: success · note · warning · danger · failure ·
info, plus `[!contested]` and `[!agreement]`.

## Components

Buttons (primary / secondary / danger-outline / ghost; sm / base / lg; one
primary per view; destructive is outlined, never filled) · gesture row ·
chips (status, badge, severity) · subject pill · confidence bar · particle
row (+ `--nested`, `--ghost`, `--stale`, `--hit`) · card (+ `--raised`) ·
banners (error / info / readonly / success) · callouts · fields (input,
select, textarea, hint, error) · top nav + tabs · site header · sheet ·
hero + CTA pair + feature grid + footer · graph legend · empty state ·
progress ("Ranking…" dots).

See `previews/` for every state in both themes; `index.html` has all of them.

## Adoption (how this lands on real surfaces)

Suggested order, each its own PR:

1. **Web UI — done.** The build copies `tokens.css` into `dist/`,
   `index.html` loads it before `styles.css`, and `styles.css` aliases its
   legacy names (`--bg→--p-page`, `--accent→--p-accent`, …) onto the tokens
   with no literal colour left; chips use `--p-status-*`. `src/theme.ts`
   owns the `<html data-theme>` stamp (Settings → Appearance: system / light
   / dark) and reads token values for the Cytoscape styles, so the graph
   ramp (`--p-graph-node-l-weak/-strong`) flips with the theme. `design/`
   rides the engine export so the public tree can rebuild the UI.
2. **Docs site — done.** `hooks/copy_design_tokens.py` materialises
   `docs/stylesheets/tokens.css` (this file + the primitives re-scoped under
   Material's `body[data-md-color-scheme="default"|"slate"]`, via
   `build.py: scoped_tokens`) and the favicon; the authored
   `docs/stylesheets/particles.css` maps every `--md-*` variable onto the
   tokens, flattens the header to the page ground with a hairline, and turns
   admonitions into the callout family. `mkdocs.yml` carries the light/dark
   toggle (`primary`/`accent: custom`), Inter / JetBrains Mono, and the mark
   as an inline custom icon (`overrides/.icons/particles/mark.svg`, a tracked
   copy the hook checks) so its core takes the header ink.
3. **Graph exporter — done.** `tokens.css` is force-included in the wheel
   (`particles/_artifacts/design/tokens.css`; source-tree fallback) and
   inlined into the self-contained artifact, which pins
   `<html data-theme="light">`; `render.py`'s CSS and Cytoscape styles are
   all `--p-*` tokens (read off `<html>` at load), so the `SUPERSEDED` chip
   drift is gone and the light graph greys (`--p-graph-node-border` /
   `-edge` / `-edge-ghost`) are the exporter's values.
4. **Obsidian plugin** stays theme-native by design; nothing to do.

`tokens.css` is the file to change when a value changes; everything else
follows.

## Working with Claude Design

The `previews/` are pushed to the Claude Design project **"Particles design
system"** with the DesignSync tool (`/design-sync`); each preview's first-line
`@dsCard` marker builds the pane's card index (groups: Brand, Colors, Type,
Spacing & shape, Epistemic encodings, Components, Layouts). Iterate here —
edit `cards/` / `tokens.css` / `components.css`, run `build.py`, re-sync — so
the repo stays the source of truth and the design tool is a view of it.
