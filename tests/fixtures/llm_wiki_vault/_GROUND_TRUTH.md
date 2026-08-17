# Ground truth manifest — llm_wiki_vault fixture

This directory is a small synthetic LLM-Wiki-style Obsidian vault used to
exercise the retrospective onboarding path (`particles import vault`) end to end: deposit → extract → lint. It reads like a personal
numismatics research vault, but its defects are planted on purpose so a
lint run against it is a measurement, not a vibe check.

This file's leading underscore means `deposit_vault()` skips it — it is
documentation for humans and tests, never corpus content. The same goes
for `.obsidian/` and `_templates/` (skip-rule scaffolding) and
`inventory.csv` (non-Markdown filter).

## Expected deposit set (11 files)

`Home.md`, `Morgan Dollar.md`, `Coin Silver Standards.md`,
`1933 Double Eagle.md`, `Mint Records.md`, `US Mint History.md`,
`Philadelphia Mint.md`, `Most Expensive Coins.md`,
`Flowing Hair Dollar.md`, `Grading and Condition.md`, and
`Seated Liberty Dollar.md` — the last carries deliberately malformed
YAML frontmatter to exercise the drop-dict-keep-body path in
`_strip_obsidian_frontmatter()`.

## Planted contradictions (ground truth)

| ID | Claim A (note) | Claim B (note) | Kind |
|---|---|---|---|
| C1 | Morgan dollar is 90% silver / 10% copper (`Morgan Dollar.md`) | Morgan dollars were struck in sterling, 92.5% silver (`Coin Silver Standards.md`) | composition conflict |
| C2 | 1933 Double Eagle mintage was 445,500 (`1933 Double Eagle.md`) | Mint ledgers record 312,000 struck (`Mint Records.md`) | numeric conflict |
| C3 | The first US Mint was established in Philadelphia in 1792 (`US Mint History.md`) | The Philadelphia Mint was established in 1794 (`Philadelphia Mint.md`) | date conflict |
| S1 | The 1794 Flowing Hair dollar's $10.0M (2013) remains the record price; "no coin has ever sold for more" (`Most Expensive Coins.md`) | The 1933 Double Eagle sold for $18.9M at Sotheby's in 2021 (`1933 Double Eagle.md`) | stale claim |

The truth, for the record: A is true and B is planted-false in C1 and C2;
in C3, A is true (Coinage Act of 1792). In S1 the stale note was once true
(written "as of 2019") and the Double Eagle note reflects the later fact.

## Clean controls (no findings expected)

`Flowing Hair Dollar.md`, `Grading and Condition.md`, `Home.md` (MOC —
mostly links), `Seated Liberty Dollar.md`.

## Running it by hand

Use a scratch store so store-wide findings don't drown the fixture's:

```bash
export DATABASE_URL=sqlite+aiosqlite:///./scratch.db
uv run particles db init
uv run particles import vault tests/fixtures/llm_wiki_vault --verbose
uv run particles extract --all-pending   # requires ANTHROPIC_API_KEY
uv run particles lint
```

A correct run today deposits 11 entries and extracts particles whose
content echoes the notes' claims — frontmatter keys like `tags:` must
NOT surface as claims. `Home.md`, a links-only map-of-content note,
correctly yields 0 particles and is then flagged by lint's
`EMPTY_COMPLETE_SNAPSHOT` check.

**The planted conflicts C1–C3 are detected as cross-source CONTRADICTION
findings by lint's L-SEM-01 (active 1.18.0).** Before that
(discovered on this fixture's first end-to-end run, 2026-06-10) NO surface
caught them: extract-time §6.6 scopes its candidate set to the *same*
corpus entry (`get_active_particles_for_entry`,
`particles/ingest/pipeline.py`), and L-SEM-01 then skipped any pair that
did not share a SOURCE entry. L-SEM-01 was generalized to a
store-wide, embedding-similarity-gated candidate set
(`lint.contradiction_candidate_threshold`, default 0.6): C1–C3's claim
pairs embed at cosine ≈ 0.79–0.87, clearing the gate, and reach the LLM
contradiction probe across the two notes they straddle.

**S1 is *not* caught by L-SEM-01 — by design.** It is a staleness pair (a
once-true "no coin has ever sold for more" claim a later note overtakes),
not an embedding-near contradiction: its pair embeds at cosine ≈ 0.40,
below any similarity gate that still excludes unrelated controls (≈ 0.37).
S1 belongs to a recency-aware staleness lint, not the
contradiction gate (§ Deferred).

So a correct end-to-end run of this workflow surfaces **C1–C3 as
CONTRADICTION findings** and leaves **S1** for the staleness surface. The
full-store-scoped reconciliation path (`reconcile_and_insert`) still serves
only contribution / interchange imports, not this flow. (The contradiction
probe is an LLM call, so the end-to-end run requires `ANTHROPIC_API_KEY`;
the deterministic candidate-gating logic is unit-tested in
`tests/test_lint.py`.)
