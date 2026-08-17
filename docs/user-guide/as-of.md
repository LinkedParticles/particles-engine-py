# As-of time travel

*What did the store believe at instant T — and why did it stop believing
it?* The `--as-of` query lens answers that question live, on any
past instant, with the supersession chain visible. It is the sharpest
expression of "AI memory you can audit and trust": every belief already
carries its assertion instant and provenance; from then on every
retirement carries its instant too, so the store can replay its own history.

```bash
uv run particles query "How many planets are in the Solar System?" --as-of 2000-01-01
```

A bare date means the start of that day, UTC; any ISO-8601 datetime works. A
future instant is rejected — as-of is a historical lens. The same parameter
exists on all three surfaces: the CLI flag, `QueryRequest.as_of` on
`POST /query`, and `as_of` on the MCP `query` tool.

## What the lens does

With `--as-of T`, the query answers from the beliefs **believed at T**:

- A particle counts as believed at T when it had been asserted
  (`asserted_at <= T`) and had not yet been retired — a belief superseded or
  retracted *after* T still answers, which is the whole point.
- **Recency decay and the recency window are evaluated at T** — content that
  was fresh at T scores fresh, exactly as the store would have scored it
  then. Your *trust* policy stays current: trust is your present judgment
  applied to historical beliefs.
- Each hit whose belief has since ended carries an **as-of note**: its
  current status and reason, the retirement instant, the *basis* for that
  instant (`stored` / `successor` / `event` / `valid_until` — so the
  timestamp is itself auditable), and, when a successor exists, the id,
  content, and assertion instant of the replacing belief.
  `particles particle show <successor-id>` is the drill-down.
- Retired particles whose retirement instant the store **cannot
  reconstruct** (some pre-ADR-0191 automated demotions) are excluded
  fail-closed, with a disclosure line — the lens discloses a gap rather than
  manufacture history. The companion `UNDATED_RETIREMENT` lint finding counts
  such rows at hygiene time; growth of that count on a current store means
  something is writing retirements outside the SDK.

An instant before the store's first assertion is a valid question: the answer
honestly says the store held no beliefs at T.

## Live walkthrough

The demo is a real supersession chain — no backdating needed, just two steps
separated in time:

1. **t₀ — learn the old truth.** Deposit and extract a pre-2006 source:

    ```bash
    uv run particles deposit ./pluto-1996-excerpt.txt
    uv run particles extract --all
    uv run particles query "Is Pluto a planet?"
    # → answers from "Pluto is the ninth planet …"
    ```

2. **t₁ — record the revision.** The epistemically correct mechanism for
   "the IAU redefined the term" is the explicit supersede — the
   `particle_supersede` MCP write tool, which retires the
   predecessor and asserts the successor with the `supersedes` pointer in one
   transaction:

    ```text
    particle_supersede(particle_id=<planet-claim-id>,
                       content="Pluto is a dwarf planet (IAU 2006 reclassification).",
                       subjects=["Pluto"], confidence=0.95)
    ```

3. **Time-travel.** Pick any T between t₀ and t₁:

    ```bash
    uv run particles query "Is Pluto a planet?" --as-of <t0<T<t1>
    # → answers from the planet claim, with one note line:
    # ↳ Pluto is the ninth planet … — now SUPERSEDED (EXPLICIT_SUPERSESSION),
    #   retired <t1> [basis: stored]; superseded by 3fa2b1c9: Pluto is a dwarf planet …
    uv run particles query "Is Pluto a planet?"
    # → answers from the dwarf-planet claim
    ```

## The picturesque version — `--as-of 2000-01-01`

For a demo whose instants span real history (belief asserted 1996, retired
on the actual IAU date), a seed script writes the backdated Pluto chain into
a **throwaway** store — a script, not a CLI verb, because backdating is a
capability the normal write path deliberately lacks:

```bash
uv run python scripts/seed_pluto_demo.py --db ./pluto-demo.db

DATABASE_URL="sqlite+aiosqlite:///$PWD/pluto-demo.db" \
  uv run particles query "How many planets are in the Solar System?" --as-of 2000-01-01
# → the planet belief, retired 2006-08-24, superseded by the dwarf-planet belief

DATABASE_URL="sqlite+aiosqlite:///$PWD/pluto-demo.db" \
  uv run particles query "How many planets are in the Solar System?"
# → the dwarf-planet belief

DATABASE_URL="sqlite+aiosqlite:///$PWD/pluto-demo.db" \
  uv run particles query "How many planets are in the Solar System?" --as-of 1980-01-01
# → "The store held no beliefs matching this question as of 1980-01-01T00:00:00+00:00."
```

## Honest limits

The lens is the **assertion-time** (transaction-time) axis of a bitemporal
system: *what did the store believe at T*, not *what was true of the world
at T*. The Pluto store believed "planet" between the instants it *learned*
the two claims — not between 1930 and 2006. Retirement instants are exact for
everything the SDK writes going forward and for reconstructible history
(explicit supersessions, operator retractions, validity expiry); the rest is
disclosed, never guessed.
