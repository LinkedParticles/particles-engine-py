# Particles SDK

**A git-like ledger for what an AI system believes.**

Every piece of knowledge is one claim: sourced, dated, and confidence-scored.
As in a git history, nothing is overwritten: a correction *supersedes* the old
claim, a withdrawal *retracts* it, and a disagreement stays *disputed* in the
open, so the full history of what was believed, and when, is always there.
Trust, doubt, and staleness are applied as a lens at *query time*, never baked
into the stored claim.

[Get started in 5 minutes](user-guide/getting-started.md){ .md-button .md-button--primary }
[Read the whitepaper](spec/whitepaper.md){ .md-button }

## Why Particles?

Ask an AI system a question and you want to know: *Where did this come from?
Is it still true? What does it actually believe, and what happens when two of
its sources disagree?* Two established approaches each give up on half of that.

- **Formalize everything.** Cyc and the semantic web bet that machines could
  reason over knowledge once it was formalized. They stalled on the cost of
  formalizing it by hand.
- **Formalize nothing.** Retrieval-augmented systems store raw text chunks and
  return whatever looks similar. Fast to build, but a chunk has no notion of a
  claim, a source's trustworthiness, or a belief that was later corrected.

Particles takes the path between them. An LLM extracts each claim as a plain
sentence, bundled with confidence, provenance, and canonical **subjects** into
a **particle**: the smallest self-contained unit of knowledge, its uncertainty
grounded in the [PSUM](https://www.omg.org/spec/PSUM/) standard. Truth is
scoped, not absolute; contested claims stay visible under an auditable trust
policy. An agent's knowledge becomes something you can query, audit, and revise
one belief at a time.

## Three ideas carry the design

<div class="grid cards" markdown>

-   **Nothing is overwritten**

    ---

    A corrected claim *supersedes* the old one; retractions cascade as status
    changes; every lifecycle transition is validated and auditable. Version
    control for beliefs.

-   **Trust is a read-time lens**

    ---

    A particle's stored confidence is immutable. At query time it's modulated
    by extractor trust, source trust, and recency decay into an *effective*
    confidence used for ranking. Change your trust policy and nothing gets
    rewritten.

-   **Contradictions are first-class**

    ---

    When two sources disagree, the conflict becomes a visible inconsistency
    record to review, never a silent overwrite. Your rulings accumulate into a
    reusable source-trust policy.

</div>

## See it in action

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run particles db init

# Deposit a source (file or URL): append-only, SHA-256 snapshotted
uv run particles deposit https://en.wikipedia.org/wiki/Douglas_Lenat

# Extract claim-granularity particles, then ask, showing the evidence
uv run particles extract --all-pending
uv run particles query "What is Cyc, and what was Lenat's role in building it?" \
  --show-particles
```

```text
 CONF   EFF   EXTRACTOR          CONTENT
 1.00   0.49  general-extractor  Lenat worked on the Cyc program at MCC.
 1.00   0.49  general-extractor  Douglas Lenat was the founder and CEO of Cycorp, Inc. …
 1.00   0.49  general-extractor  In 1986, Lenat estimated the effort to complete Cyc …
 1.00   0.49  general-extractor  Lenat became principal scientist of MCC from 1984 to 1994.
 …

## What is Cyc, and What Was Lenat's Role in Building It?

Cyc is a large-scale AI project aimed at building a comprehensive common
sense knowledge base. Douglas Lenat was the central figure behind it,
driving the work across two institutional phases: principal scientist at
MCC (1984-1994) and founder and CEO of Cycorp from 1994 onward …
```

`--show-particles` prints the retrieved claims ranked by **effective
confidence** (the `EFF` column: each particle's immutable stored `CONF`
modulated by source trust and recency *at read time*) above the synthesized
answer, so every answer is traceable to the exact particles it was built
from. Drop the flag for just the prose answer; run `particles lint` to
surface contradictions and staleness, or `particles export obsidian ./vault`
to project the whole graph into your notes.

Full walkthrough: [User guide → getting started](user-guide/getting-started.md).

### And what did it believe in 2000?

That demo shows the loop. This one shows what makes it different. Because
nothing is overwritten, the store can replay its own history — a seed script
writes one real supersession chain into a throwaway store (*"Pluto is the ninth
planet"* learned in 1996, retired on the actual IAU date):

```bash
uv run python scripts/seed_pluto_demo.py --db ./pluto-demo.db

DATABASE_URL="sqlite+aiosqlite:///$PWD/pluto-demo.db" \
  uv run particles query "How many planets are in the Solar System?" --as-of 2000-01-01
# → the planet belief, and one note line: now SUPERSEDED, retired 2006-08-24,
#   superseded by "Pluto is a dwarf planet (IAU 2006 reclassification)."

DATABASE_URL="sqlite+aiosqlite:///$PWD/pluto-demo.db" \
  uv run particles query "How many planets are in the Solar System?"
# → the dwarf-planet belief
```

One flag, and the same question answers from the beliefs held at that instant,
each retired hit naming what replaced it and when. `--as-of` is the
**assertion-time** lens: *what did the store believe at T, and why did it stop*
— not what was true of the world at T.

Full walkthrough: [User guide → as-of time travel](user-guide/as-of.md).

## The honest tradeoff

No free lunch. Choosing extraction over hand-formalization means **no provable
inference, some extraction noise, and ongoing curation**. Particles doesn't
hide that cost; it makes it visible and manageable: `lint` surfaces
contradictions and staleness as they accumulate, and `review` turns source
disagreements into a reusable trust policy.

## Where to go next

<div class="grid cards" markdown>

-   **Use it**

    ---

    Manage knowledge: deposit, extract, query, lint, export.

    [User guide →](user-guide/index.md)

-   **Operate it**

    ---

    Run it long-term: configure, tune, review, troubleshoot.

    [Operator guide →](operator-guide/index.md)

-   **Extend it**

    ---

    Add a new extractor, exporter, or benchmark suite.

    [Plugin-author guide →](plugin-author-guide/index.md)

</div>

## Reference

- [CLI command index](cli.md) (workflow-oriented) and
  [CLI reference](cli-reference.md) (auto-generated full options)
- [Python API reference](api/schema.md) (auto-generated from docstrings)
- [Whitepaper](spec/whitepaper.md): design motivation
- [Technical specification](spec/technical-specification.md): formal schema + operations
- [Roadmap](roadmap.md): what ships in each release
- [Security and trust](security.md): the audit record, what held under
  attack, and the caveats we ask you to understand
- [Architecture decisions (ADRs)](ADR/README.md): the design log
