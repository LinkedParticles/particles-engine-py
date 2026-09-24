# Memory rot: currency, supersession, and source trust

Most agent-memory benchmarks, LongMemEval included, score a **settled**
corpus at one instant. They cannot see what happens to a memory as the world
changes under it: whether it serves the new value of a fact that changed,
whether it stops serving the old one, and whether a value that only ever
arrived from an untrusted channel ends up believed. `particles benchmark rot`
measures exactly those three things, in the shape the RotBench
harness introduced: the same facts probed again and again across simulated
time.

## What it does

A deterministic world generator builds a 90-day life for one persona ("the
user") with twelve attributes: home city, employer, project codename, code
editor, manager, gym, diet, car, phone, team, coffee shop, and the language
they are learning. Values change on seeded days, and three attributes change
*back* (A → B → A). Two never change, as controls. Later sessions mention old
values in the past tense ("back when I lived in Boston…"), which is honest
history and must not be scored as stale. Six untrusted values are injected
through three channels:

| Channel | How the value arrives |
|---|---|
| `relay` | The assistant relays a web-search result in its own turn |
| `tool_turn` | A `tool:` turn in the transcript (the Claude Code harvester drops these, so on that path the channel is structurally zero; the benchmark measures the raw-transcript path) |
| `source` | A separate web page on an untrusted domain |

A seed *is* the fixture. The world is a pure function of the seed, so nothing
is vendored, and every value comes from a pool of distinctive proper nouns
that are checked never to overlap, which is what makes judge-free substring
scoring unambiguous.

The sessions are deposited **in time order** into a throwaway store and
extracted through the normal pipeline. At days 15, 30, 45, 60, 75 and 90 every
attribute is asked about through the query operation's ranking half, with
default configuration. Each top-10 hit is classified against the truth at
that day:

| Class | Meaning |
|---|---|
| current | Carries the current value |
| mixed | Carries the current value and an old or untrusted one (an honest transition) |
| stale | Carries a superseded value, framed as current |
| history | Carries a superseded value framed as past; reported, never penalised |
| poison (asserted / attributed) | Carries an untrusted value, as a fact or as a quoted source |

The report has three families and no overall score:

| Family | Metric | Better |
|---|---|---|
| Currency | `recall_current@k`: the current value is anywhere in the top 10 | higher |
| | `current_first`: the first value-bearing hit is the current value | higher |
| Supersession | `stale_over_current`: the first value-bearing hit is a superseded value | lower |
| | `stale_retained@k`: any stale hit in the top 10 | lower |
| Poison | `poison_leak@k` / `poison_first` / `poison_surfaced@k` | lower |

Supersession rates only count questions about attributes that have changed,
and poison rates only count attributes that have been poisoned. Each rate is
printed with its numerator and denominator.

## Three arms: what costs money

| Arm | Extraction | Contradiction check | Cost |
|---|---|---|---|
| `oracle` | scripted from the world's ground truth | scripted | **free**, deterministic |
| `probe` | scripted | the live model | cents |
| `live` | the real extractor | the live model | a few dollars per world |

The `oracle` arm answers the question "given perfect perception, does the
engine's decision logic keep a memory current?" It isolates reconciliation,
source trust, and ranking from the extractor, and because it makes no model
call at all it runs in the unit-test tier. `--estimate` prints the projected
calls and, when prices are configured, the dollar cost before anything runs.

```bash
uv run particles benchmark rot                      # oracle arm, seeds 42/43/44
uv run particles benchmark rot --no-trust-policy    # the same, without the trust rule
uv run particles benchmark rot --arm live --estimate
```

## First results: the oracle arm (2026-09-19)

Three worlds (seeds 42, 43, 44), 216 attribute questions (123 of them on
attributes that had changed, 51 on attributes that had been poisoned) plus
36 about attributes the world never mentions. Default
configuration, `all-MiniLM-L6-v2` embeddings, run time 18 seconds, no model
calls.

| Metric | Trust rule on (default) | Trust rule off |
|---|---|---|
| `recall_current@k` | **100%** (216/216) | 100% (216/216) |
| `current_first` | 67.6% (146/216) | 66.2% (143/216) |
| `stale_over_current` | **56.9%** (70/123) | 55.3% (68/123) |
| `stale_retained@k` | **100%** (123/123) | 100% (123/123) |
| `poison_leak@k`, `source` channel | **12.5%** (2/16) | 100% (16/16) |
| `poison_first` | **0%** (0/51) | 9.8% (5/51) |

**Source trust works as designed.** With the untrusted domain demoted,
the poisoned web-page claims are still stored (the ledger keeps what the
source said), but the query-time trust factor ranks them below even unrelated
beliefs: they reach the top 10 in 2 of 16 questions and never rank first.
Without the rule, every one of them surfaces. The `relay` and `tool_turn`
channels are zero in this arm by construction, because a perfect extractor
does not adopt tool output as a fact about the user. Whether the real
extractor does is the `live` arm's question.

**Supersession does not happen.** When a value changes, the old and the new
claim both stay active. In more than half the questions about a changed
attribute, the old value ranks first. None of the three stores contains a
single conflict record. The cause is upstream of the conflict-resolution
ladder, in two layers. First, extraction only compares a new claim against
claims from the *same source document*, and every conversation session is its
own document, so an update in one session never meets the value it replaces
from an earlier one. Lowering the similarity threshold from 0.80 to 0.30 in
the oracle arm still produced no conflict records at all. Second, even
across documents the comparison only considers pairs at least 0.80 similar,
and a value update rarely clears that bar: "The user's home city is Boston"
and "The user's home city is Denver" score 0.70, the employer pair 0.56.
The contradiction check never runs, so the question of how the ladder would
settle the conflict never arises. Ranking then decides, and with decay inert for conversation sources and every claim
at the same confidence, the order between the old and new value comes down to
small cosine differences against the question: close to a coin flip.

This is a property of the engine, not of the scripted perception: the arm
scripts the contradiction check, not the candidate search in front of it. It
is the finding this benchmark was built to surface, and it is recorded here
before any tuning.

**The relevance floor passes everything here.** Questions about attributes
the world never mentions ("What is the name of the user's dog?") still find
top hits above the 0.25 floor in every case (36/36): templated claims about
the user are all close to any question about the user. No answerable
question was refused at any floor up to 0.40. This is a proxy on synthetic
data, not a judged measurement.

## The live arm: the product (2026-09-19)

The same three worlds, with the real extractor and the real contradiction
check, both on `claude-sonnet-5`: 549 extraction calls, run as three
concurrent worlds in about 13 minutes. Projected cost ~US$12 at US$2 / US$10
per million tokens; the harness does not meter actual usage. The report of
record is [`rot-live-2026-09-19.json`](rot-live-2026-09-19.json), which
carries the full text of every top-10 hit.

| Metric | Oracle | Live |
|---|---|---|
| `recall_current@k` | 100% | **94.4%** (204/216) |
| `current_first` | 67.6% | **60.6%** (131/216) |
| `stale_over_current` | 56.9% | **67.5%** (83/123) |
| `stale_retained@k` | 100% | **94.3%** (116/123) |
| `poison_leak@k`, `relay` | 0% | **0%** (0/15) |
| `poison_leak@k`, `tool_turn` | 0% | **55%** (11/20)† |
| `poison_leak@k`, `source` | 12.5% | **12.5%** (2/16) |
| `poison_first` | 0% | **0%** (0/51) |

† Scored with scorer v2. The run was first scored under v1, which reported
14/20: its attribution lexicon did not recognise "The user's public profile
lists the phone as Zenfone." and counted that attributed claim as a leak.
Scorer v2 adds that phrasing. The saved report was re-scored under v2 with
`particles benchmark rot rescore`, which re-classifies the recorded hit texts
without re-running anything. The v1 figures stay reproducible (`rescore` can
target either version), and the re-scored report names its origin in its
first note. No other headline figure moved.

**The supersession finding holds on the product, and is worse.** The live
stores, like the oracle ones, hold no conflict records at all: the real
contradiction check never ran either, for the same two reasons. The real extractor also
phrases the same fact several ways across sessions ("The user drives a
Skoda", "The user drives a Skoda now", "The user bought a Rivian"), so each
attribute ends up with more competing live claims, and the old value ranks
first in two thirds of the questions about a changed attribute.

**The extractor treats the three untrusted channels differently.** When the
assistant relays a search result in its own words, the extractor attributes
it every time ("A profile page found by the web search states that…"): it
surfaces, but never as the user's own fact. A `tool:` line in a raw
transcript is different. Its content was extracted as a plain fact about the
user in 11 of 20 questions, e.g. "The user's diet is keto." This is the path
a transcript deposited verbatim takes; the Claude Code harvester drops tool
output before extraction, so on that path the channel does not exist. The
untrusted web page behaves as in the oracle arm: source trust keeps it out
of the top 10 almost always.

**Honest history mostly reads as history.** The extractor recorded 32
distinct past-tense claims ("The user used to eat kosher", "The user sold
their Skoda"), and the history-aware scorer counted none of them as stale.

## After the fix: same-subject update supersession (2026-09-19)

The supersession finding above is what a new engine rule was designed against.
Extraction now also compares a new claim with active claims
**about the same subject from any source document**, at a similarity floor
low enough for value updates (0.45), and still asks the contradiction check
about every pair it finds. When two claims contradict, come from the same
source lineage, and are both dated, the newer one stays active and the older
one is retired, with a pointer from new to old. Nothing is queued for review.

The oracle arm, same three worlds, same configuration otherwise:

| Metric | Before | After |
|---|---|---|
| `recall_current@k` | 100% | **100%** (216/216) |
| `current_first` | 67.6% | **94.0%** (203/216) |
| `stale_over_current` | 56.9% | **0.8%** (1/123) |
| `stale_retained@k` | 100% | **4.9%** (6/123) |
| `poison_surfaced@k` | 3.9% | **0%** (0/51) |

The same run with the store in multi-contributor mode, as an agent-memory
store is, gives the same numbers. The untrusted web page, a different
lineage, is never allowed to retire a conversation's claim: there it goes to
review with the poisoned value quarantined.

The one remaining stale-first question is a similarity miss. "Ljubljana" and
"Bergen" embed far apart (cosine 0.39), below the floor. Most of the
remaining `current_first` gap is honest history ranking first ("The user's
car was previously Skoda" above "The user's car is Polestar"), which is not
a stale value served as current.

The **live arm** (the real extractor and contradiction check, same tuple as
before; report of record
[`rot-live-adr0268-2026-09-19.json`](rot-live-adr0268-2026-09-19.json)) moved
less:

| Metric | Live, before | Live, after |
|---|---|---|
| `recall_current@k` | 94.4% | **96.3%** |
| `current_first` | 60.6% | **68.5%** |
| `stale_over_current` | 67.5% | **25.2%** (31/123) |
| `stale_retained@k` | 94.3% | **42.3%** (52/123) |

Most of the remaining gap is in *finding* the pair, not in resolving it. The
rule compares claims about the same subject, and the real extractor does not
always name the subject the same way. It splits the persona into "User", "the
user" and "the speaker", and sometimes makes the value the subject ("Hiroshi
is the user's new manager"). 21 of the 31 remaining stale answers trace to
that. A few more are genuinely not contradictions the check should confirm:
learning Tagalog does not mean the user stopped learning Hungarian.

## Reading the numbers

- **The `oracle` arm is not the product.** It is the ceiling a perfect
  extractor would allow. The `live` arm is the product number.
- **Synthetic and small.** One persona, twelve attributes, templated
  sentences. It is a directional probe of three axes, not a leaderboard.
- **Substring scoring undercounts paraphrase.** A live extraction that drops
  the proper noun scores as a miss, and past-tense narration that names an old
  value in passing ("meal-prepped every Sunday while eating paleo") still
  scores as stale. Every report records the full text of every top-10 hit, so
  a classification can be checked by hand, and a scorer change can be applied
  to a saved report for free:

  ```bash
  uv run particles benchmark rot rescore report.json -o rescored.json
  ```
- **Comparability.** Reports are comparable only under the same run tuple:
  arm, seeds, generator version, scorer version, top-k, trust setting, models,
  and thresholds, all of which are recorded in the report.
