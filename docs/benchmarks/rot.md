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

The live column is one run of each world (n=1). One world run five times
under one configuration spans 32 points of `stale_over_current` (see
[How wide one live world is](#how-wide-one-live-world-is-2026-11-08)), so a
single live figure here is an observation with that much room around it.

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

Both columns are one run of each world (n=1), and each world's extraction is
a separate sample. Seed 42 alone, re-run five times on a later engine, gave
`stale_over_current` from 20.5% to 52.3%
([How wide one live world is](#how-wide-one-live-world-is-2026-11-08)). The
fall from 67.5% to 25.2% is far outside that width; a difference of a few
points between two such columns is not.

Most of the remaining gap is in *finding* the pair, not in resolving it. The
rule compares claims about the same subject, and the real extractor does not
always name the subject the same way. It splits the persona into "User", "the
user" and "the speaker", and sometimes makes the value the subject ("Hiroshi
is the user's new manager"). 21 of the 31 remaining stale answers trace to
that. A few more are genuinely not contradictions the check should confirm:
learning Tagalog does not mean the user stopped learning Hungarian.

## The slot check on the live probe (2026-09-30)

A later change (1.157.0) added a second check before an update retires
anything: the newer claim must give a new value for the *same slot* as the
older one. On LongMemEval that cut the sweep's retirements from 203 to 33
and ended its loss on multi-session questions (see
[the re-measurement there](../benchmarks.md#the-re-measurement-with-the-slot-check-2026-09-30)).
This benchmark measures the other side, whether real updates are still
retired, and the `probe` arm is the one that answers it: extraction stays
scripted, so every difference from the oracle is the live checks' judgement.
Same three worlds, compiled defaults except the checks on
`claude-haiku-4-5`, engine 1.168.1, about 13 minutes and a projected ~US$0.65.
Report of record:
[`rot-probe-adr0282-2026-09-30.json`](rot-probe-adr0282-2026-09-30.json).
The oracle arm under the same configuration reproduces the figures above to
the question.

| Metric | Oracle | Probe |
|---|---|---|
| `recall_current@k` | 100% (216/216) | **99.5%** (215/216) |
| `current_first` | 94.0% (203/216) | **93.5%** (202/216) |
| `stale_over_current` | 0.8% (1/123) | **0.8%** (1/123) |
| `stale_retained@k` | 4.9% (6/123) | **8.1%** (10/123) |
| `poison_surfaced@k` | 0% (0/51) | **0%** (0/51) |

**Real updates are still retired.** Of the 58 values the oracle retires, the
live checks retired 57. The one they kept was a diet change ("kosher" left
active after "Mediterranean"), which is the four extra `stale_retained`
questions.

**The live checks read a past-tense statement as a new value.** They retired
79 claims where the oracle retires 58. The extra 21 are all honest history
retired by later history: "The user's manager was previously Tomasz" retired
in favour of "The user's manager was previously Ingrid", though both remain
true. The scorer counts history as neither current nor stale, so those 21 move
no metric. Once, the same misreading cost the current value: "The user's diet
is vegetarian" was retired in favour of "The user's diet was previously
paleo", which says nothing about the current diet, and the world's last
checkpoint lost its current answer. A first run under a different
configuration showed the same failure once, on a car. It is the coexisting
facts defect again, in a narrower form: two past values are two facts, and a
statement about the past does not replace a statement about the present.

## How wide one live world is (2026-11-08)

Every `live` run extracts each session afresh from a model that samples, so
one world's result is one draw from a distribution. Until these runs no world
had been run twice under one configuration, and differences between single
runs had been read as the effects of code changes. This section
measures the width of that distribution for one world.

Seed 42, the full 90-day world, run five times on engine 1.251.2 with
compiled defaults except the contradiction check, which ran on
`claude-haiku-5-5`; extraction and the second slot reading ran on
`claude-sonnet-5-5`. Trust policy on, top-k 10, scorer v2, real pairs off.
Runs 1 to 3 started together on 2026-11-08 and runs 4 and 5 on 2026-11-09,
from the same checkout and configuration, each as a separate process with its
own store and its own extraction cache; each took about 16 minutes. The world
fingerprint is identical in all five, so only the model's sampling differs.
Reports of record: [run 1](rot-live-n3-2026-11-08-run1.json),
[run 2](rot-live-n3-2026-11-08-run2.json),
[run 3](rot-live-n3-2026-11-08-run3.json),
[run 4](rot-live-n5-2026-11-09-run4.json),
[run 5](rot-live-n5-2026-11-09-run5.json).

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | s.d. |
|---|---|---|---|---|---|---|---|
| `recall_current@k` (of 72) | 72 | 72 | 72 | 72 | 72 | 100% | 0 |
| `current_first` (of 72) | 46 | 55 | 43 | 51 | 42 | 65.8% | 7.6 pts |
| `stale_over_current` (of 44) | 22 | 9 | 12 | 17 | 23 | 37.7% | 13.9 pts |
| `stale_retained@k` (of 44) | 22 | 14 | 16 | 22 | 27 | 45.9% | 11.9 pts |
| `poison_leak@k`, every channel (of 19) | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| `poison_surfaced@k` (of 19) | 10 | 13 | 13 | 13 | 9 | 61.1% | 10.3 pts |
| claims retired as updates | 15 | 19 | 20 | 12 | 16 | 16.4 | 3.2 |

**One world's `stale_over_current` spans 32 points.** The same code, world
and models gave 20.5% (run 2) and 52.3% (run 5). Seed 42's earlier single-run
figures, 31.8% at 1.145.0 and 34.1% on the 1.146.4 replay, both fall inside
that range. The two later runs widened the range found by the first three
(20.5% to 50.0%) and moved the mean from 32.6% to 37.7%, a shift of the size
the spread predicts for a mean of three.

**The spread is per attribute, not per question.** A world asks about a
changed attribute at every checkpoint after the change, so one update that is
missed, or caught, counts four or five times. Stale answers ranked first, by
attribute, of the questions about an attribute that has a superseded value:

| Attribute | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| team | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |
| project | 5/5 | 0/5 | 5/5 | 0/5 | 5/5 |
| gym | 4/4 | 4/4 | 0/4 | 4/4 | 4/4 |
| car | 4/5 | 0/5 | 0/5 | 4/5 | 4/5 |
| phone | 4/4 | 0/4 | 0/4 | 4/4 | 3/4 |
| language | 0/4 | 0/4 | 2/4 | 0/4 | 2/4 |
| home city | 0/6 | 0/6 | 0/6 | 1/6 | 0/6 |
| diet, manager, editor | 0 | 0 | 0 | 0 | 0 |

The team change was missed in all five runs. Five others were caught in some
runs and missed in others, and each such flip moves the rate by about 10
points at once. The 44 questions behind the rate are therefore closer to ten
independent observations than to 44.

**What a live target can resolve.** At a standard deviation of 13.9 points,
telling a 10-point change in `stale_over_current` apart at two standard errors
of the difference takes about 16 runs per arm. A standard deviation measured
from five runs is itself loose: at 95% the true one lies between about 0.6
and 2.9 times it. A live figure is therefore stated with its n, and a live
target is met by a mean over n runs read beside its spread, never by one run.

**What held in every run.** Recall of the current value was complete in all
five. No poisoned value was taken as the user's own in any channel; the
`tool_turn` channel leaked 0 of 6 in each run (0 of 30 together), so the tool
output target of at most 10% holds at n=5. Every store also held
at least one INCONSISTENCY record with a matching claim awaiting review (two
in runs 1 to 3, one in run 4, four in run 5).

**Measured cost.** The `rot` verb does not meter usage, so each run was
wrapped in a usage scope. The five cost US$1.17, US$1.22, US$1.17, US$1.14
and US$1.22 at list price, US$5.92 together, against a projection of US$4.27
each. The projection assumes 1,500 output tokens per extraction call; the
runs averaged about 390, and most of the input was read from the prompt
cache. The "~US$12" projection for the 2026-09-19 live run rests on the same
assumptions.

## Real pairs: your own demotion rulings

The worlds above are synthetic. A store also accumulates real labels: each
time you affirm or dismiss a `demotion` card in the curation queue, the pair
and your ruling are appended to `demotion-rulings.jsonl` in
`benchmark.runs_dir` (see
[Lint and review: rulings on replaced claims](../operator-guide/lint-and-review.md#rulings-on-replaced-claims)).
On the `probe` and `live` arms the benchmark reads that file and asks the
update sweep's two checks about every ruled pair: do the claims conflict, and
does the later one give the earlier one's slot a new value. The sweep retires
only when both say yes.

| Ruling | Checks retire | Checks keep |
|---|---|---|
| `coexist` (you dismissed the card) | **false positive** | agree |
| `replacement` (you affirmed it) | agree | **miss** |

A pair whose checks could not complete is counted as undecided. The result
is printed as its own block, "Real pairs", and stored as `real_pairs` in the
JSON report. It is never pooled into the world metrics, because a templated
world and a real store share nothing but the checks under test. The `oracle`
arm does not score real pairs: its scripted check answers from a world's
value pools and knows nothing about a real claim.

```bash
uv run particles benchmark rot --arm probe --estimate     # counts 2 calls per ruled pair
uv run particles benchmark rot --arm probe --no-real-pairs
uv run particles benchmark rot --arm probe --rulings path/to/rulings.jsonl
```

The file is read when it exists and skipped silently when it does not. Claim
texts appear in the report under `benchmark.record_claim_text`, like every
other hit text.

## Comparing other memories: the door arm

The arms above measure Particles against itself. The door arm puts other
agent memories on the same world. Each memory is built fresh,
the world's sessions are written in time order through the memory's own write
path, and at each checkpoint every question is asked through the memory's
default door. Only the text that door returns is scored: it is cut into
segments at line breaks (including line breaks escaped inside JSON strings)
and sentence ends, and each segment is classified as above. Nothing is
credited unless the text shows it.

| Family | Metrics |
|---|---|
| Currency | `current_in_door`, `current_first`, `recall_current_at_2k`, `recall_current_at_8k` (the current value begins within the first 2,000 or 8,000 characters) |
| Supersession | `stale_over_current`, `stale_in_door` |
| Source trust | `poison_first`, `poison_leak`, `poison_surfaced`, per channel |
| Abstention | `negative_silent` (the door names no value of any attribute), `negative_chars` |
| Cost and speed | `chars_per_door`, `write_ms` (a write including the memory's own settling), `door_ms`, `build_minutes`, `build_usd` |

Competitor memories run through the adapters of the adebench project,
unmodified, from a clone pinned to one commit, each in its own process with
only the environment variables its entry in the arms manifest names.
Particles runs the same way, as a child process with its own store and
configuration, behind the MCP `query` tool at its defaults: the door an agent
receives. Two zero-cost baselines run beside them: `raw-chunks` (the ten
sessions closest to the question by cosine) and `recent-mentions` (the ten
most recent lines sharing a word with the question).

Three tiers share one driver and never change a memory's configuration:

| Tier | World | Seeds | Builds |
|---|---|---|---|
| `quick` | six attributes, three checkpoints, sparse filler (about a quarter of the sessions) | 42 | one; competitor builds are cached on their pin and the world, so a rerun rebuilds only Particles |
| `standard` | full | 42, 43, 44 | one per seed |
| `milestone` | full | three held-out seeds, committed by hash before any build | three per seed for a memory with a model on its write path |

```bash
uv run particles benchmark rot door run --estimate                        # what a quick run would build and cost
uv run particles benchmark rot door run --particles-scripted --no-cache   # free plumbing run
uv run particles benchmark rot door run --tier standard --manifest arms.toml
uv run particles benchmark rot door rescore report.json -o rescored.json
uv run particles benchmark rot --profile quick                            # the internal arms on the quick world
```

A milestone run refuses unless its seeds match the committed freeze record
and every pinned version still holds (`door seeds`, `door freeze`). The freeze
also records, for every metric, whether Particles is predicted to beat, trail
or tie the median competitor; each report prints the prediction beside the
result. Before the freeze, `door label-sample` and `door label-score` check the
segmenter and classifier against blind hand labels on other memories' text:
at least 95% agreement per memory, and at least 90% recall on stale and on
historical segments.

The development tiers come from seeds Particles was tuned on, so their numbers
are internal by construction, and every door report says so in its first
lines. Reports are written outside any repository by default.

### What a reader answers: the answer stage

The door-text tables say what reached the model. They cannot say what a model
does with it: whether a correct refusal followed by forty hits is read as a
refusal, or what a long door costs in the unit an agent pays. The answer stage
asks. One pinned reader model is handed each probe's question, the
checkpoint's date and the door text exactly as recorded, and answers in one
sentence, or `NOT IN MEMORY`. The reply is recorded whole and scored without a
judge: it is cut and classified like a door, then given one verdict by the
first rule that matches.

| Probe | Verdicts, in order |
|---|---|
| unanswerable | `abstain` (exactly `NOT IN MEMORY`), `confabulated` (any value of the world), `miss` |
| answerable | `poison` (asserted), `hedged` (a stale value beside the current one, not marked as history), `stale`, `current` (history and a sourced poisoned value allowed beside it), `poison` (adopted with its source, no current value), `abstain` (a false abstention), `miss` |

A reply that gives the current value and mentions a poisoned one with its
source is `current` and counts as a disclosure. A reply whose only answer is
the poisoned value is a leak whether or not it names the source.

| Metric | Over |
|---|---|
| `answer_current`, `answer_false_abstain` | answerable probes |
| `answer_stale`, `answer_hedged` | probes on attributes with a superseded value |
| `answer_poison` (asserted and adopted), `answer_poison_disclosed` | probes on poisoned attributes |
| `answer_abstain` | unanswerable probes |
| `reader_missed` | answerable probes whose door carried the current value: the reply did not |
| `reader_tokens_in`, `reader_usd`, `reader_overflow` | every probe |

The answer tables render below the door-text tables, never in their place,
headed with the reader, the prompt version and its hash. A summary that quotes
an answer metric quotes its door-text counterpart beside it. Particles' row
is also read twice more from the same recorded door, once without its composed
answer (`hits_only`) and once as the answer alone (`answer_only`), so the
ranking and the answer model can be told apart; those readings compare
Particles with itself and are never a comparative row. A competitor's door is
never edited.

The reader is named in the arms manifest; a read refuses without one rather
than fall back to the operator's default model:

```toml
[reader]
provider = "gemini"               # an llm.providers entry, or "anthropic"
model = "gemini-3-flash-preview"
temperature = 0.0
max_tokens = 200
reasoning_effort = "minimal"      # must match the provider entry's setting
[reader.sensitivity]              # optional: a free local reader, never a headline
provider = "local"
model = "qwen3:14b"
digest = "sha256:..."
context_tokens = 32768            # a longer door counts as an overflow, never cut
```

```bash
uv run particles benchmark rot door run --estimate                  # the reader's line prints apart from the builds
uv run particles benchmark rot door read report.json -o read.json   # read a saved report; no memory runs
uv run particles benchmark rot door read read.json -o again.json --no-cache
uv run particles benchmark rot door reader-floor read.json again.json   # the reader's own noise
```

Replies are cached by content beside the builds, under
`benchmark_rot.door_cache_dir`, so a rerun is free and a competitor's door is
read once per reader. `door rescore` re-scores recorded replies at no cost. The
sensitivity reader runs only under the benchmark host's run lock. Before a
freeze, `door label-replies` and `door label-replies-score` check the reply
scorer against blind hand labels: per reader, at least 95% agreement over
replies from at least three memories, and at least 90% recall on `stale`,
`hedged` and `poison`. The freeze then records the reader, the prompt, the
scorer versions and a prediction for every answer metric.

## Reading the numbers

- **The `oracle` arm is not the product.** It is the ceiling a perfect
  extractor would allow. The `live` arm is the product number.
- **One live run is one sample.** A `live` figure is quoted with the number
  of runs behind it, and one run of one world has a wide envelope: seed 42
  spans 32 points of `stale_over_current` over five runs
  ([How wide one live world is](#how-wide-one-live-world-is-2026-11-08)). A
  difference between two single live runs is an observation until it is
  larger than that spread. The `oracle` arm is deterministic, and the
  `probe` arm scripts extraction and samples only the checks.
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
