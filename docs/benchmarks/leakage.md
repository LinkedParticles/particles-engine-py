# Model-prior leakage in query answers

The query operation answers in prose: it retrieves the top-k particles for a
question and asks a model to compose an answer from them. That composing step
is the one place the engine runs inference over what it retrieved, and the
model brings its own background with it. Anything it adds that the retrieved
particles did not supply reaches the reader with no record of where it came
from. When the store holds nothing relevant, the relevance floor and the
answer step's own refusal rule say so. When the store holds *something* and
the model fills in the rest, nothing used to say so.

`particles benchmark leakage` measures that blended case. It reports the
share of answer sentences that the particles the answer was composed from do
not support.

## What it does

1. **Answer.** Each question runs through the query operation exactly as
   configured: the same retrieval depth, the same relevance floor, the same
   composing model (`llm.query_response`). A refused answer is never judged.
   Refusals are reported as their own rate.
2. **Split.** The answer is split into sentences by a deterministic,
   markdown-aware splitter. List items, headings, table rows, and code blocks
   are their own units, and abbreviations, initials, decimals, and version
   strings never end a sentence. The same answer always yields the same
   sentences.
3. **Judge.** Each sentence goes to the entailment judge that abstraction
   promotion already uses, with a rubric for answer sentences. The judge sees
   the sentence, the question and the two preceding sentences as context (to
   resolve an "it"), and the claims the composer was given: the rendered top-k
   plus any narrative's constituent claims. It first decides whether the
   sentence asserts anything. A lead-in, a remark about the knowledge base, or
   a caveat is counted apart and is outside every rate. For a sentence that
   does assert something, it decides whether the claims support every part of
   it. A paraphrase, a summary, a less specific restatement, a combination of
   claims, or an inference that follows directly is supported. Anything else
   is unsupported, **including things that are true**: the measure is
   attribution, not accuracy.
4. **Report.** The headline is the pooled unsupported rate over claim-bearing
   sentences. Beside it are the rate per question source, the mean per-answer
   fraction (each answer weighted equally), and the share of answers with at
   least one unsupported sentence.

The judge runs on `llm.benchmark` and must resolve to a **different model**
from the composer. A model judging its own answers shares the background being
measured and tends to find its own additions supported, so a run where the two
resolve to the same model is refused unless
`benchmark_leakage.require_distinct_judge` is turned off. A self-judged report
says its number is a lower bound.

The questions are the private held-out set the
[relevance-floor benchmark](relevance-floor.md) harvests from an operator's
own agent transcripts. `--question` measures ad hoc questions instead and
prints one row per question. The rendered table never prints a question, an
answer, a sentence, or a judge's reason.

## The first run (2026-10-03)

*Judged under rubric protocol 1, which asks for the verdict before the reason.*

The owner's store, 39,571 ACTIVE particles. All 338 held-out questions, at the
query surfaces' defaults: top-k 40, the general audience, a relevance floor of
0.25. The composer was `claude-sonnet-5` and the judge was `claude-opus-5-5`,
under rubric protocol 1 with two context sentences.

| Measure | Value |
|---|---|
| Unsupported sentences, pooled | **23.6%** (379 of 1,609 claim-bearing) |
| Mean per-answer fraction | 23.2% |
| Answers with at least one unsupported sentence | 66.8% (157 of 235) |
| Refused answers, never judged | 30.5% (103 of 338) |
| Sentences that assert nothing, counted apart | 519 |
| Sentences the judge could not score | 0 |

| Question source | Unsupported sentences |
|---|---|
| `user_prompt` | 23.6% (376 of 1,596) |
| `cli_query` | 60.0% (3 of 5) |
| `mcp_query` | 0.0% (0 of 8) |

Almost every judged sentence comes from typed prompts, which are a proxy for
memory queries. The explicit-query rows are too small to read.

**What the unsupported sentences are.** This store is about a private
software project, so the composing model cannot have memorised its facts.
A sample of unsupported sentences shows the leakage is the model's own
*interpretation*, written in the same voice as the recorded claims. Typically
the first half of a sentence restates a claim and the second half adds a gloss
the store does not hold:

- a motive ("presumably a dev-only default", a refactor "could suggest the
  current structure was seen as a problem");
- a status ("this issue seems to remain a live concern");
- a conclusion ("not yet reliable enough … pending further tuning");
- occasionally an inference the store contradicts (that the project had moved
  past a milestone the store records as re-ranked to come later).

On a store about public knowledge, the same measurement would also catch
memorised facts. On this one, it is catching editorial inference passed off
as recalled fact.

**How to read it.** The judge samples: the Opus-class judge rejects the
`temperature` parameter, so verdicts are not greedy, and a repeat run over the
same answers would differ by a few sentences. The rubric is strict by design.
"Follows directly" is supported, a plausible reading is not. The number is one
operator's store and one question set, not a benchmark.

## Grounded answers and the reason-first judge (2026-10-03)

*Both tables in this section were judged under rubric protocol 2, which asks
for the reason first and the verdicts last. Protocol 2 is now the default.*

Two changes were measured together, so that the comparison between them is
made once, on the corrected judge.

- **Protocol 2.** The first run's rubric asked for the verdict before the
  reason, the order that lets a judge commit before it reasons. Protocol 2 is
  the same rubric with the reason first. Protocol 1 is kept unchanged so the
  first number stays reproducible.
- **Grounded answers.** `particles query --grounded` has the composer cite the
  retrieved particle IDs behind every sentence and label each uncited sentence
  as its own `inference` or `background` (see
  [Grounded answers](../user-guide/querying.md#grounded-answers)). The
  benchmark's `--grounded` mode judges a cited sentence against the particles it
  cites, specifically, and every other sentence against everything the composer
  saw. Its headline counts only **silent** unsupported sentences, cited or
  unattributed, over every claim-bearing sentence. A labelled sentence is
  disclosed, so it is reported beside the headline rather than in it.

Both runs used the first run's frozen copy of the store, its questions, and
its settings: top-k 40, the general audience, a relevance floor of 0.25,
composer `claude-sonnet-5`, judge `claude-opus-5-5`.

**Both runs are partial.** The account's API credit ran out with about four
fifths of the questions measured, and the remainder was not re-run because it
could not change the decision below. The tables report the 286 questions both
runs completed, compared question for question.

| Measure | Ungrounded | Grounded |
|---|---|---|
| Silent unsupported sentences | **25.3%** (367 of 1,452) | **21.7%** (344 of 1,585) |
| Mean per-answer fraction | 26.0% | 19.8% |
| Answers with at least one silent unsupported sentence | 72.0% (144 of 200) | 70.5% (141 of 200) |
| Refused answers, never judged | 30.1% (86 of 286) | 30.1% (86 of 286) |
| Sentences that assert nothing, counted apart | 397 | 66 |

The grounded mode lowers the silent unsupported rate by 3.6 points. A paired
bootstrap over the questions puts the 95% interval at 0.1 to 7.2 points.

| Grounded attribution | Count |
|---|---|
| Sentences citing a retrieved particle | 1,370, of which 344 are unsupported by the particles they cite |
| Sentences labelled `inference` | 262 (16.5% of claim-bearing), of which the judge found 139 supported anyway |
| Sentences labelled `background` | 0 |
| Sentences with no valid citation and no label | 19, none unsupported |
| Citations naming a particle that was not retrieved | 2 |

**What it shows.** The reason-first judge reads about the same as the first
run's: 25.3% here against 23.6% under protocol 1, over a slightly different
question set. The grounded mode does not materially reduce silent leakage.
About one sentence in five is still unsupported by what it cites, every one of
them a sentence that cites a particle, and the model's own labels are loose:
it marked a sixth of its sentences as inference, and the judge found half of
those were supported after all. It never used `background`, which fits this
store: the leakage here is interpretation, not memorised fact. The default
therefore stays off (`query.grounded_answers: false`).

**What it cannot show.** A grounded sentence is checked only against the
particles it cites, which is a stricter test than an ungrounded sentence gets.
Part of the 344 may be miscitation, a sentence supported somewhere in the
retrieved set but not by the particle it names, rather than content from
nowhere. Separating the two needs a second judge call per such sentence and
was not run.
