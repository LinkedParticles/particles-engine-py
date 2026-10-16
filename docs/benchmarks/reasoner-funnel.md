# The reasoner funnel: what the deterministic checks dispose of before the probe (2026-10-15)

**A dated snapshot, as of 2026-10-15.** Lenat and Marcus (2023, footnote 9)
report that Cyc's general theorem prover timed out on over a million
consecutive queries, was switched off, and was not missed: the specialised
reasoners carried the load. The engine's general reasoner is the LLM
contradiction probe, and its specialised reasoners are the deterministic
checks that sit in front of it. This page measures, on a copy of the owner's
store and on the code as it is, how much of the probe's candidate set those
checks dispose of (a), what the probe itself yields on a full run as
configured (b), how many stored pairs the planned deterministic rungs could
ever apply to (c), and the labelled set those rungs will be measured against
(d). It is the measurement the reasoner-ladder proposal's acceptance plan
registered before any rung is built. It sets
no bar and changes no default: with the probe off, every
check in front of it is a filter that says no, and the rungs that could say
yes do not exist yet.

The raw counts are in `reasoner-funnel-2026-10-15.json` beside this page; the
committed script is `scripts/measure_reasoner_funnel.py`.

## Setup

| | |
|---|---|
| Store | a read-only copy of the owner's store, taken 2026-10-15: 66,451 particles, 55,578 of them ACTIVE; 664 corpus entries; 15,334 subjects; schema at Alembic head `051` |
| Code | `main` at `92aa1cb5` (1.191.4), before any port; extractor `general-extractor` 0.18.0; embeddings `all-MiniLM-L6-v2` |
| Models as configured | contradiction probe on `claude-haiku-5-5` (`llm.semantic_lint`); second reading on `claude-sonnet-5-5` (`llm.verification`, falling back to `llm.default`) |
| Census as configured | `lint.contradiction_candidate_threshold` 0.6; `audit.max_contradiction_probes` 200; `audit.verify_contradictions` on, `audit.max_contradiction_verifications` 25 |
| Isolation | own database and blob directory, Claude Code state directory and `benchmark.runs_dir`; projection off; MCP writes off. No census write or ledger row reached the owner's store |
| Instrumentation | none in the library. The script re-applies the census's own predicates to every pair above the similarity gate and attributes each pair to the first check that drops it; its survivor count is cross-checked against the census run with the probe cap at 0 |

## (a) The funnel

The census forms its candidate set from every ACTIVE particle that carries a
current-model embedding and pairs them by cosine similarity at or above 0.6.
On this store that is 55,533 particles and **2,816,564 pairs**. The checks
below run in the census's order: the three per-particle gates before any pair
forms, then the pair rules. "In order" attributes each pair to the first check
that disposes of it; "alone" counts every pair the check would drop by itself.

| Check | Where it runs | In order | Share | Alone | Share |
|---|---|---|---|---|---|
| DOCUMENT_META scope | per particle, before pairing | 116,817 | 4.15 % | 116,817 | 4.15 % |
| Adjudicability gate, `is_truth_apt` | per particle, before pairing | 63,226 | 2.24 % | 70,272 | 2.49 % |
| Non-asserted polarity, DECLINED or HYPOTHETICAL | per particle, before pairing | 34,641 | 1.23 % | 36,836 | 1.31 % |
| CO_EVIDENTIAL component, a paraphrase pair | per pair | 1 | 0.00 % | 1 | 0.00 % |
| CONTRADICTS already recorded | per pair | 87 | 0.00 % | 87 | 0.00 % |
| Stance holder differs | per pair | 0 | 0.00 % | 0 | 0.00 % |
| Generic-versus-instance guard | per pair | 127,560 | 4.53 % | 147,781 | 5.25 % |
| **Survive every check** | | **2,474,232** | **87.85 %** | | |
| Probe ledger: cleared under the current prompt | per pair, after the checks | 236 | 0.01 % | | |
| **Candidate pairs the census reports** | | **2,473,996** | **87.84 %** | | |
| Probe cap (`audit.max_contradiction_probes`) | | 200 planned | 0.008 % | | |

The census itself, run on the copy with the cap at 0, reported 2,473,993
candidate pairs and 236 cleared. The three-pair difference is float rounding
at the gate: 167 pairs sit within 2 × 10⁻⁶ of 0.6, and four of them land on
different sides of it under the census's row-by-row product and the script's
blocked one. The enumeration took 7.9 s in the script and 52.9 s in the
census.

**The deterministic checks dispose of 12.2 % of the pairs, and the cap does
the rest.** Every check in front of the probe together removes 342,332 of
2.82 million pairs. The adjudicability gate, the one the ladder would wrap
first, removes 2.2 %; the generic guard, the newest, removes the most of any
pair rule at 4.5 %. The structural filters that were expected to carry weight
(co-evidential links, recorded contradictions, stance holders) remove 88 pairs
between them. What keeps a nightly census affordable is not the checks but
the cap: 200 of 2.47 million candidate pairs are probed, in similarity order,
cross-source pairs first.

### The checks that do not run in the census

Three of the checks the measurement was asked about live on the write path or
in a planned rung, so the census never applies them. They are counted here
over the candidate pairs the census would probe, and over the 200 pairs the
cap actually sends.

| Check | Where it runs today | Of 2,473,996 candidate pairs | Of the 200 planned |
|---|---|---|---|
| Duplicate key: same normalised text, subject set and holder | the extraction pass and `links dedup`; never the census | 138 | 44 |
| Same normalised text, any subject set | not run anywhere on the contradiction path | 440 | 170 |
| Stored slot key: both triples resolve to one subject id and one predicate | nowhere yet; the key the comparator rung would use | 1,631 | 3 |
| Both sides carry a triple at all | | 492,559 | 25 |
| Past-value marker on either side, the guard in front of the slot reading | the write path, before the update probe | 23,689 | 0 |
| Observer precondition declines the pair | the write path, after the probe: a declined pair is probed only to record a divergence | 1,732,911 | 131 |
| Observer precondition sends the pair to review | the write path, after the probe | 343 | 0 |
| Observer precondition reconciles the pair | | 740,742 | 69 |
| Disclosure test fails: no reader has both claims in view | the nightly disclosure pass, over confirmed pairs only | 12,926 | 11 |
| Both claims from one source | ordered last under the cap, never dropped | 118,644 | 0 |

Two of these rows say more than the funnel does. **The cap's budget goes to
restatements.** 170 of the 200 pairs the census probes are the same sentence
twice, normalised; 44 of them are full duplicate-key identities (same text,
same subject set, same holder) and the other 126 are the same text attached to
different subject sets, which the subject resolver split. The probe answers
"no contradiction" on each, at a call apiece, and the ledger then clears it.
A duplicate key in front of the probe would have disposed of 85 % of this
run's budget without a call. **Slot keying today is a model rung.** The
same-slot question on the write path is answered by the update probe, a
completion; its deterministic gates (the precondition, two datable claims, no
past-value marker) select which pairs are asked. The stored triple key, which
is what the comparator rung would read, covers 0.07 % of candidate pairs and 3
of the 200 the cap sends.

The observer precondition is counted with the newer claim as the candidate
and the older as the existing one, since the census orders no pair. On this
store it would decline 70 % of candidate pairs and 131 of the planned 200,
because most pairs join claims observed from different Claude Code projects.
On the write path that is not a saving: a declined pair is still probed, and
a confirmed one is recorded as `CONTRADICTS` with `OBSERVER_DIVERGENCE`. The
disclosure test is the looser one and fails for 0.5 %.

## (b) The probe's yield

The full arm ran the census's own probe loop on the copy, sequentially, as
`particles audit` runs it, with the owner's approval of the estimate. The cap
was raised from the configured 200 to 1,000 for a usable sample; the first
200 of the run are the configured census, since the order is the same.

| | 1,000 probes (this run) | First 200 (the configured cap) |
|---|---|---|
| Candidate pairs, after 236 cleared by the ledger | 2,473,993 | |
| Similarity of the last pair probed | 0.985 | 0.995 |
| Pairs that are the same sentence, normalised | 170 | 170 |
| Probe flagged a contradiction | 107 (10.7 per hundred) | 5 (2.5 per hundred) |
| Second readings (cap 25) | 25 | 5 |
| Confirmed by the second reading | **0** | 0 |
| Rejected by the second reading | 25 | 5 |
| Flags left unread past the cap | 82 | 0 |
| **Confirmed pairs per hundred probe calls** | **0.0** (ceiling 2.5 under the reading cap) | 0.0 |
| Wall time | 1,278 s (21.3 min): 1.19 s a probe, 1.41 s a reading, about a minute to enumerate | |
| Tokens | probe 604,938 in / 108,244 out; reading 39,300 in / 1,592 out | |
| Spend at list price | US$0.21 (probe US$0.11, readings US$0.09); the nightly census batches the probes at half price | |

Every flag came from a cross-source pair, as the probe order intends, and
none from a restatement: the 170 identical sentences in the budget were all
answered "no contradiction" at a call each. What the probe flagged has one
shape. 83 of the 107 are session records that say "the pull request was
created at" two different URLs: claims harvested from Claude Code sessions
whose referent, which pull request, was never resolved at extraction, so two
true records of two events read as one slot with two values. The other 24
are six pairs on the case of a PR template's filename, seven pairs of
version-bump records naming different versions, four on an ADR's path before
and after activation, three on which models a research preview covers, and
four near-identical restatements. The second reading rejected all 25 it
read, 19 of them as separate events in separate sessions, four because the
cited passage never names a path, one as a change over time, and one as a
recurring action. Its reasons are in the results JSON beside this page.

On this store, then, a thousand probe calls and 25 readings at US$0.21
confirmed nothing, and the flags they produced are an extraction defect
(unresolved references in session records, with a definite article standing
in for the referent) rather than contradictions. The configured census, at
200 probes, flags five pairs a night of the same shape.

## (c) The applicability census for the planned rungs

Every rung the milestone plans reads the stored structured triple, so the
triple's coverage bounds all of them. The counts below are over the whole
store, not the census's pairs; the last two columns restrict them to the
census's candidate pairs and to the 200 it probes.

| Particles | Count | Share of ACTIVE |
|---|---|---|
| ACTIVE | 55,578 | |
| … with a stored triple | 20,638 | 37.1 % |
| … with a triple, FALSIFIABLE and asserted (the ladder's inputs) | 19,789 | 35.6 % |
| … whose triple resolves to a subject id | 16,593 | 29.9 % |
| … with a literal object | 19,573 | 35.2 % |
| … with a comparable object: a bare number | 1,068 | 1.9 % |
| … a date | 160 | 0.3 % |
| … a boolean | 154 | 0.3 % |
| … a number with a unit (needs unit normalisation first) | 1,307 | 2.4 % |
| … dated (`valid_from` set) | 420 | 0.8 % |
| … bounded (`valid_until` set) | 0 | 0 % |
| … on a subject with a recorded class | 5,152 | 9.3 % |
| … with a negated predicate | 402 | 0.7 % |
| Subjects with a recorded class | 3,770 of 15,334 | 24.6 % |

A slot is a resolved subject id and a normalised predicate; a slot-keyed pair
is two of the ladder's inputs in one slot.

| Rung | Pairs over the store | Of the census's candidate pairs | Of the 200 planned |
|---|---|---|---|
| Slot-keyed pairs (the base every rung selects from) | 2,462 | 1,631 | 3 |
| Comparator: both objects numbers, dates or booleans of one kind | 57 (45 numbers, 6 dates, 6 booleans) | 45 | 0 |
| … of which equal (corroboration) | 37 | 31 | 0 |
| … of which unequal (contradiction candidate) | 20 | 14 | 0 |
| … both numbers with units, out of scope until units are normalised | 182 | 112 | 0 |
| Polarity opposition: same slot and object, one side DECLINED | 1 (0 among asserted inputs) | 0 | 0 |
| … by a negated predicate ("does not", "never") against its base form | 0 | 0 | 0 |
| Interval succession: same slot, both dated, disjoint intervals | 0 | 0 | 0 |
| … same slot, both dated, different `valid_from` (succession-shaped) | 18 of 43 both-dated | 18 of 36 | 0 |
| Class closure: slot-keyed pairs on a classed subject | 667 | 463 | 0 |

With the subject's lexical token standing in where no id resolved, the
slot-keyed set grows to 2,658 pairs and the comparator's to 62; nothing else
moves.

**What this says about the rungs.** The comparator can apply to 57 pairs on
this store, and the admission bar needs 30 applicable labelled inputs, so it
can be measured; it reaches 0 of the 200 pairs a capped census probes, so on
the probe's own input it would decide nothing. Its 20 unequal pairs are not
contradictions on inspection: they are successive values of one slot (a probe
cap raised from 50 to 200 to 1,000; a test count that grew with the suite;
two calibration runs), which is the case the succession rung exists for. That
rung has nothing to read: no particle in the store carries a `valid_until`,
so no two intervals are disjoint, and the 18 succession-shaped pairs are told
apart only by their start dates. The polarity rung has one applicable pair,
and that pair's DECLINED side is removed by the asserted gate before any
pair forms; the negated-predicate form never meets its base form in a slot.
The class closure has a quarter of subjects to start from, almost all of
them the artifact classes (file, command, record, symbol) the extractor
assigns.

## (d) The labelled set

The owner labelled the full arm's 107 flagged pairs in the session on
2026-10-16, by claim shape with exceptions by id, under the fixture's own
label meaning (both claims cannot be true at once, read as the production
probe instruction defines it). They are now in the kept fixture
`contradiction-probe-pairs-2026-10-07.json` under `store_labelled`, beside the
30 authored pairs, which are unchanged; each entry carries the two particle
ids, the probe's verdicts and the owner's ruling, and the block records the
store, date, labeller and the per-shape reasoning.

| Shape | Pairs | Ruling |
|---|---|---|
| Pull-request URL pairs | 83 | compatible: unique URLs name different pull requests or comments |
| PR template filename case | 6 | compatible: on a case-insensitive file system both paths hold |
| Version-bump records | 8 | 7 compatible (past-tense records of successive versions coexist; one past-tense against one present-tense record coexists); 1 contradicts (two present-tense "main is now at" records) |
| ADR path, proposed versus active | 4 | contradicts: an ADR cannot be in both states at once |
| Fast-mode model list | 3 | contradicts: each says "only" on a list, and the lists differ |
| Restatements | 3 | 2 compatible (similar but distinct actions); 1 not labelled (one side carries an extra statement the other neither supports nor refutes) |
| **Labelled** | **106** | **8 contradict, 98 compatible, 1 skipped** |

Against these labels the probe's flags were right 8 times in 106, and the
second reading agreed with the owner on 24 of the 25 pairs it read: its one
disagreement is a fast-mode pair the owner labels a contradiction and the
reading rejected as a change over time. The labelled set is what the planned
rungs' admission bar is measured on. It holds 8 contradictions, so no rung
can yet meet the bar's 30 applicable inputs on store-drawn pairs alone, and
the 30 authored pairs remain the larger half.
