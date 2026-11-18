# When sources disagree

An agent's memory says one thing, and a source it reads today says another.
The user has moved, a document was revised, a page was wrong, or two notes
were written from different points of view. A memory that silently keeps the
old value acts on stale facts; a memory that silently takes the new one can be
rewritten by any page the agent happens to read.

Particles takes neither shortcut. This page explains which side wins, what
condition changes that, what happens to the losing claim, how a disagreement
reaches you, how stale claims are caught, and where every decision is
recorded.

## The short version

- **Nothing is overwritten.** A claim is never edited in place. A claim that
  loses is retired with a status and a reason, keeps its source and
  confidence, and stays readable. Its `confidence.value` never changes.
- **A winner is picked only when there are grounds for one.** Three
  situations give grounds: one source restating its own value later, a
  document that declares it replaces another, and a large enough gap in how
  much you trust the two sources. Each is described below.
- **Otherwise both are kept and you are asked.** The existing claim stays
  active. The new claim is stored in full but held out of query results, and
  a conflict record pairing the two waits for your ruling.
- **Every decision can be audited and most can be undone.** Retirements are
  dated, automatic ones write an event naming both claims, and a claim
  retired by an automatic replacement can be restored.

## Which side wins

When a new claim arrives, the pipeline looks for existing claims it might
conflict with: claims from the same source that are close in meaning, and
claims about the same subject from any source. Each pair is then checked in a
fixed order, and the first check that applies decides it. The specification
calls this order the conflict-resolution ladder, and the rung numbers in
parentheses below are its numbers
([technical specification §6.4](https://linkedparticles.org/spec/technical-specification/#conflict-resolution-with-source-trust)).

| Situation | Outcome | What happens to the other claim |
|---|---|---|
| One document declares that it replaces the other's document, and the two claims really conflict (rung 1.5) | The claim from the replacing document wins | Retired: `PROVENANCE_STALE`, reason `DOCUMENT_SUPERSEDED` |
| The two sources' trust ranks differ by at least 0.15 (rung 2) | The more trusted source wins | Retired: `PROVENANCE_STALE`, reason `LOWER_TRUST_SOURCE` |
| The same source gives a newer value for something that holds one value at a time, such as where someone lives (rung 2.5) | The newer claim wins, by the source's own dates | Retired: `PROVENANCE_STALE`, reason `SUPERSEDED_BY_UPDATE`; the winner's `supersedes` names it |
| None of the above (rung 3) | Neither wins | The new claim is held for review; see [Kept and flagged](#kept-and-flagged) |

A few details decide which row applies:

- **Trust** comes from your trust policy: statements about a particular
  source, an author, a source type, or a site, most specific first. Rulings
  you make in review become trust statements, so the store learns your
  preferences over time. When the existing claim is the more trusted one, the
  new claim is never stored at all; an event records what was dropped and why
  (see [Auditing a decision later](#auditing-a-decision-later)). The trust
  rung applies only in a single-owner store. In a store several contributors
  write to (`reconciliation.store_mode: multi`, the default for a store your
  agents write to directly), one contributor's trust never retires another's
  claim, and such a pair goes to review.
- **"The same source"** means the two claims share every trust key except the
  document itself: the same site, source type, author, and contributor. Your
  own notes and your own agent sessions count as one source only within one
  project. In a store several contributors write to, the two claims must
  also name who wrote them: the same recorded author, or two captures by
  your own harvesters (the Claude Code hook and `particles audit`), which
  count as you. A newer claim never replaces someone else's older one this
  way. Both claims must carry known source dates in a strict order, and a
  model must judge, reading twice, that both fill one slot and the newer one
  gives it a new value. A value that once true stays true (who wrote a book,
  a release date) is never updated this way: two values for it are a real
  conflict.
- **A document replacing another** is a declared relation, such as an
  architecture decision record's `supersedes:` field. A claim from the old
  document that the new one does not contradict is left alone.

## When nothing needs resolving

Many pairs that look like conflicts are not, and the pipeline lets them
stand rather than manufacture work for you:

- **The same claim again.** A second source stating exactly the same claim
  adds its source to the existing claim. No copy is made.
- **Similar is not contradictory.** Two claims that read alike, such as a
  paraphrase or "X said Y" beside "Y", are not a disagreement. A model must
  confirm a contradiction, and on extraction a second reading checks it
  against each claim's source passage, kind and date. Without a confirmed
  contradiction both claims stay active.
- **Opinions, preferences and definitions.** A claim nobody can settle by
  weighing two sources (an opinion, a feeling, a document's own rule) is not
  arbitrated. Both stay active. Each claim records this as its
  `assertion_modality`, which you can correct.
- **Genuine variation.** A claim marked `ALEATORY` describes something that
  varies by nature: a forecast, a range, a schedule. Trust and document
  replacement never retire one. A later report of what actually happened can
  still retire an earlier plan from the same source.
- **Different projects.** When stores are scoped by project, two projects'
  different values for one subject are not an update of one by the other.
  Both stay active, and the contradiction is recorded as a `CONTRADICTS`
  relation between them.

## Kept and flagged

When no rule can settle a real disagreement, the pipeline keeps both sides
and asks you:

1. **The existing claim stays active** and keeps answering queries.
2. **The new claim is quarantined.** It is stored in full (content, source,
   confidence, subjects) with status `PROVENANCE_STALE` and reason
   `CONFLICT_PENDING`, so queries do not return it, but review can restore it
   exactly.
3. **A conflict record is created** with status `INCONSISTENCY`, pointing at
   both claims in a fixed order (A the existing claim, B the newcomer).
4. **The existing claim is marked contested.** Query results and the
   session-start digest show it with a `contested` badge naming the conflict
   record, so an agent reading it knows it is disputed.

You rule on the record with `particles review`, or from the curation queue
(`particles curate`):

| Ruling | Effect |
|---|---|
| `PREFER_A` | The existing claim stands; the newcomer is retired. |
| `PREFER_B` | The newcomer wins: it is restored as a new active claim and the existing one is retired. |
| `BOTH_VALID` | Both claims are kept, marked as genuine variation. |
| `DEFER` | A note is added and the record stays open. |
| `DISCARD` | Neither claim is worth keeping; both are retracted. |

A `PREFER_A` or `PREFER_B` ruling also writes a trust statement for the
preferred source, which the trust rung uses to settle similar conflicts
without asking again. The full behaviour of each ruling is in
[Operator guide → lint and review](../operator-guide/lint-and-review.md#the-review-workflow).

**A value you already rejected.** When a source restates a claim that you, or
a review, retracted or replaced by judgment, the store does not let it back in
on the source's word. The new copy is held behind a conflict record that
reads "a candidate re-asserts a claim retired by judgment". Its curation card
is titled "Restated after retraction" (or replacement, or review) rather than
"Open conflict", and says when and why the claim was retired and which source
states it again. `PREFER_A` keeps the retirement and `PREFER_B` lifts it,
restoring the claim.

## Catching a stale claim before it is used

A retired claim leaves query results immediately, so the question is how
claims get retired before an agent acts on them. Several mechanisms do this:

- **Newer values from the same source** retire older ones as they arrive
  (the update rule above), and a nightly sweep applies the same rule to
  claims already in the store, including other phrasings of a retired value.
- **Claims that depended on a retired value** are found and replaced with a
  dated restatement. "A ten-minute walk from the user's flat" becomes a claim
  about the old flat, as of when it was true.
- **Edited sources are re-read.** A source registered as one you keep editing
  is re-extracted when it changes, and claims the new version no longer
  states are retired.
- **Expiry and broken sources.** `particles lint` flags a claim whose
  `valid_until` has passed, a claim whose evidence was retracted, and a claim
  whose source snapshot is gone; `lint --fix` retires them.
- **Age lowers ranking.** Older content ranks lower at query time, without
  changing the stored confidence.
- **Contested claims are labelled** in every query result and in the digest,
  as described above.
- **Automatic retirements are reported.** The session-start digest lists
  every claim an automatic rule retired since your last session, grouped by
  the source that caused it, with the command to restore them.

One limit is worth knowing: a claim that nothing newer contradicts, with no
expiry date and an unchanged source, is not detected as stale. It only ranks
lower as it ages. Setting `valid_until` on time-bound facts, and asking an
agent to supersede a claim it finds wrong, closes that gap.

To bound the damage a hostile or broken source can do, one source may win at
most five update retirements in any 24 hours
(`reconciliation.update_supersession.max_retirements_per_entry`). Past that,
its conflicts go to review instead.

## Auditing a decision later

Every claim keeps its own record of what happened to it:

```bash
particles particle show <id>               # status, reason, sources, and when it was retired
particles query "..." --as-of 2026-09-01   # what the store believed at that instant
```

[As-of time travel](as-of.md) replays any past instant with the supersession
chain visible.

Automatic decisions that retire or drop a claim also write an event naming
both claims:

| Event | Written when |
|---|---|
| `UPDATE_SUPERSESSION` | A newer value from the same source retires an older one |
| `TRUST_SUPERSESSION` | The trust rung retires the less trusted claim |
| `DOCUMENT_SUPERSESSION` | A replacing document's claim retires the old document's claim |
| `CONFLICT_CANDIDATE_DROPPED` | A new claim is not stored because a more trusted claim already holds; the event keeps its text |
| `REVIEW_RESOLVED` | You rule on a conflict record |
| `INCONSISTENCY_CLOSED` | A conflict record closes without review, for example because a later reading no longer confirms the contradiction |
| `RETIREMENT_RESTORED` | You restore a claim an automatic rule retired |

```bash
particles events list --particle <id>
particles events list --type UPDATE_SUPERSESSION --limit 20
```

Each extraction run and each nightly consolidation run also writes one
summary event (`EXTRACT_RUN`, `CONSOLIDATION_RUN`) with counts of what it
settled and what it held for review. The event log is described in
[Operator guide → auditing](../operator-guide/auditing.md).

## Undoing an automatic decision

A claim retired by an automatic replacement (an update, a replacing
document, or a restatement) can be restored:

```bash
particles particle restore <id> --replacement wrong   # the replacing claim was wrong
particles particle restore <id> --replacement holds   # both claims are true
particles particle restore --from-entry <entry-id> --replacement wrong
```

The retired claim is not revived in place; a new active claim is minted from
it, so the history stays intact. `wrong` retracts the replacing claim, and
`holds` keeps both and protects the restored claim from being replaced again.
`--from-entry` restores everything one source replaced, which is the undo for
a source that should not have been trusted. Details are in
[Operator guide → restoring a replaced claim](../operator-guide/lint-and-review.md#restoring-a-replaced-claim).

## Settings that change the outcome

| Setting | Default | Effect |
|---|---|---|
| `trust.differential_threshold` | `0.15` | The trust gap at which the more trusted source wins without asking you |
| `reconciliation.store_mode` | `single` | `multi` turns off the trust rung, so contributors' claims never retire each other |
| `reconciliation.update_supersession.enabled` | `true` | Whether a source's newer value retires its older one |
| `reconciliation.update_supersession.max_retirements_per_entry` | `5` | The most update retirements one source may cause in 24 hours |
| `document_supersession.enabled` | `true` | Whether a declared replacement between documents retires the old document's conflicting claims |
| `extraction.retired_value_quarantine.enabled` | `true` | Whether a restated, already-rejected value is held for review |
| `extraction.verify_conflicts` | `true` | Whether a contradiction must be confirmed by a second reading against the source passages |

Every setting is listed in
[Operator guide → configuration](../operator-guide/configuration.md), and trust
policy in [Operator guide → tuning](../operator-guide/tuning.md#source-trust-rank).
