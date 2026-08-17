# Citation-signal deposit suggestions

Particles never crawls the web — deposit is operator-driven. But a
URL **mentioned repeatedly** across your corpus is a strong curation signal: a
primary source many of your already-deposited discussions lean on, that the
corpus itself doesn't contain. Until you deposit it, the database represents
*hearsay about* that source rather than the source itself.

This surfaces that gap. Every URL mentioned in deposited content — in
comments and prose, not just a post's primary link — is recorded as a
**citation**. The ones you haven't deposited are ranked as suggestions.

## Seeing the suggestions

```bash
particles corpus links suggest
```

Each row is an undeposited URL ranked by **trust-weighted distinct-source
diversity × recency**:

- **Distinct-source diversity, not raw frequency.** One spammer linking the
  same URL fifty times counts once. A URL cited by five *different* sources
  ranks above one cited by two. Raw frequency is gameable; diversity isn't.
- **Source trust.** A citation from a source you trust (your trust policy +
  adopted lenses, computed at query time) weighs more than one from a source
  you don't.
- **Recency.** Recent citations rank above stale ones — but a single old
  high-trust citation never disappears, it just sinks.

A URL must be cited by at least `citation_signal.min_distinct_sources` distinct
sources (default 2) to appear, and the list is rank-capped
(`citation_signal.rank_cap`, default 20). URLs cited only from their own site
(navigation/footer chrome) are filtered (`filter_site_internal`).

`-o json` emits the full report for scripting.

## Acting on a suggestion

Depositing the primary source is an ordinary deposit:

```bash
particles deposit https://press.example/the-release
```

When you do, two things reconcile automatically:

1. The URL's mentions bind to the new entry, so it stops appearing as a
   suggestion.
2. Each source that cited it gains a `COMMENT_LINK` follow edge to the new
   entry (visible via `corpus links list`).

Now the corpus holds the primary source. At query time, §6.6 conflict
resolution and `effective_confidence` prefer the higher-trust primary over the
hearsay — the database's **accuracy** improves, not just its size.

## Silencing a suggestion

Not every cited URL is worth depositing (paywalls, dead links, off-topic
references). Dismiss it:

```bash
particles corpus links dismiss https://example.com/not-worth-it
particles corpus links dismiss https://example.com/maybe-later --snooze 30
```

A plain dismiss is permanent; `--snooze N` hides it for N days. Both are
recorded in the operator event log (`particles events`).

## The lint angle

`particles lint` emits an INFO finding (`UNDEPOSITED_CITED_SOURCE`,
"L-CITE-01") for each URL cited by at least
`citation_signal.lint_min_distinct_sources` distinct sources (default 3 — more
conservative than the verb). It anchors the suggestion to a real grounding gap,
not URL popularity.

## Turning it off

Set `citation_signal.capture_enabled: false` to stop recording new citations.
Existing citations and suggestions stay readable; nothing is captured going
forward. See [Configuration](configuration.md) for the full
`citation_signal.*` knob set.

!!! note "Suggestion-only — never auto-deposit"
    This feature only *surfaces* candidates. It never fetches, crawls, or
    deposits on its own — depositing is always your explicit action. Opaque
    link shorteners (`t.co`, `bit.ly`) are kept as-is rather than resolved,
    because following them to their target would be crawling.
