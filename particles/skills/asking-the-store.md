# Asking the Particles store

Use this when you need something the store might already know.

## Which read

| You want | Use |
|---|---|
| An answer to a question, synthesized from the evidence | `query` |
| The beliefs themselves, to reason over yourself | `query` and read the returned particles, or `particle_search` for a text match |
| Everything about one entity | `subjects_show` (and `subjects_search` to find its id) |
| One belief's full record — provenance, status, supersession chain | `particle_show` |
| The shape of the knowledge around a topic | `graph_view` |

`query` is semantic: ask it the actual question, not keywords. Reserve
`particle_search` for when you know a literal string is in the content.

## Read the confidence, not just the content

Every hit carries two numbers, and they are different quantities:

- **Stored confidence** is immutable — what was believed at the moment the
  claim was recorded. It is never rewritten.
- **Effective confidence** is computed at read time: the stored value modulated
  by how much the source is trusted and how old the claim is. This is the
  ranking key, and it is the one to reason with.

A belief can therefore sink without anyone touching it — that is staleness
working correctly, not data loss.

## Contested beliefs are a signal, not an error

A hit may be flagged **contested**: the store holds claims that disagree, and
how far apart they sit depends on whose trust policy is applied. When you see
it, say so in your answer. Presenting one side of a contested pair as settled
fact is the specific failure the flag exists to prevent.

## Time travel

`query` takes an as-of instant. With it, the same question answers from the
beliefs held *at that moment*, and each hit that has since been retired names
what replaced it and when.

It is the **assertion-time** axis — *what did the store believe at T, and why
did it stop* — not *what was true of the world at T*. Those differ, and
conflating them in an answer is an overclaim.

## Say where it came from

Every belief traces to a source. When you use one, cite it. The whole point of
this store over a pile of text chunks is that the chain back to the origin
survives; an answer that drops it has thrown away the thing that made the
answer checkable.

## Canonical documentation

- Querying, filters, and ranking —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/querying.md>
- As-of time travel, including its honest limits —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/as-of.md>
- Concepts (confidence, subjects, status) —
  <https://github.com/LinkedParticles/particles-engine-py/blob/main/docs/user-guide/concepts.md>

If this file and those pages disagree, the pages are right — report the drift.
