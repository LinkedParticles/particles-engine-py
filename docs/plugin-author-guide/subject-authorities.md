# Writing a Subject Authority

A **Subject Authority** owns one external-ID namespace (`wikidata`,
`numista`, `isbn`, `doi`, …): it decides how a subject name an extractor
produced gets linked to an identifier in that namespace. Authorities are what
align the store's [Subjects](../user-guide/concepts.md#subject) to external
ontologies at extraction time.

To add a new ID source you register an authority. You do **not** edit the
subject resolver; it iterates the registry.

## The contract (in 30 seconds)

```python
class MyAuthority:
    NAMESPACE = "myns"                 # stored in ExternalRef.namespace; never rename
    PRIORITY = 60                      # arbitration rank: lower wins
    LIVE = True                        # True if resolve() does a network lookup
    DEFAULT_LINK_CONFIDENCE = 1.0      # feeds ExternalRef.confidence
    APPLICABILITY: list[ApplicabilityClause] = []   # [] = applies to every domain

    def uri_for(self, external_id: str) -> str | None: ...
    def recognize(self, name: str) -> ExternalRef | None: ...
    async def resolve(
        self, session, name: str, *, particle_content: str | None, domain: str | None
    ) -> AuthorityResolution | None: ...
    async def canonical_name_for(self, session, external_id: str) -> str | None: ...
```

Then add one entry to `_make_authorities()` in
[`particles/ingest/authorities/registry.py`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/ingest/authorities/registry.py),
which also defines the `SubjectAuthority` protocol and the
`AuthorityResolution` result type. That is the only file outside your
authority that changes.

Subject resolution matches names against the *store*, so authorities are
Engine-layer code: they live in `particles/ingest/authorities/` and ship with
the engine, not with the Client-layer extractors.

## Where an authority runs

For each subject name an extractor emits, the resolver runs this cascade and
stops at the first hit:

1. **Cache**, then the **local alias index**: an existing Subject with that
   name or alias.
2. **Recognize pass.** Every authority's `recognize(name)`, in `PRIORITY`
   order. The first authority that recognises the name wins; if a Subject
   already carries that `(namespace, id)`, it is returned.
3. **Resolve pass.** Every authority with `LIVE = True` that is applicable to
   the claim's domain calls `resolve(...)`, in `PRIORITY` order, until one
   returns a result.
4. **Bare local Subject.** Otherwise a new Subject is created from the
   extracted name, carrying the ref from step 2 if there was one. Before it is
   created, the *recognising* authority's `canonical_name_for(...)` may supply
   a better name than the raw extracted text.

The resolver **owns every write**: inserting Subjects, merging aliases,
caching. An authority only recognises and looks up. Never write to the store
from an authority.

## Attributes and their rules

| Attribute | Rule |
|---|---|
| `NAMESPACE` | Lower-case slug, unique across the registry, **never renamed**. It is stored on every `ExternalRef` the authority produces and is the key of its `authorities` config entry. Renaming it strands every stored ref: step 2 can no longer find existing Subjects by `(namespace, id)`, and duplicates follow. |
| `PRIORITY` | Arbitration rank when more than one authority recognises the same name, and the order of the live pass. **Lower wins**; registry order breaks ties. Built-ins use 10–50 in steps of 10 (`numista` 10, `km_catalog` 20, `wikidata` 30, `isbn` 40, `doi` 50). An operator can override it. |
| `LIVE` | `True` only if `resolve()` performs a lookup. Recognise-only authorities are `False` and never reach step 3. |
| `DEFAULT_LINK_CONFIDENCE` | Your default for `ExternalRef.confidence`. Exact-identifier authorities use `1.0`. |
| `APPLICABILITY` | `ApplicabilityClause`s gating the live pass by domain. An empty list means "every domain". A `MUST_NOT` clause for a domain excludes you; otherwise, if you declare any `MUST` / `SHOULD` clause, the domain must match one of them. A claim whose domain is unknown always passes. The domain is derived from the entry's source type (see [Extractors](extractors.md#identity-constants-what-must-never-change)). |

## Methods and their rules

- **`recognize(name)`** runs on every name that misses the local index, so it
  must be fast, synchronous, and side-effect-free: a pattern over the name,
  returning `ExternalRef(namespace=self.NAMESPACE, id=...)` or `None`. No
  network.
- **`resolve(session, name, *, particle_content, domain)`** is the live
  lookup. Return `None` on a miss, and always `None` from a recognise-only
  authority. On a hit return an `AuthorityResolution` with **either**
  `existing=` (a Subject already in the store under your ref, which the
  resolver will alias-merge the searched name into) **or** the new-subject
  fields: `external_ref` (required), `canonical_name`, `aliases`,
  `description`. Never both.
- **Score your links honestly.** The resolver abstains from any live link
  whose `external_ref.confidence` is below the operator's
  `subjects.external_link_abstain_threshold` and falls through to the next
  authority, then to a bare local Subject. Use `particle_content` to judge
  whether a candidate entity fits the claim, and return a low confidence for a
  plausible-but-wrong match rather than asserting it at `1.0`.
- **`canonical_name_for(session, external_id)`** is optional enrichment:
  return a better display name for an id you recognised (Wikidata uses its
  label cache), or `None`.
- **`uri_for(external_id)`** returns the namespace's canonical IRI for an id,
  or `None` if the namespace publishes none.

Producing `(namespace, id)` refs is the whole job. Deciding that two refs in
different namespaces denote the *same* entity (`sameAs`) is out of scope for
an authority.

## Two shapes

**1. Recognise-only (a regex over the name).** No new class: instantiate
`PatternAuthority` (in `particles/ingest/authorities/_shared.py`) inside
`_make_authorities()`, as `numista`, `km_catalog`, `isbn`, and `doi` do:

```python
PatternAuthority(
    namespace="doi",
    pattern=re.compile(r"\bDOI:\s*(\S+)", re.I),   # group 1 is the id
    priority=50,
    uri_template="https://doi.org/{id}",           # optional
)
```

It also accepts `applicability=` and `default_link_confidence=`.

**2. Live (an API).** Write a class like `WikidataAuthority`
(`particles/ingest/authorities/wikidata.py`). Its rules:

- Keep network helpers **at module scope**, not as methods, so tests can
  patch them without a network.
- Rate-limit every call through `get_limiter(self.NAMESPACE, rps)` from
  `_shared.py`; each live authority owns one limiter keyed by its namespace.
- Put tuneables (rate limit, timeouts, thresholds) in the configuration model,
  and read them **at call time** with `get_config()`, never at import or in
  `__init__`. Secrets such as API keys come from `particles.secrets`, never
  from config.

## Registering and operator control

Add your authority to the list in `_make_authorities()`; the registry sorts by
`PRIORITY` after applying operator overrides. Import your class inside the
factory, as the existing entries do, so a broken import cannot break the
registry.

Operators can disable an authority or re-rank it without code, keyed by your
`NAMESPACE`:

```yaml
authorities:
  myns:
    enabled: false   # turn it off
    priority: 15     # override PRIORITY
```

Two further levers skip the live pass regardless of your authority: source
types listed in `subjects.skip_live_authorities_source_types` (conversational
and bulk-import sources, whose names are private referents or would cost one
lookup per entity), and a short-lived negative cache for names every
applicable live authority recently missed. Recognise-only authorities still
run for those sources.

## Checklist

- [ ] `NAMESPACE` is a unique lower-case slug that will not change
- [ ] `PRIORITY` chosen relative to the built-ins (lower wins)
- [ ] `LIVE` is `True` only if `resolve()` does a lookup; recognise-only authorities return `None` from `resolve()`
- [ ] `recognize()` is pure and network-free
- [ ] `resolve()` returns `existing=` or new-subject fields, never both, and never writes to the store
- [ ] Live links carry an honest `confidence`
- [ ] Network helpers are module-level and rate-limited per namespace; config read at call time
- [ ] Tests cover `recognize()`, `resolve()` with the network patched, and `uri_for()`
- [ ] Registered in `_make_authorities()`
