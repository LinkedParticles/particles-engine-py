# Writing an exporter

An exporter walks the particle store and writes an external
representation: Obsidian vault, Anki deck, JSON Lines file, …

## The contract (in 30 seconds)

```python
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from particles.exporters.summaries import BaseExporterSummary

class MyExporter:
    FORMAT = "myformat"                    # unique lowercase slug

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> BaseExporterSummary:
        # Extract only the options you recognise; ignore the rest.
        my_option = str(options.get("my_option", "default"))
        ...
        return MyFormatSummary(...)
```

Plus a Pydantic `MyFormatSummary(BaseExporterSummary)` in
`particles/exporters/summaries.py`. Add one line to
`particles/exporters/registry.py::_make_exporters()`. A new option or an
API target also needs the `particles export` command edited; the next
section walks through every file.

## Adding an exporter, file by file

An exporter is one new module plus two small registrations. Nothing else
discovers it.

**1. The module: `particles/exporters/<name>.py`** (or a
`particles/exporters/<name>/` package once it grows). It holds the class
satisfying the `ExporterPlugin` protocol above.

- **`FORMAT`** is a lowercase slug, unique across every registered
  exporter. It is the registry key and the `particles export <FORMAT>`
  argument, so two exporters with the same slug silently shadow each other
  (the registry builds a plain `{e.FORMAT: e}` dict and the later entry
  wins).
- **`options`** is a shared bag: the CLI passes the whole set of `export`
  flags to every exporter, whatever the format. Read the keys you
  recognise with `options.get(key, default)`, coerce them to the type you
  need, and ignore everything else. Never fail on an unknown key.
- **Usage errors raise `ValueError`** (a missing output path, a missing
  required option, an unknown id). The CLI turns a `ValueError` into one
  clean line on stderr and exit code 2; any other exception surfaces as a
  traceback.

**2. The summary: `particles/exporters/summaries.py`.** Subclass
`BaseExporterSummary`, pin `format: Literal["<name>"] = "<name>"`, and add
your slug to the `format` `Literal[...]` on `BaseExporterSummary` itself,
which enumerates every format. Fields that only exist on some runs (a
dry-run-only count, a synthesis-only count) are typed `X | None = None`:
the CLI prints `summary.model_dump(exclude_none=True)`, so a `None` field
simply does not appear.

**3. The registry line: `particles/exporters/registry.py`.** Add one
import and one instance to `_make_exporters()`:

```python
def _make_exporters() -> dict[str, ExporterPlugin]:
    from particles.exporters.anki import AnkiExporter
    ...
    from particles.exporters.myformat import MyExporter   # ← add

    exporters: list[ExporterPlugin] = [
        ObsidianExporter(),
        AnkiExporter(),
        ...
        MyExporter(),   # ← add
        # Register new exporters here
    ]
    return {e.FORMAT: e for e in exporters}
```

The import stays inside the function so the registry costs nothing until
an export actually runs.

**What the CLI does and does not pick up automatically.** `particles export
<FORMAT>` dispatches through the registry, so your format is reachable as
soon as it is registered, and the missing-credential pre-flight is generic
(see below). Three things in
[`particles/api/cli/export.py`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/api/cli/export.py)
are still written out by hand, and you edit them when they apply to you:

- a **new option** needs its Typer flag and an entry in the `options`
  dict the verb builds; an option you only accept programmatically needs
  neither;
- an **API-target exporter** (one that takes no output path) must be added
  to the check that sets `output_path = None`, which today names `notion`
  explicitly; otherwise the verb demands a path;
- the format list in the `format` argument's help text.

## Worked examples in the tree

| Output shape | File | Notes |
|---|---|---|
| Directory of `.md` files (vault) | `particles/exporters/obsidian/` | Template dispatch; pivot vs coin vs generic |
| Single flat file (deck) | `particles/exporters/anki.py` | Cards from `properties` or content |
| Single flat file (one particle per line) | `particles/exporters/jsonl.py` | The smallest complete exporter: threshold filter, non-asserted filter, summary. Start here. |
| Directory of cited articles | `particles/exporters/wiki.py` | LLM synthesis via the shared `particles.render.article_synthesis` |
| Directory of bullet-outline pages | `particles/exporters/logseq/` | Logseq's native format; particle IDs as block UUIDs for cross-page citation |
| External HTTP API (no file) | `particles/exporters/notion.py` | The first API-target exporter; idempotent upsert into one Notion database. Reference for the credential pattern below. |
| Single self-contained `.html` graph | `particles/exporters/graph/` | Scoped epistemic graph view: mandatory scope + disclosed caps; vendored Cytoscape.js inlined so the artifact works offline. What it renders and why is described in [User guide → graph view](../user-guide/graph-view.md); and the flags it accepts are in that page's options table. |

A package-shaped exporter (`obsidian/`, `logseq/`) keeps the plugin class
in `exporter.py` and splits the rest by role: `vault.py` (walk the store
and write files), `format.py` (format-specific Markdown shaping),
`synthesis.py` / `narrative.py` (prose splicing). Copy that layout rather
than inventing a new one.

## Cross-exporter contract

Every shipped exporter, and yours, must honour these options. They are
not on the `ExporterPlugin` protocol, so no type checker enforces them;
the shipped exporters are held to them by tests, and a user who passes
the flag to your format expects the same behaviour.

| Option | Type | Default | Behaviour |
|---|---|---|---|
| `min_particle_confidence` | `float` | `0.0` | Drop particles below this `effective_confidence` BEFORE any per-exporter step: prompt input, cache key, the `min_particles` count, rendered output, references. Filter input is *effective* confidence, never raw `confidence.value`. |
| `min_particles` | `int` | per-exporter | Minimum *post-filter* particle count required to render a subject. Counted over the `min_particle_confidence`-filtered set. Per-subject exporters only; a per-particle exporter (Anki, JSON Lines) has no subject count to check. |
| `include_non_asserted` | `bool` | `False` | Keep non-asserted particles (a document's declined, superseded, deferred, or counterfactual prose). They are off the default factual surface, so drop them unless this is truthy. |

Your summary always reports `particles_dropped_below_threshold: int`. The
field is inherited from `BaseExporterSummary` with a default of `0`; set it
to the number of particles the threshold removed, so dry-run and audit
readers see how much the filter bit.

The confidence threshold reaches you from two directions, and it is worth seeing each:
the operator sets a standing floor in `config.yaml`
([Operator guide → cross-exporter quality threshold](../operator-guide/tuning.md#cross-exporter-quality-threshold)),
and the user overrides it per run with `--min-particle-confidence`
([User guide → exporting](../user-guide/exporting.md)). The CLI resolves
that precedence before calling you, but a programmatic caller may omit the
key, so fall back to `get_config().exporter_common.min_particle_confidence`
(or `0.0`) when it is absent. Filter on
`effective_confidence`, never on the stored `confidence.value`; the two are
[deliberately different quantities](../user-guide/concepts.md#confidence).
The worked computation (trust-weight cache, source-trust ranks,
`compute_effective_confidence`) is in
[`particles/exporters/jsonl.py`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/exporters/jsonl.py).

For the non-asserted filter, don't re-derive the rule: call
`exclude_non_asserted(particles, options)` from `particles.render.markdown`
([source](https://github.com/LinkedParticles/particles-core-py/blob/main/particles/render/markdown.py)),
which returns the list unchanged when `include_non_asserted` is truthy.

## Output shape

The `output` path is:
- a **directory** for vault-style outputs (Obsidian, Wiki, Logseq)
- a **file** for single-file outputs (Anki, JSON Lines, graph)
- `None` for API-based outputs (Notion)

Your exporter is responsible for `output.parent.mkdir(...)` / atomic
writes / cleanup of stale entries from prior runs. If your exporter needs
a path and receives `None` (a programmatic caller can pass anything),
raise `ValueError`. `particles.render.markdown.atomic_write_text` is the
shared write-then-rename helper; use it for files a user may have open
(a vault note) so a crash never leaves a half-written file.

## API-target exporters & credentials

If your exporter writes to an external HTTP API (it has `output=None`)
and needs a credential, follow the Notion exporter's pattern. The pattern
is uniform so every API target fails the same way and no token ever leaks
into a file, an argv, or a log.

1. **One getter per target in `particles/secrets.py`.** Add
   `get_<x>_api_key()` (raises when missing, like `get_notion_api_key()`)
   when the target has no anonymous mode, or `get_<x>_api_key_optional()`
   (returns `None`, like `get_github_api_key_optional()`) when it does.
   The token lives **only** there: never in `config.yaml`, never on
   `ParticlesConfig`, never a CLI flag, never a field of your summary.
2. **Declare the secret; don't read it in the declaration.** Set a class
   attribute `REQUIRES_SECRET = "<ENV_VAR>"` naming the environment
   variable. It is optional and not part of the protocol, so filesystem
   exporters simply omit it; the registry reads it with
   `required_secret(exporter)`, and the CLI uses that to verify the
   variable is set before any store read or network call.
3. **Call the getter as the first statement of `export()`.** The CLI
   pre-flight is a courtesy; a programmatic caller bypasses it. Reading the
   token before anything else is the authoritative check, and it means a
   missing or invalid token can never produce a half-written workspace.
4. **Non-secret target parameters go in a config sub-model.** Database
   ids, property names, and similar settings live in a `<X>Config`
   sub-model in `particles/config.py` plus `config.yaml.sample`, read with
   `get_config()` at call time. A per-run override rides `**options`
   (Notion's `database_id`). Keep these strictly separate from the token.
5. **`output` is `None`**, and the CLI must know it (see
   [What the CLI does and does not pick up automatically](#adding-an-exporter-file-by-file)).
6. **`dry_run` makes zero writes.** Read the store and compute the plan,
   but issue no API write, so a dry run needs only a readable token and
   never mutates the target. Report the planned totals, and leave the
   run-only summary fields `None` so they drop out of the printed summary.

The worked reference is
[`particles/exporters/notion.py`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/exporters/notion.py);
the operator's side of the same credential is
[Operator guide → configuration → secrets](../operator-guide/configuration.md#secrets).

## Per-subject exporters: naming subjects

Two distinct Subjects can legitimately share a `canonical_name`
("Prometheus" the software and the Greek Titan). An exporter that names
files by `subject_slug(canonical_name)` alone collides them onto one file
and silently overwrites one with the other. The fix is shared, in
`particles.render.markdown`
([source](https://github.com/LinkedParticles/particles-core-py/blob/main/particles/render/markdown.py)):

- `build_subject_naming(subjects) -> SubjectNaming`: call it **once per
  export, over the full subject set**, before any `min_particles` or
  confidence filter drops a member (a collision must be detected even when
  one side won't render). `naming.display_name(subject)` returns the bare
  canonical name when it is unique and a qualified one
  (`"Prometheus (software)"`) when two or more share a base slug. The
  qualifier comes from the first tier that makes the whole group distinct:
  subject class, then description, then external id, then subject id.
  `naming.groups` lists the collision groups.
- `disambiguation_name(base)` returns the `"<base> (disambiguation)"` name
  for a group's disambiguation note.

The rules for a per-subject exporter:

- **Route every subject name through the display name**, never
  `canonical_name` directly: filenames, the page title, cross-note links,
  any index, the synthesis cache key, and the stale-link `known_names` set
  below.
- **On collision, emit one disambiguation note per group** linking to each
  qualified note. Where the format supports aliases (Obsidian's `aliases`
  frontmatter, Logseq's `alias::`), give that note an alias for the bare
  name so an existing `[[Prometheus]]` link still resolves.
- A per-particle exporter with no per-subject files (Anki, JSON Lines) is
  exempt.

Reusing `build_subject_naming` rather than re-deriving disambiguation is
what keeps paths consistent across the Obsidian, Wiki, and Logseq outputs
of the same store.

## Reaching across the seam

Exporters may call `store/` helpers directly for simple list / get
queries (the rule of thumb: the second time you copy a query, lift
it into `store/`). A one-off query in your exporter is fine; if the query
shape already appears in another exporter, move it into the matching
`store/` module as a public `async` helper and switch both call sites
over. The shared helpers already exposed:

- `get_particles_by_status(session, Status.ACTIVE)`: ACTIVE particles
  (`particles/store/particle_store.py`)
- `list_all_subjects(session)`: every Subject as a Pydantic model
  (`particles/store/subject_store.py`)
- `list_particle_subject_pairs(session)`: the full `(particle_id,
  subject_id)` join table (`particles/store/subject_store.py`)
- `get_entry_uri_map(session, entry_ids=None)`: `{entry_id: uri_r}` corpus
  URLs; `None` for every entry, or a set for only the entries you cite
  (`particles/corpus/store.py`)

Don't recreate these in your exporter.

## Synthesis-ready exporters

If your exporter wants per-Subject LLM-synthesised prose, import
from `particles.render.article_synthesis`
([source](https://github.com/LinkedParticles/particles-engine-py/tree/main/particles/render/article_synthesis))
rather than reimplementing. The helper provides the cache key, citation
validation, Layer-B judge, and the fallback structured-listing
render. The Obsidian, Wiki, and Logseq exporters all share this
machinery. The public surface:

| Symbol | What it does |
|---|---|
| `render_article(*, subject, particles, eff, input_hash, corpus_uris, max_tokens, layer_b_enabled, session=None, without_synthesis=False, …)` | The entry point. Tries LLM synthesis with a strict-prompt retry on citation failure, checks semantic alignment, and falls back to the structured listing. Returns `(body, used_synthesis)`. Pass `session` to use the shared synthesis cache; pass `without_synthesis=True` for a deterministic, LLM-free render. |
| `compute_input_hash(particles, subject=None, *, ordered=False)` | The cache key: particle `(id, status, confidence)` triples, the subject's identity (name, description, class, aliases, external ids) when given, and the prompt version. |
| `validate_citations(body, allowed_short_ids)` | Deterministic check that every citation in the body names an input particle; returns `(seen, invalid)`. |
| `layer_b_check(body, particles, *, max_tokens=1024, unrelated_tolerance=None)` | The per-sentence semantic-alignment judge; returns a `LayerBResult` whose `passed` is `True`, `False`, or `None` (judge not applicable). |
| `render_synthesised_article(...)` / `render_structured_listing(...)` | The two body renderers `render_article` chooses between. |
| `split_rendered_article(rendered)` | Decomposes a rendered article into `(frontmatter, h1_line, prose, references)` so you can splice the pieces into your own layout (Obsidian does this). |
| `invalidate_stale_link_articles(output_dir, known_names, *, hash_field="input_hash", recursive=False)` | Strips the cache hash from every cached article that links to a name no longer in `known_names`; see below. |
| `find_unresolved_wikilinks(output_dir, *, recursive=False)` | Read-only check that every `[[X]]` in an export resolves to a written page or alias. |

The helper is **filesystem-blind**: it returns Markdown bodies, and your
exporter writes them wherever its output shape dictates. It is also blind
to your file cache: it computes the hash, and your exporter decides where
to store it (a frontmatter field) and when a previously written article
with a matching hash can be reused.

### Cross-subject cache staleness

The cache key is per-subject. It covers this subject's particles and
identity, but **not the names of other subjects this article links to**.
When an operator renames a subject (for example with
`particles subjects fix-labels`), that subject's own article regenerates,
but every other cached article containing `[[Old Name]]` keeps a dead link
and its hash does not notice.

A synthesis-ready exporter must therefore accept the
`invalidate_stale_links` option (the `--invalidate-stale-links` flag the
Wiki, Obsidian, and Logseq exporters share). When it is set, before
rendering:

1. Build `known_names`: every subject's `canonical_name` and aliases,
   every display name from `build_subject_naming` **and its slug**, every
   `disambiguation_name(...)` **and its slug**, and the name of every
   other page you write (narrative pages, index pages). Leave one out and
   every note that links to it is spuriously invalidated on every run.
2. Call `invalidate_stale_link_articles(output_dir, known_names,
   hash_field=..., recursive=...)` with the frontmatter field your
   exporter stores its hash in (Wiki uses the default `input_hash`;
   Obsidian and Logseq use `article_input_hash`) and `recursive=True` if
   your layout nests directories.
3. Report the count in your summary
   (`stale_link_articles_invalidated: int | None`).

The next export then regenerates only the invalidated articles, which is
cheaper than a full regenerate when a few subjects were renamed.

The canonical contract is the protocol itself, in
[`particles/exporters/registry.py`](https://github.com/LinkedParticles/particles-engine-py/blob/main/particles/exporters/registry.py).
