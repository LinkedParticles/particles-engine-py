# Query latency: where the time goes, from 10k to 1M beliefs

A semantic query reads every ACTIVE belief that has an embedding, scores each
one against the question, and keeps the best `top_k`. There is no vector index.
Latency and memory therefore grow with the store, and a fair reader asks how
large a store can get before a query is slow. This page answers with numbers,
measured before any change to the query path and published whatever they
showed.

The run measures the selection half of a query (`retrieve_ranked`): it embeds
the question, loads the candidates, filters and scores them, sorts, and
collapses co-evidential duplicates. It does not generate an answer and makes no
LLM call, so the figures are the floor under every `particles query`, the MCP
`query` tool and the session digest.

## The answer, in four lines

On the benchmark host (an Apple M4 Mac mini with 16 GB), one unfiltered query:

- takes **5.7 s at 56k beliefs**, the size of the owner's own store, and grows
  linearly, at about 100 µs per belief, to **10.3 s at 100k**;
- takes **61 s at 300k**, worse than linear, because the 3.5 GB store no longer
  stays in the operating system's file cache;
- **did not finish at 1M**: it needs about 20 GiB of memory, and the run stopped
  it after 21 minutes of swapping on a 16 GB machine;
- spends about three quarters of its time **decoding embedding JSON and building
  Python models**, and about 1.4% on the cosine similarity the store is usually
  suspected of.

A query filtered by subject reads about ten rows, yet it also grows with the
store (0.13 s at 56k, 0.78 s at 300k, 2.8 s at 1M), because SQLite answers it
by walking every ACTIVE row.

These figures were measured before any change to the query path. They are the
baseline the next changes are measured against, and they are published as they
came out.

**Since then.** Release 1.192.23 made the changes this baseline pointed at. On
the same machine and stores, a warm unfiltered query takes 358 ms at 56k
(5.69 s before), 2.25 s at 300k (61 s) and 8.9 s at 1M, which now finishes; a
fresh process's first query costs what it did. The measurement is in
[After the candidate cache](#after-the-candidate-cache-2026-10-17-119223),
beside these figures, which are unchanged.

## Results (2026-10-17)

> **One run of the grid, on one machine.** Each row is one cell; warm rows pool
> 50 questions (40 at 300k, where the 40-minute cell budget ran out) and cold
> rows pool 5 fresh processes. p95 over 5 samples describes those 5, not a
> tail. Reproduction tuple: seed `20261017`; sizes 10k, 56k, 100k, 300k, 1M
> ACTIVE beliefs; harness commit `83dafe37`, released in 1.192.21 (its result
> files record the branch's earlier label, 1.192.16, a version number that
> release tagged on a different commit); Python 3.11.15, NumPy
> 2.4.6, SQLite 3.50.4, SQLAlchemy 2.1.4, Pydantic 2.14.1; macOS 27.0.1 on an
> Apple M4 (10 cores, 16 GB); encoder `all-MiniLM-L6-v2`; load
> average 1.9 / 1.6 / 1.5 at the start (Docker Desktop and the host's own
> services were running; the Ollama server, version 0.40.3, was idle with no
> model loaded). The run held the benchmark host's lock throughout, so no other
> measurement shared the machine. Raw JSON is linked from every row, one file
> per store size.

**Which figures travel.** The warm rows below 300k and every count (rows read,
bytes decoded, models hydrated, peak memory) are the portable result: they
depend on the code and the store, and on the hardware only through a per-row
cost. The cold rows, the 300k rows and the 1M result depend on the disk, the
file cache and the memory of this machine, and are labelled that way. The
reference run on a GitHub-hosted runner, below, is the second platform.

### The benchmark host

**Warm, unfiltered.** The figure most readers want: a long-running process (the
MCP server, the API) answering an ordinary semantic query.

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 50 | 993 ms | 1.05 s | 10,000 | 80 MiB | 872 MiB | ok | [json](query-latency-mini-2026-10-17-10k.json) |
| 56k | 50 | 5.69 s | 5.85 s | 56,000 | 450 MiB | 1.98 GiB | ok | [json](query-latency-mini-2026-10-17-56k.json) |
| 100k | 50 | 10.28 s | 10.69 s | 100,000 | 803 MiB | 3.04 GiB | ok | [json](query-latency-mini-2026-10-17-100k.json) |
| 300k | 40 | 60.57 s | 61.82 s | 300,000 | 2.35 GiB | 6.56 GiB | ok | [json](query-latency-mini-2026-10-17-300k.json) |
| 1M | 0 | — | — | 0 | — | — | skipped | [json](query-latency-mini-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 5.4 ms (1%) | 6.1 ms (1%) | 79 ms (10%) | 514 ms (52%) | 243 ms (25%) | 6.3 ms (4%) | 11 ms (1%) | 1.6 ms (0%) | 46 ms (5%) | 2.7 ms (0%) | 8.0 ms (1%) | 13 ms (1%) |
| 56k | 27 ms (0%) | 18 ms (0%) | 572 ms (10%) | 2.93 s (51%) | 1.48 s (26%) | 54 ms (2%) | 65 ms (1%) | 11 ms (0%) | 270 ms (6%) | 20 ms (0%) | 8.2 ms (0%) | 87 ms (2%) |
| 100k | 48 ms (0%) | 18 ms (0%) | 1.09 s (11%) | 5.26 s (51%) | 2.80 s (27%) | 120 ms (2%) | 120 ms (1%) | 19 ms (0%) | 495 ms (6%) | 36 ms (0%) | 8.3 ms (0%) | 161 ms (2%) |
| 300k | 4.37 s (7%) | 114 ms (0%) | 26.16 s (43%) | 16.34 s (27%) | 9.82 s (16%) | 526 ms (2%) | 419 ms (1%) | 57 ms (0%) | 1.60 s (3%) | 114 ms (0%) | 15 ms (0%) | 643 ms (1%) |

**Warm, subject-filtered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 50 | 28 ms | 72 ms | 18 | 0 MiB | 639 MiB | ok | [json](query-latency-mini-2026-10-17-10k.json) |
| 56k | 50 | 127 ms | 293 ms | 13 | 0 MiB | 683 MiB | ok | [json](query-latency-mini-2026-10-17-56k.json) |
| 100k | 50 | 237 ms | 375 ms | 12 | 0 MiB | 714 MiB | ok | [json](query-latency-mini-2026-10-17-100k.json) |
| 300k | 50 | 784 ms | 964 ms | 10 | 0 MiB | 712 MiB | ok | [json](query-latency-mini-2026-10-17-300k.json) |
| 1M | 50 | 2.79 s | 3.12 s | 16 | 0 MiB | 1.15 GiB | ok | [json](query-latency-mini-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 4.4 ms (13%) | 6.4 ms (19%) | 11 ms (34%) | 1.1 ms (13%) | 0.3 ms (4%) | 0.0 ms (0%) | 0.8 ms (3%) | 0.0 ms (0%) | 0.1 ms (1%) | 0.0 ms (0%) | 3.5 ms (12%) | 0.2 ms (1%) |
| 56k | 27 ms (18%) | 5.7 ms (4%) | 90 ms (61%) | 0.8 ms (8%) | 0.2 ms (5%) | 0.0 ms (0%) | 0.9 ms (1%) | 0.0 ms (0%) | 0.1 ms (1%) | 0.0 ms (0%) | 2.5 ms (3%) | 0.2 ms (0%) |
| 100k | 46 ms (18%) | 5.6 ms (2%) | 179 ms (70%) | 0.7 ms (5%) | 0.2 ms (2%) | 0.0 ms (0%) | 0.9 ms (1%) | 0.0 ms (0%) | 0.1 ms (0%) | 0.0 ms (0%) | 2.2 ms (1%) | 0.2 ms (0%) |
| 300k | 140 ms (17%) | 5.9 ms (1%) | 633 ms (79%) | 0.6 ms (2%) | 0.2 ms (1%) | 0.0 ms (0%) | 0.9 ms (0%) | 0.0 ms (0%) | 0.1 ms (0%) | 0.0 ms (0%) | 1.9 ms (0%) | 0.2 ms (0%) |
| 1M | 466 ms (16%) | 15 ms (1%) | 2.31 s (81%) | 1.0 ms (2%) | 0.3 ms (1%) | 0.0 ms (0%) | 1.0 ms (0%) | 0.0 ms (0%) | 0.1 ms (0%) | 0.0 ms (0%) | 3.8 ms (0%) | 0.2 ms (0%) |

**Cold, unfiltered.** One question per fresh process, five processes per row.
The encoder load (3.5 to 5.2 s) and the import of the query path (0.5 to 0.8 s)
are not in these figures. At 1M the first query never finished: the run
stopped it after 20 minutes 44 seconds of swapping, the remaining 1M
unfiltered cells were recorded as skipped, and the peak memory shown is the
resident part only, sampled before the stop.

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 5 | 1.14 s | 1.57 s | 10,000 | 80 MiB | 870 MiB | ok | [json](query-latency-mini-2026-10-17-10k.json) |
| 56k | 5 | 5.96 s | 6.10 s | 56,000 | 450 MiB | 1.96 GiB | ok | [json](query-latency-mini-2026-10-17-56k.json) |
| 100k | 5 | 10.72 s | 10.93 s | 100,000 | 803 MiB | 2.94 GiB | ok | [json](query-latency-mini-2026-10-17-100k.json) |
| 300k | 5 | 48.20 s | 49.29 s | 300,000 | 2.35 GiB | 6.65 GiB | ok | [json](query-latency-mini-2026-10-17-300k.json) |
| 1M | 0 | — | — | 0 | — | 5.40 GiB | memory_guard | [json](query-latency-mini-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 14 ms (1%) | 102 ms (15%) | 83 ms (7%) | 505 ms (41%) | 344 ms (28%) | 6.7 ms (1%) | 12 ms (1%) | 1.5 ms (0%) | 51 ms (4%) | 2.6 ms (0%) | 8.8 ms (1%) | 14 ms (1%) |
| 56k | 37 ms (1%) | 71 ms (1%) | 662 ms (12%) | 2.90 s (49%) | 1.44 s (24%) | 379 ms (6%) | 63 ms (1%) | 11 ms (0%) | 275 ms (5%) | 19 ms (0%) | 9.5 ms (0%) | 89 ms (1%) |
| 100k | 58 ms (1%) | 69 ms (1%) | 1.18 s (11%) | 5.25 s (49%) | 2.71 s (25%) | 629 ms (6%) | 117 ms (1%) | 19 ms (0%) | 492 ms (5%) | 34 ms (0%) | 10 ms (0%) | 159 ms (1%) |
| 300k | 4.49 s (8%) | 83 ms (0%) | 14.03 s (30%) | 16.38 s (34%) | 10.23 s (21%) | 373 ms (1%) | 416 ms (1%) | 56 ms (0%) | 1.57 s (3%) | 109 ms (0%) | 16 ms (0%) | 591 ms (1%) |

**Cold, subject-filtered.** The 1M row's p95 (23.9 s) is the first query to touch
the 11.6 GB store after the unfiltered attempt evicted it from the file cache.

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 5 | 115 ms | 117 ms | 25 | 0 MiB | 624 MiB | ok | [json](query-latency-mini-2026-10-17-10k.json) |
| 56k | 5 | 214 ms | 238 ms | 10 | 0 MiB | 630 MiB | ok | [json](query-latency-mini-2026-10-17-56k.json) |
| 100k | 5 | 319 ms | 325 ms | 4 | 0 MiB | 623 MiB | ok | [json](query-latency-mini-2026-10-17-100k.json) |
| 300k | 5 | 863 ms | 5.15 s | 12 | 0 MiB | 624 MiB | ok | [json](query-latency-mini-2026-10-17-300k.json) |
| 1M | 5 | 2.84 s | 23.89 s | 25 | 0 MiB | 639 MiB | ok | [json](query-latency-mini-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 13 ms (12%) | 71 ms (62%) | 14 ms (12%) | 1.7 ms (2%) | 0.4 ms (0%) | 0.0 ms (0%) | 1.7 ms (1%) | 0.0 ms (0%) | 5.8 ms (5%) | 0.0 ms (0%) | 5.6 ms (5%) | 0.3 ms (0%) |
| 56k | 38 ms (17%) | 70 ms (32%) | 93 ms (42%) | 1.0 ms (2%) | 0.2 ms (1%) | 0.0 ms (0%) | 1.7 ms (1%) | 0.0 ms (0%) | 6.0 ms (3%) | 0.0 ms (0%) | 2.9 ms (2%) | 0.3 ms (0%) |
| 100k | 58 ms (18%) | 69 ms (22%) | 182 ms (57%) | 0.6 ms (0%) | 0.1 ms (0%) | 0.0 ms (0%) | 1.6 ms (1%) | 0.0 ms (0%) | 6.0 ms (2%) | 0.0 ms (0%) | 1.5 ms (1%) | 0.3 ms (0%) |
| 300k | 154 ms (56%) | 68 ms (6%) | 626 ms (38%) | 1.0 ms (0%) | 0.3 ms (0%) | 0.0 ms (0%) | 1.8 ms (0%) | 0.0 ms (0%) | 6.0 ms (0%) | 0.0 ms (0%) | 4.3 ms (0%) | 0.3 ms (0%) |
| 1M | 474 ms (65%) | 67 ms (1%) | 2.28 s (34%) | 1.7 ms (0%) | 0.4 ms (0%) | 0.0 ms (0%) | 2.8 ms (0%) | 0.0 ms (0%) | 5.9 ms (0%) | 0.0 ms (0%) | 8.7 ms (0%) | 0.3 ms (0%) |

**Scaling.** The slope of `log p50` on `log N` is 1.0 for linear growth. The
per-row cost is the p50 divided by the rows the loader read; it is flat while
the store fits in the file cache and doubles at 300k.

| Mode, variant | sizes | slope | µs per row read |
|---|---|---|---|
| warm, unfiltered, 10k to 100k | 10k, 56k, 100k | 1.01 | 99, 102, 103 |
| warm, unfiltered, with 300k | 10k to 300k | 1.18 | 300k: 202 |
| cold, unfiltered, with 300k | 10k to 300k | 1.08 | 114, 106, 107, 161 |
| warm, subject | 10k to 1M | 1.01 | (about ten rows read; the cost is the scan, see below) |

### The reference run: a GitHub-hosted runner

> **One run, on a shared runner.** Commit `50a98fcb`, which measures with the
> same harness and the same query path as the host run (it adds only a fix to
> how the harness reads swap on Linux, and workflow files); Python 3.11.17,
> NumPy 2.4.6, SQLite 3.53.1, SQLAlchemy 2.1.4, Pydantic 2.14.1; Linux 6.17 on
> an AMD EPYC 9V74 with 2 vCPUs and 7.75 GiB; `--no-encoder`, so the query
> vector is a fixed seeded one and no model is loaded (the embed column is
> zero and the peak memory has no encoder in it); load average 1.0 / 0.7 / 0.3
> at the start. A shared runner holds no lock, and its neighbours are unknown.
> 1M was not attempted: the runner has 7.75 GiB of memory and 14 GB of disk.

**Warm, unfiltered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 50 | 2.17 s | 2.29 s | 10,000 | 80 MiB | 344 MiB | ok | [json](query-latency-ci-2026-10-18-10k.json) |
| 56k | 50 | 12.05 s | 12.54 s | 56,000 | 450 MiB | 1.36 GiB | ok | [json](query-latency-ci-2026-10-18-56k.json) |
| 100k | 50 | 21.19 s | 22.14 s | 100,000 | 803 MiB | 2.22 GiB | ok | [json](query-latency-ci-2026-10-18-100k.json) |
| 300k | 16 | 149.59 s | 170.73 s | 300,000 | 2.35 GiB | 6.56 GiB | ok | [json](query-latency-ci-2026-10-18-300k.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 10 ms (0%) | 0.0 ms (0%) | 192 ms (10%) | 901 ms (42%) | 621 ms (29%) | 18 ms (5%) | 33 ms (3%) | 5.4 ms (0%) | 104 ms (7%) | 9.1 ms (0%) | 19 ms (1%) | 35 ms (2%) |
| 56k | 47 ms (0%) | 0.0 ms (0%) | 1.51 s (12%) | 5.01 s (41%) | 4.12 s (33%) | 130 ms (3%) | 206 ms (2%) | 38 ms (0%) | 600 ms (6%) | 57 ms (0%) | 19 ms (0%) | 234 ms (2%) |
| 100k | 81 ms (0%) | 0.0 ms (0%) | 2.85 s (13%) | 8.94 s (42%) | 5.97 s (29%) | 1.30 s (6%) | 368 ms (2%) | 65 ms (0%) | 1.07 s (5%) | 100 ms (0%) | 19 ms (0%) | 420 ms (2%) |
| 300k | 12.16 s (9%) | 0.0 ms (0%) | 79.62 s (53%) | 27.06 s (18%) | 19.62 s (13%) | 806 ms (1%) | 1.40 s (1%) | 198 ms (0%) | 3.24 s (3%) | 298 ms (0%) | 35 ms (0%) | 1.41 s (1%) |

**Warm, subject-filtered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 50 | 50 ms | 136 ms | 18 | 0 MiB | 277 MiB | ok | [json](query-latency-ci-2026-10-18-10k.json) |
| 56k | 50 | 242 ms | 462 ms | 13 | 0 MiB | 292 MiB | ok | [json](query-latency-ci-2026-10-18-56k.json) |
| 100k | 50 | 430 ms | 734 ms | 12 | 0 MiB | 295 MiB | ok | [json](query-latency-ci-2026-10-18-100k.json) |
| 300k | 50 | 1.45 s | 2.42 s | 10 | 0 MiB | 315 MiB | ok | [json](query-latency-ci-2026-10-18-300k.json) |

The runner is about 2.1 times slower per belief than the M4 (215 µs against
100 µs), and the split is the same: decode 41 to 42%, hydration 29 to 33%, SQL
fetch 10 to 13%, scoring 5 to 7%, up to 100k. At 300k the runner's 7.75 GiB
cannot hold the file and the process together, and the SQL fetch becomes 53%
of a 150 s query, the same break the host shows at the same size, sooner and
harder. Every count (rows read, bytes decoded, models hydrated) is identical on
the two platforms, because it depends only on the store. The cold figures are
in the raw files.

## After the candidate cache (2026-10-17, 1.192.23)

The baseline above pointed at three changes. Release 1.192.23 makes them and
the two smaller ones, and this section measures the result with the same
command, the same seed and the same stores (the cached files of the baseline
run, reused). Nothing above this heading was changed by it.

**What changed.**

- **The candidates stay decoded between queries.** A process now keeps each
  store's ACTIVE candidates, their `Particle` models and decoded vectors, in
  memory, and on each query re-reads only the rows that changed since the last
  one. The store says which: a change journal, kept by SQLite triggers, records
  the id of every row any process inserts, updates or deletes, so a write by
  the session hooks or the CLI is seen by a running MCP server. The cache is
  checked against the journal on every query and returns the uncached loader's
  rows, vectors and order exactly; the test suite compares the two directly,
  and whole rankings with the cache on and off.
- **The similarity is one matrix pass** over all candidates instead of one call
  per candidate.
- **A subject-filtered query starts from the subject's links** instead of
  walking every ACTIVE row; the subject's existing index was enough once SQLite
  was kept off the status index.
- **The schema-version check is two index lookups** instead of a scan of the
  table.
- **The source lookup sends its ids in chunks** (released separately in
  1.192.22), so no store size makes a query fail.

**Why a cache rather than a new storage format.** Storing embeddings as
`float32` bytes would remove the JSON decode but not the model hydration, a
quarter of the baseline query, which only keeping the models avoids. It would
also rewrite every stored embedding in a migration and change a column that
every embedding reader reads, including an older install sharing the same
store. The cache needs no change to stored rows: its migration adds the journal
and an index, an older install neither reads nor writes the journal, and the
triggers record that install's writes as well. The as-of lens and the window
route read across statuses with predicates of their own, so they keep their own
loaders and are untouched; the cache covers only the ACTIVE set the ordinary
query reads.

### The benchmark host, after

> **One run of the grid, on the same machine.** Each row is one cell; warm rows
> pool 50 questions and cold rows 5 fresh processes, as above. Reproduction
> tuple: seed `20261017`; the baseline's stores (unchanged rows; the journal
> and the new index added in place); harness commit `60c01b64` (1.192.22 plus
> this change, released as 1.192.23); Python 3.11.15, NumPy 2.4.6, SQLite
> 3.50.4, SQLAlchemy 2.1.4, Pydantic 2.14.1; macOS 27.0.1 on the same Apple M4
> (10 cores, 16 GB); encoder `all-MiniLM-L6-v2`; load average 1.4 / 1.6 / 1.6
> at the start (the Ollama server, version 0.40.3, idle with no model loaded).
> The run held the benchmark host's lock throughout. Raw JSON is linked from
> every row.

**Before and after, p50.** The baseline's figures beside this run's, from the
two tables of record.

| Store | warm, unfiltered | warm, subject | cold, unfiltered | cold, subject | peak RSS, warm unfiltered |
|---|---|---|---|---|---|
| 10k | 993 ms → 68 ms | 28 ms → 13 ms | 1.14 s → 1.23 s | 115 ms → 102 ms | 872 MiB → 803 MiB |
| 56k | 5.69 s → 358 ms | 127 ms → 12 ms | 5.96 s → 6.02 s | 214 ms → 95 ms | 1.98 GiB → 1.37 GiB |
| 100k | 10.28 s → 660 ms | 237 ms → 12 ms | 10.72 s → 10.84 s | 319 ms → 93 ms | 3.04 GiB → 1.92 GiB |
| 300k | 60.57 s → 2.25 s | 784 ms → 12 ms | 48.20 s → 36.49 s | 863 ms → 101 ms | 6.56 GiB → 4.36 GiB |
| 1M | did not finish → 8.88 s | 2.79 s → 21 ms | did not finish → 373 s | 2.84 s → 116 ms | (about 20 GiB) → 6.96 GiB |

**Warm, unfiltered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 50 | 68 ms | 71 ms | 10,000 | 0 MiB | 803 MiB | ok | [json](query-latency-mini-after-2026-10-17-10k.json) |
| 56k | 50 | 358 ms | 368 ms | 56,000 | 0 MiB | 1.37 GiB | ok | [json](query-latency-mini-after-2026-10-17-56k.json) |
| 100k | 50 | 660 ms | 669 ms | 100,000 | 0 MiB | 1.92 GiB | ok | [json](query-latency-mini-after-2026-10-17-100k.json) |
| 300k | 50 | 2.25 s | 2.29 s | 300,000 | 0 MiB | 4.36 GiB | ok | [json](query-latency-mini-after-2026-10-17-300k.json) |
| 1M | 50 | 8.88 s | 9.38 s | 1,000,000 | 0 MiB | 6.96 GiB | ok | [json](query-latency-mini-after-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 1.2 ms (2%) | 6.2 ms (9%) | 0.4 ms (1%) | 1.2 ms (2%) | 0.0 ms (0%) | 4.4 ms (7%) | 11 ms (15%) | 1.8 ms (3%) | 31 ms (45%) | 2.6 ms (4%) | 7.8 ms (11%) | 1.3 ms (2%) |
| 56k | 1.3 ms (0%) | 5.7 ms (2%) | 0.4 ms (0%) | 6.8 ms (2%) | 0.0 ms (0%) | 46 ms (13%) | 72 ms (20%) | 13 ms (4%) | 184 ms (51%) | 18 ms (5%) | 7.8 ms (2%) | 5.1 ms (1%) |
| 100k | 1.3 ms (0%) | 6.1 ms (1%) | 0.4 ms (0%) | 12 ms (2%) | 0.0 ms (0%) | 101 ms (15%) | 127 ms (19%) | 23 ms (3%) | 338 ms (51%) | 34 ms (5%) | 7.9 ms (1%) | 8.9 ms (1%) |
| 300k | 1.4 ms (0%) | 16 ms (1%) | 0.5 ms (0%) | 61 ms (3%) | 0.0 ms (0%) | 453 ms (20%) | 405 ms (18%) | 69 ms (3%) | 1.11 s (49%) | 100 ms (4%) | 8.1 ms (0%) | 28 ms (1%) |
| 1M | 1.7 ms (0%) | 199 ms (2%) | 0.9 ms (0%) | 343 ms (4%) | 0.0 ms (0%) | 1.76 s (20%) | 1.57 s (18%) | 234 ms (3%) | 4.19 s (49%) | 351 ms (4%) | 8.8 ms (0%) | 103 ms (1%) |

**Warm, subject-filtered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 50 | 13 ms | 62 ms | 18 | 0 MiB | 642 MiB | ok | [json](query-latency-mini-after-2026-10-17-10k.json) |
| 56k | 50 | 12 ms | 162 ms | 13 | 0 MiB | 685 MiB | ok | [json](query-latency-mini-after-2026-10-17-56k.json) |
| 100k | 50 | 12 ms | 173 ms | 12 | 0 MiB | 725 MiB | ok | [json](query-latency-mini-after-2026-10-17-100k.json) |
| 300k | 50 | 12 ms | 271 ms | 10 | 0 MiB | 719 MiB | ok | [json](query-latency-mini-after-2026-10-17-300k.json) |
| 1M | 50 | 21 ms | 913 ms | 16 | 0 MiB | 1.17 GiB | ok | [json](query-latency-mini-after-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 1.1 ms (6%) | 5.3 ms (27%) | 0.8 ms (10%) | 1.0 ms (23%) | 0.2 ms (6%) | 0.0 ms (0%) | 0.6 ms (4%) | 0.0 ms (0%) | 0.1 ms (1%) | 0.0 ms (0%) | 3.5 ms (20%) | 0.5 ms (3%) |
| 56k | 1.1 ms (3%) | 5.2 ms (14%) | 0.7 ms (18%) | 0.8 ms (33%) | 0.2 ms (14%) | 0.0 ms (0%) | 0.7 ms (4%) | 0.0 ms (0%) | 0.1 ms (2%) | 0.0 ms (0%) | 2.3 ms (10%) | 0.4 ms (2%) |
| 100k | 1.1 ms (3%) | 5.3 ms (14%) | 0.8 ms (17%) | 0.7 ms (34%) | 0.2 ms (14%) | 0.0 ms (0%) | 0.6 ms (4%) | 0.0 ms (0%) | 0.1 ms (2%) | 0.0 ms (0%) | 2.1 ms (9%) | 0.4 ms (2%) |
| 300k | 1.0 ms (2%) | 5.3 ms (9%) | 1.5 ms (43%) | 0.7 ms (22%) | 0.1 ms (12%) | 0.0 ms (0%) | 0.6 ms (3%) | 0.0 ms (0%) | 0.1 ms (1%) | 0.0 ms (0%) | 1.6 ms (5%) | 0.3 ms (1%) |
| 1M | 1.1 ms (0%) | 5.5 ms (1%) | 5.7 ms (78%) | 0.9 ms (11%) | 0.2 ms (5%) | 0.0 ms (0%) | 0.7 ms (1%) | 0.0 ms (0%) | 0.1 ms (1%) | 0.0 ms (0%) | 2.9 ms (1%) | 0.5 ms (0%) |

**Cold, unfiltered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 5 | 1.23 s | 1.27 s | 10,000 | 80 MiB | 806 MiB | ok | [json](query-latency-mini-after-2026-10-17-10k.json) |
| 56k | 5 | 6.02 s | 10.24 s | 56,000 | 450 MiB | 1.36 GiB | ok | [json](query-latency-mini-after-2026-10-17-56k.json) |
| 100k | 5 | 10.84 s | 20.35 s | 100,000 | 803 MiB | 1.91 GiB | ok | [json](query-latency-mini-after-2026-10-17-100k.json) |
| 300k | 5 | 36.49 s | 51.52 s | 300,000 | 2.35 GiB | 4.39 GiB | ok | [json](query-latency-mini-after-2026-10-17-300k.json) |
| 1M | 5 | 373.26 s | 375.85 s | 1,000,000 | 7.84 GiB | 6.31 GiB | ok | [json](query-latency-mini-after-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 9.4 ms (1%) | 73 ms (6%) | 182 ms (15%) | 651 ms (53%) | 248 ms (20%) | 5.0 ms (0%) | 12 ms (1%) | 1.8 ms (0%) | 38 ms (3%) | 2.7 ms (0%) | 8.8 ms (1%) | 1.5 ms (0%) |
| 56k | 9.1 ms (0%) | 71 ms (1%) | 1.05 s (30%) | 3.38 s (48%) | 1.15 s (16%) | 48 ms (1%) | 71 ms (1%) | 13 ms (0%) | 199 ms (3%) | 18 ms (0%) | 8.6 ms (0%) | 5.4 ms (0%) |
| 100k | 9.3 ms (0%) | 69 ms (1%) | 1.47 s (29%) | 6.00 s (45%) | 2.65 s (20%) | 85 ms (1%) | 131 ms (1%) | 24 ms (0%) | 363 ms (3%) | 33 ms (0%) | 10 ms (0%) | 9.3 ms (0%) |
| 300k | 8.4 ms (0%) | 69 ms (0%) | 7.09 s (28%) | 17.87 s (44%) | 9.21 s (23%) | 262 ms (1%) | 415 ms (1%) | 70 ms (0%) | 1.14 s (3%) | 101 ms (0%) | 13 ms (0%) | 29 ms (0%) |
| 1M | 9.2 ms (0%) | 83 ms (0%) | 274.12 s (73%) | 61.17 s (17%) | 30.16 s (8%) | 1.12 s (0%) | 1.70 s (0%) | 237 ms (0%) | 4.65 s (1%) | 396 ms (0%) | 21 ms (0%) | 103 ms (0%) |

**Cold, subject-filtered.**

| Store | n | p50 | p95 | rows read (p50) | embedding JSON decoded (p50) | peak RSS | status | raw |
|---|---|---|---|---|---|---|---|---|
| 10k | 5 | 102 ms | 106 ms | 25 | 0 MiB | 627 MiB | ok | [json](query-latency-mini-after-2026-10-17-10k.json) |
| 56k | 5 | 95 ms | 125 ms | 10 | 0 MiB | 634 MiB | ok | [json](query-latency-mini-after-2026-10-17-56k.json) |
| 100k | 5 | 93 ms | 99 ms | 4 | 0 MiB | 626 MiB | ok | [json](query-latency-mini-after-2026-10-17-100k.json) |
| 300k | 5 | 101 ms | 126 ms | 12 | 0 MiB | 629 MiB | ok | [json](query-latency-mini-after-2026-10-17-300k.json) |
| 1M | 5 | 116 ms | 675 ms | 25 | 0 MiB | 642 MiB | ok | [json](query-latency-mini-after-2026-10-17-1M.json) |

| Store | setup | embed | SQL fetch | decode | hydrate | filters | sources | discounts | scoring | sort | collapse | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 8.4 ms (8%) | 73 ms (72%) | 3.3 ms (3%) | 2.0 ms (2%) | 0.5 ms (0%) | 0.0 ms (0%) | 1.4 ms (1%) | 0.0 ms (0%) | 6.5 ms (6%) | 0.0 ms (0%) | 5.6 ms (6%) | 0.8 ms (1%) |
| 56k | 8.9 ms (9%) | 71 ms (69%) | 3.0 ms (4%) | 1.1 ms (4%) | 0.2 ms (1%) | 0.0 ms (0%) | 1.3 ms (2%) | 0.0 ms (0%) | 5.8 ms (6%) | 0.0 ms (0%) | 3.1 ms (4%) | 0.5 ms (1%) |
| 100k | 7.9 ms (9%) | 69 ms (74%) | 5.2 ms (6%) | 0.8 ms (1%) | 0.1 ms (0%) | 0.0 ms (0%) | 1.4 ms (1%) | 0.0 ms (0%) | 5.6 ms (6%) | 0.0 ms (0%) | 1.8 ms (2%) | 0.3 ms (0%) |
| 300k | 10 ms (9%) | 71 ms (66%) | 7.0 ms (11%) | 1.2 ms (2%) | 0.3 ms (0%) | 0.0 ms (0%) | 1.4 ms (1%) | 0.0 ms (0%) | 6.4 ms (6%) | 0.0 ms (0%) | 3.3 ms (4%) | 0.5 ms (1%) |
| 1M | 10 ms (4%) | 71 ms (34%) | 20 ms (51%) | 1.8 ms (4%) | 0.4 ms (1%) | 0.0 ms (0%) | 1.4 ms (1%) | 0.0 ms (0%) | 6.1 ms (3%) | 0.0 ms (0%) | 5.6 ms (2%) | 0.7 ms (0%) |

**Scaling, after.** A warm query now costs 6.4 to 8.9 µs per candidate against
about 100 µs before, and the slope of `log p50` on `log N` is 1.06 from 10k to
1M: it still grows linearly with the store, with a smaller constant. A warm
subject-filtered query no longer grows with the store at all (12 ms from 56k
to 300k, 21 ms at 1M).

### Reading the after numbers

**A long-running process gets the gain; a fresh one does not.** The warm rows
are a process that has answered a query before: the MCP server, the API. Its
first unfiltered query builds the cache, which costs what every query used to
cost (the warm cells' untimed first query took 1.2 s at 10k, 6.0 s at 56k,
10.8 s at 100k, 34 s at 300k and 386 s at 1M), and every later one re-reads
only what changed. A cold row is a fresh process answering one question, which
is what a one-shot `particles query` on the command line does, and it still
reads and decodes every candidate: its figures are within a few percent of the
baseline's up to 100k. The cold 300k query is faster (36 s against 48 s)
because the build streams its rows and holds less memory at once, which leaves
more of the 3.5 GB store in the file cache. A cold subject-filtered query
builds no cache and is fast at every size.

**Where a warm query's time goes now.** At 56k, the 358 ms are the
per-candidate scoring loop (trust, decay, effective confidence) at 184 ms
(51%), the source lookup at 72 ms (20%), the list filters at 46 ms (13%), and
the sort at 18 ms (5%). The decode stage (7 ms) is now the selection of the
cached rows, and hydration is zero. The similarity itself is a small part of
the scoring stage: stacking 56k vectors into a matrix takes 19 ms and the
product 1.3 ms (the reference timing in the raw JSON). The remaining cost is
Python work done once per candidate, which is what a further change would have
to reduce; a vector index still has no measured stage to remove.

**Memory.** A warm process now holds the decoded candidates for its lifetime:
1.37 GiB at 56k and 6.96 GiB at 1M, peak, including the encoder. That is less
than the baseline's peak at every size (1.98 GiB at 56k), because the baseline
built the whole candidate set again on every query, JSON text and ORM rows
included, and the cache is built in partitions. It is also why 1M now finishes
on a 16 GB machine.

**The 1M cold query reads its store from disk.** Of a cold 1M query's 373 s,
274 s is the SQL fetch: an 11.6 GB store does not fit in the file cache beside
the process. That figure is this machine's, as the 300k figures were before.

**A first attempt, measured and fixed.** The first after-run, on commit
`2b518640`, stopped at 100k when its 56k warm figures came out bimodal: most
queries took about 0.75 s and the rest about 0.35 s, a median of 754 ms ([10k](query-latency-mini-after-nofreeze-2026-10-17-10k.json),
[56k](query-latency-mini-after-nofreeze-2026-10-17-56k.json),
[100k, cold only](query-latency-mini-after-nofreeze-2026-10-17-100k.json)).
The extra 0.4 s was Python's cyclic garbage collector walking the cache's
1.4 million tracked objects on most queries. The run of record ends a large
build by exempting those objects from the collector (`gc.freeze`; reference
counting still frees them, which is how a replaced cache's rows go), set by
`query.candidate_cache_freeze_rows`. It is a process-wide setting, which an
application embedding the SDK may want to turn off.

**Not repeated.** The reference run on a GitHub-hosted runner has not been
repeated after the change, so the after figures come from one platform.

## What it does

`particles benchmark query-latency run` builds one synthetic store per size,
then times queries against it in separate processes.

**The stores.** Scoring cost does not depend on what the text says, so each
store holds random unit vectors, which lets the run reach sizes no real store
has yet. What does affect cost is the shape of each row and of the tables the
query joins, so the generator copies the aggregate shape of the owner's store
(counts and column sizes only, measured 2026-10-17 over 56,196 ACTIVE rows):

| Property | Synthetic store | Owner's store |
|---|---|---|
| Embedding JSON per row | about 8.4 KB, the production writer's own format | 8,430 bytes on average |
| Subjects per belief | 5% none, 68% one, 22% two, the rest three or more | the same distribution |
| Beliefs per subject | Zipf, exponent 0.8: median 3, p90 9, p99 about 50 at 56k | median 2, p90 9, p99 52 |
| Corpus entries, snapshots | one per 82 and one per 32 beliefs | 686 and 1,755 |
| Rows that are not ACTIVE | 20% more rows (stale, superseded, retracted) | 11,458 |
| Beliefs with properties, with a structured claim | 21.6%, 36.7% | the same |
| Co-evidential pairs | 1.3% of beliefs | 724 relations |

Every row is a real `Particle` serialised by the same function the write path
uses, stamped with the current embedding model so the loader does not skip it.
A `(seed, size)` pair is the fixture: two builds are identical, and the store's
manifest records a hash of every id to prove it.

**The stages.** The harness wraps the functions `retrieve_ranked` already
calls, so nothing in the query path was changed to measure it:

| Stage | What it covers |
|---|---|
| setup | the schema-version guard, the extractor trust cache, the trust and decay policies |
| embed | encoding the question (the encoder load is reported apart) |
| SQL fetch | the loader's one `SELECT`, including building the ORM row objects |
| decode | `json.loads` of each row's embedding and the `float32` array built from it |
| hydrate | turning each row into a `Particle` model |
| filters | the list filters over the candidates (document-meta, polarity, stance) |
| sources | the batch lookup of each candidate's source snapshot and entry |
| discounts | the stale-support discount lookup |
| scoring | the per-candidate loop: trust, decay, effective confidence, cosine |
| sort | the sort of the scored list and the cut to `top_k` |
| collapse | the co-evidential collapse of the top `k` |
| other | the residual, so the stages always sum to the total |

**The cells.** Each store size runs four ways:

- **cold**: a fresh Python process answering one question, five times over,
  with the import of the query path and the encoder load timed separately. The
  operating system's file cache is not purged (that needs root), so cold means
  a cold process, not a cold disk;
- **warm**: one process answering 50 questions after one untimed warm-up
  query, one database session per question, as the query verbs open one;
- **unfiltered**: no filter, so every ACTIVE belief is a candidate;
- **subject**: `subject_id` set, the one filter the loader applies in SQL. The
  subject for each question is the first subject of a random belief, so a
  popular subject is asked about as often as it is used.

`top_k` is the default, 40. The encoder is the default `all-MiniLM-L6-v2`. The
cells load an empty configuration, so every setting is the compiled default.

**What is counted, apart from time.** For every query: rows the SQL returned,
characters of embedding JSON decoded, models hydrated, candidates scored, and
the size of the source lookup. For every cell, the process's peak resident
memory. These counts do not depend on the hardware, and they are what makes
the timings portable: a reader can multiply them by their own per-row costs.

Each cell runs in its own process and writes its own result file when it
finishes. A run interrupted by a reboot or a lost connection resumes where it
stopped, losing only the cell in flight. The run watches each cell and stops it
at a two-hour timeout, or when system swap grows by more than 8 GiB, or when
free disk falls under 4 GiB; a stopped cell is a result, and the remaining
cells of that size and variant are recorded as skipped.

## Reading the numbers

**Where the time goes.** At 56k, warm and unfiltered, a 5.7 s query is 2.9 s of
embedding decode (51%), 1.5 s of model hydration (26%), 0.6 s of SQL fetch
(10%), 0.3 s of scoring (6%), and well under a tenth of a second for every
other stage. The split is the same at 10k and 100k to within two points, which
is what linear growth looks like from the inside. Decode is the cost of
storing each embedding as JSON text: about 8.4 KB of digits per belief,
`json.loads` into 384 Python floats, then a `float32` array. Hydration is one
Pydantic `Particle` per row, built for every candidate although only `top_k`
(40) are returned.

**The cosine is not what hurts first.** The scoring stage is the whole
per-candidate loop (trust, decay, effective confidence and the similarity).
The similarity alone, timed apart over the same vectors, is 81 ms at 56k when
called once per candidate as the loop calls it, and 5.8 ms as one matrix
product (plus 21 ms to stack the vectors into a matrix). The arithmetic is
about 1.4% of the query. A vector index would remove that 1.4% and leave the
decode and the hydration of every candidate in place, so on this evidence an
index alone would not make the query fast; whether it would after the cheap
changes below is for the re-measurement to show.

**Memory.** Peak memory grows by about 21 KB per ACTIVE belief, on top of
roughly 0.9 GB for the interpreter and the encoder: 2.0 GB at 56k, 3.0 GB at
100k, 6.6 GB at 300k (the reference runner, with no encoder loaded, grows by
the same amount per belief). At that rate 1M needs about 20 GiB, which is why the 1M
query never finished on a 16 GB machine: the process swapped for 21 minutes at
under 5% CPU before the run's swap guard stopped it. The cause is that every
candidate is held as a decoded array and a model at once.

**Where 300k breaks the line.** Decode and hydration stay close to linear at
300k (16.3 s and 9.8 s, 3.1 and 3.5 times their 100k values). The two stages that read the file
do not: the SQL fetch is 26 s where linear growth predicts about 3.3 s, and
setup is 4.4 s where it predicts 0.15 s. Setup includes the schema-version
guard, which counts the ACTIVE rows by schema version on every query and so
reads the whole table too. A 3.5 GB store plus a 6.6 GB process does not fit in
this machine's file cache, so each query reads from the SSD. That part of the
figure is the machine's, and a host with more memory will show less of it.

**The subject filter does not narrow the read.** With `subject_id` set, the
loader returns about ten rows, yet the SQL fetch grows with the store: 11 ms at
10k, 90 ms at 56k, 179 ms at 100k. SQLite's plan for the loader's query, read
with `EXPLAIN QUERY PLAN` against the owner's own store as well as the
synthetic ones, searches `ix_particles_status_confidence` for every ACTIVE row
and probes `particle_subjects` once per row, rather than starting from the
subject's handful of links. The filter therefore saves the decode and the
hydration, which is why a filtered query is 45 times faster at 56k, but not the
scan: it grows linearly (slope 1.01) to 2.3 s of SQL fetch at 1M. The
schema-version guard in setup adds another O(N) read (0.47 s at 1M).

**Cold is the import and the encoder, not the query.** A fresh process spends
about 0.5 s importing the query path and 3.5 s loading the encoder (which, run
this way, also revalidates the model against the Hugging Face Hub). Its first
query then costs within 5% of a warm one at 56k and 100k (15% at 10k, where
the first encode and the first models are a larger share), because the file
cache was already warm (purging it needs root).

**A limit found on the way.** The source lookup sends the snapshot ids of every
candidate in one `IN` list, and SQLite refuses a statement with more than
32,766 variables (`too many SQL variables`, checked directly against the
lookup). At the owner's ratio, one source snapshot per 32 beliefs, that is
about 1.05M beliefs: past it an unfiltered query fails instead of slowing down.
A store whose sources each yield fewer beliefs reaches it sooner. No size in
this grid crossed it (1M sends 31,250). Release 1.192.22 removes the limit:
the lookups now send their ids in chunks of 500.

**What this measurement does not cover.** The stores are freshly written, so
their pages are in insertion order, where a store that has lived for months is
fragmented. The vectors are random, so the similarity values (and which beliefs
rank first) mean nothing; only the cost does. Every other query feature (tags,
`as_of`, windows, the project observer, structured claims) is off. The answer
step, when a query has one, adds a model call on top.

## What comes next

The baseline points at three changes, none of them made here: store embeddings
as `float32` bytes or keep a matrix in memory, so decoding stops being per row;
score in one vectorised step; and build models only for the `top_k` survivors.
The subject filter's plan and the `IN`-list limit are separate, smaller fixes.
Each change is a separate decision, and each is re-measured with the same
command and the same seed, with the before and after figures on this page. The
vector index (deferred until a measured stage says the similarity is the cost)
waits for that re-measurement.

Release 1.192.23 made the first three together, with the two smaller fixes;
the re-measurement is
[above](#after-the-candidate-cache-2026-10-17-119223). It finds the remaining
warm cost in per-candidate Python work rather than in the similarity, and
leaves one-shot processes as slow as before, so the vector index still waits.


## Reproducing

```bash
# The run of record (the encoder runs; about two hours on an M4 mini):
particles benchmark query-latency run --work-dir ~/bench/query-latency/work

# The hardware-reference run: a fixed seeded query vector, no model loaded.
particles benchmark query-latency run --no-encoder --sizes 10k,56k,100k,300k

# The after-measurement: a new work directory whose stores/ is a link to the
# baseline run's, so the same store files are reused.
mkdir -p ~/bench/query-latency/after
ln -s ~/bench/query-latency/work/stores ~/bench/query-latency/after/stores
particles benchmark query-latency run --work-dir ~/bench/query-latency/after

# The tables on this page, and one raw JSON per store size:
particles benchmark query-latency report --work-dir ~/bench/query-latency/work \
  --export-dir out --prefix query-latency-mini-2026-10-17 \
  --raw-link "query-latency-mini-2026-10-17-{size}.json"
```

The seed (`20261017`) and the sizes are the defaults, so the commands above
rebuild the same stores. A re-run with the same `--work-dir` skips every cell
already finished; `--retry-failed` re-runs the ones that stopped.
