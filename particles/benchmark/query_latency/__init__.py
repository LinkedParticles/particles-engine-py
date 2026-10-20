# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Query latency of the semantic retrieval path, by stage and by store size.

Times ``retrieve_ranked`` (the selection half of a query, no answer generation
and no LLM call) over seeded synthetic stores from 10k to 1M ACTIVE rows, split
into the stages the path runs: setup, query embedding, SQL fetch, embedding
decode, model hydration, list filters, source lookup, support discounts, the
scoring loop, the sort, and the co-evidential collapse. Each cell runs in its
own process and writes its own result, so a run resumes after a reboot.
Report-only; it never touches a user store. CLI:
``particles benchmark query-latency``.

Deliberately re-exports nothing. A cold cell runs as
``python -m particles.benchmark.query_latency.cell``, which imports this package
first, and the cell reports how long importing the query path takes: anything
imported here would be loaded before that clock starts. Import the submodules.
"""
