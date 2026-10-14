# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The trip-versus-move case and the update-gate pair sets.

Two suites over one package. ``particles benchmark trip-vs-move`` runs the
reader-proposed case end to end, one store per run in a child process with a
private environment, and scores each store with judge-free checks
(:mod:`.checks`, :mod:`.case`, :mod:`._child`). ``particles benchmark
update-gate`` replays the registered pair sets through the production probes
and scores three gates from the same samples (:mod:`.pairs`,
:mod:`.scoring`). Both are report-only: neither ever touches the operator's
store.
"""

from __future__ import annotations
