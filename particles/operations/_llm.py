# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Back-compat alias: the circuit-breaker seam moved to :mod:`particles.llm.breaker`.

Extraction's second reading of a contradiction calls the seam from
below ``operations``, so the seam moved down into the Client-layer ``llm``
package. This module replaces itself in ``sys.modules`` with that one, so the
breaker's process-wide state (the trip deadline, the failure count) stays one
state: a caller or a test that reads, patches or resets
``particles.operations._llm`` is acting on :mod:`particles.llm.breaker`. The
explicit imports below are what static analysis sees.
"""

from __future__ import annotations

import sys

from particles.llm import breaker as _breaker
from particles.llm.breaker import (
    _is_account_level,
    _llm_call,
    _llm_call_many,
    _reset_llm_state,
    llm_circuit_open,
    llm_failure_count,
    llm_trip_count,
    llm_unavailable_cause,
    record_unusable_reply,
)

__all__ = [
    "_is_account_level",
    "_llm_call",
    "_llm_call_many",
    "_reset_llm_state",
    "llm_circuit_open",
    "llm_failure_count",
    "llm_trip_count",
    "llm_unavailable_cause",
    "record_unusable_reply",
]

sys.modules[__name__] = _breaker
