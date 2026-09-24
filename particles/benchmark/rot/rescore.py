# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-score a saved memory-rot report under the current scorer.

The retrieval stage of a rot run — which particles came back for which probe,
in what order — is the expensive part of a ``live`` run and is fully recorded
in the report: every hit's text, rank, cosine, and badge, and every probe's
ground truth at its checkpoint. Classification is a pure function of those, so
a scorer change (:data:`~particles.benchmark.rot.scoring.SCORER_VERSION`) can be
applied to a saved report without re-running anything: no store, no encoder,
no LLM call. This is the ``rejudge`` pattern one harness over, and it
is the only way to isolate the scorer's effect — a re-run would also move the
retrieval being scored.

The output is a complete report of record: every hit re-classified, each
probe's first value-bearing hit recomputed, every metric family and breakdown
rebuilt, ``selection.scorer_version`` set to what ran, and a first quality note
naming the source and both scorer versions. Retrieval fields are copied as
recorded.
"""

from __future__ import annotations

from particles.benchmark.rot.schema import (
    HitClass,
    ProbeResult,
    RotBenchmarkReport,
    SlotState,
    WorldResult,
)
from particles.benchmark.rot.scoring import (
    SCORER_VERSION,
    breakdowns,
    classify,
    first_value_bearing,
    floor_sweep,
    merge_metrics,
    metrics_for,
)


class RescoreError(ValueError):
    """A report cannot be re-scored (hit texts were not recorded)."""


def _state_of(probe: ProbeResult) -> SlotState:
    """The probe's recorded ground truth, as the scorer takes it."""
    return SlotState(
        slot=probe.slot,
        current=probe.current,
        superseded=list(probe.superseded),
        poison=list(probe.poison),
        poison_channels=list(probe.poison_channels),
        last_phrasing=probe.last_phrasing,
    )


def _rescore_probe(probe: ProbeResult, scorer_version: int) -> ProbeResult:
    state = _state_of(probe)
    hits = [
        h.model_copy(
            update={
                "hit_class": (
                    HitClass.NONE
                    if probe.negative
                    else classify(h.text or "", state, scorer_version=scorer_version)
                )
            }
        )
        for h in probe.hits
    ]
    out = probe.model_copy(update={"hits": hits})
    out.first_hit, out.first_hit_contested = first_value_bearing(out)
    return out


def rescore_report(
    report: RotBenchmarkReport,
    *,
    source: str,
    scorer_version: int = SCORER_VERSION,
) -> RotBenchmarkReport:
    """Re-classify every recorded hit under ``scorer_version`` and rebuild the metrics.

    Raises:
        RescoreError: when any non-negative probe's hit carries no text — a
            report produced with ``benchmark.record_claim_text: false`` cannot
            be re-scored, and a partial re-score would silently mix versions.
    """
    missing = sum(
        1
        for w in report.worlds
        for p in w.probes
        if not p.negative
        for h in p.hits
        if h.text is None
    )
    if missing:
        raise RescoreError(
            f"{missing} hit(s) have no recorded text (benchmark.record_claim_text was "
            f"off); this report cannot be re-scored."
        )

    worlds: list[WorldResult] = []
    for w in report.worlds:
        probes = [_rescore_probe(p, scorer_version) for p in w.probes]
        by_cp, by_ph, by_ch = breakdowns(probes)
        worlds.append(
            w.model_copy(
                update={
                    "probes": probes,
                    "metrics": metrics_for(probes),
                    "by_checkpoint": by_cp,
                    "by_phrasing": by_ph,
                    "by_channel": by_ch,
                }
            )
        )

    all_probes = [p for w in worlds for p in w.probes]
    by_cp, by_ph, by_ch = breakdowns(all_probes)
    floors = [row.floor for row in report.floor_sweep]
    old = report.selection.scorer_version
    note = (
        f"Re-scored from {source}: scorer v{old} → v{scorer_version}. Retrieval is "
        f"as recorded (no store, encoder, or LLM call); only hit classes and the "
        f"metrics built from them changed."
    )
    return report.model_copy(
        update={
            "selection": report.selection.model_copy(update={"scorer_version": scorer_version}),
            "metrics": merge_metrics(w.metrics for w in worlds),
            "by_checkpoint": by_cp,
            "by_phrasing": by_ph,
            "by_channel": by_ch,
            "floor_sweep": floor_sweep(all_probes, floors),
            "worlds": worlds,
            "quality_notes": [note, *report.quality_notes],
        }
    )
