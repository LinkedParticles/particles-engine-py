# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Held-out seeds and the freeze record of a milestone comparison.

Development seeds were tuned on, so a comparative number comes from a
held-out set the authors have not seen results on. The owner generates it
(:func:`new_held_out`), keeps it outside the repository, and commits only its
SHA-256 commitment, inside a **freeze record** (:class:`FreezeRecord`) that
also pins every version the run depends on and the pre-registered predictions.
A milestone run refuses unless the seeds it is handed match the commitment
and every pinned version still holds. The seeds are revealed with the
published results, so anyone can check the commitment.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime
from importlib.metadata import version

from pydantic import BaseModel, Field

from particles.benchmark.rot.answer import (
    ANSWER_STRIP_VERSION,
    PREDICTED_ANSWER_METRICS,
    READER_PROMPT_VERSION,
    REPLY_SCORER_VERSION,
    prompt_sha256,
)
from particles.benchmark.rot.door import DOOR_SEGMENTER_VERSION, METRIC_DIRECTIONS
from particles.benchmark.rot.generator import GENERATOR_VERSION
from particles.benchmark.rot.manifest import ArmsManifest
from particles.benchmark.rot.scoring import SCORER_VERSION

#: The predictions a freeze record may carry, per metric.
PREDICTIONS = ("beat", "trail", "tie")


class FreezeError(ValueError):
    """A held-out run whose seeds, versions or predictions do not match the freeze."""


class HeldOutSeeds(BaseModel):
    """The owner's private held-out set: never in a repository until revealed."""

    seeds: list[int]
    salt: str
    commitment: str


def commitment_for(seeds: list[int], salt: str) -> str:
    """SHA-256 over the seed list and the salt (order matters)."""
    blob = json.dumps({"salt": salt, "seeds": list(seeds)}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def new_held_out(count: int = 3) -> HeldOutSeeds:
    """Fresh random seeds and a salt, from the operating system's CSPRNG."""
    seeds = [secrets.randbelow(2**31 - 1) + 1 for _ in range(count)]
    salt = secrets.token_hex(16)
    return HeldOutSeeds(seeds=seeds, salt=salt, commitment=commitment_for(seeds, salt))


class FreezeRecord(BaseModel):
    """What a milestone run is pinned to, committed before any held-out build."""

    commitment: str
    created_at: str
    generator_version: int
    scorer_version: int
    door_segmenter_version: int
    particles_version: str
    adebench_sha: str | None = None
    manifest_sha256: str | None = None
    #: metric → ``beat`` / ``trail`` / ``tie`` for Particles' headline row
    #: against the median competitor.
    predictions: dict[str, str] = Field(default_factory=dict)
    #: The segmenter validation result the freeze rests on.
    segmenter_validation: dict[str, object] | None = None
    #: The answer stage's pins, set when the manifest names a
    #: reader: the reader table itself is pinned by ``manifest_sha256``.
    reader: dict[str, object] | None = None
    reader_prompt_version: int | None = None
    reader_prompt_sha256: str | None = None
    answer_strip_version: int | None = None
    reply_scorer_version: int | None = None
    #: The reply validation result the answer metrics rest on.
    reply_validation: dict[str, object] | None = None


def check_predictions(predictions: dict[str, str], *, answer: bool = False) -> None:
    """Every prediction names a known metric and a known outcome.

    With ``answer`` (a reader is pinned), every answer metric must be
    predicted too.

    Raises:
        FreezeError: Otherwise.
    """
    required = set(METRIC_DIRECTIONS) | (set(PREDICTED_ANSWER_METRICS) if answer else set())
    known = set(METRIC_DIRECTIONS) | set(PREDICTED_ANSWER_METRICS)
    for metric, outcome in predictions.items():
        if metric not in known:
            raise FreezeError(f"prediction for unknown metric {metric!r}")
        if outcome not in PREDICTIONS:
            raise FreezeError(f"prediction {outcome!r} for {metric} is not one of {PREDICTIONS}")
    missing = sorted(required - set(predictions))
    if missing:
        raise FreezeError(f"the freeze must predict every metric; missing: {', '.join(missing)}")


def build_freeze_record(
    held_out: HeldOutSeeds,
    *,
    manifest: ArmsManifest,
    predictions: dict[str, str],
    segmenter_validation: dict[str, object] | None,
    reply_validation: dict[str, object] | None = None,
) -> FreezeRecord:
    """The record to commit: the commitment, the current versions, the predictions.

    When the manifest names a reader, the answer stage's versions and the
    reply validation join the record and every answer metric must be
    predicted.

    Raises:
        FreezeError: When the seeds do not match their own commitment, or a
            prediction is malformed.
    """
    if commitment_for(held_out.seeds, held_out.salt) != held_out.commitment:
        raise FreezeError("the held-out file's seeds do not match its commitment")
    check_predictions(predictions, answer=manifest.reader is not None)
    record = FreezeRecord(
        commitment=held_out.commitment,
        created_at=datetime.now(UTC).isoformat(),
        generator_version=GENERATOR_VERSION,
        scorer_version=SCORER_VERSION,
        door_segmenter_version=DOOR_SEGMENTER_VERSION,
        particles_version=version("linkedparticles"),
        adebench_sha=manifest.adebench.sha if manifest.adebench else None,
        manifest_sha256=manifest.sha256,
        predictions=dict(predictions),
        segmenter_validation=segmenter_validation,
    )
    if manifest.reader is not None:
        record.reader = manifest.reader.model_dump(mode="json")
        record.reader_prompt_version = READER_PROMPT_VERSION
        record.reader_prompt_sha256 = prompt_sha256()
        record.answer_strip_version = ANSWER_STRIP_VERSION
        record.reply_scorer_version = REPLY_SCORER_VERSION
        record.reply_validation = reply_validation
    return record


def verify_against_freeze(
    held_out: HeldOutSeeds, record: FreezeRecord, *, manifest: ArmsManifest
) -> list[str]:
    """Refuse a milestone run that does not match its freeze; return notes otherwise.

    The seeds must match the committed hash, and the generator, scorer,
    segmenter, adebench pin and manifest must be the frozen ones, and so must
    the answer stage's versions when the freeze pinned a reader. The
    Particles version may differ (a later release may be measured on the same
    set), and that is returned as a note to print in the report; an engine
    change made after a held-out result was seen burns the set, which only the
    owner can know.

    Raises:
        FreezeError: On any mismatch above.
    """
    if commitment_for(held_out.seeds, held_out.salt) != record.commitment:
        raise FreezeError("the held-out seeds do not match the freeze record's commitment")
    checks: dict[str, tuple[object, object]] = {
        "generator_version": (record.generator_version, GENERATOR_VERSION),
        "scorer_version": (record.scorer_version, SCORER_VERSION),
        "door_segmenter_version": (record.door_segmenter_version, DOOR_SEGMENTER_VERSION),
        "adebench_sha": (
            record.adebench_sha,
            manifest.adebench.sha if manifest.adebench else None,
        ),
        "manifest_sha256": (record.manifest_sha256, manifest.sha256),
    }
    if record.reader_prompt_version is not None:
        checks |= {
            "reader_prompt_version": (record.reader_prompt_version, READER_PROMPT_VERSION),
            "reader_prompt_sha256": (record.reader_prompt_sha256, prompt_sha256()),
            "answer_strip_version": (record.answer_strip_version, ANSWER_STRIP_VERSION),
            "reply_scorer_version": (record.reply_scorer_version, REPLY_SCORER_VERSION),
        }
    for name, (frozen, now) in checks.items():
        if frozen != now:
            raise FreezeError(f"{name} is {now!r} now but {frozen!r} in the freeze record")
    notes: list[str] = []
    current = version("linkedparticles")
    if current != record.particles_version:
        notes.append(
            f"Particles {current} measured on a set frozen at {record.particles_version}; "
            "an engine change made after a held-out result was seen burns the set."
        )
    return notes
