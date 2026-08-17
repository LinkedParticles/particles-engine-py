"""LongMemEval dataset acquisition, parsing, and subset selection.

**Download-on-demand, never vendored**: the ``s``-variant file is part of a
~3 GB dataset, so the loader fetches the pinned revision from the HuggingFace
*resolve* endpoint over the existing :func:`particles.http.particles_client`
(no ``huggingface_hub`` dependency), verifies its SHA-256 against the
compiled-in pin, and caches it under ``~/.particles/benchmark/longmemeval/``.
Bumping ``benchmark_memory.dataset_revision`` (or a file pin below) is a
deliberate diff.

Parsing follows the published LongMemEval format (Wu et al., ICLR 2025; the
``xiaowu0162/LongMemEval`` repository README): a JSON array of question
instances carrying ``question_id`` / ``question_type`` / ``question`` /
``answer`` / ``question_date`` / ``haystack_session_ids`` /
``haystack_dates`` / ``haystack_sessions`` (each a list of
``{"role", "content"}`` turns) / ``answer_session_ids``. Required fields
raise on absence (gold data must not silently degrade); *unknown* keys are
ignored — the dataset is upstream-owned, so forward-compat tolerance beats
the sibling harnesses' unknown-key strictness here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from particles.benchmark.memory.schema import MemoryQuestion, MemorySession, MemoryTurn
from particles.config import get_config
from particles.http import particles_client

log = logging.getLogger(__name__)

#: The HuggingFace dataset repo the ADR pins (v1 cleaned; owner-resolved
#: 2026-07-12 — the richer published-baseline record wins over v2).
DATASET_REPO = "xiaowu0162/longmemeval-cleaned"

#: Variant → filename inside the dataset repo, verified against the pinned
#: revision's file listing (2026-07-18): the cleaned release renames the s/m
#: files with a ``_cleaned`` suffix; oracle keeps the original name.
VARIANT_FILES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

#: Sentinel for a not-yet-finalized SHA-256 pin.
UNPINNED = "UNPINNED"

#: Compiled-in SHA-256 pins per variant file, finalized
#: 2026-07-18 against revision ``98d7416c…`` (the repo's only commit at pin
#: time). ``oracle`` (15 MB) and ``s`` (277 MB) were downloaded, digest-
#: computed locally, cross-checked against the commit's own git-LFS oids
#: (identical), and parse-validated (500 questions each, all six type
#: strata). ``m`` (2.7 GB) is pinned from the same commit's LFS oid without
#: a local download — the digest's provenance is the identical trust anchor,
#: and a future ``--variant m`` fetch verifies against it as usual.
EXPECTED_SHA256: dict[str, str] = {
    "oracle": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
    "s": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "m": "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f",
}


class MemoryDatasetLoadError(ValueError):
    """Raised when the LongMemEval dataset cannot be acquired or parsed."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _require(raw: dict[str, Any], key: str, path: str) -> Any:
    if key not in raw:
        raise MemoryDatasetLoadError(f"{path}: missing required field {key!r}")
    return raw[key]


def _parse_turn(raw: Any, path: str) -> MemoryTurn:
    if not isinstance(raw, dict):
        raise MemoryDatasetLoadError(f"{path}: turn must be a mapping")
    return MemoryTurn(
        role=str(_require(raw, "role", path)),
        content=str(_require(raw, "content", path)),
        has_answer=bool(raw.get("has_answer", False)),
    )


def parse_question(raw: dict[str, Any], *, path: str = "<dataset>") -> MemoryQuestion:
    """Parse one LongMemEval question instance into a :class:`MemoryQuestion`.

    ``haystack_sessions`` / ``haystack_session_ids`` / ``haystack_dates`` are
    the dataset's parallel lists; they are zipped into per-session records
    (dates may be shorter — a missing date parses to ``None``). Raises
    :class:`MemoryDatasetLoadError` on any structural problem.
    """
    if not isinstance(raw, dict):
        raise MemoryDatasetLoadError(f"{path}: question must be a mapping")
    question_id = str(_require(raw, "question_id", path))
    qpath = f"{path}[{question_id}]"

    sessions_raw = _require(raw, "haystack_sessions", qpath)
    session_ids_raw = _require(raw, "haystack_session_ids", qpath)
    if not isinstance(sessions_raw, list) or not isinstance(session_ids_raw, list):
        raise MemoryDatasetLoadError(f"{qpath}: haystack_sessions/ids must be lists")
    if len(sessions_raw) != len(session_ids_raw):
        raise MemoryDatasetLoadError(
            f"{qpath}: haystack_sessions ({len(sessions_raw)}) and haystack_session_ids "
            f"({len(session_ids_raw)}) differ in length"
        )
    dates_raw = raw.get("haystack_dates") or []
    if not isinstance(dates_raw, list):
        raise MemoryDatasetLoadError(f"{qpath}: haystack_dates must be a list")

    sessions: list[MemorySession] = []
    for i, (sid, turns_raw) in enumerate(zip(session_ids_raw, sessions_raw, strict=True)):
        if not isinstance(turns_raw, list):
            raise MemoryDatasetLoadError(f"{qpath}.haystack_sessions[{i}]: must be a turn list")
        date = str(dates_raw[i]) if i < len(dates_raw) and dates_raw[i] is not None else None
        turns = [
            _parse_turn(t, f"{qpath}.haystack_sessions[{i}][{j}]") for j, t in enumerate(turns_raw)
        ]
        sessions.append(MemorySession(session_id=str(sid), date=date, turns=turns))

    answer_ids_raw = raw.get("answer_session_ids", [])
    if not isinstance(answer_ids_raw, list):
        raise MemoryDatasetLoadError(f"{qpath}: answer_session_ids must be a list")

    answer_raw = raw.get("answer")
    return MemoryQuestion(
        question_id=question_id,
        question_type=str(_require(raw, "question_type", qpath)),
        question=str(_require(raw, "question", qpath)),
        answer=str(answer_raw) if answer_raw is not None else None,
        question_date=(str(raw["question_date"]) if raw.get("question_date") is not None else None),
        sessions=sessions,
        answer_session_ids=[str(s) for s in answer_ids_raw],
    )


def load_dataset_file(path: Path) -> list[MemoryQuestion]:
    """Read a LongMemEval-format JSON file into :class:`MemoryQuestion` records."""
    if not path.exists():
        raise MemoryDatasetLoadError(f"Dataset file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MemoryDatasetLoadError(f"{path.name}: JSON parse error: {exc}") from exc
    if not isinstance(raw, list):
        raise MemoryDatasetLoadError(f"{path.name}: top-level must be a JSON array")
    return [parse_question(q, path=path.name) for q in raw]


# ---------------------------------------------------------------------------
# Download-on-demand acquisition (never vendored)
# ---------------------------------------------------------------------------


def resolve_url(variant: str, revision: str) -> str:
    """The HuggingFace *resolve* URL for one variant file at a pinned revision."""
    try:
        filename = VARIANT_FILES[variant]
    except KeyError:
        raise MemoryDatasetLoadError(
            f"Unknown LongMemEval variant {variant!r}; expected one of {sorted(VARIANT_FILES)}"
        ) from None
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{revision}/{filename}"


def default_cache_dir() -> Path:
    """``~/.particles/benchmark/longmemeval/``."""
    return Path.home() / ".particles" / "benchmark" / "longmemeval"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def ensure_dataset(
    variant: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Return the local path of the variant file, downloading + verifying if absent.

    ``revision`` defaults to the pinned ``benchmark_memory.dataset_revision``
    (a HF commit sha); ``expected_sha256`` defaults to the compiled-in
    :data:`EXPECTED_SHA256` pin for the variant. An :data:`UNPINNED` pin
    refuses to download — verification-first, so a 3 GB fetch is never
    trusted on faith. The cache is keyed by revision, so a revision bump
    re-downloads rather than silently reusing the old file.
    """
    if revision is None:
        revision = get_config().benchmark_memory.dataset_revision
    if expected_sha256 is None:
        expected_sha256 = EXPECTED_SHA256.get(variant, UNPINNED)

    url = resolve_url(variant, revision)  # also validates the variant name
    target = (cache_dir or default_cache_dir()) / revision / VARIANT_FILES[variant]

    if target.exists():
        if expected_sha256 != UNPINNED:
            actual = _sha256_file(target)
            if actual != expected_sha256:
                raise MemoryDatasetLoadError(
                    f"Cached dataset {target} fails its SHA-256 pin "
                    f"(expected {expected_sha256}, got {actual}); delete the file "
                    f"and re-download, or fix the pin."
                )
        return target

    if expected_sha256 == UNPINNED:
        raise MemoryDatasetLoadError(
            f"The SHA-256 pin for LongMemEval variant {variant!r} is not finalized "
            f"(placeholder {UNPINNED!r} in particles.benchmark.memory.loader."
            f"EXPECTED_SHA256). Record the digest of the file at {url} before the "
            f"first download, or pass --dataset-file with a locally verified copy."
        )

    log.info("Downloading LongMemEval %s variant from %s", variant, url)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    async with particles_client() as client, client.stream("GET", url) as response:
        if response.status_code != 200:
            raise MemoryDatasetLoadError(
                f"Dataset download failed: HTTP {response.status_code} for {url} "
                f"(is benchmark_memory.dataset_revision a valid commit sha?)"
            )
        with tmp.open("wb") as fh:
            async for chunk in response.aiter_bytes():
                digest.update(chunk)
                fh.write(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise MemoryDatasetLoadError(
            f"Downloaded dataset fails its SHA-256 pin (expected {expected_sha256}, "
            f"got {actual}) — refusing to cache an unverified file."
        )
    tmp.replace(target)
    return target


# ---------------------------------------------------------------------------
# Stratified subset selection (§5/§6)
# ---------------------------------------------------------------------------


def select_questions(
    questions: list[MemoryQuestion],
    *,
    seed: int,
    limit: int | None,
    types: list[str] | None = None,
) -> list[MemoryQuestion]:
    """Deterministic question-type-stratified subset selection.

    Filters to ``types`` (when given), then — when ``limit`` cuts the set —
    allocates the limit across the question types proportionally (largest
    remainder) and samples each stratum with ``random.Random(seed)``, walking
    the strata in sorted-type order so the RNG stream is order-stable. Two
    calls with the same ``(questions, seed, limit, types)`` return the same
    subset (the reproducibility contract behind the published subset table).
    Result order is ``(question_type, question_id)`` for stable rendering.
    """
    pool = questions
    if types:
        wanted = set(types)
        pool = [q for q in pool if q.question_type in wanted]

    if limit is None or limit >= len(pool):
        return sorted(pool, key=lambda q: (q.question_type, q.question_id))

    groups: dict[str, list[MemoryQuestion]] = {}
    for q in pool:
        groups.setdefault(q.question_type, []).append(q)

    # Proportional allocation with largest remainder; every stratum with at
    # least one question keeps representation when the limit allows.
    total = len(pool)
    quotas: dict[str, float] = {t: limit * len(g) / total for t, g in groups.items()}
    counts: dict[str, int] = {t: int(quota) for t, quota in quotas.items()}
    remainder = limit - sum(counts.values())
    for t in sorted(groups, key=lambda t: (-(quotas[t] - counts[t]), t)):
        if remainder <= 0:
            break
        if counts[t] < len(groups[t]):
            counts[t] += 1
            remainder -= 1

    rng = random.Random(seed)
    selected: list[MemoryQuestion] = []
    for t in sorted(groups):
        k = min(counts.get(t, 0), len(groups[t]))
        if k:
            selected.extend(rng.sample(groups[t], k))
    return sorted(selected, key=lambda q: (q.question_type, q.question_id))
