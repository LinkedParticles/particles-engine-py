# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The door arm's arms manifest: which memories run, and exactly how.

A TOML file the operator keeps outside the repository during the internal
phase. It is embedded whole in every report's run tuple, so the configuration
of every row is published with the row. Shape::

    [adebench]
    clone = "/Users/jeff/bench/adebench"      # a clone pinned to `sha`
    sha = "6b23060"
    python = "/Users/jeff/bench/adebench/.venv/bin/python"

    [particles]
    parity_provider = "gemini"                # an llm.providers entry
    parity_model = "gemini-3-flash-preview"

    [memories.mem0]
    adapter = "adebench.mem0:Mem0Adapter"
    version = "mem0ai 2.0.1"
    model = "gemini-3-flash-preview"
    model_backed = true
    wait_s = 120
    usd_per_full_build = [0.3, 0.8]           # the estimate's figure, scaled by world size
    maintainer_review = "offered 2026-11-20, no reply"
    patched = false                           # true: a labelled "patched adapter" row only
    prepare = [["python", "-c", "..."]]       # run in the build's store dir first
    [memories.mem0.env]                       # keys only as {env:NAME}: a literal
    MEM0_STORE = "{store}"                    # value is published in every report
    GOOGLE_API_KEY = "{env:GEMINI_API_KEY}"
    [memories.mem0.server]                    # optional: a per-build server
    cmd = ["docker", "run", "--rm", "..."]
    cwd = "{store}"

    [reader]                                  # the answer stage's reader
    provider = "gemini"                       # an llm.providers entry, or "anthropic"
    model = "gemini-3-flash-preview"
    temperature = 0.0
    max_tokens = 200
    reasoning_effort = "minimal"              # must match the provider entry's setting
    [reader.sensitivity]                      # optional: the free local reader
    provider = "local"
    model = "qwen3:14b"
    digest = "sha256:..."                     # as `ollama list` reports it
    context_tokens = 32768                    # the host's OLLAMA_CONTEXT_LENGTH
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

#: Names the door arm reserves for its own rows.
RESERVED_NAMES = frozenset({"particles", "raw-chunks", "recent-mentions"})


class ServerSpec(BaseModel):
    """A per-build server: started on the store dir, stopped after the build."""

    cmd: list[str]
    cwd: str = "{store}"
    env: dict[str, str] = Field(default_factory=dict)


class MemorySpec(BaseModel):
    """One competitor memory's manifest entry."""

    adapter: str
    version: str = ""
    model: str | None = None
    model_backed: bool = False
    wait_s: float = 120.0
    usd_per_full_build: tuple[float, float] | None = None
    maintainer_review: str | None = None
    patched: bool = False
    env: dict[str, str] = Field(default_factory=dict)
    prepare: list[list[str]] = Field(default_factory=list)
    server: ServerSpec | None = None


class AdebenchSpec(BaseModel):
    """Where the pinned adebench clone is and which interpreter runs its adapters."""

    clone: str
    sha: str
    python: str


class ParticlesSpec(BaseModel):
    """Particles' variant rows: the model-parity provider and model."""

    parity_provider: str | None = None
    parity_model: str | None = None
    maintainer_review: str | None = None


class SensitivityReaderSpec(BaseModel):
    """The free local reader that reads the headline readings a second time."""

    provider: str
    model: str
    digest: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = Field(default=200, ge=1)
    reasoning_effort: str | None = None
    #: The local server's context, in tokens: a longer prompt is an overflow,
    #: never truncated.
    context_tokens: int = Field(ge=1)
    #: Characters per token the overflow check assumes; low, so a door near
    #: the limit counts as an overflow rather than being cut by the server.
    chars_per_token: float = Field(default=3.0, gt=0)


class ReaderSpec(BaseModel):
    """The answer stage's headline reader."""

    provider: str
    model: str
    digest: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = Field(default=200, ge=1)
    #: When given, must match the provider entry's ``reasoning_effort``, which
    #: is what the adapter sends.
    reasoning_effort: str | None = None
    sensitivity: SensitivityReaderSpec | None = None


class ArmsManifest(BaseModel):
    """The whole manifest; ``raw`` is what the report embeds."""

    adebench: AdebenchSpec | None = None
    particles: ParticlesSpec = Field(default_factory=ParticlesSpec)
    reader: ReaderSpec | None = None
    memories: dict[str, MemorySpec] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    sha256: str | None = None


class ManifestError(ValueError):
    """A manifest the door arm cannot run."""


def load_manifest(path: Path | None) -> ArmsManifest:
    """Read and check a manifest; ``None`` means Particles and the baselines only.

    Raises:
        ManifestError: On unreadable TOML, a reserved memory name, or a
            competitor with no adebench clone to run it from.
    """
    if path is None:
        return ArmsManifest()
    if not path.is_file():
        raise ManifestError(f"{path}: no such manifest")
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
        manifest = ArmsManifest.model_validate(
            {k: v for k, v in raw.items() if k in ("adebench", "particles", "memories", "reader")}
        )
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    manifest.raw = raw
    manifest.sha256 = hashlib.sha256(data).hexdigest()
    clash = RESERVED_NAMES & set(manifest.memories)
    if clash:
        raise ManifestError(f"{path}: memory names {sorted(clash)} are reserved")
    if manifest.memories and manifest.adebench is None:
        raise ManifestError(f"{path}: competitor memories need an [adebench] clone")
    return manifest
