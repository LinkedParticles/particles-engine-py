# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The no-egress posture report: what each configured destination will do.

``particles config validate`` and ``particles hook doctor`` both print from
:func:`posture`, so the two cannot disagree. It reads config and the local
Hugging Face cache and makes no network call. Kept apart from
:mod:`particles.egress` because it reads the LLM routing and the encoder, both
of which import that module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from particles.config import get_config
from particles.egress import (
    HUGGINGFACE_HOST,
    WIKIDATA_HOST,
    Posture,
    current_policy,
    entry_labels,
    host_port,
)
from particles.embeddings import EMBEDDING_MODEL_ID
from particles.llm.registry import LLMPurpose, purpose_endpoint

# ---------------------------------------------------------------------------

Severity = Literal["ok", "warn", "error"]


@dataclass(frozen=True)
class PostureItem:
    """One configured destination and whether the posture allows it."""

    kind: str
    name: str
    host: str
    port: int | None
    allowed: bool
    severity: Severity
    note: str

    def line(self) -> str:
        mark = {"ok": "✓", "warn": "⚠", "error": "✗"}[self.severity]
        where = self.host if self.port is None else f"{self.host}:{self.port}"
        return f"{mark} {self.kind} {self.name} → {where}: {self.note}"


@dataclass(frozen=True)
class EgressPosture:
    """Every configured destination with its verdict; both reports print from this."""

    posture: Posture
    allowed: tuple[str, ...]
    items: tuple[PostureItem, ...]

    @property
    def errors(self) -> tuple[PostureItem, ...]:
        return tuple(i for i in self.items if i.severity == "error")

    @property
    def warnings(self) -> tuple[PostureItem, ...]:
        return tuple(i for i in self.items if i.severity == "warn")

    def header(self) -> str:
        if self.posture == "open":
            return "network: egress=open (no check)"
        listed = ", ".join(self.allowed) if self.allowed else "nothing"
        return f"network: egress={self.posture} · allowed: {listed}"

    def summary(self) -> str:
        """One line for ``hook doctor``."""
        if self.posture == "open":
            return "open (no check)"
        refused = [i for i in self.items if not i.allowed]
        if not refused:
            return f"{self.posture} · every configured destination is allowed"
        names = ", ".join(f"{i.kind} {i.name}" for i in refused)
        return f"{self.posture} · refused: {names}"


def encoder_cached() -> bool:
    """Whether the embedding model's files are in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - a base dependency
        return False
    repo = (
        EMBEDDING_MODEL_ID
        if "/" in EMBEDDING_MODEL_ID
        else f"sentence-transformers/{EMBEDDING_MODEL_ID}"
    )
    found = try_to_load_from_cache(repo, "config.json")
    return isinstance(found, str)


def posture() -> EgressPosture:
    """List every configured destination with ``allowed`` or ``refused``.

    Reads config; makes no network call. The encoder's cache state is read from
    the local Hugging Face cache.
    """
    cfg = get_config()
    policy = current_policy()
    items: list[PostureItem] = []

    by_endpoint: dict[tuple[str, int | None, str], list[str]] = {}
    for purpose in get_args(LLMPurpose):
        host, port, provider = purpose_endpoint(purpose)
        by_endpoint.setdefault((host, port, provider), []).append(purpose)
    for (host, port, provider), purposes in by_endpoint.items():
        ok = policy.allows(host, port)
        notes = "allowed" if ok else "refused; these purposes are unavailable"
        if not ok and "semantic_lint" in purposes:
            notes += (
                "; agent-side near pairs are held for review and background "
                "contradiction detection is off"
            )
        items.append(
            PostureItem(
                "llm",
                f"{', '.join(purposes)} ({provider})",
                host,
                port,
                ok,
                "ok" if ok else "warn",
                notes,
            )
        )

    if cfg.engine.base_url:
        host, port = host_port(cfg.engine.base_url)
        ok = policy.allows(host, port)
        items.append(
            PostureItem(
                "engine",
                "engine.base_url",
                host,
                port,
                ok,
                "ok" if ok else "error",
                "allowed" if ok else "refused; every verb would fail",
            )
        )

    obs = cfg.observability
    if obs.enabled and obs.exporter == "otlp":
        host, port = host_port(obs.endpoint or "http://localhost:4318")
        ok = policy.allows(host, port)
        items.append(
            PostureItem(
                "otlp",
                "observability.endpoint",
                host,
                port,
                ok,
                "ok" if ok else "warn",
                "allowed" if ok else "refused; no OTLP exporter is installed",
            )
        )

    dsn = cfg.storage.database_url
    if dsn and not dsn.startswith("sqlite"):
        host, port = host_port(dsn.split("+", 1)[-1] if "://" not in dsn else dsn)
        if host:
            ok = policy.allows(host, port)
            items.append(
                PostureItem(
                    "store",
                    "storage.database_url",
                    host,
                    port,
                    ok,
                    "ok" if ok else "error",
                    "allowed" if ok else "refused; the store cannot be reached",
                )
            )

    wikidata = cfg.authorities.get("wikidata")
    if wikidata is None or wikidata.enabled:
        ok = policy.allows(*WIKIDATA_HOST)
        items.append(
            PostureItem(
                "authority",
                "wikidata",
                *WIKIDATA_HOST,
                ok,
                "ok" if ok else "warn",
                "allowed" if ok else "refused; subjects resolve locally",
            )
        )

    hub_ok = policy.allows(*HUGGINGFACE_HOST)
    cached = encoder_cached()
    if cached:
        note, severity = ("cached, loads offline", "ok") if policy.refusing else ("cached", "ok")
    elif hub_ok and not policy.refusing:
        note, severity = "not cached; downloads on first use", "ok"
    else:
        note, severity = "not cached; semantic features are disabled", "warn"
    items.append(
        PostureItem(
            "encoder",
            EMBEDDING_MODEL_ID,
            *HUGGINGFACE_HOST,
            hub_ok or cached,
            severity,  # type: ignore[arg-type]
            note,
        )
    )

    return EgressPosture(
        posture=policy.posture,
        allowed=entry_labels(policy),
        items=tuple(items),
    )
