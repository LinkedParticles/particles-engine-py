# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The scripted arm: the case end to end with every model answer read from a table.

The arm runs the same child, the same verbs and the same checks as the live
arm, with extraction and every probe answered from ``scripted_table.json``
through :func:`~particles.llm.override_providers`, opened in the child.
It costs nothing and needs no key. It proves the plumbing,
the isolation, the checks, and that the pipeline turns these verdicts into the
expected store. It proves nothing about any model's reading.

**The table is keyed on the data a request carries, never on its
instructions.** Extraction is keyed on the SHA-256 of the text inside the
request's ``source`` fence; a probe on the two claim texts every probe prompt
ends with (``Claim A:`` / ``Claim B:``, read from inside their fences), and on
whether the system turn asks for the update probe's ``SLOT`` line. A change to
a prompt's instructions therefore changes nothing here. A change to the fence,
to that claim-pair tail, to a reply grammar, or to how the notes are chunked
breaks the arm, and the unit tier catches it: a request the table does not
hold is refused, counted, and logged with its key, and the test requires none.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from particles.llm import CompletionError, VisionImage

#: The table the arm reads, beside this module (it ships in the wheel).
TABLE_PATH = Path(__file__).with_name("scripted_table.json")

_TOKEN = re.compile(r"[a-z0-9']+")
#: Every data fence in a request: ``<label nonce="…">`` text ``</label nonce="…">``.
_FENCES = re.compile(
    r'<(?P<tag>[a-z][a-z0-9_]*) nonce="(?P<n>[0-9a-f]+)">\n'
    r'(?P<text>.*?)\n</(?P=tag) nonce="(?P=n)">',
    re.DOTALL,
)
#: A re-anchor candidate's label (``c1``, ``c2``, …).
_CANDIDATE_LABEL = re.compile(r"\Ac\d+\Z")
_CLAIMS = re.compile(r"Claim A:\s*(?P<a>.*?)\n\s*\nClaim B:\s*(?P<b>.*)\Z", re.DOTALL)
#: A claim the probe fenced: ``<claim_a nonce="…">`` … ``</claim_a nonce="…">``.
_CLAIM_FENCE = re.compile(
    r'\A<(?P<tag>claim_[ab]) nonce="(?P<n>[0-9a-f]+)">\n(?P<text>.*)\n</(?P=tag) nonce="(?P=n)">\Z',
    re.DOTALL,
)
#: The update probe asks for this line (``particles.ingest.pipeline.UPDATE_SLOT_REQUEST``).
_UPDATE_REQUEST = "SLOT: CHANGES or SLOT: FIXED"


class BagOfWordsEncoder:
    """Deterministic stand-in encoder: hashed bag of tokens, L2-normalised.

    The scripted arm's child installs it, so the arm downloads no model. A text
    the table gives features for is encoded from those features; any other text
    from its own words, as the rot and observer unit tiers' stand-in does.
    Embeddings are perception, scripted like the rest: the candidate search
    and its floors, which are decision logic, still run for real.
    """

    def __init__(self, features: dict[str, list[str]] | None = None, dim: int = 384) -> None:
        self._features = features or {}
        self._dim = dim

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """Encode ``texts`` as hashed, L2-normalised token counts."""
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._features.get(text.strip()) or _TOKEN.findall(text.lower())
            for tok in tokens:
                h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self._dim] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm:
                out[i] /= norm
        return out


def unfence_claim(text: str) -> str:
    """A probe's claim text with its data fence, if any, removed."""
    match = _CLAIM_FENCE.match(text.strip())
    return (match["text"] if match else text).strip()


def source_key(text: str) -> str:
    """The extraction key: SHA-256 of the fenced source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ScriptedTable:
    """The answers the arm gives, keyed on request data."""

    #: Fenced-source SHA-256 → the candidate array the extractor returns.
    extraction: dict[str, list[dict[str, Any]]]
    #: ``(claim A, claim B)`` → ``YES`` / ``NO`` for the contradiction probe.
    contradiction: dict[tuple[str, str], str]
    #: ``(claim A, claim B)`` → ``CHANGES`` / ``FIXED`` / ``DIFFERENT`` for the update probe.
    update: dict[tuple[str, str], str]
    #: Claim text → the feature tokens the stand-in encoder embeds it from.
    embedding: dict[str, list[str]] = field(default_factory=dict)
    #: ``(retired claim, candidate claim)`` → ``(HOLDS|DEPENDS, restatement)``
    #: for the re-anchor probe. Its second reading confirms exactly the
    #: restatement written here, and nothing else.
    reanchor: dict[tuple[str, str], tuple[str, str | None]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = TABLE_PATH) -> ScriptedTable:
        """Read the table file."""
        data = json.loads(path.read_text())
        return cls(
            extraction={e["sha256"]: e["candidates"] for e in data.get("extraction", [])},
            contradiction={(p["a"], p["b"]): p["verdict"] for p in data.get("contradiction", [])},
            update={(p["a"], p["b"]): p["verdict"] for p in data.get("update", [])},
            embedding={e["text"]: e["features"] for e in data.get("embedding", [])},
            reanchor={
                (e["retired"], e["claim"]): (e["verdict"], e.get("restatement"))
                for e in data.get("reanchor", [])
            },
        )


@dataclass
class RequestLog:
    """Every request the scripted providers saw, answered or not."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def misses(self) -> list[dict[str, Any]]:
        """The requests no table entry answered."""
        return [e for e in self.entries if not e["answered"]]

    def refused_by_purpose(self) -> dict[str, int]:
        """Refused requests counted by purpose, as the report records them."""
        counts: dict[str, int] = {}
        for e in self.misses():
            counts[e["purpose"]] = counts.get(e["purpose"], 0) + 1
        return counts


def _reply_for_probe(kind: str, verdict: str) -> str:
    if kind == "update":
        if verdict == "DIFFERENT":
            return "REASON: scripted\nSLOT: NONE\nVERDICT: NO"
        return f"REASON: scripted\nSLOT: {verdict}\nVERDICT: YES"
    return f"REASON: scripted\nVERDICT: {verdict}"


class ScriptedProvider:
    """Answer one purpose's requests from the table; refuse and log the rest."""

    def __init__(self, purpose: str, table: ScriptedTable, log: RequestLog) -> None:
        self._purpose = purpose
        self._table = table
        self._log = log

    @property
    def provider_model(self) -> str:
        """Disclosure key: what answered, recorded on usage rows."""
        return f"scripted:{self._purpose}"

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        images: Sequence[VisionImage] | None = None,
        response_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        **opts: object,
    ) -> str:
        """Answer from the table, or refuse, counting the miss under its key."""
        fences = {m["tag"]: m["text"] for m in _FENCES.finditer(prompt)}
        if "retired" in fences and "restatement" in fences:
            return self._reading(fences)
        if "retired" in fences:
            return self._reanchor(fences)
        if "source" in fences:
            key = source_key(fences["source"])
            candidates = self._table.extraction.get(key)
            self._record("extraction", {"sha256": key}, candidates is not None, fences["source"])
            if candidates is None:
                raise CompletionError(f"scripted arm: no extraction entry for source {key}")
            return json.dumps(candidates)
        claims = _CLAIMS.search(prompt)
        if claims is not None:
            kind = "update" if _UPDATE_REQUEST in (system or "") else "contradiction"
            pair = (unfence_claim(claims["a"]), unfence_claim(claims["b"]))
            table = self._table.update if kind == "update" else self._table.contradiction
            verdict = table.get(pair)
            self._record(kind, {"a": pair[0], "b": pair[1]}, verdict is not None, None)
            if verdict is None:
                raise CompletionError(f"scripted arm: no {kind} entry for this pair")
            return _reply_for_probe(kind, verdict)
        self._record("other", {"prompt_sha256": source_key(prompt)}, False, prompt[-600:])
        raise CompletionError(f"scripted arm: purpose {self._purpose!r} has no scripted answer")

    def _reanchor(self, fences: dict[str, str]) -> str:
        """The re-anchor probe: one verdict per candidate, each looked up on its own."""
        retired = fences["retired"].strip()
        claims = []
        for label, text in fences.items():
            if not _CANDIDATE_LABEL.match(label):
                continue
            entry = self._table.reanchor.get((retired, text.strip()))
            self._record(
                "reanchor", {"retired": retired, "claim": text.strip()}, entry is not None, None
            )
            if entry is None:
                raise CompletionError("scripted arm: no re-anchor entry for this candidate")
            verdict, restatement = entry
            item: dict[str, str] = {"claim": label, "verdict": verdict, "reason": "scripted"}
            if restatement:
                item["restatement"] = restatement
            claims.append(item)
        return json.dumps({"claims": claims})

    def _reading(self, fences: dict[str, str]) -> str:
        """The second reading: confirms only the restatement the table itself wrote."""
        retired = fences["retired"].strip()
        original = fences.get("original", "").strip()
        restatement = fences["restatement"].strip()
        entry = self._table.reanchor.get((retired, original))
        ours = entry is not None and entry[0] == "DEPENDS" and entry[1] == restatement
        self._record(
            "reading",
            {"retired": retired, "original": original, "restatement": restatement},
            ours,
            None,
        )
        if not ours:
            raise CompletionError("scripted arm: a restatement the table did not write")
        return json.dumps({"depended": True, "faithful": True, "reason": "scripted"})

    def _record(self, kind: str, key: dict[str, str], answered: bool, excerpt: str | None) -> None:
        entry: dict[str, Any] = {
            "purpose": self._purpose,
            "kind": kind,
            "key": key,
            "answered": answered,
        }
        if not answered and excerpt is not None:
            entry["excerpt"] = excerpt
        self._log.entries.append(entry)


def scripted_providers(
    purposes: Sequence[str], table: ScriptedTable, log: RequestLog
) -> dict[str, ScriptedProvider]:
    """One scripted provider per purpose, sharing the table and the log."""
    return {p: ScriptedProvider(p, table, log) for p in purposes}
