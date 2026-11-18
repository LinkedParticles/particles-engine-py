# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The door arm's two zero-cost baselines, written by the Particles authors.

* ``raw-chunks``: each written session is one chunk, embedded with the local
  encoder; the door is the ten chunks closest to the question by cosine, with
  no other logic. It shows what plain retrieval over raw text scores here.
* ``recent-mentions``: every session line is kept in write order; the door is
  the ten most recent lines sharing a content word with the question, persona
  words excluded. It shows how much of a currency score recency alone buys,
  which bounds how easy the world is.

Both are in-process and make no model call.
"""

from __future__ import annotations

import re
from typing import Any

from particles.benchmark.rot.contract import DoorMemory
from particles.embeddings import EmbeddingModel, cosine_similarities, get_embedding_model

#: How many chunks or lines a baseline door returns.
BASELINE_TOP = 10

_WORD = re.compile(r"[a-z0-9']+")
#: Words that carry no content for a question about the persona.
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "do",
        "does",
        "for",
        "go",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "use",
        "usually",
        "what",
        "which",
        "who",
        "whom",
        "with",
        "currently",
        "working",
        "work",
        "user",
        "user's",
        "users",
        "i",
        "my",
        "me",
    }
)


def content_words(text: str) -> set[str]:
    """Lower-cased words of ``text`` minus stop and persona words."""
    return {w.strip("'") for w in _WORD.findall(text.lower())} - _STOP - {""}


class RawChunksMemory(DoorMemory):
    """Plain cosine retrieval over raw session chunks."""

    name = "raw-chunks"

    def __init__(self, encoder: EmbeddingModel | None = None) -> None:
        model = encoder if encoder is not None else get_embedding_model()
        if model is None:
            raise RuntimeError("raw-chunks needs the local embedding model")
        self._encoder = model
        self._chunks: list[str] = []
        self._vectors: list[Any] = []

    def doors(self) -> list[str]:
        """One door, ``chunks``."""
        return ["chunks"]

    def import_memory(self, text: str, written_at: str) -> None:
        """Keep the session text as one chunk and embed it."""
        self._chunks.append(text)
        self._vectors.append(self._encoder.encode([text])[0])

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        """The ten nearest chunks, nearest first, separated by blank lines."""
        if not self._chunks:
            return "", {"hits": 0}
        q = self._encoder.encode([query])[0]
        sims = cosine_similarities(q, self._vectors)
        order = sorted(range(len(self._chunks)), key=lambda i: sims[i], reverse=True)
        top = order[:BASELINE_TOP]
        return "\n\n".join(self._chunks[i] for i in top), {"hits": len(top)}


class RecentMentionsMemory(DoorMemory):
    """The most recent session lines that share a content word with the question."""

    name = "recent-mentions"

    def __init__(self) -> None:
        self._lines: list[str] = []

    def doors(self) -> list[str]:
        """One door, ``recent``."""
        return ["recent"]

    def import_memory(self, text: str, written_at: str) -> None:
        """Keep every non-blank line, in write order."""
        self._lines += [line for line in text.splitlines() if line.strip()]

    def door_text(self, query: str, door: str) -> tuple[str, Any]:
        """The ten most recent matching lines, newest first."""
        words = content_words(query)
        hits: list[str] = []
        for line in reversed(self._lines):
            if words & content_words(line):
                hits.append(line)
                if len(hits) == BASELINE_TOP:
                    break
        return "\n".join(hits), {"hits": len(hits)}
