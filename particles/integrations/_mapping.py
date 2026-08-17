"""Pure ``QueryResponse`` → LangChain return-shape mapping.

These helpers carry **all** the adapter's logic that does *not* need
``langchain_core`` — they consume a :class:`~particles.core.schema.QueryResponse`
and return plain Python (a string, or a list of ``(page_content, metadata)``
pairs). ``langchain.py`` wraps the output in the real ``StructuredTool`` /
``BaseRetriever`` / ``Document`` types.

Keeping the mapping here means the unit gate — which runs **without** the
optional ``langchain`` extra installed (like ``otel``) — can cover
"``QueryResponse.answer`` flows through the query tool" and "each ranked
particle maps to one ``Document``'s content + metadata" with the ``Backend``
mocked and no dependency on ``langchain_core``. The thin wrappers in
``langchain.py`` then only have to be exercised by the optional
integration-tier test.
"""

from __future__ import annotations

from typing import Any

from particles.core.schema import QueryResponse


def query_answer_text(response: QueryResponse) -> str:
    """Render a ``QueryResponse`` as the cited NL answer string a tool returns.

    ``QueryResponse.answer`` is already the cited-prose answer the query
    operation synthesises. When the response carries coverage gaps or a
    ``top_k`` truncation warning, a short trailing ``Note:`` is appended so the
    agent sees retrieval limits rather than silently trusting a thin answer
    .
    """
    notes: list[str] = []
    if response.truncation_warning:
        notes.append(response.truncation_warning)
    for gap in response.subject_coverage_gaps:
        notes.append(gap.detail)
    answer = response.answer
    if notes:
        answer = f"{answer}\n\nNote: {' '.join(notes)}"
    return answer


def query_documents(response: QueryResponse) -> list[tuple[str, dict[str, Any]]]:
    """Map a ``QueryResponse`` to one ``(page_content, metadata)`` pair per particle.

    Each ranked particle becomes one LangChain ``Document``:

    * ``page_content`` is the particle's ``content`` (the claim text a RAG chain
      embeds / stuffs into context).
    * ``metadata`` is a flat dict carrying provenance + ranking so a downstream
      chain can cite and filter: ``particle_id``, ``effective_confidence`` (from
      the parallel ``effective_confidences[i]``), ``confidence`` (the stored,
      immutable ``confidence.value``), ``subject_ids``, and ``status``.

    The particles arrive already ranked by the engine; no relevance-score field
    LangChain does not ask for is invented (``effective_confidence`` rides in
    metadata for any chain that wants to threshold on it).
    """
    docs: list[tuple[str, dict[str, Any]]] = []
    for particle, effective in zip(response.particles, response.effective_confidences, strict=True):
        metadata: dict[str, Any] = {
            "particle_id": particle.id,
            "effective_confidence": effective,
            "confidence": particle.confidence.value,
            "subject_ids": list(particle.subject_ids),
            "status": str(particle.status),
        }
        docs.append((particle.content, metadata))
    return docs


def deposit_confirmation_text(entry_id: str, snapshot_id: str) -> str:
    """Render the ``deposit_text`` outcome as a short confirmation string.

    ``Backend.deposit_text`` returns ``(entry_id, snapshot_id)``;
    the deposit tool surfaces both so the agent can cite the corpus entry it
    just created.
    """
    return f"Deposited corpus entry {entry_id} (snapshot {snapshot_id})."
