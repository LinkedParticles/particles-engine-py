# Using Particles from LangChain

Particles ships a thin **LangChain adapter** so a LangChain agent or
RAG chain can read from — and deposit into — a Particles store without
hand-rolling the HTTP / MCP surface. It lives in `particles.integrations` and is
gated behind an optional extra:

```bash
pip install particles[langchain]   # installs langchain-core>=0.3
```

The adapter imports `langchain_core` lazily, so the base install never pays its
cost; using the adapter without the extra raises an actionable `ImportError`.

## Tools — drop into an agent's toolset

`get_langchain_tools()` returns two `StructuredTool`s:

```python
from particles.integrations import get_langchain_tools

tools = get_langchain_tools()   # [particles_query, particles_deposit]
agent = create_agent(llm, tools)
```

- **`particles_query`** — answers a natural-language question from the store and
  returns the cited prose answer (ranked by effective confidence). Optional
  args: `tags`, `subject_id`, `min_confidence`, `top_k`.
- **`particles_deposit`** — archives text into the corpus as a new source entry
  (no belief is asserted) and returns the created entry / snapshot ids.

## Retriever — plug into a RAG chain

`ParticlesRetriever` is a `BaseRetriever`: each ranked particle becomes one
`Document` whose `page_content` is the claim text and whose `metadata` carries
`particle_id`, `effective_confidence`, `confidence`, `subject_ids`, and
`status` — so a downstream chain can cite and filter on believability.

```python
from particles.integrations import ParticlesRetriever

retriever = ParticlesRetriever(top_k=20, min_confidence=0.3)
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

## Transport, auth, and writes are inherited

The adapter reaches the engine only through the [remote-engine backend
seam](../operator-guide/remote-engine.md): with no engine configured it runs
in-process against your local store; with `engine.base_url` set it talks to a
remote engine over HTTP, presenting the `PARTICLES_ENGINE_TOKEN` bearer. Whether
a deposit is allowed is decided by the **engine's** write-enablement — the
adapter never overrides it.
