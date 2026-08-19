# Design notes

The original design sketch for JJRAG, kept for the record, followed by what the
current implementation does differently and why.

## Original sketch

1. **Document loading** — support several document types.
2. **Document splitting** — chunk to fit the model and improve efficiency.
   Chose `RecursiveCharacterTextSplitter`, since the inputs are unstructured.
3. **Vector storage** — local, for efficiency. Chose FAISS: fast similarity
   search, no server or cloud dependency, well integrated, and appropriate for a
   project that does not need a distributed architecture.
4. **Query** — choose the model type, and choose the embedding backend
   (local via Ollama, or hosted).
5. **UI and `main()`** — Streamlit.

General principles from the sketch:

- **Error handling** — `try/except` around each function, logging per area, and
  defensive handling of empty Streamlit state.
- **Further optimisation** — persist the vector store (`save_local()` /
  `load_local()`) so runs do not have to re-embed: persistence between runs
  without external services, no reprocessing, offline operation after setup.

## What changed, and why

The sketch was sound about structure; the current system keeps its shape and
hardens each part.

| Area | Sketch | Now |
| --- | --- | --- |
| Model choice | local *or* hosted | **local only** — the hosted path is not disabled, it does not exist, and CI rejects any pull request that adds one |
| Embeddings | Ollama or OpenAI | local sentence-transformers, Ollama, or a dependency-free hashing backend |
| Chunking | 40 characters | 1,000 with 150 overlap; 40 characters cannot hold a sentence, so retrieval returned fragments |
| Splitter | LangChain | equivalent recursive splitter implemented in-tree — one fewer dependency to clear, and behaviour pinned by our own tests |
| Vector store | FAISS | versioned numpy index by default (no native dependency, exact search, trivially auditable); FAISS still available via config |
| Persistence | `save_local()` / `load_local()` | versioned index directories with an atomic `current` pointer, plus a SQLite catalog for lineage, dedup and erasure |
| Error handling | try/except + logging | that, plus a validation rule engine with gates that fail a run instead of publishing a bad index |
| Privacy | API key in `.env` | PII redaction before storage, socket-level egress guard, redacting log filter, audit log |
| UI | Streamlit | FastAPI service with a dependency-free web UI, plus a CLI for automation |

Two choices from the sketch proved especially load-bearing and were kept:
**local persistence** (now the versioned index, which is what makes atomic
publish and rollback possible) and **local-first model execution** (now the
only option, which is what makes the compliance story defensible).

The original prototypes (`JJrag.py`, `rag_app.py`) were removed once the
pipeline replaced them: both contained hosted-model API calls, which contradict
the guarantee the project now makes. They remain in git history at commit
`3339bfd` if you need to refer back to them.
