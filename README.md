# 🔒 JJRAG — local-only document Q&A pipeline

Ask questions of your own documents, with answers grounded in — and cited to —
their actual contents. Every stage runs on hardware you control: extraction,
cleaning, validation, embedding, indexing, retrieval and generation.

**There is no hosted-model API in this codebase.** Not disabled behind a flag —
absent. No `anthropic`, no `openai`, no cloud embedding service, in the source
or the dependencies. On top of that, the running process installs a
socket-level guard that refuses outbound connections to anything except
loopback and your configured local model, so a dependency that tries to phone
home fails loudly instead of succeeding quietly.

```
upload ─▶ scan ─▶ extract ─▶ transform ─▶ validate ─▶ embed ─▶ load ─▶ publish
          │        │          │            │           │        │        │
      allowlist   PDF·DOCX   normalise   20+ rules   local    versioned  atomic
      magic bytes PPTX·MD    de-hyphen   gate the    model    index +    pointer
      zip bombs   HTML·CSV   redact PII  run         embeds   catalog    swap
      macros      JSON·EML   chunk·dedup
```

---

## Why this exists

Sending a document to a hosted model means a third party receives its contents,
logs the request, and holds it under their retention policy rather than yours.
For regulated data that is usually the end of the conversation. JJRAG removes
the question instead of answering it.

| | |
| --- | --- |
| 🔒 **Local generation** | [Ollama](https://ollama.com) on your host. No key, no account, no egress. |
| 🧮 **Local embeddings** | sentence-transformers, Ollama, or a dependency-free hashing backend for air-gapped hosts. |
| 🛡️ **Egress guard** | Non-local sockets raise `EgressBlocked` before a byte leaves. Blocked attempts are reported at `/api/compliance`. |
| ✅ **Validation gates** | 20+ rules across the stages. A breach fails the run instead of publishing a broken index. |
| 🧾 **Auditable** | Per-run manifests, an append-only audit log, and a SQLite catalog tracing every chunk to the bytes it came from. |
| 🗑️ **Real erasure** | Deleting a document rebuilds the index without it — the deletion reaches the vectors. |
| 🔁 **Atomic publish** | New index versions are built beside the live one and swapped only after gates pass. Rollback is a pointer move. |

---

## Quick start

```bash
git clone https://github.com/jfjustin/JJRAG.git
cd JJRAG
pip install -r requirements.txt

# Install a local model (once)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

jjrag doctor                              # check the environment
jjrag ingest ~/Documents/policies         # run the pipeline
jjrag query "What is our retention period?"
jjrag serve                               # → http://localhost:8000
```

Or the whole stack in containers, on a Docker network with no route to the
internet:

```bash
cp deploy/.env.example deploy/.env        # set JJRAG_API_TOKEN
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml exec ollama ollama pull llama3.1:8b
```

---

## What it reads

PDF (page-level citations, optional local OCR), DOCX, PPTX, TXT, Markdown,
HTML, CSV/TSV, JSON/JSONL, EML.

Tabular and structured formats are serialised so each chunk stays
self-describing — a CSV row keeps its column names, an email keeps its headers.
Without that, retrieved rows are unusable numbers with no labels.

---

## The pipeline

| Stage | Does | Fails the run when |
| --- | --- | --- |
| **scan** | Extension allowlist, magic-byte match, size caps, zip-bomb and macro checks. Rejects go to quarantine with a written reason. | — (rejections are recorded, not fatal) |
| **extract** | Per-format parsers producing labelled segments (`p. 12`, `Slide 3`, `rows 51–100`). | A file yields no text; output is garbled; most of a batch fails. |
| **transform** | Unicode normalise, de-hyphenate PDFs, strip repeated headers, redact PII, chunk recursively, drop exact and near duplicates. | A document produces no chunks; a chunk is oversized; PII survives redaction. |
| **assemble** | Combines new chunks with the still-active corpus, so deletions and config changes take effect. | — |
| **embed** | Local embeddings, batched and cached by `(model, text hash)`. | Zero vectors, NaN/inf, or a changed dimensionality. |
| **load** | Builds a new index version, checks parity against the catalog, then publishes atomically. | Vector/chunk/catalog counts disagree; the index would be empty. |

A failed run leaves the previous index serving traffic, untouched.

Retrieval is hybrid — semantic search and BM25 keyword search, fused with
Reciprocal Rank Fusion, then MMR for diversity. Dense-only search misses exact
terms like a policy code (`DR-14`); keyword-only misses paraphrase.

---

## Commands

```bash
jjrag doctor                       # environment check
jjrag ingest [PATHS...]            # run the pipeline (default: the inbox)
jjrag ingest --force ~/docs        # re-ingest already-indexed files
jjrag rebuild                      # re-index after a config change
jjrag query "..."                  # ask, streamed to the terminal
jjrag query "..." --retrieve-only  # inspect retrieval without generating
jjrag docs                         # list indexed documents
jjrag rm doc_abc123                # erase a document and rebuild
jjrag runs [RUN_ID]                # history, or one full validation report
jjrag compliance                   # print the privacy attestation
jjrag serve                        # start the web app
```

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/documents` | Upload files (scanned on arrival) |
| `POST /api/ingest` | Run the pipeline; returns the full manifest |
| `GET /api/runs`, `GET /api/runs/{id}` | Run history and validation reports |
| `GET /api/documents` | Indexed documents |
| `DELETE /api/documents/{id}` | Erase a document and rebuild the index |
| `POST /api/query` | Ask a question (JSON) |
| `POST /api/query/stream` | Ask a question (server-sent events) |
| `GET /api/health` | Index and local-model status |
| `GET /api/compliance` | Live privacy attestation |
| `GET /api/audit` | Recent audit entries (token required) |

Interactive schema at `/api/docs`.

---

## Hosting it on your domain

Two different things, with different requirements — see
[`docs/hosting.html`](docs/hosting.html) for the full walkthrough.

**The docs** (`docs/`) are static and go on GitHub Pages, fronted by Cloudflare
or GoDaddy DNS. Enable *Settings → Pages → Source: GitHub Actions*; the workflow
publishes on every push to `main`. Then add a `CNAME` for `docs` →
`<user>.github.io` (for a GoDaddy apex domain, use GitHub's four `A` records
instead).

**The app** holds your documents and runs the model, so it stays on a machine
you control. The best fit is a **Cloudflare Tunnel**: your host makes an
outbound connection only, and serves `https://app.example.com` with no inbound
firewall rule, no public IP and no port forwarding.

```bash
docker compose -f deploy/docker-compose.yml --profile tunnel up -d
```

Put **Cloudflare Access** in front of it for SSO and a login audit trail. A VPS
with the provided nginx config works too if you would rather point an `A` record
at your own server.

> **Never commit `data/`.** It holds the uploaded documents, the index and the
> catalog. It is gitignored — keep it that way.

Before exposing anything publicly: set `security.api_token`, decide whether
anonymous reads are acceptable, and check `trusted_proxy_header` matches your
actual proxy (it defaults to Cloudflare's `cf-connecting-ip`; a wrong value lets
clients forge their address and defeat rate limiting).

---

## Configuration

Everything lives in [`config/jjrag.yaml`](config/jjrag.yaml), overridable by
environment variables (`JJRAG_LLM__MODEL=mistral:7b`). The settings that matter
most:

```yaml
transform:  { chunk_size: 1000, chunk_overlap: 150 }
privacy:    { redact_pii: true, fail_on_residual_pii: true }
embedding:  { backend: sentence-transformers, device: cpu }
llm:        { model: llama3.1:8b, host: http://localhost:11434 }
validation: { enabled: true, max_error_issues: 0 }
security:   { enforce_local_only: true, api_token: null, retention_days: null }
```

### Choosing a model

| Host | Model | Expect |
| --- | --- | --- |
| 8 GB RAM, CPU | `llama3.2:3b`, `phi3:mini` | ~5–15 s per answer |
| 16 GB RAM, CPU | `llama3.1:8b` | ~15–40 s per answer |
| GPU, 8 GB+ VRAM | `llama3.1:8b` | ~1–3 s per answer |
| GPU, 24 GB+ VRAM | `qwen2.5:14b`, quantised 70B | Best quality |

Changing the generation model needs no re-index; changing the *embedding* model
does, and the pipeline detects the dimensionality change and rebuilds rather
than silently corrupting the index.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                 # 164 tests
ruff check jjrag tests
```

The suite runs with the `hashing` embedding backend and a stub local-model
server, so it needs no model weights, no GPU and no network — the same on a
laptop and in CI. CI additionally asserts that no hosted-model SDK has been
reintroduced, in the source or the dependencies.

## Documentation

- [Overview](docs/index.html)
- [Pipeline reference](docs/pipeline.html) — every stage and every gate
- [Hosting](docs/hosting.html) — Cloudflare, GoDaddy, VPS, GitHub Pages
- [Compliance](docs/compliance.html) — the controls, and how to verify each one
- [Design notes](DESIGN_NOTES.md) — the original sketch and what changed

## Limits

Worth stating plainly: encryption at rest is the host's job (use full-disk
encryption); the API token is a shared secret, not user management; PII
detection is pattern-based and deliberately strict about structure, so review
the redaction counts on your own corpus; and a local model still makes
mistakes — citations make its errors checkable, not impossible.

## License

MIT.
