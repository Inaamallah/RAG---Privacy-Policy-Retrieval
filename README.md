# Document Q&A

A local RAG pipeline over PDFs, with a FastAPI backend and a React chat page.

docling converts the PDF, docling's `HybridChunker` splits it, `BAAI/bge-m3` embeds the chunks,
ChromaDB stores them on disk, and a Groq-hosted LLM answers questions grounded on the retrieved
excerpts — citing the page each claim came from, and saying so when the document does not answer.

Ingestion is a CLI. The web app is **read-only over the vector store**: it has no upload route, and
every retrieval is pinned to one document by a metadata filter, so it can only ever answer out of
the PDF you ingested.

## Setup

```bash
uv sync
cd frontend && npm install && cd ..
```

Create `.env` at the repo root:

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b     # optional
```

Ingest the document the app is pinned to. This is the slow step (docling plus bge-m3 on CPU) and a
one-off per document:

```bash
uv run rag ingest --pdf src/rag/data/policy_removed_removed.pdf --no-ocr --replace-existing
```

## Run

One process serves both the API and the page:

```bash
cd frontend && npm run build && cd ..
uv run rag-api                      # http://127.0.0.1:8000
```

Or two, with hot reload on each side:

```bash
uv run rag-api --reload             # terminal 1  :8000
cd frontend && npm run dev          # terminal 2  :5173, proxies /api to :8000
```

Check the page still renders after editing it:

```bash
cd frontend && npm run smoke
```

## API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Whether the store can answer, and what about |
| `GET /api/document` | The same payload, for the sidebar |
| `POST /api/ask` | `{question, top_k}` in, `{answer, chunks}` out |

Interactive reference at `/docs`. Failures answer with `{"detail": "<message>"}`: 400 for the
request, 503 for the environment (nothing ingested, no API key — the message says what to run),
502 when the model call fails.

## CLI

```bash
uv run rag ask "..." --show-sources --top-k 5   # answer in the terminal
uv run rag-search "..." --show-text             # retrieval only, no LLM call
uv run rag-embed-query "..." --full             # embedding only, no search
```

Serving a different PDF means two matching changes: ingest it, and set `RAG_DOCUMENT` (or change
`DOCUMENT` in `src/rag/api/config.py`). See `CLAUDE.md` for the architecture and its invariants.
