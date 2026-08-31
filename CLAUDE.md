# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A local RAG pipeline over PDFs: docling converts the PDF, docling's `HybridChunker` splits it,
`BAAI/bge-m3` embeds the chunks, ChromaDB stores them on disk, and a Groq-hosted LLM answers
questions grounded on the retrieved excerpts. Ingestion runs through argparse CLIs; the query half
is also exposed as a read-only FastAPI service (`src/rag/api/`) with a React chat page in
`frontend/` on top of it. Python >=3.13, managed with `uv` (uv_build backend, src layout); the
frontend is React 19 + Vite, managed with `npm`.

## Commands

```bash
uv sync                                        # install/refresh the environment

uv run rag ingest                              # load, chunk, embed and store the bundled PDF
uv run rag ingest --pdf path/to.pdf --no-ocr --replace-existing
uv run rag ask "what is the leave policy?" --show-sources --top-k 5

uv run rag-api                                 # FastAPI backend on http://127.0.0.1:8000
uv run rag-api --port 8600 --reload            # different port, restart on source changes

uv run rag-search "..." --show-text            # retrieval only, no LLM call
uv run rag-embed-query "..." --full            # embedding only, no search
uv run python -m rag.ingestions.splitter       # loads + chunks the default PDF, prints chunks
```

```bash
cd frontend
npm install                                    # once
npm run dev                                    # Vite dev server on http://localhost:5173
npm run build                                  # emit frontend/dist, which rag-api then serves
npm run smoke                                  # server-render the page and assert on the markup
```

There is no linter config and no Python test suite. `npm run smoke` is the one automated check:
it bundles `frontend/smoke.jsx` with esbuild and server-renders the components, which catches an
import that does not resolve, a component that throws, and — the reason it exists — an answer's
`$$`/fenced equations being typeset instead of shown verbatim. Run it after touching anything in
`frontend/src`.

The only build step is `npm run build`, and it is optional: the backend serves `frontend/dist` when
that directory exists and runs API-only when it does not.

`GROQ_API_KEY` must be set (a `.env` at the repo root is loaded by `generation/generator.py`).
`GROQ_MODEL` optionally overrides the default model id.

## Running the app

The app is two processes' worth of code and, in production, one process: FastAPI answers `/api/*`
and also serves the built React page. Neither half ingests — there is no upload route, because
there was no uploader — so the document has to be in Chroma before anything will answer. Four
steps from a clean clone:

```bash
# 1. environment
uv sync
(cd frontend && npm install)

# 2. credentials -- create .env at the repo root
#    GROQ_API_KEY=gsk_...
#    GROQ_MODEL=openai/gpt-oss-120b     (optional)

# 3. ingest the document the API is pinned to
uv run rag ingest --pdf src/rag/data/policy_removed_removed.pdf --no-ocr --replace-existing

# 4a. production: build the page once, then one process serves everything
(cd frontend && npm run build)
uv run rag-api                             # http://127.0.0.1:8000 serves page and API

# 4b. development: two terminals, hot reload on both sides
uv run rag-api --reload                    # terminal 1, :8000
(cd frontend && npm run dev)               # terminal 2, :5173, proxies /api to :8000
```

In development the browser only ever talks to `http://localhost:5173`, because `vite.config.js`
proxies `/api` to the backend. That is deliberate: the fetch paths in `src/api.js` are relative, so
the same code works unchanged against the production build, where there is no proxy and no
cross-origin request at all. `config.DEV_ORIGINS` only matters if you bypass the proxy.

Only step 3's ingest is slow (docling conversion plus bge-m3 on CPU); it is a one-off per document.
The server binds its port in about a second because it never imports docling, then loads bge-m3 in
a background thread — `/api/health` reports `embedder_ready: false` until that finishes, and the
page shows "Loading the embedding model...". Both the collection handle and the embedding model are
process-wide singletons behind a lock in `api/service.py`, so that cost is paid once per process,
not once per question. (Streamlit's `st.cache_resource` used to do this; a server has no rerun to
cache against, but it does have concurrent requests, hence the lock.)

**Serving a different PDF** means two matching changes: ingest it, and change `DOCUMENT` in
`api/config.py` — or set `RAG_DOCUMENT` in the environment, which the constant reads. Changing only
one of them gives an empty page — the retrieval filter and the stored `source` metadata have to
agree.

If the page reports no stored chunks, the collection is empty or holds a different `source`; re-run
step 3 and the page will pick it up on its next health poll, without a reload. Check what is
actually stored with `uv run rag-search "..." --show-text`, which hits the same collection without
the LLM. `http://127.0.0.1:8000/docs` is the generated API reference.

## Architecture

The flow is one direction, each stage a plain function that takes and returns data — no classes,
no shared state object:

`ingestions/loader.py` (docling `DocumentConverter` → docling document)
→ `ingestions/splitter.py` (`HybridChunker` with a bge-m3 tokenizer → `DocChunk`s)
→ `ingestions/embedding.py` (`HuggingFaceEmbeddings` → one vector per chunk)
→ `vectorstore/chroma_store.py` (`store_embeddings` → persistent Chroma collection)

and on the query side:

`retrieval/query_embedder.py` (question → vector)
→ `retrieval/retriever.py` (Chroma `query` → ranked dicts)
→ `generation/prompts.py` + `generation/generator.py` (Groq chat completion → answer text)

`main.py` wires both halves into `ingest()` / `ask()` and the `rag` CLI. `retriever.py` and
`query_embedder.py` also have their own `main()` entry points registered in `pyproject.toml`,
so each stage can be exercised in isolation.

The HTTP layer sits on the query side only, in four small modules:

`api/config.py` (the `DOCUMENT` pin, CORS origins, `top_k` bounds)
→ `api/service.py` (the cached collection and embedder; `answer_question`, `health`)
→ `api/schemas.py` (pydantic request/response shapes)
→ `api/app.py` (`create_app`, three routes, the static mount)

`server.py` is the `rag-api` shim: it hands uvicorn the import string `rag.api.app:app`, which is
what `--reload` needs. `frontend/src/App.jsx` holds the page's state (health, messages, `topK`) and
`frontend/src/api.js` is the only module that calls the backend.

The API is **read-only over the vector store**: it imports the retrieval half only, never docling,
so it starts in about a second instead of loading converter models it would never use. There is
deliberately no upload route — `DOCUMENT` names the one PDF it serves, and every `retrieve()` call
passes `where={"source": DOCUMENT}`, so the pin is enforced at the query rather than merely
displayed. Ingesting a second document therefore cannot leak into its answers; changing which
document it serves means changing that one constant. Adding a write route would also mean
importing docling into the server, which is what the startup time depends on.

Three routes, and the client needs no others:

- `GET /api/health` — `ready`, `document`, `model`, `chunks`, `embedder_ready`, the `top_k` bounds,
  and `detail` (what to run) when it is not ready. It never raises: a page that cannot reach the
  store still has to render the reason, which is what the old `st.error` block did.
- `GET /api/document` — the same payload, named for how the sidebar reads it.
- `POST /api/ask` — `{question, top_k}` in; `{answer, chunks}` out.

`post_ask` is a sync `def` on purpose. Embedding and the Groq call both block, so FastAPI runs it
in its threadpool and one slow answer cannot stall the event loop; making it `async` would.

### Invariants that cut across modules

- **One embedding model, both sides.** `ingestions/embedding.MODEL_ID` (`BAAI/bge-m3`) is imported
  by `retrieval/query_embedder.py` on purpose. Changing it in one place only makes stored vectors
  and query vectors incomparable, and requires re-ingesting everything.
- **The Chroma collection has no embedding function.** Vectors are computed upstream and handed to
  Chroma as-is, both on write (`upsert(embeddings=...)`) and on read (`query(query_embeddings=...)`).
  Never let Chroma embed for you here.
- **Cosine, not L2.** The collection is created with `{"hnsw:space": "cosine"}`; `retriever.py`
  reports `score = 1.0 - distance` on that basis. Both must move together.
- **Chunk ids are content-derived** (`source + sha256(text)`), so re-ingesting an unchanged document
  upserts in place. Edited text yields *new* ids and leaves the old rows behind — that is what
  `--replace-existing` (delete by `source` first) is for.
- **Chroma metadata must be scalars.** `_chunk_metadata` flattens headings and page lists into
  strings; adding a list or dict there will fail at write time.
- **The chunker returns a one-shot iterator.** `splitter()` now `list()`s it before correcting
  headings, so it hands back a real list; `main.ingest` and `store_embeddings` still `list()`
  defensively, which is harmless.
- **Headings are corrected, not trusted.** `HybridChunker` labels a chunk with the last heading
  above it in the tree, so a chunk on a page whose own section header docling missed inherits the
  previous section's title. `splitter._drop_inherited_headings` keeps a heading only when the chunk
  overlaps the page the heading was found on, and writes nothing otherwise — an empty `headings`
  is correct, a wrong one misdirects the model via the `section:` line.
- **Contentless chunks are never stored.** `chroma_store._is_contentless` drops chunks that hold
  nothing but LaTeX filler; they embed and rank like real text otherwise.
- **The prompt is the security boundary.** `SYSTEM_PROMPT` treats CONTEXT as untrusted data and
  forbids disclosing internals; `format_context` deliberately omits scores, chunk ids and paths.
  Keep new fields out of the context block unless the model is meant to see them. `/api/ask` does
  return scores and chunk ids, for the sources panel — that is the reader's channel, not the
  model's, and the two must not be conflated: widening `format_context` to match the API response
  is exactly the mistake this separation exists to prevent.
- **One error shape.** Every failing response is `{"detail": "<one sentence>"}`, so `api.js` has a
  single path for rendering a failure. FastAPI's own 422 would be a *list* under that key, so
  `on_invalid_request` flattens it to a 400 with one string. A new route that answers differently
  breaks the client's error handling silently.
- **Status codes carry the meaning.** 400 the request, 503 the environment (store missing or empty,
  no `GROQ_API_KEY` — all fixable by the operator, and the `detail` says how), 502 the Groq call
  failing. The page shows `detail` verbatim, so it is user-facing text.

### Costs and defaults worth knowing

- docling's enrichment passes (`do_formula_enrichment`, `do_picture_classification`,
  `do_picture_description`, `generate_picture_images`) run extra vision models per page and turn a
  short PDF into a tens-of-minutes job on CPU. They are **off** by default in `loader()`,
  `main.ingest()` and every CLI flag, and each pass has a flag on `rag ingest` so the three layers
  cannot drift apart again. Leave them off unless you have a GPU: on CPU they do not fail cleanly,
  they degrade. Formula enrichment emitted runs of empty `\text { }` and picture description
  invented a bar chart that never existed in the PDF, and both were indexed as document text.
- **Formulas come from the text layer, not a model.** docling leaves `text` empty on formula items
  unless formula enrichment runs, so equations would otherwise serialise as
  `<!-- formula-not-decoded -->`. `loader.recover_formula_text` copies the unicode docling already
  parsed off `item.orig` into `item.text`, via `loader.normalize_formula_text`. That is where the
  math in the index comes from — these PDFs contain no LaTeX source, only a text layer.
- **Extracted equations are flattened, and nothing downstream may pretend otherwise.** A PDF text
  layer is positioned glyphs with no structure. A fraction bar is a drawn rule with no character
  behind it and scripts are baseline offsets, so `log(t)/t` extracts as `log(t) t` and `ξ_t` as
  `ξ t`. `normalize_formula_text` cleans the debris on top of that — it folds each stack of
  stretched-delimiter piece glyphs (`⎛⎜⎝`, `⎧⎪⎨⎪⎩`, U+239B–U+23AD) back into the one bracket it
  drew, counting hooks so two adjacent delimiters do not merge, and drops the `︷︸` of an
  underbrace — but the lost relations are gone. The rest is the prompt's job: rule 6 of
  `SYSTEM_PROMPT` forbids reconstructing these strings, because a model handed
  `K (K log(t) t) 1 / 3` inside `$$…$$` reads the delimiters as a promise of LaTeX and confidently
  emits `\bigl(K\log(t)\,t\bigr)^{1/3}` — turning a quotient into a product. STYLE therefore
  requires quoted equations in a fenced code block, never in `$…$`/`$$…$$`. The page must not
  undo that: `frontend/src/components/Markdown.jsx` runs `react-markdown` with `remark-gfm` and
  **no math plugin**, so `$$` renders as the two characters it is. Adding `remark-math`/KaTeX
  would typeset those delimiters and silently eat the braces of `min {γ, …}`.
- **Formula enrichment is not a fallback here.** `--formula-enrichment` transcribes equations with a
  model instead of reading the text layer, and is the only way to recover the lost structure — but
  measured on this machine it did not finish a 4-page PDF in 45 minutes. Treat it as GPU-only.
- `--no-ocr` is the biggest single speedup for PDFs that already carry a text layer.
- `main.py` calls `warnings.filterwarnings("ignore")` and `logging.disable(logging.INFO)` *before*
  importing docling/transformers, hence the `# noqa: E402` imports. `api/app.py` does the same for
  the same reason. Keep that ordering in both.

### Windows notes

- `console.use_utf8_stdout()` is called at the top of every `main()`, and in the API's lifespan
  startup so it applies however uvicorn was launched. Without it, model answers and PDF text raise
  `UnicodeEncodeError` on the default Windows console codepage. New entry points need the same call.
- Vite binds `localhost`, which resolves to `::1` here. `curl http://127.0.0.1:5173` gets connection
  refused while `http://localhost:5173` works — that is IPv6, not a broken dev server.
- `chroma_db/` and `__pycache__/` are listed in `.gitignore` but were committed before it existed,
  so they are still tracked — running an ingest dirties the working tree with binary diffs. Don't
  mistake that for real changes. `frontend/node_modules/`, `frontend/dist/` and `smoke.cjs` are
  covered by `frontend/.gitignore`. The root `.gitignore` matches the bare name `.gitignore`, so it
  ignores itself *and* every nested one — `frontend/.gitignore` had to be committed with
  `git add -f`, and a new one elsewhere will need the same.

### Layout gotcha

`src/rag/ingestions/` has no `__init__.py` (unlike the other subpackages) and works only as an
implicit namespace package. Add one if packaging or imports start misbehaving.
