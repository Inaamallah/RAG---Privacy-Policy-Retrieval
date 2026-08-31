# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A local RAG pipeline over PDFs: docling converts the PDF, docling's `HybridChunker` splits it,
`BAAI/bge-m3` embeds the chunks, ChromaDB stores them on disk, and a Groq-hosted LLM answers
questions grounded on the retrieved excerpts. Ingestion runs through argparse CLIs; there is one
read-only Streamlit chat page (`app.py`) over the query half. Python >=3.13, managed with `uv`
(uv_build backend, src layout).

## Commands

```bash
uv sync                                        # install/refresh the environment

uv run rag ingest                              # load, chunk, embed and store the bundled PDF
uv run rag ingest --pdf path/to.pdf --no-ocr --replace-existing
uv run rag ask "what is the leave policy?" --show-sources --top-k 5

uv run rag-ui                                  # Streamlit chat page on the ingested PDF
uv run rag-ui --server.port 8600               # extra args pass through to streamlit

uv run rag-search "..." --show-text            # retrieval only, no LLM call
uv run rag-embed-query "..." --full            # embedding only, no search
uv run python -m rag.ingestions.splitter       # loads + chunks the default PDF, prints chunks
```

There are no tests, linter config, or build step in this repo.

`GROQ_API_KEY` must be set (a `.env` at the repo root is loaded by `generation/generator.py`).
`GROQ_MODEL` optionally overrides the default model id.

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

`app.py` is the Streamlit chat page and `ui.py` the `rag-ui` shim that launches it. The page is
**read-only over the vector store**: it imports the retrieval half only, never docling, so it
starts in about a second instead of loading converter models it would never use. There is
deliberately no uploader — `DOCUMENT` names the one PDF it serves, and every `retrieve()` call
passes `where={"source": DOCUMENT}`, so the pin is enforced at the query rather than merely
displayed. Ingesting a second document therefore cannot leak into its answers; changing which
document it serves means editing that one constant.

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
  Keep new fields out of the context block unless the model is meant to see them.

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
  parsed off `item.orig` into `item.text`. That is where the math in the index comes from — these
  PDFs contain no LaTeX source, only a text layer.
- `--no-ocr` is the biggest single speedup for PDFs that already carry a text layer.
- `main.py` calls `warnings.filterwarnings("ignore")` and `logging.disable(logging.INFO)` *before*
  importing docling/transformers, hence the `# noqa: E402` imports. Keep that ordering.

### Windows notes

- `console.use_utf8_stdout()` is called at the top of every `main()`. Without it, model answers and
  PDF text raise `UnicodeEncodeError` on the default Windows console codepage. New entry points
  need the same call.
- `chroma_db/` and `__pycache__/` are listed in `.gitignore` but were committed before it existed,
  so they are still tracked — running an ingest dirties the working tree with binary diffs. Don't
  mistake that for real changes.

### Layout gotcha

`src/rag/ingestions/` has no `__init__.py` (unlike the other subpackages) and works only as an
implicit namespace package. Add one if packaging or imports start misbehaving.
