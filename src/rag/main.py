"""Entry point for the whole RAG flow.

Two commands:

    uv run rag ingest                    # load, chunk, embed, store the PDF
    uv run rag ask "what is the leave policy?"

Nothing in here executes on import.
"""

import argparse
import logging
import warnings

# Silence the noisy docling/transformers import banners before they load.
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

from .console import use_utf8_stdout  # noqa: E402
from .generation.generator import DEFAULT_MODEL, generate_answer  # noqa: E402
from .ingestions.embedding import embedding  # noqa: E402
from .ingestions.loader import DEFAULT_PDF, loader  # noqa: E402
from .ingestions.splitter import splitter  # noqa: E402
from .retrieval.retriever import DEFAULT_TOP_K, retrieve  # noqa: E402
from .vectorstore.chroma_store import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    store_embeddings,
)


def ingest(
    pdf=DEFAULT_PDF,
    collection_name=DEFAULT_COLLECTION,
    persist_dir=DEFAULT_PERSIST_DIR,
    chunk_size=512,
    chunk_overlap=200,
    replace_existing=False,
    do_ocr=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
):
    """
    Runs the ingestion end to end: load, split, embed, store.

    Args:
        pdf: Path to the PDF to ingest.
        collection_name: Chroma collection to write to.
        persist_dir: Directory the Chroma database files live in.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Token overlap between neighbouring chunks.
        replace_existing: Delete this document's previous chunks before writing.
        do_ocr: Run OCR over the pages; off is much faster for PDFs that
            already carry a text layer.
        do_formula_enrichment: Transcribe formulas with a generative model.
            Very slow without a GPU.
        do_picture_classification: Label figures with the figure classifier.

    Returns:
        The Chroma collection, or None if any step failed.
    """
    document = loader(
        pdf,
        do_ocr=do_ocr,
        do_formula_enrichment=do_formula_enrichment,
        do_picture_classification=do_picture_classification,
    )
    if document is None:
        print("Failed to load the document.")
        return None

    # chunker.chunk() hands back a one-shot iterator; materialise it so the
    # embedding and storage steps can both walk the same chunks.
    chunks = list(splitter(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    if not chunks:
        print("The document produced no chunks.")
        return None

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedding(chunks)

    return store_embeddings(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_dir=persist_dir,
        replace_existing=replace_existing,
    )


def ask(
    query,
    top_k=DEFAULT_TOP_K,
    collection_name=DEFAULT_COLLECTION,
    persist_dir=DEFAULT_PERSIST_DIR,
    model=DEFAULT_MODEL,
):
    """
    Answers a question: retrieve the closest chunks, then ground the LLM on them.

    Args:
        query: The user's question.
        top_k: How many chunks to retrieve as context.
        collection_name: Chroma collection to search.
        persist_dir: Directory the Chroma database files live in.
        model: Groq model id.

    Returns:
        A dict with the `answer` text and the `chunks` it was grounded on.

    Raises:
        ValueError: If the query is empty or `top_k` is not positive.
        RuntimeError: If no Groq API key is configured.
    """
    chunks = retrieve(query, top_k=top_k, collection_name=collection_name, persist_dir=persist_dir)
    answer = generate_answer(query, chunks, model=model)
    return {"answer": answer, "chunks": chunks}


def _sources(chunks):
    """Deduplicated 'file, p.pages' labels for the chunks an answer used."""
    labels = []
    for hit in chunks:
        meta = hit.get("metadata") or {}
        label = meta.get("source", "unknown")
        pages = meta.get("pages")
        if pages:
            label += f", p.{pages}"
        if label not in labels:
            labels.append(label)
    return labels


def _add_store_arguments(parser):
    """Adds the collection/persist-dir options shared by both commands."""
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Directory holding the Chroma database")


def _build_parser():
    """Builds the two-command CLI."""
    parser = argparse.ArgumentParser(description="Ingest documents into a vector store and ask questions about them.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Load, chunk, embed and store a PDF")
    ingest_parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF to ingest (default: the bundled policy document)")
    ingest_parser.add_argument("--chunk-size", type=int, default=512, help="Maximum tokens per chunk")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200, help="Token overlap between chunks")
    ingest_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete the document's previous chunks before writing the new ones",
    )
    ingest_parser.add_argument(
        "--no-ocr",
        dest="do_ocr",
        action="store_false",
        help="Skip OCR; much faster when the PDF already has a text layer",
    )
    ingest_parser.add_argument(
        "--formula-enrichment",
        action="store_true",
        help="Transcribe formulas with a generative model (very slow without a GPU)",
    )
    ingest_parser.add_argument(
        "--picture-classification",
        action="store_true",
        help="Label figures with the figure classifier (slow without a GPU)",
    )
    _add_store_arguments(ingest_parser)

    ask_parser = commands.add_parser("ask", help="Answer a question from the stored documents")
    ask_parser.add_argument("query", nargs="*", help="The question; you are prompted if it is omitted")
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to use as context")
    ask_parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model id")
    ask_parser.add_argument("--show-sources", action="store_true", help="List the documents the answer drew on")
    _add_store_arguments(ask_parser)

    return parser


def main():
    """CLI entry point. Returns a process exit code."""
    use_utf8_stdout()
    args = _build_parser().parse_args()

    if args.command == "ingest":
        collection = ingest(
            pdf=args.pdf,
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            replace_existing=args.replace_existing,
            do_ocr=args.do_ocr,
            do_formula_enrichment=args.formula_enrichment,
            do_picture_classification=args.picture_classification,
        )
        return 0 if collection is not None else 1

    query = " ".join(args.query) or input("Question: ")
    try:
        result = ask(
            query,
            top_k=args.top_k,
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            model=args.model,
        )
    except (ValueError, RuntimeError) as e:
        print(e)
        return 1

    print(result["answer"])
    if args.show_sources and result["chunks"]:
        print("\nSources:")
        for label in _sources(result["chunks"]):
            print(f"  - {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
