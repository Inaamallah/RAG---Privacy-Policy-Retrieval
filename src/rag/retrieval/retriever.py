"""Fetch the chunks most relevant to a user's question.

The collection was built with cosine distance and holds vectors produced by
`rag.ingestions.embedding`, so the query is embedded with that same model and
handed to Chroma as a raw vector -- the collection has no embedding function
of its own.

    uv run rag-search "what is the leave policy?"
    uv run rag-search "..." --top-k 5 --show-text
"""

import argparse

from ..console import use_utf8_stdout
from ..vectorstore.chroma_store import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, get_collection
from .query_embedder import embed_query

DEFAULT_TOP_K = 5


def retrieve(
    query,
    top_k=DEFAULT_TOP_K,
    collection_name=DEFAULT_COLLECTION,
    persist_dir=DEFAULT_PERSIST_DIR,
    collection=None,
):
    """
    Returns the chunks closest to the query, nearest first.

    Args:
        query: The user's question.
        top_k: How many chunks to return; fewer come back if the collection
            holds fewer.
        collection_name: Chroma collection to search.
        persist_dir: Directory the Chroma database files live in.
        collection: An already-open collection to reuse; one is opened if
            omitted.

    Returns:
        A list of dicts with `id`, `text`, `metadata`, `distance` and `score`,
        ordered from most to least relevant. Empty if the collection is empty.

    Raises:
        ValueError: If the query is empty or `top_k` is not positive.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    collection = collection or get_collection(collection_name, persist_dir)

    # Checked before embedding: loading the embedding model is expensive and
    # there is nothing to compare the vector against anyway.
    available = collection.count()
    if not available:
        print(f"Collection '{collection_name}' is empty -- run the ingestion first.")
        return []

    vector = embed_query(query)  # raises ValueError on an empty query

    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, available),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma nests one list per query; there is only ever one query here.
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {},
            "distance": distance,
            # Cosine distance runs 0 (identical) to 2 (opposite); flip it so
            # bigger reads as better.
            "score": 1.0 - distance,
        }
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def format_results(results, show_text=False, snippet_chars=300):
    """
    Renders retrieval results as a readable block of text.

    Args:
        results: The list `retrieve()` returned.
        show_text: Print each chunk in full instead of a snippet.
        snippet_chars: Characters of each chunk to show when not in full.

    Returns:
        A printable string.
    """
    if not results:
        return "No matching chunks."

    lines = []
    for rank, hit in enumerate(results, start=1):
        meta = hit["metadata"]
        where = meta.get("source", "unknown")
        pages = meta.get("pages")
        if pages:
            where += f" p.{pages}"
        headings = meta.get("headings")

        lines.append(f"[{rank}] score {hit['score']:.4f}  {where}")
        if headings:
            lines.append(f"    {headings}")

        text = hit["text"]
        if not show_text and len(text) > snippet_chars:
            text = text[:snippet_chars].rstrip() + "..."
        lines.append("    " + text.replace("\n", "\n    "))
        lines.append("")

    return "\n".join(lines).rstrip()



def main():
    """CLI wrapper around `retrieve()`. Returns a process exit code."""
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Fetch the chunks most relevant to a query.")
    parser.add_argument("query", nargs="*", help="The question to search for; you are prompted if it is omitted")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to return")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection to search")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Directory holding the Chroma database")
    parser.add_argument("--show-text", action="store_true", help="Print each chunk in full instead of a snippet")
    args = parser.parse_args()

    query = " ".join(args.query) or input("Query: ")

    try:
        results = retrieve(
            query,
            top_k=args.top_k,
            collection_name=args.collection,
            persist_dir=args.persist_dir,
        )
    except ValueError as e:
        print(e)
        return 1

    print(format_results(results, show_text=args.show_text))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
