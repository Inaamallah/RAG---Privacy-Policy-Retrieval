"""Turn a user's question into a vector.

Nothing is searched here -- this module only produces the query embedding, so
it has to use the exact model the chunks were embedded with, otherwise the two
sets of vectors are not comparable.

    uv run python -m rag.retrieval.query_embedder "what is the leave policy?"
    uv run python -m rag.retrieval.query_embedder          # prompts for input
"""

import argparse

from langchain_huggingface import HuggingFaceEmbeddings

from ..console import use_utf8_stdout
from ..ingestions.embedding import MODEL_ID

# Loading bge-m3 takes a few seconds, so the model is built once and reused
# across calls in the same process.
_embeddings = None


def get_embedder(model_name=MODEL_ID):
    """
    Returns the shared embedding model, building it on first use.

    Args:
        model_name: Hugging Face id of the embedding model.

    Returns:
        A HuggingFaceEmbeddings instance.
    """
    global _embeddings
    if _embeddings is None or _embeddings.model_name != model_name:
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings


def embed_query(query, model_name=MODEL_ID):
    """
    Embeds a single user query.

    Args:
        query: The question to embed.
        model_name: Hugging Face id of the embedding model.

    Returns:
        One vector, as a list of floats.

    Raises:
        ValueError: If the query is empty or only whitespace.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("The query is empty.")
    return get_embedder(model_name).embed_query(query)



def main():
    """CLI wrapper around `embed_query()`. Returns a process exit code."""
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Embed a user query with the document embedding model.")
    parser.add_argument("query", nargs="*", help="The question to embed; you are prompted if it is omitted")
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face id of the embedding model")
    parser.add_argument("--full", action="store_true", help="Print the whole vector instead of a preview")
    args = parser.parse_args()

    query = " ".join(args.query) or input("Query: ")

    try:
        vector = embed_query(query, model_name=args.model)
    except ValueError as e:
        print(e)
        return 1

    print(f"Query: {query}")
    print(f"Dimensions: {len(vector)}")
    if args.full:
        print(vector)
    else:
        preview = ", ".join(f"{value:.6f}" for value in vector[:8])
        print(f"Vector: [{preview}, ...]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
