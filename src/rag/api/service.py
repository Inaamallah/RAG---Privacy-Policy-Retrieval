"""What the API actually does: open the store once, retrieve, generate.

This is the half of the old Streamlit page that was not layout. Streamlit
re-ran the file top to bottom on every interaction and leaned on
`st.cache_resource` so the collection and the embedding model were built once
per process; a server has no rerun, but it does have concurrent requests, so
the same one-off work is done here behind a lock instead.

Retrieval is pinned to `config.DOCUMENT` with a Chroma metadata filter, so
nothing outside that PDF can reach the model regardless of what else the
collection holds.
"""

import threading

from .config import DOCUMENT
from ..generation.generator import DEFAULT_MODEL, generate_answer
from ..retrieval.query_embedder import get_embedder
from ..retrieval.retriever import retrieve
from ..vectorstore.chroma_store import (
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    get_collection,
)

NO_CONTEXT_ANSWER = "I could not find that in the provided documents."

# Both are built on first use and reused for the life of the process. The lock
# keeps two simultaneous requests from each paying the several-second cost of
# loading bge-m3, and from racing on the module global inside `get_embedder`.
_lock = threading.Lock()
_collection = None
_embedder = None


class StoreUnavailable(RuntimeError):
    """The vector store cannot answer: it is missing, unreadable, or empty.

    Every case is fixed the same way -- run the ingest -- so they share one
    exception and the message says which it was.
    """


def open_collection():
    """
    Returns the Chroma collection, opening it on first use.

    Raises:
        StoreUnavailable: If the database cannot be opened.
    """
    global _collection
    with _lock:
        if _collection is None:
            try:
                _collection = get_collection(DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR)
            except Exception as error:
                raise StoreUnavailable(
                    f"Could not open the vector store at {DEFAULT_PERSIST_DIR}: {error}"
                ) from error
        return _collection


def warm_embedder():
    """
    Loads bge-m3, once.

    `embed_query` would load it lazily on the first question anyway. Doing it
    up front means the cost lands on server startup rather than looking like a
    slow first answer, and `/api/health` can report when it is paid.

    Returns:
        The embedding model.
    """
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = get_embedder()
        return _embedder


def embedder_ready():
    """True once the embedding model is loaded."""
    return _embedder is not None


def document_rows():
    """
    Counts stored chunks belonging to `DOCUMENT`.

    Returns:
        The number of matching rows, or 0 if the lookup failed.

    Raises:
        StoreUnavailable: If the database cannot be opened at all.
    """
    collection = open_collection()
    try:
        found = collection.get(where={"source": DOCUMENT}, include=[])
        return len(found.get("ids") or [])
    except Exception:
        return 0


def require_ready():
    """
    Checks the store can answer, and says how to fix it when it cannot.

    Raises:
        StoreUnavailable: If the store is missing, unreadable, or holds no
            chunks for `DOCUMENT`.
    """
    if not document_rows():
        raise StoreUnavailable(
            f"No stored chunks for {DOCUMENT}. Ingest it first: "
            f"uv run rag ingest --pdf src/rag/data/{DOCUMENT} --no-ocr --replace-existing"
        )


def answer_question(question, top_k):
    """
    Retrieves context for a question and has the LLM answer from it.

    Args:
        question: The user's question.
        top_k: How many chunks to ground the answer on.

    Returns:
        A tuple of the answer text and the chunks it was grounded on.

    Raises:
        StoreUnavailable: If the store is missing, unreadable, or empty.
        ValueError: If the question is empty.
        RuntimeError: If no Groq API key is configured.
    """
    warm_embedder()
    chunks = retrieve(
        question,
        top_k=top_k,
        collection=open_collection(),
        where={"source": DOCUMENT},
    )
    if not chunks:
        return NO_CONTEXT_ANSWER, []
    return generate_answer(question, chunks, model=DEFAULT_MODEL), chunks


def health():
    """
    Describes whether the API can answer and what it would answer about.

    Never raises: a page that cannot reach the store still has to render the
    reason, which is exactly what the old `st.error` block did.

    Returns:
        A dict matching `schemas.HealthResponse`.
    """
    state = {
        "ready": False,
        "document": DOCUMENT,
        "model": DEFAULT_MODEL,
        "chunks": 0,
        "embedder_ready": embedder_ready(),
        "detail": None,
    }
    try:
        rows = document_rows()
    except StoreUnavailable as error:
        state["detail"] = str(error)
        return state

    state["chunks"] = rows
    if not rows:
        state["detail"] = (
            f"No stored chunks for {DOCUMENT}. Ingest it first: "
            f"uv run rag ingest --pdf src/rag/data/{DOCUMENT} --no-ocr --replace-existing"
        )
        return state

    state["ready"] = True
    return state
