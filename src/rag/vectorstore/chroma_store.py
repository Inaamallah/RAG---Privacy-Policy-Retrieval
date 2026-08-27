"""Persist chunk embeddings in a local, on-disk ChromaDB collection.

The embeddings are computed upstream by `rag.ingestions.embedding`, so this
module never re-embeds anything: vectors are handed to Chroma as-is and no
embedding function is attached to the collection.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings

# <repo root>/chroma_db -- chroma_store.py is at <root>/src/rag/vectorstore/
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[3] / "chroma_db"
DEFAULT_COLLECTION = "policy_documents"

# bge-m3 vectors are meant to be compared with cosine similarity, not the
# squared L2 distance Chroma defaults to.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_client(persist_dir=DEFAULT_PERSIST_DIR):
    """
    Opens (creating it if needed) the on-disk Chroma database.

    Args:
        persist_dir: Directory the database files live in.

    Returns:
        A chromadb PersistentClient.
    """
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(collection_name=DEFAULT_COLLECTION, persist_dir=DEFAULT_PERSIST_DIR, client=None):
    """
    Returns the named collection, creating it on first use.

    Args:
        collection_name: Name of the Chroma collection.
        persist_dir: Directory the database files live in.
        client: An existing client to reuse; one is opened if omitted.

    Returns:
        A chromadb Collection configured for cosine distance.
    """
    client = client or get_client(persist_dir)
    return client.get_or_create_collection(name=collection_name, metadata=COLLECTION_METADATA)


def _chunk_text(chunk):
    """Reads the text off a docling DocChunk or a LangChain Document."""
    text = getattr(chunk, "text", None)
    if text is None:
        text = getattr(chunk, "page_content", None)
    if text is None:
        raise TypeError(f"Chunk has neither .text nor .page_content: {type(chunk)!r}")
    return text


def _chunk_source(chunk):
    """Best-effort origin filename for a chunk, used for metadata and ids."""
    meta = getattr(chunk, "meta", None)
    origin = getattr(meta, "origin", None)
    if origin is not None and getattr(origin, "filename", None):
        return origin.filename
    # LangChain Documents keep a plain dict of metadata.
    doc_meta = getattr(chunk, "metadata", None)
    if isinstance(doc_meta, dict):
        return str(doc_meta.get("source", "unknown"))
    return "unknown"


def _chunk_pages(chunk):
    """Page numbers a docling chunk spans, as a sorted comma-joined string."""
    meta = getattr(chunk, "meta", None)
    pages = set()
    for item in getattr(meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(page_no)
    return ",".join(str(p) for p in sorted(pages))


def _chunk_metadata(chunk, index):
    """
    Flattens chunk metadata into the scalar-only shape Chroma accepts.

    Chroma rejects lists and nested dicts in metadata, so headings and page
    numbers are collapsed into strings.
    """
    headings = getattr(getattr(chunk, "meta", None), "headings", None) or []
    return {
        "source": _chunk_source(chunk),
        "chunk_index": index,
        "headings": " > ".join(headings),
        "pages": _chunk_pages(chunk),
    }


def _chunk_id(chunk, seen):
    """
    Builds a deterministic id so re-ingesting an unchanged document upserts
    in place instead of creating duplicates.

    Args:
        chunk: The chunk to identify.
        seen: Mutable dict tracking how many times each digest has appeared,
            which disambiguates chunks whose text is byte-identical.

    Returns:
        A stable id string.
    """
    source = _chunk_source(chunk)
    digest = hashlib.sha256(f"{source}\x00{_chunk_text(chunk)}".encode("utf-8")).hexdigest()[:16]
    occurrence = seen.get(digest, 0)
    seen[digest] = occurrence + 1
    return f"{source}:{digest}:{occurrence}"


def store_embeddings(
    chunks,
    embeddings,
    collection_name=DEFAULT_COLLECTION,
    persist_dir=DEFAULT_PERSIST_DIR,
    replace_existing=True,
    batch_size=1000,
):
    """
    Stores precomputed embeddings, with their text and metadata, in ChromaDB.

    Args:
        chunks: The chunks the embeddings were computed from, in the same order.
        embeddings: One vector per chunk, as returned by `embedding()`.
        collection_name: Name of the Chroma collection to write to.
        persist_dir: Directory the database files live in.
        replace_existing: Delete this document's previous chunks first. Ids are
            derived from the chunk text, so edited text produces new ids and
            the superseded rows would otherwise linger forever.
        batch_size: Rows per add call, to stay under Chroma's max batch size.

    Returns:
        The Chroma collection, or None if the write failed.
    """
    try:
        chunks = list(chunks)  # chunker.chunk() returns a one-shot iterator
        embeddings = list(embeddings)

        if not chunks:
            print("No chunks to store.")
            return None
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings; they must match 1:1.")

        seen = {}
        ids = [_chunk_id(chunk, seen) for chunk in chunks]
        documents = [_chunk_text(chunk) for chunk in chunks]
        metadatas = [_chunk_metadata(chunk, index) for index, chunk in enumerate(chunks)]

        collection = get_collection(collection_name, persist_dir)

        if replace_existing:
            for source in {meta["source"] for meta in metadatas}:
                collection.delete(where={"source": source})

        print(f"Storing {len(ids)} embeddings in Chroma collection '{collection_name}'...")
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            collection.upsert(
                ids=ids[start:stop],
                embeddings=embeddings[start:stop],
                documents=documents[start:stop],
                metadatas=metadatas[start:stop],
            )

        print(f"Stored {collection.count()} total embeddings at {persist_dir}")
        return collection

    except Exception as e:
        print(f"An error occurred while storing embeddings: {e}")
        return None
