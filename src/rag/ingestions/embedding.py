from langchain_huggingface import HuggingFaceEmbeddings

# Queries must be embedded with this same model, see rag.retrieval.query_embedder.
MODEL_ID = "BAAI/bge-m3"


def _chunk_text(chunk):
    """Reads the text off a docling DocChunk or a LangChain Document."""
    text = getattr(chunk, "text", None)
    if text is None:
        text = getattr(chunk, "page_content", None)
    if text is None:
        raise TypeError(f"Chunk has neither .text nor .page_content: {type(chunk)!r}")
    return text


def embedding(chunks):
    """
    Embeds the chunks using the BAAI/bge-m3 model.

    Args:
        chunks: Chunks from `splitter()`, or any objects carrying text.

    Returns:
        One vector per chunk, in the same order.
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_ID)
    return embeddings.embed_documents([_chunk_text(chunk) for chunk in chunks])
