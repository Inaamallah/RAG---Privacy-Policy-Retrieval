"""Streamlit chat interface over the already-ingested policy document.

    uv run rag-ui                          # or:
    uv run streamlit run src/rag/app.py

The document is fixed. There is no uploader, and retrieval is pinned to
`DOCUMENT` with a Chroma metadata filter, so the page can only ever answer out
of that one PDF even if the collection later holds others.

Ingestion stays in `uv run rag ingest`; this page is read-only over the vector
store. That is why it imports the retrieval half only and never pulls in
docling -- the app starts in about a second instead of loading converter
models it would never use.
"""

import logging
import warnings

# Match main.py: silence the import banners before transformers loads.
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import streamlit as st  # noqa: E402

from rag.console import use_utf8_stdout  # noqa: E402
from rag.generation.generator import DEFAULT_MODEL, generate_answer  # noqa: E402
from rag.retrieval.query_embedder import get_embedder  # noqa: E402
from rag.retrieval.retriever import DEFAULT_TOP_K, retrieve  # noqa: E402
from rag.vectorstore.chroma_store import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    get_collection,
)

# The one document this UI serves. It is matched against the `source` metadata
# written by `chroma_store._chunk_metadata`, which stores the file name only.
DOCUMENT = "policy_removed_removed.pdf"

GREETING = (
    f"Ask me anything about **{DOCUMENT}**. I answer only from that document, "
    "and I cite the page each claim came from."
)


@st.cache_resource(show_spinner=False)
def open_collection():
    """
    Opens the Chroma collection once and reuses it across reruns.

    Streamlit re-executes this file top to bottom on every interaction, so
    without the cache each message would reopen the database.

    Returns:
        The Chroma collection.
    """
    return get_collection(DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR)


@st.cache_resource(show_spinner=False)
def warm_embedder():
    """
    Loads bge-m3 once, up front.

    `embed_query` would load it lazily on the first question anyway; doing it
    here means the cost lands on a visible spinner at startup rather than
    looking like a slow first answer.

    Returns:
        The embedding model.
    """
    return get_embedder()


def document_rows(collection):
    """
    Counts stored chunks belonging to `DOCUMENT`.

    Args:
        collection: The Chroma collection to inspect.

    Returns:
        The number of matching rows, or 0 if the lookup failed.
    """
    try:
        found = collection.get(where={"source": DOCUMENT}, include=[])
        return len(found.get("ids") or [])
    except Exception:
        return 0


def answer_question(question, top_k):
    """
    Retrieves context for a question and has the LLM answer from it.

    Retrieval is filtered to `DOCUMENT`, so nothing outside that PDF can reach
    the model regardless of what else the collection holds.

    Args:
        question: The user's question.
        top_k: How many chunks to ground the answer on.

    Returns:
        A tuple of the answer text and the chunks it was grounded on.
    """
    chunks = retrieve(
        question,
        top_k=top_k,
        collection=open_collection(),
        where={"source": DOCUMENT},
    )
    if not chunks:
        return "I could not find that in the provided documents.", []
    return generate_answer(question, chunks, model=DEFAULT_MODEL), chunks


def render_sources(chunks):
    """
    Shows the excerpts an answer was grounded on, in a collapsed expander.

    Scores and chunk ids are shown here but never enter the prompt: this is
    for the reader, and `format_context` still keeps them from the model.

    Args:
        chunks: The list `retrieve()` returned.
    """
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} excerpt{'s' if len(chunks) != 1 else ''})"):
        for hit in chunks:
            meta = hit.get("metadata") or {}
            pages = meta.get("pages")
            where = f"p.{pages}" if pages else "page unknown"
            heading = meta.get("headings")
            st.markdown(
                f"**{meta.get('source', DOCUMENT)} — {where}**"
                + (f"  \n*{heading}*" if heading else "")
                + f"  \nmatch {hit['score']:.3f}"
            )
            st.text(hit["text"])
            st.divider()


def main():
    """Runs the chat page. Streamlit calls this on every rerun."""
    use_utf8_stdout()

    st.set_page_config(page_title="Document Q&A", page_icon="📄")
    st.title("📄 Document Q&A")
    st.caption(f"Grounded on {DOCUMENT} · {DEFAULT_MODEL} via Groq")

    with st.sidebar:
        st.header("Settings")
        st.markdown(f"**Document**  \n`{DOCUMENT}`")
        st.caption("Fixed by design — this app has no upload, so the answers always come from this one PDF.")
        st.markdown(f"**Model**  \n`{DEFAULT_MODEL}`")
        top_k = st.slider(
            "Excerpts per answer",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="How many retrieved chunks are given to the model as context.",
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Fail loudly and usefully rather than throwing a traceback into the page.
    try:
        collection = open_collection()
    except Exception as error:
        st.error(f"Could not open the vector store at `{DEFAULT_PERSIST_DIR}`.\n\n{error}")
        st.info("Run `uv run rag ingest --pdf src/rag/data/policy_removed_removed.pdf` first.")
        st.stop()

    rows = document_rows(collection)
    if not rows:
        st.error(f"No stored chunks for `{DOCUMENT}`.")
        st.info(
            "Ingest it first:\n\n"
            "```\nuv run rag ingest --pdf src/rag/data/policy_removed_removed.pdf --replace-existing\n```"
        )
        st.stop()

    with st.spinner("Loading the embedding model..."):
        warm_embedder()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info(GREETING)

    # Replay the conversation so far; Streamlit rebuilds the page each rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("chunks") or [])

    # chat_input stays pinned at the bottom and re-arms itself after every
    # answer, so the conversation loops without any extra wiring.
    question = st.chat_input(f"Ask about {DOCUMENT}...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                text, chunks = answer_question(question, top_k)
            except RuntimeError as error:
                # generate_answer raises this when GROQ_API_KEY is missing.
                text, chunks = f"⚠️ {error}", []
            except Exception as error:
                text, chunks = f"⚠️ Something went wrong: {error}", []
        st.markdown(text)
        render_sources(chunks)

    st.session_state.messages.append({"role": "assistant", "content": text, "chunks": chunks})


main()
