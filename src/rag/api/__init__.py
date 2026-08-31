"""HTTP layer over the query half of the RAG pipeline.

`app.py` builds the FastAPI application, `service.py` holds the retrieval and
generation calls it serves, `schemas.py` the request and response shapes and
`config.py` the one document the whole API is pinned to.

Nothing here ingests. Like the page it replaces, this is read-only over the
vector store: it imports the retrieval half only and never pulls in docling,
so the server starts in about a second instead of loading converter models it
would never use.
"""
