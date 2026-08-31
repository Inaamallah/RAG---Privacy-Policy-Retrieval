"""Settings for the HTTP layer.

The document pin lives here rather than inside a route, so the value the
retrieval filter uses and the value the page displays are the same string.
"""

import os
from pathlib import Path

# The one document this API serves. It is matched against the `source`
# metadata written by `chroma_store._chunk_metadata`, which stores the file
# name only. Serving a different PDF means ingesting it and changing this --
# the pin is enforced in the Chroma query, not merely displayed.
DOCUMENT = os.environ.get("RAG_DOCUMENT", "policy_removed_removed.pdf")

# Origins the browser may call the API from. The Vite dev server proxies /api
# to this process, so same-origin requests are the normal path; these cover a
# page loaded straight off the dev server's own origin.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# <repo root>/frontend/dist -- config.py sits at <root>/src/rag/api/.
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

# Bounds on the client-supplied excerpt count, matching the old sidebar slider.
MIN_TOP_K = 1
MAX_TOP_K = 10
