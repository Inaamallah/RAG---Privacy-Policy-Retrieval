"""Console entry point that serves the FastAPI backend.

    uv run rag-api
    uv run rag-api --port 8600 --reload

This is the shim `ui.py` used to be, pointed at uvicorn instead of Streamlit.
Uvicorn is handed an import string rather than the application object, because
that is what `--reload` needs to re-import the module on a change.
"""

import argparse

from .console import use_utf8_stdout

APP = "rag.api.app:app"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main():
    """
    Runs the API server. Returns a process exit code.
    """
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Serve the document Q&A API.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Interface to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Restart on source changes")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(APP, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
