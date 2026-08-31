"""Console entry point that starts the Streamlit chat UI.

    uv run rag-ui
    uv run rag-ui --server.port 8600

A Streamlit page has to be launched by the `streamlit run` machinery rather
than executed directly, so this shim rewrites argv and hands over to
Streamlit's own CLI. Extra arguments are passed through untouched.
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "app.py"


def main():
    """
    Runs `streamlit run src/rag/app.py`, forwarding any extra arguments.

    Returns:
        Streamlit's exit status. In practice Streamlit calls sys.exit itself.
    """
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    return streamlit_cli.main()


if __name__ == "__main__":
    main()
