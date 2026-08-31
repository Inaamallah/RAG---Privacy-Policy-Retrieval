"""The FastAPI application.

    uv run rag-api                         # or:
    uv run uvicorn rag.api.app:app --reload

Three routes, all the React client needs:

    GET  /api/health   can the store answer, and what about
    POST /api/ask      a question in, an answer plus its excerpts out
    GET  /api/document the pinned document and model, for the sidebar

The document is fixed. There is no upload route, because there was no
uploader: ingestion stays in `uv run rag ingest` and this process is read-only
over the vector store. Adding a write route here would also mean importing
docling, which is what keeps startup at about a second.
"""

import logging
import threading
import warnings

# Match main.py: silence the import banners before transformers loads.
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import service  # noqa: E402
from .config import DEV_ORIGINS, DOCUMENT, FRONTEND_DIST, MAX_TOP_K, MIN_TOP_K  # noqa: E402
from .schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse  # noqa: E402
from ..console import use_utf8_stdout  # noqa: E402
from ..generation.generator import DEFAULT_MODEL  # noqa: E402
from ..retrieval.retriever import DEFAULT_TOP_K  # noqa: E402


def _warm_up():
    """Loads the embedding model in the background, ignoring failures.

    A failure here is not fatal: `ask` loads the model itself if it has to,
    and reports the error to the request that triggered it rather than to a
    startup thread nobody is watching.
    """
    try:
        service.warm_embedder()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app):
    """Warms the embedder off the startup path.

    Loading bge-m3 takes several seconds. Doing it in a thread means the
    server binds its port immediately and `/api/health` can report
    `embedder_ready` while it finishes, instead of the first question paying
    for it.
    """
    use_utf8_stdout()
    threading.Thread(target=_warm_up, name="warm-embedder", daemon=True).start()
    yield


def create_app():
    """
    Builds the application.

    Returns:
        The FastAPI instance.
    """
    app = FastAPI(
        title="Document Q&A",
        description=f"Answers questions grounded on {DOCUMENT}.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def on_invalid_request(request: Request, error: RequestValidationError):
        """Flattens FastAPI's list of field errors into one sentence.

        Every other failure answers with `{"detail": "<message>"}`; without
        this, a rejected field would answer with a list under the same key and
        the client would have two error shapes to render.
        """
        first = (error.errors() or [{}])[0]
        field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = first.get("msg", "Invalid request.")
        return JSONResponse(
            status_code=400,
            content={"detail": f"{field}: {message}" if field else message},
        )

    @app.get("/api/health", response_model=HealthResponse)
    def get_health():
        """Whether the store can answer, and what it would answer about."""
        return service.health()

    @app.get("/api/document", response_model=HealthResponse)
    def get_document():
        """Alias of /api/health, named for how the client reads it."""
        return service.health()

    @app.post(
        "/api/ask",
        response_model=AskResponse,
        responses={
            400: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def post_ask(request: AskRequest):
        """
        Answers a question from the pinned document.

        The endpoint is sync on purpose: embedding and the Groq call both
        block, so FastAPI runs this in its threadpool and one slow answer
        cannot stall the event loop.
        """
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="The question is empty.")

        try:
            service.require_ready()
            answer, chunks = service.answer_question(question, request.top_k)
        except service.StoreUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            # generate_answer raises this when GROQ_API_KEY is missing.
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502, detail=f"The answer service failed: {error}"
            ) from error

        return {"answer": answer, "chunks": chunks}

    # Serve the built React page from the same origin when it exists, so a
    # production run is one process and needs no CORS. In development Vite
    # serves the page instead and proxies /api here.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


app = create_app()

__all__ = ["app", "create_app", "DEFAULT_MODEL", "DEFAULT_TOP_K", "MIN_TOP_K", "MAX_TOP_K"]
