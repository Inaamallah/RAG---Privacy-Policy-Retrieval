"""Request and response shapes for the API.

These are the contract the React client is written against. What a response
carries is a deliberate choice, not a dump of the retrieval dict: scores and
chunk ids are shown to the *reader* in the sources panel, exactly as the old
expander did, but `generation.prompts.format_context` still keeps them out of
the prompt, so the model never sees them.
"""

from pydantic import BaseModel, Field

from .config import MAX_TOP_K, MIN_TOP_K
from ..retrieval.retriever import DEFAULT_TOP_K


class AskRequest(BaseModel):
    """A question to answer from the pinned document."""

    question: str = Field(min_length=1, max_length=2000, description="The user's question.")
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="How many retrieved excerpts to ground the answer on.",
    )


class ChunkMetadata(BaseModel):
    """The scalar metadata Chroma stores alongside a chunk."""

    source: str = ""
    pages: str = ""
    headings: str = ""
    chunk_index: int | None = None


class Chunk(BaseModel):
    """One retrieved excerpt, as shown in the sources panel."""

    id: str
    text: str
    score: float
    metadata: ChunkMetadata


class AskResponse(BaseModel):
    """An answer and the excerpts it was grounded on."""

    answer: str
    chunks: list[Chunk]


class HealthResponse(BaseModel):
    """Whether the API can answer questions, and what it would answer about.

    `ready` is the single thing the client branches on. When it is false,
    `detail` says what to do about it in the same words the old page used.
    """

    ready: bool
    document: str
    model: str
    chunks: int
    embedder_ready: bool
    default_top_k: int = DEFAULT_TOP_K
    min_top_k: int = MIN_TOP_K
    max_top_k: int = MAX_TOP_K
    detail: str | None = None


class ErrorResponse(BaseModel):
    """The body every failing request returns."""

    detail: str
