"""Send retrieved chunks to a Groq-hosted LLM and get an answer back.

The credential is read from GROQ_API_KEY, via a .env file if one is present.
It is never logged, never placed in a prompt, and never returned to the caller.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from .prompts import SYSTEM_PROMPT, build_user_message

load_dotenv()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
NO_CONTEXT_ANSWER = "I could not find that in the provided documents."

_client = None


def get_client(api_key=None):
    """
    Returns the shared Groq client, building it on first use.

    Args:
        api_key: Key to use; taken from GROQ_API_KEY when omitted.

    Returns:
        A Groq client.

    Raises:
        RuntimeError: If no API key is configured.
    """
    global _client
    if api_key:
        return Groq(api_key=api_key)
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Put it in your environment or in a .env file.")
        _client = Groq(api_key=key)
    return _client


def generate_answer(
    query,
    chunks,
    model=DEFAULT_MODEL,
    temperature=0.0,
    max_tokens=1024,
    client=None,
    api_key=None,
):
    """
    Answers a question from the retrieved chunks alone.

    Args:
        query: The user's question.
        chunks: The list `retrieve()` returned.
        model: Groq model id.
        temperature: Sampling temperature; kept at 0 so answers stay tied to
            the excerpts rather than invented.
        max_tokens: Cap on the generated answer length.
        client: An existing Groq client to reuse.
        api_key: Key to use instead of the environment one.

    Returns:
        The answer text. Returns the standard "not found" line, without
        calling the model, when there are no chunks to reason over.

    Raises:
        ValueError: If the query is empty.
        RuntimeError: If no API key is configured.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("The query is empty.")
    if not chunks:
        return NO_CONTEXT_ANSWER

    client = client or get_client(api_key)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, chunks)},
        ],
    )
    return (completion.choices[0].message.content or "").strip()
