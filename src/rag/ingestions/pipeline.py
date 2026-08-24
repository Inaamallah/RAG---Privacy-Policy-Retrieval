import warnings
import logging

warnings.filterwarnings("ignore")

# Hide INFO and lower-level logs
logging.disable(logging.INFO)
from .splitter import splitter
from .loader import loader
from .embedding import embedding

def pipeline():
    """
    Executes the document loading and splitting pipeline.

    Returns:
        A list of document chunks.
    """
    # Load the document
    document = loader()
    if document is None:
        print("Failed to load the document.")
        return []

    # Split the document into chunks
    chunks = splitter(document)
    embeddings = embedding(chunks)
    return embeddings

chunks = pipeline()
