import warnings
import logging

warnings.filterwarnings("ignore")

# Hide INFO and lower-level logs
logging.disable(logging.INFO)
from .splitter import splitter
from .loader import loader

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
    return chunks # Return only the first 5 chunks for demonstration purposes

chunks = pipeline()
print(f"Number of chunks created: {len(chunks)}")
print(f"First 5 chunks: {chunks[:5]}")