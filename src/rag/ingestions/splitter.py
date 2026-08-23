from transformers import AutoTokenizer

from .loader import loader
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

def splitter(doc, chunk_size=512, chunk_overlap=200):
    """
    Splits a document into chunks using the HybridChunker.

    Args:
        doc: The document to be split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of document chunks.
    """
    model_id = "BAAI/bge-m3"
    try:        
        tokenizer = AutoTokenizer.from_pretrained(model_id) # Downloads and initializes the tokenizer associated with BAAI/bge-m3
        hf_tokenizer = HuggingFaceTokenizer(tokenizer = tokenizer, max_tokens=chunk_size)  # Standardizes the Hugging Face tokenizer so Docling can call token-counting methods
        # Split the document into chunks
        chunker = HybridChunker(tokenizer=hf_tokenizer) 
        print("Splitting the document into chunks...")
        chunks = chunker.chunk(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks

    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        return []

