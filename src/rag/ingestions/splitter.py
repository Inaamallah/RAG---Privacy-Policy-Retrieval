from transformers import AutoTokenizer
from .loader import loader

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


def _item_pages(item):
    """Page numbers a single document item was found on."""
    return {
        prov.page_no
        for prov in (getattr(item, "prov", None) or [])
        if getattr(prov, "page_no", None) is not None
    }


def _heading_pages(doc):
    """
    Maps each section heading's text to the set of pages it appears on.

    Args:
        doc: The docling document the chunks came from.

    Returns:
        A dict of heading text to page numbers.
    """
    pages = {}
    for item, _level in doc.iterate_items():
        if "header" not in str(getattr(item, "label", "")):
            continue
        text = (getattr(item, "text", "") or "").strip()
        if text:
            pages.setdefault(text, set()).update(_item_pages(item))
    return pages


def _drop_inherited_headings(chunks, doc):
    """
    Strips headings a chunk only inherited from an earlier page.

    HybridChunker labels a chunk with the last heading above it in the document
    tree. When docling's layout model misses a section header -- as it does for
    the mid-document headers in a two-column paper -- every later chunk keeps
    the previous section's title, so results and appendix text end up labelled
    with the introduction. A heading is kept only when the chunk overlaps the
    page the heading itself was found on; otherwise the chunk is left
    unlabelled, which is less misleading than a false `section:` line.

    Args:
        chunks: The chunks to correct, modified in place.
        doc: The docling document the chunks came from.

    Returns:
        The same chunks.
    """
    heading_pages = _heading_pages(doc)
    for chunk in chunks:
        chunk_pages = set()
        for item in getattr(chunk.meta, "doc_items", None) or []:
            chunk_pages |= _item_pages(item)
        kept = [
            heading
            for heading in (getattr(chunk.meta, "headings", None) or [])
            if heading_pages.get(heading.strip(), set()) & chunk_pages
        ]
        chunk.meta.headings = kept or None
    return chunks


def splitter(doc, chunk_size=512, chunk_overlap=200):
    """
    Splits a document into chunks using the HybridChunker.

    Args:
        doc: The document to be split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of document chunks. Headings a chunk merely inherited from an
        earlier page are removed, so `meta.headings` is either right or empty.
    """
    model_id = "BAAI/bge-m3"
    try:        
        if doc is None:
            doc = loader()  # Load the document if not provided

        tokenizer = AutoTokenizer.from_pretrained(model_id) # Downloads and initializes the tokenizer associated with BAAI/bge-m3
        hf_tokenizer = HuggingFaceTokenizer(tokenizer = tokenizer, max_tokens=chunk_size)  # Standardizes the Hugging Face tokenizer so Docling can call token-counting methods
        # Split the document into chunks
        chunker = HybridChunker(tokenizer=hf_tokenizer) 
        print("Splitting the document into chunks...")
        # chunk() hands back a one-shot iterator; materialise it so the heading
        # correction and the callers can each walk the same chunks.
        chunks = list(chunker.chunk(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        return _drop_inherited_headings(chunks, doc)

    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    document = loader()  # Load the document
    chunks = splitter(document)  # Split the document into chunks
    print(chunks)
