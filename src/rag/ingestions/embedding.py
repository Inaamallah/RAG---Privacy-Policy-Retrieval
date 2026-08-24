from langchain_huggingface import HuggingFaceEmbeddings
def embedding(chunks):
    """
    Embeds the chunks using the BAAI/bge-m3 model.
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return embeddings.embed_documents([chunk.page_content for chunk in chunks])