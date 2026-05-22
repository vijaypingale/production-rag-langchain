"""
Embedding Generation Module

Responsibilities:
-----------------
- Generate embeddings from chunked documents
- Prepare vectors for FAISS indexing
- Maintain provider abstraction

Enterprise Principles:
----------------------
- provider abstraction
- reusable embedding layer
- centralized configuration
- structured observability
"""

from app.utils.logger import logger

from app.utils.openai_client import get_embedding_model


def generate_embeddings(chunks):

    logger.info(
        "embedding_generation_started",
        embedding_model="text-embedding-3-small",
        total_chunks=len(chunks)
    )
    embedding_model = get_embedding_model()

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(texts)

    logger.info(
        "embedding_generation_completed",

        # Total vectors generated
        total_embeddings=len(embeddings),

        # OpenAI embedding vector dimension
        embedding_dimension=len(embeddings[0]),

        # Validate vector data exists
        sample_embedding_preview=embeddings[0][:5],

        # Validate chunk to embedding mapping
        source_chunks=len(chunks)
    )

    return embeddings