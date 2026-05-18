"""
Document Chunking Module

Responsibilities:
-----------------
- Split large documents into semantic chunks
- Preserve contextual overlap
- Prepare retrieval-ready documents

This module directly impacts:
- retrieval quality
- semantic relevance
- hallucination reduction

Why Chunking Matters:
---------------------
Large documents (100+ pages) can't fit in LLM context.
Break them into 1000-char chunks with 200-char overlap.
This preserves context while enabling efficient retrieval.

Example:
--------
Original PDF: "Page 1 talks about X. Page 2 discusses Y."
After chunking:
  Chunk 1: "X. Page 2 discusses Y."  (1000 chars)
  Chunk 2: "discusses Y. Page 3..."  (1000 chars, 200 overlap)
                    ^^^^^^^^
                (Context preserved via overlap)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_loader import load_pdf

from app.config.settings import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from app.utils.logger import logger


# ============================================================================
# Document Chunking Function
# ============================================================================

def split_documents(documents):
    """
    Split LangChain Documents into semantic chunks.
    
    Why RecursiveCharacterTextSplitter:
    - Splits recursively: paragraphs → sentences → words
    - Preserves logical boundaries (not arbitrary splits)
    - Adds overlap for context preservation
    
    Configuration:
    - chunk_size: 1000 chars per chunk
    - chunk_overlap: 200 chars overlap between chunks
    - length_function: Use standard len() for character count
    
    Args:
        documents: List[LangChain Document] from pdf_loader.py
                  Each Document has: page_content (text) + metadata
                  
    Returns:
        List[LangChain Document]: Semantic chunks ready for:
        - Embedding generation
        - Vector DB storage (FAISS, Pinecone)
        - Similarity search in retrieval
        
    Impact on RAG:
    - Better chunks → Better retrieval
    - Overlap → No context loss at boundaries
    - Logging → Full observability
    """

    # Log pipeline start with configuration
    logger.info(
        "document_chunking_started",
        chunk_size=CHUNK_SIZE,           # 1000 chars
        chunk_overlap=CHUNK_OVERLAP      # 200 chars
    )

    # Create splitter with semantic boundary detection
    # RecursiveCharacterTextSplitter: smart, not naive splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,              # Max 1000 characters per chunk
        chunk_overlap=CHUNK_OVERLAP,        # Preserve 200 chars context
        length_function=len                 # Count characters
    )

    # Apply splitter to all documents
    # Converts: List[Document(full_page)] → List[Document(chunks)]
    chunked_documents = text_splitter.split_documents(documents)

    # Log completion with chunk statistics
    logger.info(
        "document_chunking_completed",
        total_chunks=len(chunked_documents)
    )

    # Log sample metadata for debugging/audit
    # Verify metadata preserved through chunking
    logger.info(
        "sample_chunk_metadata",
        metadata=chunked_documents[0].metadata if chunked_documents else None
    )

    # Log sample chunk content (first 500 chars) for validation
    # Ensure chunking didn't corrupt content
    logger.info(
        "sample_chunk_preview",
        content=chunked_documents[0].page_content[:500] if chunked_documents else ""
    )

    # Return chunks ready for downstream processing
    # Next step: Embedding generation → Vector DB storage
    return chunked_documents


# ============================================================================
# NOTE:
# ----------------------------------------------------------------------------
# Previously this module was executed directly using:
#
#     python -m app.ingestion.pdf_loader
#
# or
#
#     python -m app.ingestion.text_splitter
#
# That approach was useful during the initial development and debugging phase
# to independently validate:
# - PDF ingestion
# - metadata extraction
# - document chunking
# - logging
#
# In the enterprise FastAPI architecture, execution flow changes to:
#
# Client/UI
#     ↓
# FastAPI Endpoint
#     ↓
# Service Layer
#     ↓
# Reusable Ingestion Modules
#
# Therefore:
# - APIs trigger execution
# - service layer orchestrates workflow
# - ingestion modules become reusable components
#
# We are intentionally keeping the old standalone execution block commented
# for:
# - future debugging
# - local testing
# - development reference
# - independent module validation
#
# This follows enterprise backend engineering practices where:
# - business logic remains modular
# - APIs remain thin
# - services coordinate workflows
# - modules stay independently testable
# ============================================================================

# if __name__ == "__main__":

#     documents = load_pdf(PDF_PATH)

#     chunks = split_documents(documents)