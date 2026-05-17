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
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_loader import load_pdf

from app.config.settings import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from app.utils.logger import logger


def split_documents(documents):

    logger.info(
        "document_chunking_started",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunked_documents = text_splitter.split_documents(documents)

    logger.info(
        "document_chunking_completed",
        total_chunks=len(chunked_documents)
    )

    logger.info(
        "sample_chunk_metadata",
        metadata=chunked_documents[0].metadata
    )

    logger.info(
        "sample_chunk_preview",
        content=chunked_documents[0].page_content[:500]
    )

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