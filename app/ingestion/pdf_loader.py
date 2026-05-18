"""
PDF Ingestion Module

Responsibilities:
-----------------
- Load PDF documents
- Extract metadata
- Validate ingestion
- Return LangChain Document objects

This module represents the first layer
of the enterprise RAG pipeline.
"""

from pathlib import Path
import fitz

from langchain_community.document_loaders import PyMuPDFLoader

from app.config.settings import PDF_PATH
from app.utils.logger import logger


def extract_pdf_metadata(pdf_path: str):
    """
    Extract PDF metadata using PyMuPDF (fitz).
    
    Purpose: Diagnostic information about PDF before processing
    
    Extracts:
    - Total page count
    - Image detection (if contains images)
    - Encryption status
    - PDF metadata (title, author, etc.)
    
    Args:
        pdf_path (str): Full path to PDF file
    """

    pdf_document = fitz.open(pdf_path)

    metadata = pdf_document.metadata

    total_pages = pdf_document.page_count

    has_images = False

    # Scan through pages to detect images
    for page_num in range(total_pages):

        page = pdf_document.load_page(page_num)

        if page.get_images():
            has_images = True
            break

    # Log metadata details for audit trail
    logger.info(
        "pdf_metadata_extracted",
        file_name=Path(pdf_path).name,
        total_pages=total_pages,
        contains_images=has_images,
        encrypted=pdf_document.is_encrypted,
        metadata=metadata
    )

    pdf_document.close()


def load_pdf(pdf_path: str):
    """
    Load PDF and return LangChain Document objects.
    
    Process:
    --------
    1. Validate file exists
    2. Extract metadata using PyMuPDF (fitz) for diagnostics
    3. Create LangChain PyMuPDFLoader
    4. Load PDF → Returns standardized Document objects
    5. Log results and return
    
    Args:
        pdf_path (str): Full path to PDF file
        
    Returns:
        List[LangChain Document]: One Document per page with metadata
        
    Why Two Libraries:
    - PyMuPDF (fitz): Direct access to PDF internals (metadata, images, encryption)
    - PyMuPDFLoader (LangChain): Standardized Document extraction for RAG pipeline
    """

    path = Path(pdf_path)

    if not path.exists():

        logger.error(
            "pdf_file_not_found",
            file_path=pdf_path
        )

        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(
        "pdf_loading_started",
        file_name=path.name
    )

    # Step 1: Extract metadata using PyMuPDF (fitz)
    # This gives us diagnostic info: page count, images, encryption
    extract_pdf_metadata(pdf_path)

    # Step 2: Load PDF using LangChain's PyMuPDFLoader
    # PyMuPDFLoader uses PyMuPDF internally but returns standardized Documents
    loader = PyMuPDFLoader(str(path))

    # Returns: List[Document] - one Document per page
    documents = loader.load()

    logger.info(
        "pdf_loaded_successfully",
        total_documents=len(documents),
        sample_metadata=documents[0].metadata if documents else None
    )

    if documents:
        logger.info(
            "sample_document_preview",
            content=documents[0].page_content[:500]
        )

    return documents


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

#     load_pdf(PDF_PATH)