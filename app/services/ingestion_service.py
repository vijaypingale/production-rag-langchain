"""
Ingestion Service Layer

Responsibilities:
-----------------
- Save uploaded files
- Trigger PDF ingestion
- Trigger chunking pipeline
- Return ingestion summary

Enterprise Principle:
---------------------
Service layer orchestrates workflows.
Business logic stays outside API routes.

Flow:
------
API Request → save_uploaded_file() → process_uploaded_pdf()
    ↓              ↓                       ↓
Upload       Save to disk           Load PDF → Chunk → Response
"""

from pathlib import Path
import shutil

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_splitter import split_documents

from app.utils.logger import logger


# ============================================================================
# Configuration
# ============================================================================

# Directory where uploaded PDFs are stored on disk
DOCUMENTS_DIR = "data/documents"


# ============================================================================
# File Persistence Layer
# ============================================================================

def save_uploaded_file(upload_file):
    """
    Save uploaded file from memory to disk.
    
    Process:
    --------
    1. Create destination path: data/documents/{filename}
    2. Open file in write-binary mode
    3. Copy file from memory buffer to disk (streaming)
    4. Log successful save
    5. Return file path for downstream processing
    
    Args:
        upload_file: FastAPI UploadFile object with .filename and .file
        
    Returns:
        str: Full path to saved file (e.g., "data/documents/sample.pdf")
    """

    # Construct destination path: data/documents/{filename}
    upload_path = Path(DOCUMENTS_DIR) / upload_file.filename

    # Stream copy from memory → disk (efficient for large files)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    # Log success with filename for audit trail
    logger.info(
        "file_uploaded_successfully",
        file_name=upload_file.filename
    )

    # Return path for next step in pipeline
    return str(upload_path)


# ============================================================================
# RAG Pipeline Orchestration
# ============================================================================

def process_uploaded_pdf(file_path: str):
    """
    Orchestrate complete PDF ingestion & chunking pipeline.
    
    Pipeline Steps:
    ---------------
    1. Log pipeline start with file path
    2. Load PDF → Extract text into LangChain Documents (one per page)
    3. Split Documents → Break into semantic chunks (1000 chars, 200 char overlap)
    4. Log pipeline completion with statistics
    5. Return summary with document & chunk counts
    
    Args:
        file_path (str): Full path to PDF file (e.g., "data/documents/sample.pdf")
        
    Returns:
        dict: {
            "status": "success",
            "total_documents": N,    # Number of pages/documents
            "total_chunks": M        # Number of semantic chunks
        }
        
    Flow:
    -----
    PDF File → load_pdf() → LangChain Documents (pages)
             ↓ split_documents() → LangChain Documents (chunks)
             ↓ Return results
    """

    # Log pipeline initialization for observability
    logger.info(
        "pdf_ingestion_pipeline_started",
        file_path=file_path
    )

    # Step 1: Load PDF using LangChain's PyMuPDFLoader
    # Also extracts metadata using PyMuPDF (fitz) for diagnostics
    # Returns: List[Document] with one Document per page
    documents = load_pdf(file_path)

    # Step 2: Split documents into semantic chunks
    # Uses RecursiveCharacterTextSplitter (1000 char chunks, 200 char overlap)
    # Returns: List[Document] with smaller chunks for efficient retrieval
    chunks = split_documents(documents)

    # Log pipeline completion with metrics
    logger.info(
        "pdf_ingestion_pipeline_completed",
        total_documents=len(documents),  # Total pages extracted
        total_chunks=len(chunks)         # Total chunks created
    )

    # Return structured response for API client
    return {
        "status": "success",
        "total_documents": len(documents),  # Pages/Documents from PDF
        "total_chunks": len(chunks)         # Semantic chunks for RAG
    }