"""
Ingestion Service Layer

Responsibilities:
-----------------
- Trigger PDF ingestion
- Trigger chunking pipeline
- Return ingestion summary

Enterprise Principle:
---------------------
Service layer orchestrates workflows.
Business logic stays outside API routes.

Flow:
------
API Request → process_uploaded_file_in_memory()
    ↓                       ↓
Upload           Load PDF (temp in-memory) → Chunk → Response

Legacy/optional persistence:
----------------------------
- save_uploaded_file() still exists for explicit disk persistence
  but is not used by the current upload API route.
"""

from pathlib import Path
import shutil
import tempfile
import time

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_splitter import split_documents

from app.utils.logger import logger


# ============================================================================
# Configuration
# ============================================================================

# Directory where uploaded PDFs are stored on disk when persistence is required
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

    Notes:
        This helper persists uploads to disk for legacy or audit use.
        The current API upload path prefers in-memory processing.
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


async def process_uploaded_file_in_memory(upload_file):
    """
    Process uploaded PDF entirely in-memory (uses a temporary file for PyMuPDF compatibility).

    This function does NOT persist the PDF in the project's data folders.
    It writes a short-lived temp file, loads it with existing loaders, then removes the temp file.

    Args:
        upload_file: FastAPI UploadFile

    Returns:
        dict: ingestion summary (same shape as `process_uploaded_pdf`)
    """

    logger.info(
        "pdf_processing_in_memory_started",
        file_name=upload_file.filename,
        content_type=getattr(upload_file, "content_type", None)
    )

    overall_start = time.perf_counter()
    try:
        # Read bytes from the upload (async)
        read_start = time.perf_counter()
        file_bytes = await upload_file.read()
        read_time = time.perf_counter() - read_start

        logger.info(
            "file_bytes_read_from_memory",
            file_name=upload_file.filename,
            size_bytes=len(file_bytes),
            read_duration_seconds=round(read_time, 4)
        )

        # Write to a platform-safe temporary file (deleted after processing)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = Path(tmp.name)
        try:
            tmp.write(file_bytes)
            tmp.flush()
        finally:
            tmp.close()

        logger.info(
            "temp_pdf_written",
            temp_path=str(tmp_path),
            temp_file_size_bytes=len(file_bytes)
        )

        # Load PDF and measure load duration
        load_start = time.perf_counter()
        documents = load_pdf(str(tmp_path))
        load_time = time.perf_counter() - load_start

        logger.info(
            "pdf_loaded_from_memory",
            file_name=upload_file.filename,
            total_documents=len(documents),
            load_duration_seconds=round(load_time, 4)
        )

        # Split documents into semantic chunks and measure chunking duration
        chunk_start = time.perf_counter()
        chunks = split_documents(documents)
        chunk_time = time.perf_counter() - chunk_start

        logger.info(
            "documents_split_into_chunks",
            file_name=upload_file.filename,
            total_chunks=len(chunks),
            chunk_duration_seconds=round(chunk_time, 4)
        )

        cleanup_start = time.perf_counter()
        try:
            tmp_path.unlink()
            cleanup_time = time.perf_counter() - cleanup_start
            logger.info(
                "temp_file_deleted",
                temp_path=str(tmp_path),
                cleanup_duration_seconds=round(cleanup_time, 4)
            )
        except Exception as ex:
            cleanup_time = time.perf_counter() - cleanup_start
            logger.warning(
                "temp_file_delete_failed",
                temp_path=str(tmp_path),
                error=str(ex),
                cleanup_duration_seconds=round(cleanup_time, 4)
            )

        total_time = time.perf_counter() - overall_start
        logger.info(
            "pdf_processing_in_memory_completed",
            file_name=upload_file.filename,
            total_documents=len(documents),
            total_chunks=len(chunks),
            total_duration_seconds=round(total_time, 4),
            read_duration_seconds=round(read_time, 4),
            load_duration_seconds=round(load_time, 4),
            chunk_duration_seconds=round(chunk_time, 4),
            cleanup_duration_seconds=round(cleanup_time, 4)
        )

        return {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(chunks)
        }

    except Exception as ex:
        logger.error(
            "pdf_processing_in_memory_failed",
            file_name=upload_file.filename,
            error=str(ex)
        )
        # Attempt cleanup if temp exists
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


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