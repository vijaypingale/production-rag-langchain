"""
Ingestion Service Layer

Responsibilities:
-----------------
- Trigger PDF ingestion
- Trigger chunking pipeline
- Trigger embedding generation
- Return ingestion summary

Enterprise Principle:
---------------------
Service layer orchestrates workflows.
Business logic stays outside API routes.

Flow:
------
API Request → process_uploaded_file_in_memory()
    ↓
Load PDF → Chunk Documents → Generate Embeddings → Response

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
from app.embeddings.embedding_generator import generate_embeddings

from app.utils.logger import logger


# ============================================================================
# Configuration
# ============================================================================

DOCUMENTS_DIR = "data/documents"


# ============================================================================
# File Persistence Layer
# ============================================================================

def save_uploaded_file(upload_file):
    """
    Save uploaded file from memory to disk.
    """

    upload_path = Path(DOCUMENTS_DIR) / upload_file.filename

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    logger.info(
        "file_uploaded_successfully",
        file_name=upload_file.filename
    )

    return str(upload_path)


# ============================================================================
# In-Memory Upload Processing Pipeline
# ============================================================================

async def process_uploaded_file_in_memory(upload_file):
    """
    Process uploaded PDF entirely in-memory.

    Pipeline:
    ---------
    Upload
        ↓
    PDF Loading
        ↓
    Chunking
        ↓
    Embedding Generation
        ↓
    Response
    """

    logger.info(
        "pdf_processing_in_memory_started",
        file_name=upload_file.filename,
        content_type=getattr(upload_file, "content_type", None)
    )

    overall_start = time.perf_counter()

    try:

        # =====================================================================
        # Step 1: Read uploaded file bytes
        # =====================================================================

        read_start = time.perf_counter()

        file_bytes = await upload_file.read()

        read_time = time.perf_counter() - read_start

        logger.info(
            "file_bytes_read_from_memory",
            file_name=upload_file.filename,
            size_bytes=len(file_bytes),
            read_duration_seconds=round(read_time, 4)
        )

        # =====================================================================
        # Step 2: Write temporary PDF file
        # =====================================================================

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

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

        # =====================================================================
        # Step 3: Load PDF documents
        # =====================================================================

        load_start = time.perf_counter()

        documents = load_pdf(str(tmp_path))

        load_time = time.perf_counter() - load_start

        logger.info(
            "pdf_loaded_from_memory",
            file_name=upload_file.filename,
            total_documents=len(documents),
            load_duration_seconds=round(load_time, 4)
        )

        # =====================================================================
        # Step 4: Split documents into semantic chunks
        # =====================================================================

        chunk_start = time.perf_counter()

        chunks = split_documents(documents)

        chunk_time = time.perf_counter() - chunk_start

        logger.info(
            "documents_split_into_chunks",
            file_name=upload_file.filename,
            total_chunks=len(chunks),
            chunk_duration_seconds=round(chunk_time, 4)
        )

        # =====================================================================
        # Step 5: Generate semantic embeddings
        # =====================================================================

        embedding_start = time.perf_counter()

        embeddings = generate_embeddings(chunks)

        embedding_time = time.perf_counter() - embedding_start

        logger.info(
            "embeddings_generated_successfully",

            # File Context
            file_name=upload_file.filename,

            # Embedding Statistics
            total_embeddings=len(embeddings),

            # OpenAI Embedding Vector Dimension
            embedding_dimension=len(embeddings[0]),

            # Validate Chunk-to-Embedding Mapping
            source_chunks=len(chunks),

            # Sample Vector Preview
            sample_embedding_preview=embeddings[0][:5],

            # Performance Metrics
            embedding_duration_seconds=round(embedding_time, 4)
        )

        # =====================================================================
        # Step 6: Cleanup temporary file
        # =====================================================================

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

        # =====================================================================
        # Final Pipeline Completion Log
        # =====================================================================

        total_time = time.perf_counter() - overall_start

        logger.info(
            "pdf_processing_in_memory_completed",
            file_name=upload_file.filename,
            total_documents=len(documents),
            total_chunks=len(chunks),
            total_embeddings=len(embeddings),
            total_duration_seconds=round(total_time, 4),
            read_duration_seconds=round(read_time, 4),
            load_duration_seconds=round(load_time, 4),
            chunk_duration_seconds=round(chunk_time, 4),
            embedding_duration_seconds=round(embedding_time, 4),
            cleanup_duration_seconds=round(cleanup_time, 4)
        )

        return {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings)
        }

    except Exception as ex:

        logger.error(
            "pdf_processing_in_memory_failed",
            file_name=upload_file.filename,
            error=str(ex)
        )

        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()

        except Exception:
            pass

        raise


# ============================================================================
# Legacy Disk-Based Processing Flow
# ============================================================================

def process_uploaded_pdf(file_path: str):
    """
    Legacy disk-based ingestion pipeline.
    """

    logger.info(
        "pdf_ingestion_pipeline_started",
        file_path=file_path
    )

    documents = load_pdf(file_path)

    chunks = split_documents(documents)

    embeddings = generate_embeddings(chunks)

    logger.info(
        "pdf_ingestion_pipeline_completed",
        total_documents=len(documents),
        total_chunks=len(chunks),
        total_embeddings=len(embeddings)
    )

    return {
        "status": "success",
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings)
    }