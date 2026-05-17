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
"""

from pathlib import Path
import shutil

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_splitter import split_documents

from app.utils.logger import logger


DOCUMENTS_DIR = "data/documents"


def save_uploaded_file(upload_file):

    upload_path = Path(DOCUMENTS_DIR) / upload_file.filename

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    logger.info(
        "file_uploaded_successfully",
        file_name=upload_file.filename
    )

    return str(upload_path)


def process_uploaded_pdf(file_path: str):

    logger.info(
        "pdf_ingestion_pipeline_started",
        file_path=file_path
    )

    documents = load_pdf(file_path)

    chunks = split_documents(documents)

    logger.info(
        "pdf_ingestion_pipeline_completed",
        total_documents=len(documents),
        total_chunks=len(chunks)
    )

    return {
        "status": "success",
        "total_documents": len(documents),
        "total_chunks": len(chunks)
    }