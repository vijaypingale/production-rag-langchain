from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingestion_service import (
    process_uploaded_pdf,
    process_uploaded_file_in_memory
)

from app.utils.logger import logger


router = APIRouter(
    tags=["Document Ingestion"]
)


@router.get("/ingest/test")
async def test_ingestion_api():

    return {
        "message": "Ingestion API is working successfully"
    }


@router.post("/ingest/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    logger.info("UPLOAD API HIT")
    try:
        
        # =====================================================
        # Validate file type
        # =====================================================

        if not file.filename.endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        logger.info(
            "pdf_upload_received",
            file_name=file.filename
        )

        # =====================================================
        # Trigger ingestion pipeline in memory
        # =====================================================

        result = await process_uploaded_file_in_memory(file)

        return result

    except Exception as ex:

        logger.error(
            "pdf_upload_failed",
            error=str(ex)
        )

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )