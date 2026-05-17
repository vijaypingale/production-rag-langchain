from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingestion_service import (
    save_uploaded_file,
    process_uploaded_pdf
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
        # Save uploaded file
        # =====================================================

        saved_file_path = save_uploaded_file(file)

        # =====================================================
        # Trigger ingestion pipeline
        # =====================================================

        result = process_uploaded_pdf(saved_file_path)

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