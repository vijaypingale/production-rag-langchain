"""
Main FastAPI Application Entry Point

Responsibilities:
-----------------
- Initialize FastAPI application
- Register API routers
- Configure application metadata
- Serve as backend entry point for the RAG platform

Enterprise Architecture:
------------------------
Client/UI
    ↓
FastAPI Routes
    ↓
Service Layer
    ↓
RAG Pipeline Components
"""

from fastapi import FastAPI

from app.api.ingestion_api import router as ingestion_router

from app.utils.logger import logger


# ============================================================================
# Initialize FastAPI Application
# ============================================================================

app = FastAPI(
    title="Production RAG LangChain API",
    description="Enterprise-grade RAG backend using LangChain, FAISS, FastAPI, and OpenAI",
    version="1.0.0"
)


# ============================================================================
# Register API Routers
# ============================================================================

app.include_router(
    ingestion_router,
    prefix="/api/v1"
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():

    logger.info(
        "health_check_called"
    )

    return {
        "status": "healthy",
        "application": "production-rag-langchain"
    }


# ============================================================================
# Application Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():

    logger.info(
        "FASTAPI APPLICATION STARTED"
    )


# ============================================================================
# Application Shutdown Event
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():

    logger.info(
        "fastapi_application_stopped"
    )