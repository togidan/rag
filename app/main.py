from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

from app.api import chat, documents, health
from app.core.settings import get_settings
from app.core.logging import setup_logging
from app.core.error_handlers import ErrorHandlingMiddleware, EXCEPTION_HANDLERS

# Initialize logging
settings = get_settings()
setup_logging(settings)

app = FastAPI(
    title="RAG-Enhanced ChatGPT",
    description="A Retrieval-Augmented Generation system using OpenAI and Milvus",
    version="2.0.0"
)

# Add error handling middleware
app.add_middleware(ErrorHandlingMiddleware)

# Add exception handlers
for exception_class, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exception_class, handler)

# Include API routers
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve the main web interface."""
    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


# Legacy endpoints for backward compatibility
from typing import Annotated, List
from fastapi import Depends, UploadFile, File
from app.models.schemas import ChatRequest, TextUploadRequest
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.vector_store_service import VectorStoreService
from app.core.dependencies import get_chat_service, get_document_service, get_vector_store_service


@app.post("/ask")
async def legacy_ask(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)]
):
    """Legacy endpoint - redirects to /chat/ask"""
    return await chat_service.generate_rag_response(request)


@app.post("/upload")
async def legacy_upload(
    files: List[UploadFile] = File(...),
    document_service: Annotated[DocumentService, Depends(get_document_service)] = None
):
    """Legacy endpoint - redirects to /documents/upload"""
    return await document_service.process_file_uploads(files)


@app.post("/upload-text")
async def legacy_upload_text(
    request: TextUploadRequest,
    document_service: Annotated[DocumentService, Depends(get_document_service)]
):
    """Legacy endpoint - redirects to /documents/upload-text"""
    return await document_service.process_text_document(request.title, request.text)


@app.get("/status")
async def legacy_status(
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store_service)]
):
    """Legacy endpoint - redirects to /health/status"""
    from app.api.health import get_system_status
    return await get_system_status(chat_service, vector_store)


@app.get("/documents")
async def legacy_documents(
    document_service: Annotated[DocumentService, Depends(get_document_service)]
):
    """Legacy endpoint - redirects to /documents/list"""
    documents = await document_service.get_document_list()
    return {"documents": documents}


if __name__ == "__main__":
    import uvicorn
    from app.core.settings import get_settings
    
    settings = get_settings()
    # uvicorn.run(
    #     "app.main:app",
    #     host=settings.app_host,
    #     port=settings.app_port,
    #     reload=settings.debug
    # )