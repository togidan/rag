from typing import Annotated, List
from fastapi import APIRouter, Depends, UploadFile, File

from app.models.schemas import TextUploadRequest, DocumentUploadResponse
from app.services.document_service import DocumentService
from app.core.dependencies import get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    document_service: Annotated[DocumentService, Depends(get_document_service)] = None
) -> DocumentUploadResponse:
    """
    Upload and process multiple files.
    
    Args:
        files: List of uploaded files
        document_service: Injected document service
        
    Returns:
        Document upload response with processing results
    """
    return await document_service.process_file_uploads(files)


@router.post("/upload-text", response_model=DocumentUploadResponse)
async def upload_text(
    request: TextUploadRequest,
    document_service: Annotated[DocumentService, Depends(get_document_service)]
) -> DocumentUploadResponse:
    """
    Upload and process text content.
    
    Args:
        request: Text upload request with title and content
        document_service: Injected document service
        
    Returns:
        Document upload response with processing results
    """
    return await document_service.process_text_document(request.title, request.text)


@router.get("/list")
async def get_documents(
    document_service: Annotated[DocumentService, Depends(get_document_service)]
) -> dict:
    """
    Get list of all uploaded documents.
    
    Args:
        document_service: Injected document service
        
    Returns:
        Dictionary containing list of documents
    """
    documents = await document_service.get_document_list()
    return {"documents": documents}