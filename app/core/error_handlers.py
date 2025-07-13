from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
from typing import Dict, Any
import time

from app.core.exceptions import (
    RAGChatGPTException,
    ConfigurationError,
    EmbeddingError,
    VectorStoreError,
    DocumentProcessingError,
    ChatServiceError,
    ExternalServiceError,
    ValidationError,
    NotFoundError,
    RateLimitError
)
from app.core.logging import get_logger

logger = get_logger("error_handlers")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling uncaught exceptions and logging requests."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log successful requests
            logger.info(
                "Request completed",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "url": str(request.url),
                        "status_code": response.status_code,
                        "process_time": round(process_time, 4)
                    }
                }
            )
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            
            # Log the error
            logger.error(
                "Request failed with unhandled exception",
                extra={
                    "extra_fields": {
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": round(process_time, 4),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)
                    }
                },
                exc_info=True
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": id(request)
                }
            )


def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create a standardized error response."""
    content = {
        "error": error_type,
        "message": message
    }
    
    if details:
        content["details"] = details
    
    return JSONResponse(status_code=status_code, content=content)


async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
    """Handle configuration errors."""
    logger.error(
        "Configuration error",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=500,
        error_type="configuration_error",
        message="Application configuration error",
        details=exc.details if exc.details else None
    )


async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    """Handle embedding service errors."""
    logger.error(
        "Embedding service error",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=502,
        error_type="embedding_error",
        message="Failed to generate embeddings",
        details=exc.details
    )


async def vector_store_error_handler(request: Request, exc: VectorStoreError) -> JSONResponse:
    """Handle vector store errors."""
    logger.error(
        "Vector store error",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=502,
        error_type="vector_store_error",
        message="Vector database operation failed",
        details=exc.details
    )


async def document_processing_error_handler(request: Request, exc: DocumentProcessingError) -> JSONResponse:
    """Handle document processing errors."""
    logger.error(
        "Document processing error",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=422,
        error_type="document_processing_error",
        message="Failed to process document",
        details=exc.details
    )


async def chat_service_error_handler(request: Request, exc: ChatServiceError) -> JSONResponse:
    """Handle chat service errors."""
    logger.error(
        "Chat service error",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=502,
        error_type="chat_service_error",
        message="Chat service operation failed",
        details=exc.details
    )


async def external_service_error_handler(request: Request, exc: ExternalServiceError) -> JSONResponse:
    """Handle external service errors."""
    logger.error(
        "External service error",
        extra={
            "extra_fields": {
                "service_name": exc.service_name,
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=502,
        error_type="external_service_error",
        message=f"External service error: {exc.service_name}",
        details=exc.details
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    logger.warning(
        "Validation error",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=422,
        error_type="validation_error",
        message="Invalid input data",
        details=exc.details
    )


async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    """Handle not found errors."""
    logger.warning(
        "Resource not found",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=404,
        error_type="not_found",
        message="Resource not found",
        details=exc.details
    )


async def rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Handle rate limit errors."""
    logger.warning(
        "Rate limit exceeded",
        extra={
            "extra_fields": {
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=429,
        error_type="rate_limit_error",
        message="Rate limit exceeded",
        details=exc.details
    )


async def generic_rag_exception_handler(request: Request, exc: RAGChatGPTException) -> JSONResponse:
    """Handle generic RAG ChatGPT exceptions."""
    logger.error(
        "RAG ChatGPT application error",
        extra={
            "extra_fields": {
                "exception_type": type(exc).__name__,
                "exception_message": exc.message,
                "details": exc.details
            }
        }
    )
    
    return create_error_response(
        status_code=500,
        error_type="application_error",
        message=exc.message,
        details=exc.details
    )


# Exception handler mapping
EXCEPTION_HANDLERS = {
    ConfigurationError: configuration_error_handler,
    EmbeddingError: embedding_error_handler,
    VectorStoreError: vector_store_error_handler,
    DocumentProcessingError: document_processing_error_handler,
    ChatServiceError: chat_service_error_handler,
    ExternalServiceError: external_service_error_handler,
    ValidationError: validation_error_handler,
    NotFoundError: not_found_error_handler,
    RateLimitError: rate_limit_error_handler,
    RAGChatGPTException: generic_rag_exception_handler,
}