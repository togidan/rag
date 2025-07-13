from typing import Any, Dict, Optional


class RAGChatGPTException(Exception):
    """Base exception class for RAG ChatGPT application."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(RAGChatGPTException):
    """Raised when there's a configuration issue."""
    pass


class EmbeddingError(RAGChatGPTException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGChatGPTException):
    """Raised when vector store operations fail."""
    pass


class DocumentProcessingError(RAGChatGPTException):
    """Raised when document processing fails."""
    pass


class ChatServiceError(RAGChatGPTException):
    """Raised when chat service operations fail."""
    pass


class ExternalServiceError(RAGChatGPTException):
    """Raised when external service calls fail."""
    
    def __init__(self, service_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        super().__init__(f"{service_name}: {message}", details)


class ValidationError(RAGChatGPTException):
    """Raised when data validation fails."""
    pass


class NotFoundError(RAGChatGPTException):
    """Raised when a requested resource is not found."""
    pass


class RateLimitError(RAGChatGPTException):
    """Raised when rate limits are exceeded."""
    pass