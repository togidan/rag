from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.models.schemas import DocumentChunk, RAGSource


class DocumentRepository(ABC):
    """Abstract repository for document operations."""
    
    @abstractmethod
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Insert document chunks into storage."""
        pass
    
    @abstractmethod
    async def search_similar(self, query_embedding: List[float], top_k: int) -> List[RAGSource]:
        """Search for similar documents using vector similarity."""
        pass
    
    @abstractmethod
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents metadata."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if repository is healthy and accessible."""
        pass


class MilvusRepository(DocumentRepository):
    """Milvus implementation of document repository."""
    
    def __init__(self, vector_store_service):
        """Initialize with vector store service."""
        self.vector_store = vector_store_service
    
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Insert document chunks into Milvus."""
        return await self.vector_store.insert_document_chunks(chunks)
    
    async def search_similar(self, query_embedding: List[float], top_k: int) -> List[RAGSource]:
        """Search for similar documents in Milvus."""
        return await self.vector_store.search_similar_documents(query_embedding, top_k)
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from Milvus."""
        return await self.vector_store.list_documents()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Milvus collection statistics."""
        return await self.vector_store.get_collection_stats()
    
    def is_healthy(self) -> bool:
        """Check if Milvus connection is healthy."""
        return self.vector_store.is_connected() and self.vector_store.collection_exists()