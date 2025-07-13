from typing import Annotated

from fastapi import Depends

from app.core.settings import Settings, get_settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService
from app.db.repositories import MilvusRepository


def get_embedding_service(
    settings: Annotated[Settings, Depends(get_settings)]
) -> EmbeddingService:
    """Dependency provider for EmbeddingService."""
    return EmbeddingService(settings)


def get_vector_store_service(
    settings: Annotated[Settings, Depends(get_settings)]
) -> VectorStoreService:
    """Dependency provider for VectorStoreService."""
    return VectorStoreService(settings)


def get_milvus_repository(
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store_service)]
) -> MilvusRepository:
    """Dependency provider for MilvusRepository."""
    return MilvusRepository(vector_store)


def get_document_service(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store_service)]
) -> DocumentService:
    """Dependency provider for DocumentService."""
    return DocumentService(settings, embedding_service, vector_store)


def get_chat_service(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store_service)]
) -> ChatService:
    """Dependency provider for ChatService."""
    return ChatService(settings, embedding_service, vector_store)