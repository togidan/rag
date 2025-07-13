from typing import Annotated
from fastapi import APIRouter, Depends

from app.models.schemas import SystemStatus
from app.services.chat_service import ChatService
from app.services.vector_store_service import VectorStoreService
from app.core.dependencies import get_chat_service, get_vector_store_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    vector_store: Annotated[VectorStoreService, Depends(get_vector_store_service)]
) -> SystemStatus:
    """
    Get comprehensive system health status.
    
    Args:
        chat_service: Injected chat service
        vector_store: Injected vector store service
        
    Returns:
        System status with connection and data information
    """
    try:
        # Check Milvus connection and collection
        milvus_connected = vector_store.is_connected()
        collection_exists = vector_store.collection_exists()
        
        # Get document count
        total_documents = 0
        if milvus_connected and collection_exists:
            try:
                stats = await vector_store.get_collection_stats()
                total_documents = stats.get('row_count', 0)
            except Exception as e:
                print(f"Error getting collection stats: {e}")
                milvus_connected = False
        
        # Test OpenAI connection
        openai_connected = await chat_service.test_openai_connection()
        
        return SystemStatus(
            milvus_connected=milvus_connected,
            collection_exists=collection_exists,
            total_documents=total_documents,
            openai_connected=openai_connected
        )
    except Exception as e:
        print(f"Status check failed: {e}")
        return SystemStatus(
            milvus_connected=False,
            collection_exists=False,
            total_documents=0,
            openai_connected=False
        )