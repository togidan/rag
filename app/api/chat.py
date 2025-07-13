from typing import Annotated
from fastapi import APIRouter, Depends

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.core.dependencies import get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_chatgpt(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResponse:
    """
    Generate a RAG-enhanced response to the user's message.
    
    Args:
        request: Chat request containing the user's message
        chat_service: Injected chat service
        
    Returns:
        Chat response with RAG sources
    """
    return await chat_service.generate_rag_response(request)