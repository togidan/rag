from typing import List, Optional
import openai
from openai import OpenAI

from app.core.settings import Settings
from app.models.schemas import ChatRequest, ChatResponse, RAGSource
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


class ChatService:
    """Service for handling RAG-enhanced chat interactions."""
    
    def __init__(
        self, 
        settings: Settings, 
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService
    ):
        """Initialize the chat service with dependencies."""
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.max_tokens
    
    async def generate_rag_response(self, request: ChatRequest) -> ChatResponse:
        """
        Generate a RAG-enhanced response to the user's message.
        
        Args:
            request: Chat request containing the user's message
            
        Returns:
            Chat response with RAG sources
        """
        try:
            # Get relevant documents using vector search
            rag_sources = await self._search_relevant_documents(request.message)
            
            # Prepare context from retrieved documents
            context = self._prepare_context(rag_sources)
            
            # Generate response using OpenAI
            response_text = await self._generate_openai_response(request.message, context)
            
            return ChatResponse(
                response=response_text,
                rag_sources=rag_sources
            )
            
        except Exception as e:
            return ChatResponse(
                response=f"I apologize, but I encountered an error: {str(e)}",
                rag_sources=[]
            )
    
    async def _search_relevant_documents(self, query: str) -> List[RAGSource]:
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query: User query to search for
            
        Returns:
            List of relevant RAG sources
        """
        # Generate embedding for the query
        query_embedding = await self.embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        
        # Search for similar documents
        return await self.vector_store_service.search_similar_documents(
            query_embedding, 
            top_k=self.settings.max_search_results
        )
    
    def _prepare_context(self, rag_sources: List[RAGSource]) -> str:
        """
        Prepare context string from RAG sources.
        
        Args:
            rag_sources: List of relevant documents
            
        Returns:
            Formatted context string
        """
        if not rag_sources:
            return ""
        
        context = "\n\nRelevant information from your knowledge base:\n"
        for i, source in enumerate(rag_sources, 1):
            context += f"\n{i}. {source.text}\n"
        
        return context
    
    async def _generate_openai_response(self, user_message: str, context: str) -> str:
        """
        Generate response using OpenAI with context.
        
        Args:
            user_message: Original user message
            context: Relevant context from documents
            
        Returns:
            Generated response text
        """
        # Enhanced system message with context
        system_message = (
            "You are a helpful assistant. Use the provided context information "
            "to enhance your responses when relevant."
        )
        
        if context:
            system_message += (
                " If the context contains relevant information, incorporate it "
                "into your answer and mention the source documents."
            )
        
        # Prepare messages for ChatGPT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message + context}
        ]
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def test_openai_connection(self) -> bool:
        """
        Test the OpenAI connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        return await self.embedding_service.test_connection()