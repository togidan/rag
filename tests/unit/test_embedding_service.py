import pytest
from unittest.mock import Mock, patch
import openai

from app.services.embedding_service import EmbeddingService
from app.core.exceptions import EmbeddingError, ExternalServiceError


@pytest.mark.unit
class TestEmbeddingService:
    """Unit tests for EmbeddingService."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_success(self, mock_embedding_service):
        """Test successful embedding generation."""
        result = await mock_embedding_service.get_embedding("test text")
        
        assert result is not None
        assert len(result) == 3072
        assert all(isinstance(x, float) for x in result)
        mock_embedding_service.client.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embedding_openai_error(self, test_settings):
        """Test OpenAI API error handling."""
        service = EmbeddingService(test_settings)
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = openai.APIError(
            message="Rate limit exceeded",
            response=Mock(),
            body={}
        )
        service.client = mock_client
        
        with pytest.raises(ExternalServiceError) as exc_info:
            await service.get_embedding("test text")
        
        assert "OpenAI" in str(exc_info.value)
        assert "API error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_embedding_generic_error(self, test_settings):
        """Test generic error handling."""
        service = EmbeddingService(test_settings)
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Connection error")
        service.client = mock_client
        
        with pytest.raises(EmbeddingError) as exc_info:
            await service.get_embedding("test text")
        
        assert "Failed to generate embedding" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_embeddings_batch_success(self, mock_embedding_service):
        """Test successful batch embedding generation."""
        texts = ["text 1", "text 2", "text 3"]
        
        # Mock batch response
        mock_embedding_service.client.embeddings.create.return_value.data = [
            Mock(embedding=[0.1] * 3072),
            Mock(embedding=[0.2] * 3072),
            Mock(embedding=[0.3] * 3072)
        ]
        
        result = await mock_embedding_service.get_embeddings_batch(texts)
        
        assert len(result) == 3
        assert all(len(embedding) == 3072 for embedding in result)
        mock_embedding_service.client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=texts
        )
    
    @pytest.mark.asyncio
    async def test_get_embeddings_batch_empty_input(self, mock_embedding_service):
        """Test batch embedding with empty input."""
        result = await mock_embedding_service.get_embeddings_batch([])
        assert result == []
    
    def test_validate_embedding_correct_dimension(self, mock_embedding_service):
        """Test embedding validation with correct dimension."""
        embedding = [0.1] * 3072
        assert mock_embedding_service.validate_embedding(embedding) is True
    
    def test_validate_embedding_wrong_dimension(self, mock_embedding_service):
        """Test embedding validation with wrong dimension."""
        embedding = [0.1] * 1536  # Wrong dimension
        assert mock_embedding_service.validate_embedding(embedding) is False
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_embedding_service):
        """Test successful connection test."""
        result = await mock_embedding_service.test_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, test_settings):
        """Test connection test failure."""
        service = EmbeddingService(test_settings)
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Connection failed")
        service.client = mock_client
        
        result = await service.test_connection()
        assert result is False