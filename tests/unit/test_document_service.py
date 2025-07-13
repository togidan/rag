import pytest
from unittest.mock import Mock, AsyncMock

from app.services.document_service import DocumentService
from app.models.schemas import DocumentUploadResponse


@pytest.mark.unit
class TestDocumentService:
    """Unit tests for DocumentService."""
    
    def test_chunk_text_default_settings(self, mock_document_service):
        """Test text chunking with default settings."""
        text = "A" * 250  # 250 characters
        chunks = mock_document_service.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)  # chunk_size = 100 in test settings
    
    def test_chunk_text_custom_settings(self, mock_document_service):
        """Test text chunking with custom settings."""
        text = "A" * 100
        chunks = mock_document_service.chunk_text(text, chunk_size=50, overlap=5)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_chunk_text_empty_input(self, mock_document_service):
        """Test chunking empty text."""
        chunks = mock_document_service.chunk_text("")
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_process_text_document_success(self, mock_document_service, mock_embedding_service, mock_vector_store_service):
        """Test successful text document processing."""
        # Mock embedding service to return valid embeddings
        mock_embedding_service.get_embeddings_batch = AsyncMock(return_value=[
            [0.1] * 3072,
            [0.2] * 3072
        ])
        mock_embedding_service.validate_embedding = Mock(return_value=True)
        
        # Mock vector store service
        mock_vector_store_service.insert_document_chunks = AsyncMock(return_value=True)
        
        result = await mock_document_service.process_text_document(
            "Test Document",
            "This is a test document with enough content to create multiple chunks for testing purposes."
        )
        
        assert isinstance(result, DocumentUploadResponse)
        assert "successfully uploaded" in result.message.lower()
        assert result.chunks > 0
    
    @pytest.mark.asyncio
    async def test_process_text_document_no_chunks(self, mock_document_service):
        """Test processing document with no valid chunks."""
        result = await mock_document_service.process_text_document("Empty", "")
        
        assert isinstance(result, DocumentUploadResponse)
        assert "no valid text chunks" in result.message.lower()
        assert result.chunks == 0
    
    @pytest.mark.asyncio
    async def test_process_text_document_embedding_failure(self, mock_document_service, mock_embedding_service):
        """Test processing when embedding generation fails."""
        mock_embedding_service.get_embeddings_batch = AsyncMock(return_value=[])
        
        result = await mock_document_service.process_text_document(
            "Test Document",
            "This is a test document."
        )
        
        assert isinstance(result, DocumentUploadResponse)
        assert "failed to generate embeddings" in result.message.lower()
        assert result.chunks == 0
    
    @pytest.mark.asyncio
    async def test_process_text_document_validation_failure(self, mock_document_service, mock_embedding_service):
        """Test processing when embedding validation fails."""
        mock_embedding_service.get_embeddings_batch = AsyncMock(return_value=[[0.1] * 1536])  # Wrong dimension
        mock_embedding_service.validate_embedding = Mock(return_value=False)
        
        result = await mock_document_service.process_text_document(
            "Test Document",
            "This is a test document."
        )
        
        assert isinstance(result, DocumentUploadResponse)
        assert "no valid embeddings after validation" in result.message.lower()
        assert result.chunks == 0
    
    @pytest.mark.asyncio
    async def test_process_text_document_storage_failure(self, mock_document_service, mock_embedding_service, mock_vector_store_service):
        """Test processing when vector storage fails."""
        mock_embedding_service.get_embeddings_batch = AsyncMock(return_value=[[0.1] * 3072])
        mock_embedding_service.validate_embedding = Mock(return_value=True)
        mock_vector_store_service.insert_document_chunks = AsyncMock(return_value=False)
        
        result = await mock_document_service.process_text_document(
            "Test Document",
            "This is a test document."
        )
        
        assert isinstance(result, DocumentUploadResponse)
        assert "failed to store document" in result.message.lower()
        assert result.chunks == 0
    
    @pytest.mark.asyncio
    async def test_process_file_uploads_success(self, mock_document_service):
        """Test successful file upload processing."""
        # Mock file objects
        mock_file1 = Mock()
        mock_file1.filename = "test1.txt"
        mock_file1.read = AsyncMock(return_value=b"Content of test file 1")
        
        mock_file2 = Mock()
        mock_file2.filename = "test2.txt"
        mock_file2.read = AsyncMock(return_value=b"Content of test file 2")
        
        # Mock the process_text_document method
        mock_document_service.process_text_document = AsyncMock(
            return_value=DocumentUploadResponse(message="Success", chunks=2)
        )
        
        result = await mock_document_service.process_file_uploads([mock_file1, mock_file2])
        
        assert isinstance(result, DocumentUploadResponse)
        assert "successfully uploaded" in result.message.lower()
        assert result.files is not None
        assert len(result.files) == 2
    
    @pytest.mark.asyncio
    async def test_get_document_list(self, mock_document_service, mock_vector_store_service):
        """Test getting document list."""
        mock_vector_store_service.list_documents = AsyncMock(return_value=[
            {"primary_key": 123, "title": "Test Doc", "chunk_count": 2}
        ])
        
        result = await mock_document_service.get_document_list()
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["primary_key"] == 123