import pytest
from unittest.mock import Mock, AsyncMock
from typing import AsyncGenerator
import asyncio
from fastapi.testclient import TestClient

from app.core.settings import Settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService
from app.main import app


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        openai_api_key="test_key",
        openai_model="gpt-4o",
        openai_embedding_model="text-embedding-3-large",
        embedding_dimension=3072,
        milvus_uri="test://localhost",
        milvus_token="test_token",
        milvus_collection_name="test_collection",
        app_host="127.0.0.1",
        app_port=8001,
        debug=True,
        chunk_size=100,
        chunk_overlap=10,
        max_search_results=3,
        max_tokens=500
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    
    # Mock embeddings response
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1] * 3072)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock chat response
    mock_chat_response = Mock()
    mock_chat_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_client.chat.completions.create.return_value = mock_chat_response
    
    return mock_client


@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client for testing."""
    mock_client = Mock()
    mock_client.has_collection.return_value = True
    mock_client.insert.return_value = None
    mock_client.search.return_value = [[Mock(
        entity={"text": "test document", "primary_key": 123},
        distance=0.8
    )]]
    mock_client.query.return_value = [
        {"primary_key": 123, "text": "test document"}
    ]
    mock_client.get_collection_stats.return_value = {"row_count": 1}
    return mock_client


@pytest.fixture
def mock_embedding_service(test_settings, mock_openai_client):
    """Create mock embedding service."""
    service = EmbeddingService(test_settings)
    service.client = mock_openai_client
    return service


@pytest.fixture
def mock_vector_store_service(test_settings, mock_milvus_client):
    """Create mock vector store service."""
    service = VectorStoreService(test_settings)
    service.client = mock_milvus_client
    return service


@pytest.fixture
def mock_document_service(test_settings, mock_embedding_service, mock_vector_store_service):
    """Create mock document service."""
    return DocumentService(test_settings, mock_embedding_service, mock_vector_store_service)


@pytest.fixture
def mock_chat_service(test_settings, mock_embedding_service, mock_vector_store_service):
    """Create mock chat service."""
    service = ChatService(test_settings, mock_embedding_service, mock_vector_store_service)
    service.client = mock_openai_client
    return service


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    from app.models.schemas import DocumentChunk
    
    return [
        DocumentChunk(
            text="This is a test document chunk.",
            embedding=[0.1] * 3072,
            primary_key=123,
            metadata={"chunk_index": 0, "title": "test.txt"}
        ),
        DocumentChunk(
            text="This is another test chunk.",
            embedding=[0.2] * 3072,
            primary_key=123,
            metadata={"chunk_index": 1, "title": "test.txt"}
        )
    ]


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing."""
    from app.models.schemas import ChatRequest
    return ChatRequest(message="What is machine learning?")


@pytest.fixture
def sample_text_upload():
    """Sample text upload request for testing."""
    from app.models.schemas import TextUploadRequest
    return TextUploadRequest(
        title="Test Document",
        text="This is a test document with some content for RAG processing."
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()