import pytest
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    def test_health_index_endpoint(self, test_client):
        """Test the main index endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    @patch('app.core.dependencies.get_chat_service')
    @patch('app.core.dependencies.get_vector_store_service')
    def test_health_status_endpoint(self, mock_vector_store, mock_chat_service, test_client):
        """Test the system status endpoint."""
        # Mock services
        mock_chat_service_instance = Mock()
        mock_chat_service_instance.test_openai_connection = AsyncMock(return_value=True)
        mock_chat_service.return_value = mock_chat_service_instance
        
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.is_connected.return_value = True
        mock_vector_store_instance.collection_exists.return_value = True
        mock_vector_store_instance.get_collection_stats = AsyncMock(return_value={"row_count": 10})
        mock_vector_store.return_value = mock_vector_store_instance
        
        response = test_client.get("/health/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "milvus_connected" in data
        assert "collection_exists" in data
        assert "total_documents" in data
        assert "openai_connected" in data
    
    @patch('app.core.dependencies.get_chat_service')
    def test_chat_ask_endpoint(self, mock_chat_service, test_client):
        """Test the chat ask endpoint."""
        # Mock chat service
        mock_service = Mock()
        mock_service.generate_rag_response = AsyncMock(return_value={
            "response": "Test response",
            "rag_sources": []
        })
        mock_chat_service.return_value = mock_service
        
        response = test_client.post(
            "/chat/ask",
            json={"message": "What is machine learning?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "rag_sources" in data
    
    @patch('app.core.dependencies.get_document_service')
    def test_document_upload_text_endpoint(self, mock_document_service, test_client):
        """Test the text upload endpoint."""
        # Mock document service
        mock_service = Mock()
        mock_service.process_text_document = AsyncMock(return_value={
            "message": "Successfully uploaded",
            "chunks": 3
        })
        mock_document_service.return_value = mock_service
        
        response = test_client.post(
            "/documents/upload-text",
            json={
                "title": "Test Document",
                "text": "This is a test document for uploading."
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "chunks" in data
    
    @patch('app.core.dependencies.get_document_service')
    def test_document_list_endpoint(self, mock_document_service, test_client):
        """Test the document list endpoint."""
        # Mock document service
        mock_service = Mock()
        mock_service.get_document_list = AsyncMock(return_value=[
            {
                "primary_key": 123,
                "title": "Test Document",
                "chunk_count": 3,
                "preview": "This is a test..."
            }
        ])
        mock_document_service.return_value = mock_service
        
        response = test_client.get("/documents/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) == 1
        assert data["documents"][0]["primary_key"] == 123
    
    def test_chat_ask_invalid_input(self, test_client):
        """Test chat endpoint with invalid input."""
        response = test_client.post(
            "/chat/ask",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_document_upload_text_invalid_input(self, test_client):
        """Test text upload endpoint with invalid input."""
        response = test_client.post(
            "/documents/upload-text",
            json={"title": "Missing text field"}
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.core.dependencies.get_chat_service')
    def test_legacy_ask_endpoint(self, mock_chat_service, test_client):
        """Test legacy /ask endpoint for backward compatibility."""
        mock_service = Mock()
        mock_service.generate_rag_response = AsyncMock(return_value={
            "response": "Legacy response",
            "rag_sources": []
        })
        mock_chat_service.return_value = mock_service
        
        response = test_client.post(
            "/ask",
            json={"message": "Legacy test question"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
    
    @patch('app.core.dependencies.get_document_service')
    def test_legacy_upload_text_endpoint(self, mock_document_service, test_client):
        """Test legacy /upload-text endpoint."""
        mock_service = Mock()
        mock_service.process_text_document = AsyncMock(return_value={
            "message": "Legacy upload successful",
            "chunks": 2
        })
        mock_document_service.return_value = mock_service
        
        response = test_client.post(
            "/upload-text",
            json={
                "title": "Legacy Document",
                "text": "Legacy document content."
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data