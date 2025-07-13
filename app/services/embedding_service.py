from typing import List, Optional
import openai
from openai import OpenAI

from app.core.settings import Settings
from app.models.schemas import EmbeddingResponse
from app.core.logging import LoggerMixin
from app.core.exceptions import EmbeddingError, ExternalServiceError


class EmbeddingService(LoggerMixin):
    """Service for handling OpenAI embeddings."""
    
    def __init__(self, settings: Settings):
        """Initialize the embedding service with settings."""
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimension = settings.embedding_dimension
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            self.log_debug(f"Generating embedding for text of length {len(text)}")
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            self.log_debug("Successfully generated embedding")
            return [float(x) for x in embedding]
        except openai.APIError as e:
            self.log_error(
                "OpenAI API error while generating embedding",
                extra_fields={"error_code": e.code, "error_type": e.type}
            )
            raise ExternalServiceError("OpenAI", f"API error: {e.message}")
        except Exception as e:
            self.log_error("Unexpected error generating embedding")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
            
        try:
            self.log_debug(f"Generating embeddings for batch of {len(texts)} texts")
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = [[float(x) for x in data.embedding] for data in response.data]
            self.log_debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except openai.APIError as e:
            self.log_error(
                "OpenAI API error while generating batch embeddings",
                extra_fields={"error_code": e.code, "error_type": e.type, "batch_size": len(texts)}
            )
            raise ExternalServiceError("OpenAI", f"API error: {e.message}")
        except Exception as e:
            self.log_error("Unexpected error generating batch embeddings")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the correct dimension.
        
        Args:
            embedding: The embedding to validate
            
        Returns:
            True if valid, False otherwise
        """
        return len(embedding) == self.dimension
    
    async def test_connection(self) -> bool:
        """
        Test the OpenAI connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.log_debug("Testing OpenAI connection")
            test_response = self.client.embeddings.create(
                model=self.model,
                input="test"
            )
            success = len(test_response.data) > 0
            self.log_info(f"OpenAI connection test {'successful' if success else 'failed'}")
            return success
        except Exception as e:
            self.log_error(
                "OpenAI connection test failed",
                extra_fields={"error": str(e)}
            )
            return False