from typing import List, Optional, Dict, Any
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from app.core.settings import Settings
from app.models.schemas import DocumentChunk, RAGSource
from app.utils.hash_utils import generate_text_hash


class VectorStoreService:
    """Service for handling Milvus vector database operations."""
    
    def __init__(self, settings: Settings):
        """Initialize the vector store service with settings."""
        self.settings = settings
        self.client: Optional[MilvusClient] = None
        self.collection_name = settings.milvus_collection_name
        self.dimension = settings.embedding_dimension
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Milvus client and collection."""
        try:
            self.client = MilvusClient(
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token
            )
            self._ensure_collection_exists()
            print(f"Successfully connected to Milvus collection '{self.collection_name}'")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            self.client = None
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if not."""
        if not self.client:
            return
            
        if not self.client.has_collection(collection_name=self.collection_name):
            fields = [
                FieldSchema(name="primary_key", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            schema = CollectionSchema(fields, description="Document collection for RAG")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params={
                    "field_name": "vector",
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                }
            )
            print(f"Created collection '{self.collection_name}' with explicit schema")
    
    async def insert_document_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Insert document chunks into the vector store.
        
        Args:
            chunks: List of document chunks to insert
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client or not chunks:
            return False
        
        try:
            # Insert each chunk individually
            for chunk in chunks:
                data = {
                    "primary_key": chunk.primary_key,
                    "text": chunk.text,
                    "vector": chunk.embedding
                }
                
                self.client.insert(
                    collection_name=self.collection_name,
                    data=data
                )
            
            print(f"Successfully inserted {len(chunks)} chunks into Milvus")
            return True
        except Exception as e:
            print(f"Error inserting chunks: {e}")
            return False
    
    async def search_similar_documents(self, query_embedding: List[float], top_k: int = 3) -> List[RAGSource]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        if not self.client or not query_embedding:
            return []
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "primary_key"]
            )
            
            rag_sources = []
            for result in results:
                for hit in result:
                    rag_sources.append(RAGSource(
                        text=hit.get("entity", {}).get("text", ""),
                        score=hit.get("distance", 0),
                        primary_key=str(hit.get("entity", {}).get("primary_key", "unknown"))
                    ))
            
            return rag_sources
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.client:
            return {"error": "Client not available"}
        
        try:
            if self.client.has_collection(collection_name=self.collection_name):
                stats = self.client.get_collection_stats(collection_name=self.collection_name)
                return stats
            else:
                return {"error": "Collection does not exist"}
        except Exception as e:
            return {"error": f"Failed to get stats: {str(e)}"}
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all uploaded documents with metadata.
        
        Returns:
            List of document information dictionaries
        """
        if not self.client:
            return []
        
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="primary_key >= 0",
                output_fields=["primary_key", "text"]
            )
            
            # Group by primary_key to get unique documents
            documents_dict = {}
            for result in results:
                pk = result.get("primary_key")
                text = result.get("text", "")
                
                if pk is None:
                    continue
                
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                
                if pk not in documents_dict:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    documents_dict[pk] = {
                        "primary_key": pk,
                        "title": f"Document {pk}",
                        "upload_date": "Unknown",
                        "chunk_count": 0,
                        "preview": preview
                    }
                documents_dict[pk]["chunk_count"] += 1
            
            return list(documents_dict.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self.client is not None
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        if not self.client:
            return False
        return self.client.has_collection(collection_name=self.collection_name)