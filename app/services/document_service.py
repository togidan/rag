from typing import List, Optional
from fastapi import UploadFile

from app.core.settings import Settings
from app.models.schemas import DocumentChunk, DocumentUploadResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.utils.hash_utils import generate_text_hash


class DocumentService:
    """Service for document processing, chunking, and storage."""
    
    def __init__(
        self, 
        settings: Settings, 
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService
    ):
        """Initialize the document service with dependencies."""
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (defaults to settings value)
            overlap: Overlap between chunks (defaults to settings value)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.settings.chunk_size
        overlap = overlap or self.settings.chunk_overlap
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            start = end - overlap
        return chunks
    
    async def process_text_document(self, title: str, text: str) -> DocumentUploadResponse:
        """
        Process a text document: chunk, embed, and store.
        
        Args:
            title: Document title
            text: Document content
            
        Returns:
            Document upload response
        """
        try:
            # Generate primary key for the document
            primary_key = generate_text_hash(title)
            
            # Chunk the text
            chunks = self.chunk_text(text)
            if not chunks:
                return DocumentUploadResponse(
                    message="No valid text chunks found",
                    chunks=0
                )
            
            # Generate embeddings for chunks
            embeddings = await self.embedding_service.get_embeddings_batch(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                return DocumentUploadResponse(
                    message="Failed to generate embeddings for all chunks",
                    chunks=0
                )
            
            # Validate embeddings and create document chunks
            valid_chunks = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                if self.embedding_service.validate_embedding(embedding):
                    chunk_primary_key = primary_key + i
                    valid_chunks.append(DocumentChunk(
                        text=chunk_text,
                        embedding=embedding,
                        primary_key=chunk_primary_key,
                        metadata={"chunk_index": i, "title": title}
                    ))
            
            if not valid_chunks:
                return DocumentUploadResponse(
                    message="No valid embeddings after validation",
                    chunks=0
                )
            
            # Store in vector database
            success = await self.vector_store_service.insert_document_chunks(valid_chunks)
            if not success:
                return DocumentUploadResponse(
                    message="Failed to store document in vector database",
                    chunks=0
                )
            
            return DocumentUploadResponse(
                message=f"Successfully uploaded text document '{title}'",
                chunks=len(valid_chunks)
            )
            
        except Exception as e:
            return DocumentUploadResponse(
                message=f"Error processing document: {str(e)}",
                chunks=0
            )
    
    async def process_file_uploads(self, files: List[UploadFile]) -> DocumentUploadResponse:
        """
        Process uploaded files: read, chunk, embed, and store.
        
        Args:
            files: List of uploaded files
            
        Returns:
            Document upload response
        """
        try:
            uploaded_files = []
            total_chunks = 0
            
            for file in files:
                # Read file content
                content = await file.read()
                
                # Convert to text (basic implementation for .txt files)
                if file.filename.endswith('.txt'):
                    text = content.decode('utf-8')
                else:
                    # For other file types, try to decode as text
                    text = content.decode('utf-8', errors='ignore')
                
                # Process the document
                result = await self.process_text_document(file.filename, text)
                
                if result.chunks and result.chunks > 0:
                    uploaded_files.append({
                        "filename": file.filename,
                        "chunks": result.chunks
                    })
                    total_chunks += result.chunks
            
            if not uploaded_files:
                return DocumentUploadResponse(
                    message="No files were successfully processed",
                    files=[]
                )
            
            return DocumentUploadResponse(
                message=f"Successfully uploaded {len(uploaded_files)} files",
                files=uploaded_files,
                chunks=total_chunks
            )
            
        except Exception as e:
            return DocumentUploadResponse(
                message=f"Error processing files: {str(e)}",
                files=[]
            )
    
    async def get_document_list(self) -> List[dict]:
        """
        Get list of all uploaded documents.
        
        Returns:
            List of document information
        """
        return await self.vector_store_service.list_documents()