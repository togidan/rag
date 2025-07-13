from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import numpy as np
from typing import List
import io
import mmh3

app = FastAPI()

# OpenAI client for embeddings
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv()
# Milvus configuration
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DIMENSION = int(os.getenv("DIMENSION"))

def generate_text_hash(text: str, attempt: int = 0) -> int:
    """Generate a consistent INT64 hash for text using MurmurHash3"""
    # Use MurmurHash3 to generate a 32-bit hash, with attempt-based seed for collision handling
    seed = 42 + attempt  # Modify seed if collision detected
    hash_32 = mmh3.hash(text, seed=seed)  # Use seed for consistency
    # Convert to positive 64-bit integer range
    hash_64 = abs(hash_32) % (2**63 - 1)  # Ensure it fits in INT64 range
    return hash_64

def generate_unique_hash(text: str, existing_hashes: set) -> int:
    """Generate a unique hash, handling collisions by trying different seeds"""
    attempt = 0
    max_attempts = 100  # Prevent infinite loops
    
    while attempt < max_attempts:
        hash_value = generate_text_hash(text, attempt)
        if hash_value not in existing_hashes:
            return hash_value
        attempt += 1
    
    # If we can't find a unique hash after max attempts, raise an error
    raise ValueError(f"Could not generate unique hash for text after {max_attempts} attempts")

# Initialize Milvus client
def init_milvus():
    try:
        client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        
        # Check if collection exists, create if not
        if not client.has_collection(collection_name=COLLECTION_NAME):
            # Define explicit schema
            fields = [
                FieldSchema(name="primary_key", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
            ]
            schema = CollectionSchema(fields, description="Document collection for RAG")
            
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params={
                    "field_name": "vector",
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                }
            )
            print(f"Created collection '{COLLECTION_NAME}' with explicit schema")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists")
        
        return client
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return None

# Initialize client
milvus_client = init_milvus()

def get_openai_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenAI's text-embedding-3-large model"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        embedding = response.data[0].embedding
        # Ensure all values are native Python floats
        return [float(x) for x in embedding]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def get_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in batch"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        # Ensure all values are native Python floats
        return [[float(x) for x in data.embedding] for data in response.data]
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return []

class ChatRequest(BaseModel):
    message: str

class TextUploadRequest(BaseModel):
    title: str
    text: str

class SystemStatus(BaseModel):
    milvus_connected: bool
    collection_exists: bool
    total_documents: int
    openai_connected: bool

class DocumentInfo(BaseModel):
    primary_key: int
    title: str
    upload_date: str
    chunk_count: int

@app.get("/")
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/status")
async def get_system_status():
    """Get system status including Milvus connection and document count"""
    try:
        # Check Milvus connection
        milvus_connected = milvus_client is not None
        collection_exists = False
        total_documents = 0
        
        if milvus_connected:
            try:
                collection_exists = milvus_client.has_collection(collection_name=COLLECTION_NAME)
                if collection_exists:
                    # Get document count (this is approximate since we have chunks)
                    stats = milvus_client.get_collection_stats(collection_name=COLLECTION_NAME)
                    total_documents = stats.get('row_count', 0)
            except Exception as e:
                print(f"Error checking collection: {e}")
                milvus_connected = False
        
        # Test OpenAI connection
        openai_connected = True
        try:
            # Quick test of OpenAI API
            test_response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input="test"
            )
            openai_connected = len(test_response.data) > 0
        except Exception as e:
            print(f"OpenAI connection error: {e}")
            openai_connected = False
        
        return SystemStatus(
            milvus_connected=milvus_connected,
            collection_exists=collection_exists,
            total_documents=total_documents,
            openai_connected=openai_connected
        )
    except Exception as e:
        return {"error": f"Status check failed: {str(e)}"}

@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents with metadata"""
    if not milvus_client:
        return {"error": "Vector database not available"}
    
    try:
        # Query all unique documents by primary_key
        results = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter="primary_key >= 0",  # Get all documents
            output_fields=["primary_key", "text"],
            limit=1000  # Reasonable limit
        )
        
        # Group by primary_key to get unique documents
        documents_dict = {}
        for result in results:
            # Add defensive checks for undefined values
            pk = result.get("primary_key") if result else None
            text = result.get("text", "") if result else ""
            
            # Skip if primary key is None/undefined
            if pk is None:
                continue
                
            # Ensure text is a string for safe operations
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            if pk not in documents_dict:
                # Safe string slicing with checks
                preview = text[:100] + "..." if len(text) > 100 else text
                documents_dict[pk] = {
                    "primary_key": pk,
                    "title": f"Document {pk}",  # We'll improve this
                    "upload_date": "Unknown",  # We'll add timestamps later
                    "chunk_count": 0,
                    "preview": preview
                }
            documents_dict[pk]["chunk_count"] += 1
        
        return {"documents": list(documents_dict.values())}
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}

def search_similar_documents(query: str, top_k: int = 3):
    if not milvus_client:
        return []
    
    try:
        # Generate embedding for the query using OpenAI
        query_embedding = get_openai_embedding(query)
        if not query_embedding:
            return []
        
        # Search in Milvus using client
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "primary_key"]
        )
        
        # Extract relevant documents
        relevant_docs = []
        for result in results:
            for hit in result:
                relevant_docs.append({
                    "text": hit.get("entity", {}).get("text", ""),
                    "score": hit.get("distance", 0),
                    "primary_key": hit.get("entity", {}).get("primary_key", "unknown")
                })
        
        return relevant_docs
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

@app.post("/ask")
async def ask_chatgpt(request: ChatRequest):
    try:
        # Search for relevant documents
        relevant_docs = search_similar_documents(request.message)
        
        # Prepare context from retrieved documents
        context = ""
        if relevant_docs:
            context = "\n\nRelevant information from your knowledge base:\n"
            for i, doc in enumerate(relevant_docs, 1):
                context += f"\n{i}. {doc['text']}\n"
        
        # Enhanced system message with context
        system_message = "You are a helpful assistant. Use the provided context information to enhance your responses when relevant."
        if context:
            system_message += " If the context contains relevant information, incorporate it into your answer and mention the source documents."
        
        # Prepare messages for ChatGPT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": request.message + context}
        ]
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        return {
            "response": response.choices[0].message.content,
            "rag_sources": [
                {
                    "text": doc["text"],
                    "score": doc["score"],
                    "primary_key": doc.get("primary_key", "unknown")
                }
                for doc in relevant_docs
            ]
        }
    except Exception as e:
        return {"error": str(e)}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not milvus_client:
        return {"error": "Vector database not available"}
    
    try:
        uploaded_files = []
        
        for file in files:
            # Read file content
            content = await file.read()
            
            # Convert to text (basic implementation for .txt files)
            if file.filename.endswith('.txt'):
                text = content.decode('utf-8')
            else:
                # For other file types, you might want to add more sophisticated parsing
                text = content.decode('utf-8', errors='ignore')
            
            # Chunk the text
            chunks = chunk_text(text)
            
            # Generate embeddings and prepare data for insertion
            texts = []
            
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    texts.append(chunk)
            
            # Generate embeddings in batch for efficiency
            if texts:
                chunk_embeddings = get_openai_embeddings_batch(texts)
                if chunk_embeddings:
                    embeddings = chunk_embeddings
                else:
                    # Fallback to individual embedding generation
                    embeddings = []
                    for text in texts:
                        embedding = get_openai_embedding(text)
                        if embedding:
                            embeddings.append(embedding)
                
                # Prepare data for MilvusClient insert
                if embeddings and len(texts) == len(embeddings):
                    # Validate embedding dimensions and data types
                    valid_embeddings = []
                    valid_texts = []
                    
                    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                        if len(embedding) != DIMENSION:
                            
                            print(f"Warning: Skipping embedding {i} - dimension {len(embedding)}, expected {DIMENSION}")
                            continue
                        # Ensure all values are native Python floats
                        validated_embedding = [float(x) for x in embedding]
                        valid_embeddings.append(validated_embedding)
                        valid_texts.append(text)
                    
                    if valid_embeddings and valid_texts:
                        # Generate one primary key for the entire upload based on filename
                        upload_primary_key = generate_text_hash(file.filename)
                        
                        data = {
                            "primary_key": upload_primary_key,
                            "text": valid_texts,
                            "vector": valid_embeddings[0]
                        }
                        
                        print(f"Inserting {len(valid_texts)} chunks into Milvus")
                        print(f"Data structure: text={type(valid_texts)} with {len(valid_texts)} items, vector={type(valid_embeddings)} with {len(valid_embeddings)} items")
                        
                        # Insert into Milvus using client
                        milvus_client.insert(
                            collection_name=COLLECTION_NAME,
                            data=data
                        )
                        print(f"Successfully inserted data into Milvus")
                    else:
                        print("Warning: No valid embeddings to insert")
                    
                    uploaded_files.append({
                        "filename": file.filename,
                        "chunks": len(texts)
                    })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-text")
async def upload_text(request: TextUploadRequest):
    if not milvus_client:
        return {"error": "Vector database not available"}
    
    try:
        # Use the provided text directly
        text = request.text
        filename = request.title
        
        # Chunk the text
        chunks = chunk_text(text)
        
        
        # Generate embeddings and prepare data for insertion
        texts = []
        
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                texts.append(chunk)
        
        # Generate embeddings in batch for efficiency
        if texts:
            chunk_embeddings = get_openai_embeddings_batch(texts)
            if chunk_embeddings:
                embeddings = chunk_embeddings
            else:
                # Fallback to individual embedding generation
                embeddings = []
                for text_chunk in texts:
                    embedding = get_openai_embedding(text_chunk)
                    if embedding:
                        embeddings.append(embedding)
                 
            # Prepare data for MilvusClient insert
            if embeddings and len(texts) == len(embeddings):
                # Validate embedding dimensions and data types
                valid_embeddings = []
                valid_texts = []
                
                for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                    if len(embedding) != DIMENSION:
                        print(len(embedding))
                        print(DIMENSION)
                        print(f"Warning: Skipping embedding {i} - dimension {len(embedding)}, expected {DIMENSION}")
                        continue
                    # Ensure all values are native Python floats
                    validated_embedding = [float(x) for x in embedding]
                    valid_embeddings.append(validated_embedding)
                    valid_texts.append(text)
                
                if valid_embeddings and valid_texts:
                    # Generate one primary key for the entire upload based on title
                    upload_primary_key = generate_text_hash(filename)
                    
                    data = {
                        "primary_key": upload_primary_key,
                        "text": valid_texts,
                        "vector": valid_embeddings[0]
                    }
                    
                    print(f"Inserting {len(valid_texts)} chunks into Milvus")
                    print(f"Data structure: text={type(valid_texts)} with {len(valid_texts)} items, vector={type(valid_embeddings)} with {len(valid_embeddings)} items")
                    print(valid_texts)
                    print(data)
                    
                    # Insert into Milvus using client
                    milvus_client.insert(
                        collection_name=COLLECTION_NAME,
                        data=data
                    )
                    print(f"Successfully inserted data into Milvus")
                    
                    return {
                        "message": f"Successfully uploaded text document '{filename}'",
                        "chunks": len(valid_texts)
                    }
                else:
                    return {"error": "No valid embeddings after validation"}
            else:
                return {"error": "Failed to generate embeddings for the text"}
        else:
            return {"error": "No valid text chunks found"}
    
    except Exception as e:
        return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)