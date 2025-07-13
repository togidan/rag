from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import openai
import os
from pymilvus import MilvusClient, DataType
import numpy as np
from typing import List
import io

app = FastAPI()

# OpenAI client for embeddings
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Milvus configuration
MILVUS_URI = "https://in03-eac7c2985174613.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_TOKEN = "97b4cbdb86fc8864a00950a47f22f69566f241573781f5c9294a7ffd6e28eba82806d5777eb7320c1c43a45e1a608cf8034301d6"
COLLECTION_NAME = "promp_engineer_test"
DIMENSION = 1536  # text-embedding-3-small dimension

# Initialize Milvus client
def init_milvus():
    try:
        client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        
        # Check if collection exists, create if not
        if not client.has_collection(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=DIMENSION,
                metric_type="COSINE",
                auto_id=True
            )
        
        return client
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return None

# Initialize client
milvus_client = init_milvus()

def get_openai_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenAI's text-embedding-3-small model"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def get_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in batch"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return []

class ChatRequest(BaseModel):
    message: str

class TextUploadRequest(BaseModel):
    title: str
    text: str

@app.get("/")
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

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
            output_fields=["text", "filename"]
        )
        
        # Extract relevant documents
        relevant_docs = []
        for result in results:
            for hit in result:
                relevant_docs.append({
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "score": hit.get("distance", 0)
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
                context += f"\n{i}. From {doc['filename']}:\n{doc['text']}\n"
        
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
        return {"response": response.choices[0].message.content}
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
            filenames = []
            
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    texts.append(chunk)
                    filenames.append(file.filename)
            
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
                    data = []
                    for i in range(len(texts)):
                        data.append({
                            "text": texts[i],
                            "vector": embeddings[i],
                            "filename": filenames[i]
                        })
                    
                    # Insert into Milvus using client
                    milvus_client.insert(
                        collection_name=COLLECTION_NAME,
                        data=data
                    )
                    
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
        filenames = []
        
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                texts.append(chunk)
                filenames.append(filename)
        
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
                data = []
                for i in range(len(texts)):
                    data.append({
                        "text": texts[i],
                        "vector": embeddings[i],
                        "filename": filenames[i]
                    })
                
                # Insert into Milvus using client
                milvus_client.insert(
                    collection_name=COLLECTION_NAME,
                    data=data
                )
                
                return {
                    "message": f"Successfully uploaded text document '{filename}'",
                    "chunks": len(texts)
                }
            else:
                return {"error": "Failed to generate embeddings for the text"}
        else:
            return {"error": "No valid text chunks found"}
    
    except Exception as e:
        return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)