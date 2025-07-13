Primary Prompt (Complete Application)

  Create a RAG (Retrieval-Augmented Generation) web application that allows users to upload documents and ask questions about
  them. The app should have these specific features:

  BACKEND REQUIREMENTS:
  - Use FastAPI with Python
  - Integrate OpenAI's text-embedding-3-large for embeddings and GPT-4o for chat
  - Use Milvus vector database for storing document embeddings
  - Support file uploads and direct text input
  - Chunk documents into 500-character pieces with 50-character overlap
  - Implement vector similarity search to find relevant document chunks
  - Return AI responses with source citations

  FRONTEND REQUIREMENTS:
  - Single HTML file with embedded CSS and JavaScript
  - Modern gradient design with purple/blue theme
  - Chat interface with user/assistant message styling
  - Settings panel showing system status, uploaded documents, and RAG sources
  - Modal dialog for document uploads with tabs for file upload and text input
  - Real-time status monitoring and document inventory

  API ENDPOINTS NEEDED:
  1. GET / - serve the HTML interface
  2. GET /status - system health check (Milvus, OpenAI connections)
  3. GET /documents - list uploaded documents with metadata
  4. POST /ask - RAG-powered chat that finds relevant chunks and generates responses
  5. POST /upload - handle file uploads
  6. POST /upload-text - handle direct text input

  TECHNICAL SPECIFICATIONS:
  - Use MurmurHash3 for generating document IDs
  - Store embeddings in Milvus with cosine similarity search
  - Include proper error handling and loading states
  - Make the UI responsive and professional-looking
  - Environment variable configuration for API keys and database connections

  Create the complete working application with main.py, index.html, requirements.txt, and .env template files.

  Segmented Prompts (For Step-by-Step Creation)

  Prompt 1: Backend Foundation

  Create a FastAPI backend for a RAG system with these requirements:

  1. Set up FastAPI with proper imports and basic structure
  2. Configure OpenAI client for embeddings (text-embedding-3-large) and chat (GPT-4o)
  3. Set up Milvus vector database connection with proper schema
  4. Create utility functions for:
     - Text hashing using MurmurHash3 for document IDs
     - Text chunking (500 chars with 50 char overlap)
     - OpenAI embedding generation (single and batch)
     - Vector similarity search

  Include proper error handling and use environment variables for configuration.

  Prompt 2: API Endpoints

  Add these API endpoints to the FastAPI app:

  1. GET /status - Check Milvus and OpenAI connections, return system health
  2. GET /documents - List all uploaded documents with metadata
  3. POST /ask - RAG chat endpoint that:
     - Takes user message
     - Finds similar document chunks
     - Sends context + message to GPT-4o
     - Returns response with source citations
  4. POST /upload - File upload that processes and stores documents
  5. POST /upload-text - Direct text input processing

  Use Pydantic models for request/response validation and include comprehensive error handling.

  Prompt 3: Frontend Interface

  Create a modern HTML interface for the RAG application with:

  LAYOUT:
  - Two-column design: chat interface + settings panel
  - Purple/blue gradient theme with modern styling
  - Responsive design for mobile/desktop

  CHAT INTERFACE:
  - Message history with user/assistant styling
  - Text input with send button
  - Loading indicators during processing
  - Upload button to add documents

  SETTINGS PANEL (toggleable):
  - System status indicators (Milvus, OpenAI)
  - RAG sources for last response
  - List of uploaded documents
  - Real-time status updates

  MODAL SYSTEM:
  - Document upload dialog with tabs
  - File upload and text input options
  - Success/error message handling

  Include all JavaScript for API communication, real-time updates, and UI interactions.

  Prompt 4: Configuration & Deployment

  Create the supporting files for the RAG application:

  1. requirements.txt with all necessary Python packages:
     - FastAPI, uvicorn, pydantic
     - openai, pymilvus
     - python-dotenv, python-multipart
     - mmh3 for hashing

  2. .env template file with:
     - OPENAI_API_KEY
     - MILVUS_URI and MILVUS_TOKEN
     - COLLECTION_NAME and DIMENSION settings

  3. Add proper environment variable loading in main.py
  4. Include instructions for running the application locally

  Make sure all components work together seamlessly.

  Specific Technical Prompts

  For the Vector Database Setup:

  Create a Milvus vector database setup function that:
  - Connects to Milvus cloud service
  - Creates a collection with schema: primary_key (INT64), text (VARCHAR), vector (FLOAT_VECTOR, 3072 dim)
  - Sets up IVF_FLAT index with cosine similarity
  - Handles connection errors gracefully
  - Returns a configured MilvusClient instance

  For the RAG Search Logic:

  Implement RAG (Retrieval-Augmented Generation) search that:
  1. Takes a user query and converts it to embedding using OpenAI
  2. Searches Milvus for top 3 most similar document chunks
  3. Combines user query + relevant context
  4. Sends to GPT-4o with system prompt for contextual responses
  5. Returns AI response + source citations with similarity scores

  For the Frontend Chat Interface:

  Create a professional chat interface with:
  - Styled message bubbles (user: right-aligned blue, assistant: left-aligned white)
  - Auto-scrolling message container
  - Loading animation during API calls
  - Error message styling
  - Responsive design that works on mobile and desktop
  - Modern CSS with gradients and smooth transitions

  Prompt for Specific Improvements:

  Enhance the RAG application with these professional improvements:
  1. Replace hardcoded credentials with environment variables
  2. Add comprehensive error handling with specific error types
  3. Implement input validation using Pydantic models
  4. Add logging for debugging and monitoring
  5. Improve file upload to handle PDF and DOCX formats
  6. Add rate limiting and security headers
  7. Implement proper async operations for better performance