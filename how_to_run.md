# How to Run the RAG-Enhanced ChatGPT Application

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Git (for cloning the repository)
- Terminal/Command prompt access

### Required Services
- **OpenAI API Account**: For embeddings and chat completions
- **Milvus Cloud Account**: For vector database (or local Milvus installation)

## Installation Guide

### 1. Clone and Navigate to Project
```bash
git clone <your-repository-url>
cd prompt-engineering
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

#### Create Environment File
Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

#### Configure Environment Variables
Edit `.env` file with your credentials:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Milvus Configuration  
MILVUS_URI=https://your-milvus-uri-here
MILVUS_TOKEN=your_milvus_token_here
MILVUS_COLLECTION_NAME=prompt_engineer_test

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_SEARCH_RESULTS=3

# API Configuration
MAX_TOKENS=1000
```

#### Getting Your API Keys

**OpenAI API Key:**
1. Visit [OpenAI API Platform](https://platform.openai.com)
2. Sign up/Log in to your account
3. Navigate to API Keys section
4. Create a new secret key
5. Copy the key to your `.env` file

**Milvus Cloud:**
1. Visit [Zilliz Cloud](https://cloud.zilliz.com) (Milvus managed service)
2. Create a free account
3. Create a new cluster
4. Get the URI and Token from cluster details
5. Add them to your `.env` file

### 5. Create Required Directories
```bash
mkdir -p logs
```

## Running the Application

### Development Mode

#### Start the Application
```bash
# Using Python module
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Status**: http://localhost:8000/health/status

### Production Mode

#### Using uvicorn
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Using gunicorn (Linux/macOS)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Testing the Installation

### 1. Run Automated Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run only unit tests
pytest -m unit
```

### 2. Check System Health
```bash
curl http://localhost:8000/health/status
```

Expected response:
```json
{
  "milvus_connected": true,
  "collection_exists": true,
  "total_documents": 0,
  "openai_connected": true
}
```

### 3. Test Document Upload
```bash
curl -X POST "http://localhost:8000/documents/upload-text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Document",
    "text": "This is a test document for the RAG system."
  }'
```

### 4. Test Chat Functionality
```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?"
  }'
```

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
**Problem**: `ModuleNotFoundError: No module named 'app'`
**Solution**: Ensure you're in the project root directory and virtual environment is activated

#### 2. OpenAI API Errors
**Problem**: `401 Unauthorized` or `Invalid API key`
**Solution**: 
- Verify your OpenAI API key is correct
- Check if you have sufficient credits
- Ensure no extra spaces in the `.env` file

#### 3. Milvus Connection Errors
**Problem**: `Failed to connect to Milvus`
**Solution**:
- Verify Milvus URI and token are correct
- Check if your Milvus cluster is running
- Ensure network connectivity

#### 4. Permission Errors (Logs Directory)
**Problem**: `Permission denied` when creating log files
**Solution**:
```bash
chmod 755 logs
# Or run with elevated permissions if needed
```

#### 5. Port Already in Use
**Problem**: `Address already in use`
**Solution**:
```bash
# Change port in .env file or use different port
APP_PORT=8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

### Debug Mode

Enable debug mode for detailed logging:
```env
DEBUG=true
```

Then restart the application to see detailed logs in `logs/app.log`.

### Checking Logs

View application logs:
```bash
# Real-time log viewing
tail -f logs/app.log

# View recent logs
tail -n 100 logs/app.log
```

## Development Workflow

### Code Changes
1. Make your changes to the code
2. Run tests: `pytest`
3. Check code style: `black app/` (if installed)
4. Restart the application

### Adding Dependencies
1. Add package to `requirements.txt`
2. Install: `pip install -r requirements.txt`
3. Test the application

### Database Reset
To reset your Milvus collection:
1. Delete the collection from Milvus console
2. Restart the application (it will recreate the collection)

## Performance Optimization

### For Production Use:
1. Set `DEBUG=false` in `.env`
2. Use multiple workers with gunicorn
3. Configure proper logging levels
4. Monitor resource usage
5. Set up proper SSL/TLS termination

### Resource Requirements:
- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2 CPU cores
- **Storage**: 1GB for logs and temporary files

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review application logs in `logs/app.log`
3. Verify all environment variables are set correctly
4. Ensure all external services (OpenAI, Milvus) are accessible