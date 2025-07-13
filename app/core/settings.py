from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    
    # Milvus Configuration
    milvus_uri: str
    milvus_token: str
    milvus_collection_name: str = "prompt_engineer_test"
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    
    # Document Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_search_results: int = 3
    
    # API Configuration
    max_tokens: int = 1000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()