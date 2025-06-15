from typing import List, Optional, Union
from pydantic import AnyHttpUrl, BaseSettings, validator


class Settings(BaseSettings):
    """
    Application settings using Pydantic BaseSettings for environment variable loading.
    """
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Agentic RAG API"
    PROJECT_DESCRIPTION: str = "API for Agentic RAG application"
    VERSION: str = "0.1.0"
    
    # CORS configuration
    CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = ["http://localhost:3000"]
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """
        Parse CORS_ORIGINS from string to list if needed.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    APPINSIGHTS_INSTRUMENTATION_KEY: Optional[str] = None
    
    # Redis configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Search configuration
    SEARCH_ENDPOINT: Optional[str] = None
    SEARCH_API_KEY: Optional[str] = None
    SEARCH_INDEX_NAME: str = "documents"
    
    # LLM configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_API_VERSION: str = "2023-05-15"
    OPENAI_API_TYPE: str = "azure"
    CHAT_MODEL_DEPLOYMENT: str = "gpt-4"
    EMBEDDING_MODEL_DEPLOYMENT: str = "text-embedding-ada-002"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Create a global settings object
settings = Settings()
