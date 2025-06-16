from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings.
    
    Values are loaded from environment variables or .env file.
    """
    # API settings
    API_PREFIX: str = "/api"
    DEBUG: bool = Field(default=False)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"])
    
    # LLM API selection
    USE_AZURE_OPENAI: bool = Field(default=False)
    
    # OpenAI API settings
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_API_BASE: str = Field(default="https://api.openai.com/v1")
    OPENAI_CHAT_MODEL: str = Field(default="gpt-4")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002")
    
    # Azure OpenAI API settings
    AZURE_OPENAI_KEY: str = Field(default="")
    AZURE_OPENAI_ENDPOINT: str = Field(default="")
    AZURE_OPENAI_API_VERSION: str = Field(default="2023-05-15")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(default="")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(default="")
    
    # Azure Cognitive Search settings
    SEARCH_ENDPOINT: str = Field(default="")
    SEARCH_API_KEY: str = Field(default="")
    SEARCH_INDEX_NAME: str = Field(default="documents")
    
    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    
    # Celery settings
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    
    # Application Insights settings
    APPINSIGHTS_CONNECTION_STRING: Optional[str] = Field(default=None)
    
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v):
        """
        Parse CORS_ORIGINS from string to list.
        """
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [i.strip() for i in v.split(",")]
        return v
    
    @property
    def openai_settings(self) -> Dict[str, Any]:
        """
        Get OpenAI API settings based on the selected API.
        """
        if self.USE_AZURE_OPENAI:
            return {
                "api_key": self.AZURE_OPENAI_KEY,
                "api_base": self.AZURE_OPENAI_ENDPOINT,
                "api_version": self.AZURE_OPENAI_API_VERSION,
                "chat_deployment": self.AZURE_OPENAI_CHAT_DEPLOYMENT,
                "embedding_deployment": self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                "api_type": "azure"
            }
        else:
            return {
                "api_key": self.OPENAI_API_KEY,
                "api_base": self.OPENAI_API_BASE,
                "chat_model": self.OPENAI_CHAT_MODEL,
                "embedding_model": self.OPENAI_EMBEDDING_MODEL,
                "api_type": "openai"
            }
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }


# Create settings instance
settings = Settings()
