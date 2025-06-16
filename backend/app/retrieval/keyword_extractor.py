import logging
from typing import List, Optional, Dict, Any
import httpx
import json
from fastapi import Depends

from app.core.config import settings

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Extracts keywords from text using a large language model.
    Supports both OpenAI API and Azure OpenAI API.
    Implemented as a singleton via FastAPI dependency injection.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeywordExtractor, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.initialized = False
        return cls._instance
    
    async def initialize(self):
        """
        Initialize the keyword extractor.
        """
        if self.initialized:
            return
        
        # Determine which API to use
        self.use_azure = settings.USE_AZURE_OPENAI
        
        if self.use_azure:
            # Azure OpenAI API settings
            self.api_key = settings.AZURE_OPENAI_KEY
            self.api_base = settings.AZURE_OPENAI_ENDPOINT
            self.api_version = settings.AZURE_OPENAI_API_VERSION
            self.model_deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT
        else:
            # OpenAI API settings
            self.api_key = settings.OPENAI_API_KEY
            self.api_base = settings.OPENAI_API_BASE
            self.model = settings.OPENAI_CHAT_MODEL
        
        # Create a persistent HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.initialized = True
        
        logger.info(f"KeywordExtractor initialized using {'Azure OpenAI' if self.use_azure else 'OpenAI'} API")
    
    async def close(self):
        """
        Close the HTTP client.
        """
        if self.client:
            await self.client.aclose()
            self.client = None
            self.initialized = False
    
    async def extract(self, text: str, max_k: int = 3) -> List[str]:
        """
        Extract keywords from text using a large language model.
        
        Args:
            text: The text to extract keywords from
            max_k: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        if not self.initialized:
            await self.initialize()
        
        if not text:
            return []
        
        if (self.use_azure and (not self.api_key or not self.api_base)) or \
           (not self.use_azure and not self.api_key):
            logger.error("API key or base URL not configured")
            return []
        
        try:
            # Prepare the prompt for the LLM
            system_prompt = f"Return EXACTLY {max_k} keywords, comma-separated, lower-case."
            
            user_prompt = f"Extract {max_k} keywords from the following text:\n\n{text}"
            
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Prepare the API request based on the API type
            if self.use_azure:
                # Azure OpenAI API
                url = f"{self.api_base}/openai/deployments/{self.model_deployment}/chat/completions?api-version={self.api_version}"
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.api_key,
                }
            else:
                # OpenAI API
                url = f"{self.api_base}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
            
            # Common payload
            payload = {
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 50,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": None
            }
            
            # Add model parameter for OpenAI API
            if not self.use_azure:
                payload["model"] = self.model
            
            # Make the API call
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse the response
            keywords = [kw.strip().lower() for kw in content.split(",") if kw.strip()]
            
            # Ensure we have exactly max_k keywords
            if len(keywords) > max_k:
                keywords = keywords[:max_k]
            elif len(keywords) < max_k:
                # If we have fewer keywords than requested, duplicate the last one
                keywords.extend([keywords[-1] if keywords else ""] * (max_k - len(keywords)))
            
            logger.debug(f"Extracted keywords: {keywords}")
            return keywords
                
        except Exception as e:
            logger.exception(f"Error extracting keywords: {str(e)}")
            # Return empty list on error
            return [""] * max_k


# FastAPI dependency
async def get_keyword_extractor() -> KeywordExtractor:
    """
    FastAPI dependency for the KeywordExtractor singleton.
    """
    extractor = KeywordExtractor()
    await extractor.initialize()
    try:
        yield extractor
    finally:
        # No need to close here as we want to keep the singleton alive
        pass
