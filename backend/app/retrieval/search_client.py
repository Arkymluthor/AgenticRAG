import logging
import json
import httpx
from typing import List, Dict, Any, Optional
from fastapi import Depends

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorSearchClient:
    """
    Client for Azure Cognitive Search with vector search capabilities.
    Supports both OpenAI API and Azure OpenAI API for embeddings.
    Implemented as a singleton via FastAPI dependency injection.
    """
    _instance = None
    
    def __new__(cls, endpoint: Optional[str] = None, key: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(VectorSearchClient, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.embedding_client = None
            cls._instance.initialized = False
            cls._instance._endpoint = endpoint
            cls._instance._key = key
        return cls._instance
    
    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the search client with endpoint and key.
        If not provided, will use values from settings.
        
        Args:
            endpoint: Azure Cognitive Search endpoint
            key: Azure Cognitive Search API key
        """
        # These are only used during the first initialization
        self._endpoint = endpoint or self._endpoint
        self._key = key or self._key
    
    async def initialize(self):
        """
        Initialize the search client.
        """
        if self.initialized:
            return
        
        # Use provided values or fall back to settings
        self.endpoint = self._endpoint or settings.SEARCH_ENDPOINT
        self.api_key = self._key or settings.SEARCH_API_KEY
        self.index_name = settings.SEARCH_INDEX_NAME
        
        # Determine which API to use for embeddings
        self.use_azure = settings.USE_AZURE_OPENAI
        
        if self.use_azure:
            # Azure OpenAI API settings
            self.embedding_api_key = settings.AZURE_OPENAI_KEY
            self.embedding_api_base = settings.AZURE_OPENAI_ENDPOINT
            self.embedding_api_version = settings.AZURE_OPENAI_API_VERSION
            self.embedding_deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        else:
            # OpenAI API settings
            self.embedding_api_key = settings.OPENAI_API_KEY
            self.embedding_api_base = settings.OPENAI_API_BASE
            self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        
        # Create persistent HTTP clients
        self.client = httpx.AsyncClient(timeout=30.0)
        self.embedding_client = httpx.AsyncClient(timeout=30.0)
        self.initialized = True
        
        logger.info(f"VectorSearchClient initialized for index {self.index_name}")
        logger.info(f"Using {'Azure OpenAI' if self.use_azure else 'OpenAI'} API for embeddings")
    
    async def close(self):
        """
        Close the HTTP clients.
        """
        if self.client:
            await self.client.aclose()
            self.client = None
        
        if self.embedding_client:
            await self.embedding_client.aclose()
            self.embedding_client = None
            
        self.initialized = False
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Prepare the API request based on the API type
            if self.use_azure:
                # Azure OpenAI API
                url = f"{self.embedding_api_base}/openai/deployments/{self.embedding_deployment}/embeddings?api-version={self.embedding_api_version}"
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.embedding_api_key,
                }
            else:
                # OpenAI API
                url = f"{self.embedding_api_base}/embeddings"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.embedding_api_key}",
                }
            
            # Common payload
            payload = {
                "input": text,
                "dimensions": 1536  # Default for text-embedding-ada-002
            }
            
            # Add model parameter for OpenAI API
            if not self.use_azure:
                payload["model"] = self.embedding_model
            
            # Make the API call
            response = await self.embedding_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embedding based on API type
            if self.use_azure:
                embedding = result.get("data", [{}])[0].get("embedding", [])
            else:
                embedding = result.get("data", [{}])[0].get("embedding", [])
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error generating embedding: {str(e)}")
            return []
    
    async def search(self, keywords: List[str], filter_expr: str = "", k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the index using keywords and optional filter.
        
        Args:
            keywords: List of keywords to search for
            filter_expr: OData filter expression
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()
        
        if not keywords:
            return []
        
        if not self.endpoint or not self.api_key:
            logger.error("Search endpoint or API key not configured")
            return []
        
        try:
            # Build search query by joining keywords with OR
            search_query = " OR ".join(keywords)
            
            # Prepare the API request
            url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-07-01-Preview"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            # Prepare search payload
            payload = {
                "search": search_query,
                "queryType": "full",
                "searchMode": "any",
                "top": k,
                "count": True,
                "select": "id,content,document_type,document_name,document_year,document_entity,page,source_uri",
                "orderby": "search.score() desc",
                "highlight": "content",
                "highlightPreTag": "<em>",
                "highlightPostTag": "</em>",
                "semanticConfiguration": "default"
            }
            
            # Add filter if provided
            if filter_expr:
                payload["filter"] = filter_expr
            
            # Make the API call
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract and format search results
            search_results = []
            for item in result.get("value", []):
                # Extract highlights
                highlights = []
                if "@search.highlights" in item and "content" in item["@search.highlights"]:
                    highlights = item["@search.highlights"]["content"]
                
                # Create result item
                result_item = {
                    "id": item.get("id", ""),
                    "content": item.get("content", ""),
                    "document_type": item.get("document_type", ""),
                    "document_name": item.get("document_name", ""),
                    "document_year": item.get("document_year", ""),
                    "document_entity": item.get("document_entity", ""),
                    "page": item.get("page", 0),
                    "source_uri": item.get("source_uri", ""),
                    "score": item.get("@search.score", 0.0),
                    "highlights": highlights
                }
                
                search_results.append(result_item)
            
            logger.info(f"Search for '{search_query}' returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.exception(f"Error searching index: {str(e)}")
            return []
    
    async def search_hybrid(
        self, 
        text: str,
        filter_expr: str = "", 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both text and vector embedding.
        
        Args:
            text: Text to search for
            filter_expr: OData filter expression
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()
        
        if not text:
            return []
        
        if not self.endpoint or not self.api_key:
            logger.error("Search endpoint or API key not configured")
            return []
        
        try:
            # Generate embedding for the text
            vector_query = await self.generate_embedding(text)
            
            # Extract keywords from the text (simple approach)
            keywords = [word.lower() for word in text.split() if len(word) > 3]
            keywords = list(set(keywords))[:5]  # Deduplicate and limit to 5 keywords
            
            # Build search query by joining keywords with OR
            search_query = " OR ".join(keywords) if keywords else "*"
            
            # Prepare the API request
            url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-07-01-Preview"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            # Prepare search payload with vector
            payload = {
                "search": search_query,
                "queryType": "full",
                "searchMode": "any",
                "top": k,
                "count": True,
                "select": "id,content,document_type,document_name,document_year,document_entity,page,source_uri",
                "orderby": "search.score() desc",
                "highlight": "content",
                "highlightPreTag": "<em>",
                "highlightPostTag": "</em>",
                "semanticConfiguration": "default"
            }
            
            # Add vector search if we have an embedding
            if vector_query:
                payload["vectors"] = [
                    {
                        "value": vector_query,
                        "fields": "embedding",
                        "k": k
                    }
                ]
                
                # Add hybrid scoring profile
                payload["scoringProfile"] = "hybrid"
            
            # Add filter if provided
            if filter_expr:
                payload["filter"] = filter_expr
            
            # Make the API call
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract and format search results
            search_results = []
            for item in result.get("value", []):
                # Extract highlights
                highlights = []
                if "@search.highlights" in item and "content" in item["@search.highlights"]:
                    highlights = item["@search.highlights"]["content"]
                
                # Create result item
                result_item = {
                    "id": item.get("id", ""),
                    "content": item.get("content", ""),
                    "document_type": item.get("document_type", ""),
                    "document_name": item.get("document_name", ""),
                    "document_year": item.get("document_year", ""),
                    "document_entity": item.get("document_entity", ""),
                    "page": item.get("page", 0),
                    "source_uri": item.get("source_uri", ""),
                    "score": item.get("@search.score", 0.0),
                    "highlights": highlights
                }
                
                search_results.append(result_item)
            
            logger.info(f"Hybrid search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.exception(f"Error performing hybrid search: {str(e)}")
            return []


# FastAPI dependency
async def get_search_client() -> VectorSearchClient:
    """
    FastAPI dependency for the VectorSearchClient singleton.
    """
    client = VectorSearchClient()
    await client.initialize()
    try:
        yield client
    finally:
        # No need to close here as we want to keep the singleton alive
        pass
