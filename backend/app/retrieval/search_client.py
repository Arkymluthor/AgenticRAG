import logging
from typing import List, Dict, Any, Optional
import httpx
import json
from fastapi import Depends

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorSearchClient:
    """
    Client for Azure Cognitive Search with vector search capabilities.
    Implemented as a singleton via FastAPI dependency injection.
    """
    _instance = None
    
    def __new__(cls, endpoint: Optional[str] = None, key: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(VectorSearchClient, cls).__new__(cls)
            cls._instance.client = None
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
        
        # Create a persistent HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.initialized = True
        
        logger.info(f"VectorSearchClient initialized for index {self.index_name}")
    
    async def close(self):
        """
        Close the HTTP client.
        """
        if self.client:
            await self.client.aclose()
            self.client = None
            self.initialized = False
    
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
        keywords: List[str], 
        vector_query: List[float],
        filter_expr: str = "", 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both keywords and vector embedding.
        
        Args:
            keywords: List of keywords for text search
            vector_query: Vector embedding for semantic search
            filter_expr: OData filter expression
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.endpoint or not self.api_key:
            logger.error("Search endpoint or API key not configured")
            return []
        
        try:
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
            
            # Add vector search if provided
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
