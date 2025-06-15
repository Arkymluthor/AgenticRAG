import logging
import os
import json
import time
from typing import List, Dict, Any, Optional
import requests

from ..orchestrator import RetryableError

logger = logging.getLogger(__name__)


def run(chunks: List[Dict[str, Any]]) -> None:
    """
    Upserts document chunks into Azure Cognitive Search index.
    
    Args:
        chunks: List of chunk dictionaries with content, metadata, and embeddings
        
    Raises:
        RetryableError: For transient failures that should be retried
    """
    logger.info(f"Pushing {len(chunks)} chunks to search index")
    
    try:
        # Get Azure Cognitive Search credentials from environment
        search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        search_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "documents")
        
        if not search_endpoint or not search_key:
            raise ValueError("Azure Cognitive Search credentials not configured")
        
        # Process chunks in batches to avoid throttling
        batch_size = 50  # Adjust based on your rate limits
        batched_chunks = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        total_indexed = 0
        for batch in batched_chunks:
            # Push batch to index
            indexed_count = push_batch_to_index(
                batch, 
                search_endpoint, 
                search_key, 
                index_name
            )
            total_indexed += indexed_count
            
            # Add a small delay between batches to avoid throttling
            time.sleep(1)
        
        logger.info(f"Successfully indexed {total_indexed} chunks")
        
    except Exception as e:
        # Handle transient errors
        if is_transient_error(e):
            logger.warning(f"Transient error pushing to index: {str(e)}")
            raise RetryableError(f"Transient error: {str(e)}")
        
        # Handle permanent errors
        logger.error(f"Error pushing to index: {str(e)}")
        raise


def push_batch_to_index(
    chunks: List[Dict[str, Any]],
    search_endpoint: str,
    search_key: str,
    index_name: str
) -> int:
    """
    Push a batch of chunks to the search index.
    
    Args:
        chunks: List of chunk dictionaries
        search_endpoint: Azure Cognitive Search endpoint
        search_key: Azure Cognitive Search admin key
        index_name: Name of the search index
        
    Returns:
        Number of chunks successfully indexed
    """
    # Ensure the index exists
    ensure_index_exists(search_endpoint, search_key, index_name)
    
    # Prepare documents for indexing
    documents = []
    for chunk in chunks:
        # Create a document with all the necessary fields
        document = {
            "@search.action": "mergeOrUpload",
            "id": chunk.get("chunk_id", str(time.time())),
            "content": chunk.get("content", ""),
            "embedding": chunk.get("embedding", []),
            "page": chunk.get("page", 0),
            "document_type": chunk.get("document_type", "unknown"),
            "document_name": chunk.get("document_name", ""),
            "document_year": chunk.get("document_year", ""),
            "document_entity": chunk.get("document_entity", ""),
            "source_uri": chunk.get("source_uri", ""),
        }
        documents.append(document)
    
    # Prepare API request
    url = f"{search_endpoint}/indexes/{index_name}/docs/index?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_key
    }
    payload = {
        "value": documents
    }
    
    # Call the API
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    
    if response.status_code not in [200, 201, 207]:
        error_msg = f"API error: {response.status_code} - {response.text}"
        logger.error(error_msg)
        
        # Check if this is a rate limit error (retry)
        if response.status_code == 429:
            raise RetryableError(f"Rate limit exceeded: {error_msg}")
        
        raise Exception(error_msg)
    
    # Parse the response
    result = response.json()
    
    # Check for individual document errors
    if "value" in result:
        errors = [doc for doc in result["value"] if doc.get("status") >= 300]
        if errors:
            error_details = json.dumps(errors, indent=2)
            logger.warning(f"Some documents failed to index: {error_details}")
    
    # Return the number of successfully indexed documents
    return len(documents) - len(errors) if "value" in result else len(documents)


def ensure_index_exists(
    search_endpoint: str,
    search_key: str,
    index_name: str
) -> None:
    """
    Ensure that the search index exists, creating it if necessary.
    
    Args:
        search_endpoint: Azure Cognitive Search endpoint
        search_key: Azure Cognitive Search admin key
        index_name: Name of the search index
    """
    # Check if index exists
    url = f"{search_endpoint}/indexes/{index_name}?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_key
    }
    
    response = requests.get(url, headers=headers)
    
    # If index exists, return
    if response.status_code == 200:
        return
    
    # If index doesn't exist, create it
    if response.status_code == 404:
        create_index(search_endpoint, search_key, index_name)
    else:
        # Handle other errors
        error_msg = f"Error checking index: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)


def create_index(
    search_endpoint: str,
    search_key: str,
    index_name: str
) -> None:
    """
    Create a new search index with vector search capabilities.
    
    Args:
        search_endpoint: Azure Cognitive Search endpoint
        search_key: Azure Cognitive Search admin key
        index_name: Name of the search index
    """
    # Define the index schema
    index_definition = {
        "name": index_name,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "searchable": True,
                "filterable": True
            },
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "filterable": False,
                "sortable": False,
                "facetable": False,
                "analyzer": "standard.lucene"
            },
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "filterable": False,
                "sortable": False,
                "facetable": False,
                "dimensions": 1536,
                "vectorSearchProfile": "embedding-profile"
            },
            {
                "name": "page",
                "type": "Edm.Int32",
                "searchable": False,
                "filterable": True,
                "sortable": True,
                "facetable": True
            },
            {
                "name": "document_type",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": True
            },
            {
                "name": "document_name",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": False
            },
            {
                "name": "document_year",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": True
            },
            {
                "name": "document_entity",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "facetable": True
            },
            {
                "name": "source_uri",
                "type": "Edm.String",
                "searchable": False,
                "filterable": True,
                "sortable": False,
                "facetable": False
            }
        ],
        "vectorSearch": {
            "algorithmConfigurations": [
                {
                    "name": "hnsw-config",
                    "kind": "hnsw",
                    "parameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                }
            ],
            "profiles": [
                {
                    "name": "embedding-profile",
                    "algorithmConfigurationName": "hnsw-config"
                }
            ]
        },
        "semantic": {
            "configurations": [
                {
                    "name": "semantic-config",
                    "prioritizedFields": {
                        "titleField": {
                            "fieldName": "document_name"
                        },
                        "contentFields": [
                            {
                                "fieldName": "content"
                            }
                        ],
                        "keywordsFields": [
                            {
                                "fieldName": "document_type"
                            },
                            {
                                "fieldName": "document_entity"
                            }
                        ]
                    }
                }
            ]
        }
    }
    
    # Create the index
    url = f"{search_endpoint}/indexes?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_key
    }
    
    response = requests.post(url, headers=headers, json=index_definition)
    
    if response.status_code not in [200, 201]:
        error_msg = f"Error creating index: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    logger.info(f"Successfully created index {index_name}")


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.
    """
    # Check for common transient error patterns
    error_str = str(error).lower()
    
    # Network errors
    if any(msg in error_str for msg in ['timeout', 'connection', 'network', 'temporarily']):
        return True
    
    # Azure Cognitive Search specific errors
    if any(msg in error_str for msg in ['rate limit', 'throttled', 'capacity', '429', 'busy', 'service unavailable']):
        return True
    
    return False
