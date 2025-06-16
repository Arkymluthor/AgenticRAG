import logging
import os
import json
import time
from typing import List, Dict, Any, Optional
import requests

from ..orchestrator import RetryableError

logger = logging.getLogger(__name__)


def run(input_data: Dict[str, Any]) -> None:
    """
    Upserts document chunks into Azure Cognitive Search index.
    If document_id is provided, deletes existing chunks for that document first.
    
    Args:
        input_data: Dictionary containing:
            - chunks: List of chunk dictionaries with content, metadata, and embeddings
            - document_id: (Optional) Document ID to delete existing chunks for
        
    Raises:
        RetryableError: For transient failures that should be retried
    """
    # Parse input
    if isinstance(input_data, list):
        # Backward compatibility for direct list of chunks
        chunks = input_data
        document_id = None
    else:
        chunks = input_data.get("chunks", [])
        document_id = input_data.get("document_id")
    
    logger.info(f"Pushing {len(chunks)} chunks to search index")
    
    try:
        # Get Azure Cognitive Search credentials from environment
        search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        search_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "documents")
        
        if not search_endpoint or not search_key:
            raise ValueError("Azure Cognitive Search credentials not configured")
            
        # If document_id is provided, delete existing chunks for this document
        if document_id:
            logger.info(f"Deleting existing chunks for document {document_id}")
            delete_document_chunks(document_id, search_endpoint, search_key, index_name)
        
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
            "document_id": chunk.get("document_id", ""),
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
            },
            {
                "name": "document_id",
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


def delete_document_chunks(
    document_id: str,
    search_endpoint: str,
    search_key: str,
    index_name: str
) -> None:
    """
    Delete all chunks for a specific document from the search index.
    
    Args:
        document_id: Document ID to delete chunks for
        search_endpoint: Azure Cognitive Search endpoint
        search_key: Azure Cognitive Search admin key
        index_name: Name of the search index
    """
    # Ensure the index exists
    ensure_index_exists(search_endpoint, search_key, index_name)
    
    # Prepare the filter query
    filter_query = f"document_id eq '{document_id}'"
    
    # First, get all document IDs that match the filter
    url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_key
    }
    payload = {
        "search": "*",
        "filter": filter_query,
        "select": "id",
        "top": 1000  # Adjust based on your expected number of chunks per document
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"API error when searching for chunks to delete: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        result = response.json()
        
        # Extract document IDs
        chunk_ids = [doc["id"] for doc in result.get("value", [])]
        
        if not chunk_ids:
            logger.info(f"No chunks found for document {document_id}")
            return
        
        logger.info(f"Found {len(chunk_ids)} chunks to delete for document {document_id}")
        
        # Delete chunks in batches
        batch_size = 50  # Adjust based on your rate limits
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            
            # Prepare delete actions
            delete_actions = []
            for chunk_id in batch:
                delete_actions.append({
                    "@search.action": "delete",
                    "id": chunk_id
                })
            
            # Delete batch
            delete_url = f"{search_endpoint}/indexes/{index_name}/docs/index?api-version=2023-07-01-Preview"
            delete_payload = {
                "value": delete_actions
            }
            
            delete_response = requests.post(delete_url, headers=headers, json=delete_payload, timeout=60)
            
            if delete_response.status_code not in [200, 207]:
                error_msg = f"API error when deleting chunks: {delete_response.status_code} - {delete_response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Add a small delay between batches to avoid throttling
            time.sleep(1)
        
        logger.info(f"Successfully deleted {len(chunk_ids)} chunks for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error deleting chunks for document {document_id}: {str(e)}")
        raise


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
