import logging
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import requests

from ..orchestrator import RetryableError

logger = logging.getLogger(__name__)


def run(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates embeddings for text chunks using Azure OpenAI.
    
    Args:
        chunks: List of chunk dictionaries with "content" field
        
    Returns:
        List of chunk dictionaries with added "embedding" field
        
    Raises:
        RetryableError: For transient failures that should be retried
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    
    try:
        # Get Azure OpenAI credentials from environment
        api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        
        if not api_base or not api_key:
            raise ValueError("Azure OpenAI credentials not configured")
        
        # Process chunks in batches to avoid rate limits
        batch_size = 20  # Adjust based on your rate limits
        batched_chunks = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        result_chunks = []
        for batch in batched_chunks:
            # Generate embeddings for the batch
            batch_with_embeddings = generate_embeddings_batch(
                batch, 
                api_base, 
                api_key, 
                api_version, 
                deployment_name
            )
            result_chunks.extend(batch_with_embeddings)
        
        logger.info(f"Successfully generated embeddings for {len(result_chunks)} chunks")
        return result_chunks
        
    except Exception as e:
        # Handle transient errors
        if is_transient_error(e):
            logger.warning(f"Transient error generating embeddings: {str(e)}")
            raise RetryableError(f"Transient error: {str(e)}")
        
        # Handle permanent errors
        logger.error(f"Error generating embeddings: {str(e)}")
        raise


def generate_embeddings_batch(
    chunks: List[Dict[str, Any]],
    api_base: str,
    api_key: str,
    api_version: str,
    deployment_name: str
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a batch of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        api_base: Azure OpenAI API base URL
        api_key: Azure OpenAI API key
        api_version: Azure OpenAI API version
        deployment_name: Azure OpenAI embedding model deployment name
        
    Returns:
        List of chunk dictionaries with embeddings
    """
    # Extract text from chunks
    texts = [chunk["content"] for chunk in chunks]
    
    # Prepare API request
    url = f"{api_base}/openai/deployments/{deployment_name}/embeddings?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    payload = {
        "input": texts,
        "dimensions": 1536  # Default for text-embedding-ada-002
    }
    
    # Call the API
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    
    if response.status_code != 200:
        error_msg = f"API error: {response.status_code} - {response.text}"
        logger.error(error_msg)
        
        # Check if this is a rate limit error (retry)
        if response.status_code == 429:
            raise RetryableError(f"Rate limit exceeded: {error_msg}")
        
        raise Exception(error_msg)
    
    # Parse the response
    result = response.json()
    embeddings = [item["embedding"] for item in result["data"]]
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
        
        # Add embedding metadata
        chunk["embedding_model"] = deployment_name
        chunk["vector_size"] = len(embeddings[i])
    
    return chunks


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.
    """
    # Check for common transient error patterns
    error_str = str(error).lower()
    
    # Network errors
    if any(msg in error_str for msg in ['timeout', 'connection', 'network', 'temporarily']):
        return True
    
    # Azure OpenAI specific errors
    if any(msg in error_str for msg in ['rate limit', 'throttled', 'capacity', '429', 'busy']):
        return True
    
    return False
