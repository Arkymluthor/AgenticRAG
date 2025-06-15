import logging
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob import BlobServiceClient
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """
    Custom exception for transient errors that should be retried.
    """
    pass


def orchestrator_function(context: df.DurableOrchestrationContext) -> List[Dict[str, Any]]:
    """
    Orchestrator function that coordinates the document processing pipeline.
    """
    blob_uri = context.get_input()
    logger.info(f"Starting document processing for blob: {blob_uri}")
    
    try:
        # Step 1: Detect document type
        file_type = yield context.call_activity("DetectDocumentType", blob_uri)
        logger.info(f"Detected document type: {file_type}")
        
        # Step 2: Extract text from document
        raw_text = yield context.call_activity("ExtractText", {
            "blob_uri": blob_uri,
            "file_type": file_type
        })
        logger.info(f"Extracted {len(raw_text)} characters of text")
        
        # Step 3: Chunk the text
        chunks = yield context.call_activity("ChunkText", raw_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4: Generate embeddings for chunks
        chunks_with_embeddings = yield context.call_activity("EmbedChunks", chunks)
        logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
        
        # Step 5: Push chunks to search index
        yield context.call_activity("PushIndex", chunks_with_embeddings)
        logger.info(f"Successfully pushed {len(chunks_with_embeddings)} chunks to index")
        
        return chunks_with_embeddings
        
    except Exception as e:
        logger.error(f"Error processing document {blob_uri}: {str(e)}")
        
        # Move blob to poison container
        try:
            move_to_poison_container(blob_uri)
            logger.info(f"Moved {blob_uri} to poison container")
        except Exception as move_error:
            logger.error(f"Failed to move {blob_uri} to poison container: {str(move_error)}")
        
        # Re-raise the exception to fail the orchestration
        raise


def move_to_poison_container(blob_uri: str) -> None:
    """
    Move a blob to the poison container when processing fails.
    """
    # Parse the blob URI to get account, container, and blob path
    # Example URI: https://account.blob.core.windows.net/container/path/to/blob.pdf
    parts = blob_uri.replace("https://", "").split("/")
    account_container = parts[0].split(".")
    account_name = account_container[0]
    container_name = parts[1]
    blob_path = "/".join(parts[2:])
    
    # Get connection string from environment
    connection_string = os.environ.get("AzureWebJobsStorage")
    if not connection_string:
        raise ValueError("AzureWebJobsStorage connection string not found")
    
    # Create blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Get source blob client
    source_container_client = blob_service_client.get_container_client(container_name)
    source_blob_client = source_container_client.get_blob_client(blob_path)
    
    # Get destination container client (create if not exists)
    poison_container_name = f"{container_name}-poison"
    poison_container_client = blob_service_client.get_container_client(poison_container_name)
    
    if not poison_container_client.exists():
        poison_container_client.create_container()
    
    # Get destination blob client
    destination_blob_client = poison_container_client.get_blob_client(blob_path)
    
    # Copy blob to poison container
    destination_blob_client.start_copy_from_url(source_blob_client.url)
    
    # Delete original blob
    source_blob_client.delete_blob()


main = df.Orchestrator.create(orchestrator_function)
