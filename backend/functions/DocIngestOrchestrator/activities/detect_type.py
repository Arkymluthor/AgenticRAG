import logging
import os
import mimetypes
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import re
from typing import Optional

from ..orchestrator import RetryableError

logger = logging.getLogger(__name__)


def run(blob_uri: str) -> str:
    """
    Detects the document type from a blob URI.
    
    Args:
        blob_uri: The URI of the blob to analyze
        
    Returns:
        Document type: "pdf", "docx", "txt", or "scan"
        
    Raises:
        RetryableError: For transient failures that should be retried
        ValueError: For invalid or unsupported document types
    """
    logger.info(f"Detecting document type for: {blob_uri}")
    
    try:
        # Extract file extension from URI
        file_extension = get_file_extension(blob_uri)
        
        # If no extension, try to detect from content
        if not file_extension:
            file_extension = detect_from_content(blob_uri)
        
        # Map extension to document type
        doc_type = map_extension_to_type(file_extension)
        
        logger.info(f"Detected document type: {doc_type} for {blob_uri}")
        return doc_type
        
    except Exception as e:
        # Handle transient errors
        if is_transient_error(e):
            logger.warning(f"Transient error detecting document type: {str(e)}")
            raise RetryableError(f"Transient error: {str(e)}")
        
        # Handle permanent errors
        logger.error(f"Error detecting document type: {str(e)}")
        raise


def get_file_extension(blob_uri: str) -> Optional[str]:
    """
    Extract file extension from blob URI.
    """
    # Extract filename from URI
    filename = blob_uri.split('/')[-1]
    
    # Get extension
    _, extension = os.path.splitext(filename)
    
    if extension:
        return extension.lower().lstrip('.')
    
    return None


def detect_from_content(blob_uri: str) -> str:
    """
    Detect file type by examining the content.
    """
    try:
        # Parse the blob URI to get account, container, and blob path
        parts = blob_uri.replace("https://", "").split("/")
        container_name = parts[1]
        blob_path = "/".join(parts[2:])
        
        # Get connection string from environment
        connection_string = os.environ.get("AzureWebJobsStorage")
        if not connection_string:
            raise ValueError("AzureWebJobsStorage connection string not found")
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get blob client
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        # Download a small portion of the blob to detect type
        data = blob_client.download_blob(max_concurrency=1, length=4096).readall()
        
        # Check for PDF signature
        if data.startswith(b'%PDF'):
            return 'pdf'
        
        # Check for Office Open XML (DOCX) signature
        if data.startswith(b'PK\x03\x04'):
            # This is a ZIP file, which could be DOCX or other Office formats
            # For more accurate detection, we'd need to examine the ZIP contents
            return 'docx'
        
        # Check if it's plain text
        try:
            data.decode('utf-8')
            # If we can decode as UTF-8, it's likely text
            return 'txt'
        except UnicodeDecodeError:
            pass
        
        # If we can't determine the type, assume it's a scanned document
        return 'scan'
        
    except Exception as e:
        logger.error(f"Error detecting file type from content: {str(e)}")
        # Default to scan if we can't determine the type
        return 'scan'


def map_extension_to_type(extension: str) -> str:
    """
    Map file extension to document type.
    """
    extension_map = {
        'pdf': 'pdf',
        'docx': 'docx',
        'doc': 'docx',
        'txt': 'txt',
        'text': 'txt',
        'md': 'txt',
        'jpg': 'scan',
        'jpeg': 'scan',
        'png': 'scan',
        'tiff': 'scan',
        'tif': 'scan',
        'bmp': 'scan',
    }
    
    return extension_map.get(extension.lower(), 'scan')


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.
    """
    # Check for common transient error patterns
    error_str = str(error).lower()
    
    # Network errors
    if any(msg in error_str for msg in ['timeout', 'connection', 'network', 'temporarily', 'throttled']):
        return True
    
    # Azure storage specific errors
    if any(msg in error_str for msg in ['server busy', 'operation could not be completed', '503', '500']):
        return True
    
    return False
