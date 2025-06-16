import logging
import base64
import os
import tempfile
import uuid
import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import azure.durable_functions as df
import azure.functions as func

from schemas.document import DocumentIngestionRequest, DocumentIngestionResponse, DocumentInfo
from storage.document_store import DocumentStore, get_document_store
from core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/documents/ingest", response_model=DocumentIngestionResponse, summary="Ingest a document")
async def ingest_document(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    document_store: DocumentStore = Depends(get_document_store)
) -> DocumentIngestionResponse:
    """
    Ingest a document for processing.
    
    Args:
        request: Document ingestion request with file content and metadata
        background_tasks: FastAPI background tasks
        document_store: Document store for managing document information
        
    Returns:
        Document ingestion response with status and document ID
    """
    try:
        # Decode base64 file content
        try:
            file_content = base64.b64decode(request.file_content).decode('utf-8')
        except UnicodeDecodeError:
            # Binary file, keep as bytes
            file_content = base64.b64decode(request.file_content)
        
        # Compute content hash
        content_hash = document_store.compute_content_hash(
            file_content if isinstance(file_content, str) else file_content.decode('latin1')
        )
        
        # Check if document with same content already exists
        existing_doc = document_store.document_exists(content_hash)
        if existing_doc:
            logger.info(f"Document with same content already exists: {existing_doc.document_id}")
            return DocumentIngestionResponse(
                document_id=existing_doc.document_id,
                status="exists",
                message="Document with same content already exists",
                metadata={
                    "file_name": existing_doc.file_name,
                    "file_type": existing_doc.file_type,
                    "created_at": existing_doc.created_at.isoformat(),
                    "chunk_count": existing_doc.chunk_count
                }
            )
        
        # Determine file type if not provided
        file_type = request.file_type
        if not file_type:
            file_extension = os.path.splitext(request.file_name)[1].lower()
            if file_extension:
                file_type = file_extension[1:]  # Remove the dot
            else:
                file_type = "unknown"
        
        # Create document record
        document_info = document_store.create_document(
            file_name=request.file_name,
            file_type=file_type,
            content_hash=content_hash,
            document_type=request.document_type or "unknown",
            document_name=request.document_name or "",
            document_year=request.document_year or "",
            document_entity=request.document_entity or "",
            source_uri=request.source_uri,
            metadata=request.metadata
        )
        
        # Save file to temporary location
        temp_file_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
            if isinstance(file_content, str):
                temp_file.write(file_content.encode('utf-8'))
            else:
                temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Start document processing in background
        background_tasks.add_task(
            process_document,
            document_id=document_info.document_id,
            file_path=temp_file_path,
            file_type=file_type,
            document_type=request.document_type or "unknown",
            document_name=request.document_name or "",
            document_year=request.document_year or "",
            document_entity=request.document_entity or "",
            source_uri=request.source_uri,
            metadata=request.metadata
        )
        
        return DocumentIngestionResponse(
            document_id=document_info.document_id,
            status="processing",
            message="Document ingestion started",
            metadata={
                "file_name": document_info.file_name,
                "file_type": document_info.file_type
            }
        )
        
    except Exception as e:
        logger.exception(f"Error ingesting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.post("/documents/upload", response_model=DocumentIngestionResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    document_store: DocumentStore = Depends(get_document_store)
) -> DocumentIngestionResponse:
    """
    Upload a document file for processing.
    
    Args:
        file: Uploaded file
        metadata: Optional JSON string with document metadata
        background_tasks: FastAPI background tasks
        document_store: Document store for managing document information
        
    Returns:
        Document ingestion response with status and document ID
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except Exception as e:
                logger.warning(f"Failed to parse metadata JSON: {str(e)}")
        
        # Create request object
        request = DocumentIngestionRequest(
            file_content=base64.b64encode(file_content).decode('utf-8'),
            file_name=file.filename,
            file_type=os.path.splitext(file.filename)[1][1:] if os.path.splitext(file.filename)[1] else None,
            document_type=parsed_metadata.get("document_type"),
            document_name=parsed_metadata.get("document_name"),
            document_year=parsed_metadata.get("document_year"),
            document_entity=parsed_metadata.get("document_entity"),
            source_uri=parsed_metadata.get("source_uri"),
            metadata=parsed_metadata
        )
        
        # Call the ingest_document function
        return await ingest_document(request, background_tasks, document_store)
        
    except Exception as e:
        logger.exception(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentInfo, summary="Get document information")
async def get_document(
    document_id: str,
    document_store: DocumentStore = Depends(get_document_store)
) -> DocumentInfo:
    """
    Get information about a document.
    
    Args:
        document_id: Document ID
        document_store: Document store for managing document information
        
    Returns:
        Document information
    """
    document_info = document_store.get_document(document_id)
    if not document_info:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    
    return document_info


async def process_document(
    document_id: str, 
    file_path: str, 
    file_type: str, 
    document_type: str = "unknown",
    document_name: str = "",
    document_year: str = "",
    document_entity: str = "",
    source_uri: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Process a document by starting the document ingestion orchestrator.
    
    Args:
        document_id: Document ID
        file_path: Path to the temporary file
        file_type: File type
        document_type: Document type
        document_name: Document name/title
        document_year: Document year
        document_entity: Document entity/organization
        source_uri: Source URI of the document
        metadata: Additional document metadata
    """
    try:
        logger.info(f"Processing document {document_id} at {file_path}")
        
        # Upload file to blob storage
        blob_uri = upload_to_blob_storage(file_path, document_id, file_type)
        
        # Update document record with blob URI
        document_store = get_document_store()
        document_info = document_store.get_document(document_id)
        if document_info:
            document_info.source_uri = blob_uri
            document_store.update_document(document_info)
        
        # Start the orchestrator
        function_app_url = os.environ.get("FUNCTION_APP_URL")
        if not function_app_url:
            raise ValueError("FUNCTION_APP_URL environment variable is not set")
        
        # Create a client for the durable functions
        client = df.DurableOrchestrationClient(function_app_url)
        
        # Start the orchestrator with document info
        orchestrator_input = {
            "blob_uri": blob_uri,
            "document_id": document_id,
            "file_type": file_type,
            "document_type": document_type,
            "document_name": document_name,
            "document_year": document_year,
            "document_entity": document_entity,
            "metadata": metadata or {}
        }
        
        instance_id = await client.start_new("DocIngestOrchestrator", None, orchestrator_input)
        
        logger.info(f"Started orchestration with ID = '{instance_id}' for document {document_id}")
        
    except Exception as e:
        logger.exception(f"Error processing document {document_id}: {str(e)}")
        
        # Update document status to failed
        try:
            document_store = get_document_store()
            document_info = document_store.get_document(document_id)
            if document_info:
                document_info.status = "failed"
                document_store.update_document(document_info)
        except Exception as update_error:
            logger.error(f"Failed to update document status: {str(update_error)}")
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up temporary file: {str(cleanup_error)}")


def upload_to_blob_storage(file_path: str, document_id: str, file_type: str) -> str:
    """
    Upload a file to Azure Blob Storage.
    
    Args:
        file_path: Path to the file
        document_id: Document ID
        file_type: File type
        
    Returns:
        Blob URI
    """
    from azure.storage.blob import BlobServiceClient
    
    try:
        # Get connection string from environment
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
        
        # Get container name from environment or use default
        container_name = os.environ.get("DOCUMENT_CONTAINER_NAME", "documents")
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get container client (create if not exists)
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
        
        # Generate blob name
        blob_name = f"{document_id}.{file_type}"
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Return blob URI
        return blob_client.url
        
    except Exception as e:
        logger.exception(f"Error uploading to blob storage: {str(e)}")
        raise
