import logging
import hashlib
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from azure.data.tables import TableServiceClient, TableClient, UpdateMode
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from schemas.document import DocumentInfo

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Azure Table Storage client for managing document information.
    """
    def __init__(self, connection_string: str, table_name: str = "documents"):
        """
        Initialize the document store.
        
        Args:
            connection_string: Azure Storage connection string
            table_name: Name of the Azure Table
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._ensure_table_exists()
    
    def _ensure_table_exists(self) -> None:
        """
        Ensure that the Azure Table exists, creating it if necessary.
        """
        try:
            table_service = TableServiceClient.from_connection_string(self.connection_string)
            table_service.create_table_if_not_exists(self.table_name)
            logger.info(f"Table {self.table_name} is ready")
        except Exception as e:
            logger.error(f"Error ensuring table exists: {str(e)}")
            raise
    
    def get_table_client(self) -> TableClient:
        """
        Get a table client for the documents table.
        
        Returns:
            TableClient instance
        """
        table_service = TableServiceClient.from_connection_string(self.connection_string)
        return table_service.get_table_client(self.table_name)
    
    def compute_content_hash(self, content: str) -> str:
        """
        Compute a hash of the document content.
        
        Args:
            content: Document content
            
        Returns:
            SHA-256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def document_exists(self, content_hash: str) -> Optional[DocumentInfo]:
        """
        Check if a document with the given content hash already exists.
        
        Args:
            content_hash: Hash of the document content
            
        Returns:
            DocumentInfo if the document exists, None otherwise
        """
        table_client = self.get_table_client()
        
        # Query for documents with the same content hash
        query_filter = f"ContentHash eq '{content_hash}'"
        
        try:
            entities = list(table_client.query_entities(query_filter))
            
            if entities:
                # Convert the first entity to DocumentInfo
                entity = entities[0]
                return DocumentInfo(
                    document_id=entity["RowKey"],
                    file_name=entity["FileName"],
                    file_type=entity["FileType"],
                    content_hash=entity["ContentHash"],
                    document_type=entity.get("DocumentType", "unknown"),
                    document_name=entity.get("DocumentName", ""),
                    document_year=entity.get("DocumentYear", ""),
                    document_entity=entity.get("DocumentEntity", ""),
                    source_uri=entity.get("SourceUri"),
                    document_metadata=json.loads(entity.get("DocumentMetadata", "{}")),
                    chunk_count=entity.get("ChunkCount", 0),
                    created_at=entity.get("CreatedAt", datetime.utcnow()),
                    updated_at=entity.get("UpdatedAt", datetime.utcnow()),
                    status=entity.get("Status", "processed")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking if document exists: {str(e)}")
            return None
    
    def create_document(self, file_name: str, file_type: str, content_hash: str, 
                       document_type: str = "unknown",
                       document_name: str = "",
                       document_year: str = "",
                       document_entity: str = "",
                       source_uri: Optional[str] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> DocumentInfo:
        """
        Create a new document record.
        
        Args:
            file_name: Original file name
            file_type: File type
            content_hash: Hash of the document content
            source_uri: Source URI of the document
            metadata: Document metadata
            
        Returns:
            Created DocumentInfo
        """
        document_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Create document info
        document_info = DocumentInfo(
            document_id=document_id,
            file_name=file_name,
            file_type=file_type,
            content_hash=content_hash,
            document_type=document_type,
            document_name=document_name,
            document_year=document_year,
            document_entity=document_entity,
            source_uri=source_uri,
            document_metadata=metadata or {},
            created_at=now,
            updated_at=now,
            status="processing"
        )
        
        # Create entity
        entity = {
            "PartitionKey": file_type,
            "RowKey": document_id,
            "FileName": file_name,
            "FileType": file_type,
            "ContentHash": content_hash,
            "DocumentType": document_type,
            "DocumentName": document_name,
            "DocumentYear": document_year,
            "DocumentEntity": document_entity,
            "SourceUri": source_uri,
            "DocumentMetadata": json.dumps(metadata or {}),
            "ChunkCount": 0,
            "CreatedAt": now,
            "UpdatedAt": now,
            "Status": "processing"
        }
        
        # Insert entity
        table_client = self.get_table_client()
        table_client.create_entity(entity)
        
        logger.info(f"Created document record with ID {document_id}")
        return document_info
    
    def update_document(self, document_info: DocumentInfo) -> None:
        """
        Update an existing document record.
        
        Args:
            document_info: Updated document info
        """
        # Update entity
        entity = {
            "PartitionKey": document_info.file_type,
            "RowKey": document_info.document_id,
            "FileName": document_info.file_name,
            "FileType": document_info.file_type,
            "ContentHash": document_info.content_hash,
            "DocumentType": document_info.document_type,
            "DocumentName": document_info.document_name,
            "DocumentYear": document_info.document_year,
            "DocumentEntity": document_info.document_entity,
            "SourceUri": document_info.source_uri,
            "DocumentMetadata": json.dumps(document_info.document_metadata),
            "ChunkCount": document_info.chunk_count,
            "UpdatedAt": datetime.utcnow(),
            "Status": document_info.status
        }
        
        # Update entity
        table_client = self.get_table_client()
        table_client.update_entity(mode=UpdateMode.REPLACE, entity=entity)
        
        logger.info(f"Updated document record with ID {document_info.document_id}")
    
    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """
        Get a document record by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentInfo if found, None otherwise
        """
        table_client = self.get_table_client()
        
        try:
            # We need to query since we don't know the partition key
            query_filter = f"RowKey eq '{document_id}'"
            entities = list(table_client.query_entities(query_filter))
            
            if entities:
                # Convert the entity to DocumentInfo
                entity = entities[0]
                return DocumentInfo(
                    document_id=entity["RowKey"],
                    file_name=entity["FileName"],
                    file_type=entity["FileType"],
                    content_hash=entity["ContentHash"],
                    document_type=entity.get("DocumentType", "unknown"),
                    document_name=entity.get("DocumentName", ""),
                    document_year=entity.get("DocumentYear", ""),
                    document_entity=entity.get("DocumentEntity", ""),
                    source_uri=entity.get("SourceUri"),
                    document_metadata=json.loads(entity.get("DocumentMetadata", "{}")),
                    chunk_count=entity.get("ChunkCount", 0),
                    created_at=entity.get("CreatedAt", datetime.utcnow()),
                    updated_at=entity.get("UpdatedAt", datetime.utcnow()),
                    status=entity.get("Status", "processed")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document record.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        # Get the document first to get the partition key
        document = self.get_document(document_id)
        if not document:
            logger.warning(f"Document {document_id} not found for deletion")
            return False
        
        table_client = self.get_table_client()
        
        try:
            table_client.delete_entity(partition_key=document.file_type, row_key=document_id)
            logger.info(f"Deleted document record with ID {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False


def get_document_store() -> DocumentStore:
    """
    Get a document store instance.
    
    Returns:
        DocumentStore instance
    """
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
    
    return DocumentStore(connection_string)
