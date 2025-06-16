from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import json


class DocumentIngestionRequest(BaseModel):
    """
    Request model for document ingestion endpoint.
    """
    file_content: str = Field(..., description="Base64 encoded file content")
    file_name: str = Field(..., description="Original file name")
    file_type: Optional[str] = Field(None, description="File type (e.g., pdf, docx, txt)")
    document_type: Optional[str] = Field(None, description="Document type (e.g., policy, manual, report)")
    document_name: Optional[str] = Field(None, description="Document name/title")
    document_year: Optional[str] = Field(None, description="Document year")
    document_entity: Optional[str] = Field(None, description="Document entity/organization")
    source_uri: Optional[str] = Field(None, description="Source URI of the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")


class DocumentIngestionResponse(BaseModel):
    """
    Response model for document ingestion endpoint.
    """
    document_id: str = Field(..., description="Document identifier")
    status: str = Field(..., description="Status of the ingestion process")
    message: str = Field(..., description="Status message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class DocumentInfo(BaseModel):
    """
    Model for document information stored in Azure Table.
    """
    document_id: str = Field(..., description="Document identifier")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type")
    content_hash: str = Field(..., description="Hash of the document content")
    document_type: str = Field("unknown", description="Document type (e.g., policy, manual, report)")
    document_name: str = Field("", description="Document name/title")
    document_year: str = Field("", description="Document year")
    document_entity: str = Field("", description="Document entity/organization")
    source_uri: Optional[str] = Field(None, description="Source URI of the document")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
    chunk_count: int = Field(0, description="Number of chunks created from this document")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    status: str = Field("processing", description="Processing status")
