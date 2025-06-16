from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    A message in a conversation.
    """
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    """
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="User message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    """
    message: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used for the response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class ChatStreamResponse(BaseModel):
    """
    Response model for streaming chat endpoint.
    """
    token: str = Field(..., description="Token of the assistant response")
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    is_complete: bool = Field(False, description="Whether this is the last token")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used for the response (only in last token)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata (only in last token)")
