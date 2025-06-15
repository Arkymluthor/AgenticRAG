from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Message(BaseModel):
    """
    Message model for chat conversations.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user", "assistant", "system", etc.
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    """
    message: str
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "message": "What can you tell me about retrieval augmented generation?",
                "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "metadata": {
                    "source": "web",
                    "user_id": "anonymous"
                }
            }
        }


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    """
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    history: List[Message]
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "message": "Retrieval Augmented Generation (RAG) is an approach...",
                "history": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "role": "user",
                        "content": "What can you tell me about retrieval augmented generation?",
                        "timestamp": "2023-01-01T12:00:00.000Z"
                    },
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174001",
                        "role": "assistant",
                        "content": "Retrieval Augmented Generation (RAG) is an approach...",
                        "timestamp": "2023-01-01T12:00:05.000Z"
                    }
                ],
                "sources": [
                    {
                        "title": "RAG Paper",
                        "url": "https://arxiv.org/abs/2005.11401",
                        "relevance_score": 0.92
                    }
                ],
                "metadata": {
                    "processing_time": 1.25,
                    "model": "gpt-4"
                }
            }
        }
