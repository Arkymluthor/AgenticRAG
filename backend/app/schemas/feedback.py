from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """
    Request model for feedback endpoint.
    """
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    rating: int = Field(..., description="Rating (1-5)", ge=1, le=5)
    comment: Optional[str] = Field(None, description="Optional comment")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class FeedbackResponse(BaseModel):
    """
    Response model for feedback endpoint.
    """
    success: bool = Field(..., description="Whether the feedback was successfully recorded")
    feedback_id: str = Field(..., description="Feedback identifier")
    message: str = Field("Feedback recorded", description="Response message")
