from typing import Optional
from pydantic import BaseModel, Field, validator


class FeedbackRequest(BaseModel):
    """
    Request model for feedback endpoint.
    """
    conversation_id: str
    message_id: str
    rating: int
    comments: Optional[str] = None
    
    @validator("rating")
    def validate_rating(cls, v):
        """
        Validate that rating is between 1 and 5.
        """
        if v < 1 or v > 5:
            raise ValueError("Rating must be between 1 and 5")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "message_id": "123e4567-e89b-12d3-a456-426614174001",
                "rating": 5,
                "comments": "Very helpful response with good sources"
            }
        }


class FeedbackResponse(BaseModel):
    """
    Response model for feedback endpoint.
    """
    success: bool
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Feedback received successfully"
            }
        }
