from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
from typing import Optional

from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])
logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
):
    """
    Endpoint for submitting user feedback on chat responses.
    Feedback is stored for model improvement and analytics.
    """
    logger.info(
        f"Feedback received for conversation: {request.conversation_id}, "
        f"message: {request.message_id}, rating: {request.rating}"
    )
    
    try:
        # Store feedback in database or analytics system
        # This would typically be implemented as a background task
        background_tasks.add_task(
            store_feedback,
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            rating=request.rating,
            comments=request.comments,
        )
        
        return FeedbackResponse(
            success=True,
            message="Feedback received successfully"
        )
        
    except Exception as e:
        logger.exception(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing feedback: {str(e)}"
        )


async def store_feedback(
    conversation_id: str,
    message_id: str,
    rating: int,
    comments: Optional[str] = None,
):
    """
    Background task to store feedback in database or analytics system.
    """
    try:
        # Implementation would depend on the storage system
        # For example, storing in a database or sending to an analytics service
        logger.info(f"Storing feedback for message {message_id}")
        
        # Example implementation (placeholder)
        # await db.feedback.insert_one({
        #     "conversation_id": conversation_id,
        #     "message_id": message_id,
        #     "rating": rating,
        #     "comments": comments,
        #     "timestamp": datetime.utcnow()
        # })
        
        logger.info(f"Feedback for message {message_id} stored successfully")
    except Exception as e:
        logger.exception(f"Failed to store feedback for message {message_id}: {str(e)}")
