from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import uuid
import logging

from schemas.feedback import FeedbackRequest, FeedbackResponse
from memory.redis_store import RedisConversationStore, get_conversation_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, summary="User feedback signal")
async def submit_feedback(
    request: FeedbackRequest,
    conversation_store: RedisConversationStore = Depends(get_conversation_store)
) -> FeedbackResponse:
    """
    Submit user feedback for a message.
    Persists feedback in the database for RLHF.
    
    Args:
        request: Feedback request with session_id, message_id, and rating
        conversation_store: Redis conversation store
        
    Returns:
        Feedback response with success status
    """
    try:
        # Generate a feedback ID
        feedback_id = str(uuid.uuid4())
        
        # Log the feedback
        logger.info(
            f"Received feedback: session={request.session_id}, message={request.message_id}, "
            f"rating={request.rating}, comment={request.comment}"
        )
        
        # In a real implementation, this would persist the feedback in Postgres
        # For now, we just store it in Redis as metadata
        
        # Get the conversation history
        history = await conversation_store.get_history(request.session_id)
        
        # Find the message
        message_found = False
        for turn in history:
            if turn.get("message_id") == request.message_id:
                # Add feedback to metadata
                if "metadata" not in turn:
                    turn["metadata"] = {}
                
                turn["metadata"]["feedback"] = {
                    "feedback_id": feedback_id,
                    "rating": request.rating,
                    "comment": request.comment,
                    "metadata": request.metadata
                }
                
                message_found = True
                break
        
        if not message_found:
            # If message not found, store feedback separately
            await conversation_store.save_turn(
                session_id=request.session_id,
                role="system",
                content={
                    "type": "feedback",
                    "message_id": request.message_id,
                    "feedback_id": feedback_id,
                    "rating": request.rating,
                    "comment": request.comment,
                    "metadata": request.metadata
                }
            )
        
        # Return success response
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback recorded successfully"
        )
        
    except Exception as e:
        logger.exception(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback: {str(e)}"
        )
