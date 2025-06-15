from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

from app.schemas.chat import ChatRequest, ChatResponse, Message
from app.agents.router import AgentRouter
from app.memory.redis_store import RedisStore

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    Chat endpoint that routes requests to the appropriate agent.
    """
    logger.info(f"Chat request received for conversation: {request.conversation_id}")
    
    try:
        # Initialize agent router
        agent_router = AgentRouter()
        
        # Get conversation history from Redis if conversation_id is provided
        conversation_history: List[Message] = []
        if request.conversation_id:
            redis_store = RedisStore()
            conversation_history = await redis_store.get_conversation_history(
                request.conversation_id
            )
        
        # Route to appropriate agent and get response
        response = await agent_router.route_and_process(
            user_message=request.message,
            conversation_history=conversation_history,
            conversation_id=request.conversation_id,
        )
        
        # Store conversation in Redis in the background
        if response.conversation_id:
            background_tasks.add_task(
                store_conversation,
                response.conversation_id,
                response.history
            )
        
        return response
        
    except Exception as e:
        logger.exception(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


async def store_conversation(conversation_id: str, history: List[Message]):
    """
    Background task to store conversation history in Redis.
    """
    try:
        redis_store = RedisStore()
        await redis_store.store_conversation_history(conversation_id, history)
        logger.info(f"Conversation {conversation_id} stored successfully")
    except Exception as e:
        logger.exception(f"Failed to store conversation {conversation_id}: {str(e)}")
