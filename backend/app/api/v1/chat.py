from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, AsyncGenerator
import json
import uuid
import asyncio
import aioredis

from schemas.chat import ChatRequest, ChatResponse, ChatStreamResponse
from tasks.async_tasks import chat_pipeline
from memory.redis_store import RedisConversationStore, get_conversation_store
from core.config import settings

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Stream chat completion")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    conversation_store: RedisConversationStore = Depends(get_conversation_store)
) -> ChatResponse:
    """
    Chat endpoint that processes a user message and returns a response.
    Creates a Celery task for processing and returns the first token immediately.
    The full response is streamed via Server-Sent Events (SSE).
    
    Args:
        request: Chat request with session_id and message
        background_tasks: FastAPI background tasks
        conversation_store: Redis conversation store
        
    Returns:
        Chat response with the first token
    """
    try:
        # Generate a message ID
        message_id = str(uuid.uuid4())
        
        # Store the user message
        await conversation_store.save_turn(
            session_id=request.session_id,
            role="user",
            content={"text": request.message, "timestamp": asyncio.get_event_loop().time()}
        )
        
        # Create a task to process the message
        task = chat_pipeline.delay(request.dict())
        
        # Return the initial response immediately
        initial_response = ChatResponse(
            message="I'm processing your request...",
            session_id=request.session_id,
            message_id=message_id,
            metadata={
                "task_id": task.id,
                "status": "processing"
            }
        )
        
        return initial_response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.post("/chat/stream", summary="Stream chat completion with SSE")
async def stream_chat(
    request: ChatRequest,
    conversation_store: RedisConversationStore = Depends(get_conversation_store)
) -> StreamingResponse:
    """
    Streaming chat endpoint that processes a user message and returns a response as a stream.
    Uses Redis pub/sub to stream tokens as they are generated.
    
    Args:
        request: Chat request with session_id and message
        conversation_store: Redis conversation store
        
    Returns:
        Streaming response with tokens
    """
    try:
        # Store the user message
        await conversation_store.save_turn(
            session_id=request.session_id,
            role="user",
            content={"text": request.message, "timestamp": asyncio.get_event_loop().time()}
        )
        
        # Create a task to process the message
        task = chat_pipeline.delay(request.dict())
        
        # Create a streaming response
        return StreamingResponse(
            stream_response_from_redis(request.session_id),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process streaming chat request: {str(e)}"
        )


async def stream_response_from_redis(session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream response from Redis pub/sub channel.
    
    Args:
        session_id: Session identifier
        
    Yields:
        SSE formatted response chunks
    """
    # Create Redis client
    redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    
    try:
        # Subscribe to the channel
        channel_name = f"chat:stream:{session_id}"
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel_name)
        
        # Wait for messages
        complete = False
        buffer = ""
        
        while not complete:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            
            if message is not None and message["type"] == "message":
                # Parse the token data
                token_data = json.loads(message["data"])
                
                # Create a response object
                response = ChatStreamResponse(
                    token=token_data.get("token", ""),
                    session_id=token_data.get("session_id", session_id),
                    message_id=token_data.get("message_id", ""),
                    is_complete=token_data.get("is_complete", False),
                    sources=token_data.get("sources"),
                    metadata=token_data.get("metadata")
                )
                
                # Yield the response
                yield f"data: {json.dumps(response.dict())}\n\n"
                
                # Accumulate tokens in buffer
                buffer += token_data.get("token", "")
                
                # Check if this is the last token
                if token_data.get("is_complete", False):
                    complete = True
            
            # Add a small delay to avoid busy waiting
            await asyncio.sleep(0.01)
            
            # Timeout after 60 seconds
            if not hasattr(stream_response_from_redis, "start_time"):
                stream_response_from_redis.start_time = asyncio.get_event_loop().time()
            
            if asyncio.get_event_loop().time() - stream_response_from_redis.start_time > 60:
                # Create a timeout response
                timeout_response = ChatStreamResponse(
                    token="... (response timeout)",
                    session_id=session_id,
                    message_id="timeout",
                    is_complete=True,
                    metadata={"error": "timeout"}
                )
                yield f"data: {json.dumps(timeout_response.dict())}\n\n"
                complete = True
    
    finally:
        # Unsubscribe and close connection
        if pubsub:
            await pubsub.unsubscribe(channel_name)
        
        # Close Redis connection
        await redis.close()
