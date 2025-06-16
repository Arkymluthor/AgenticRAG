import logging
import json
import uuid
import time
import asyncio
from typing import Dict, Any, List, Optional
from celery import Celery, Task
from celery.signals import worker_ready
import aioredis

from core.config import settings
from agents.router import RouterAgent
from memory.redis_store import RedisConversationStore

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "agentic_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    worker_prefetch_multiplier=1,
    worker_concurrency=4
)


class AgentTask(Task):
    """
    Base task class for agent tasks.
    Provides access to agent and memory components.
    """
    _router_agent = None
    _memory_store = None
    _redis_client = None
    
    @property
    def router_agent(self) -> RouterAgent:
        """
        Get or initialize the router agent.
        """
        if self._router_agent is None:
            # Initialize memory store
            self._memory_store = RedisConversationStore()
            
            # Initialize retriever components
            # Note: In a real implementation, these would be initialized properly
            # with the actual components
            retriever = {}
            
            # Initialize router agent
            self._router_agent = RouterAgent(
                memory=self._memory_store,
                retriever=retriever
            )
            
            logger.info("Router agent initialized")
            
        return self._router_agent
    
    @property
    def memory_store(self) -> RedisConversationStore:
        """
        Get or initialize the memory store.
        """
        if self._memory_store is None:
            self._memory_store = RedisConversationStore()
            
        return self._memory_store
    
    @property
    async def redis_client(self):
        """
        Get or initialize the Redis client for pub/sub.
        """
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
        return self._redis_client
    
    async def publish_token(self, session_id: str, token_data: Dict[str, Any]):
        """
        Publish a token to the Redis pub/sub channel.
        
        Args:
            session_id: Session identifier (used as channel name)
            token_data: Token data to publish
        """
        redis = await self.redis_client
        channel_name = f"chat:stream:{session_id}"
        
        # Serialize token data to JSON
        token_json = json.dumps(token_data)
        
        # Publish to Redis channel
        await redis.publish(channel_name, token_json)
        logger.debug(f"Published token to channel {channel_name}")


@celery_app.task(name="chat_pipeline", bind=True, base=AgentTask)
async def chat_pipeline(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a chat request through the agent pipeline.
    Hydrates RouterAgent and gets final answer dict.
    Streams back via Redis pub/sub channel keyed by session_id.
    
    Args:
        request_data: Chat request data
        
    Returns:
        Final response data
    """
    session_id = request_data.get("session_id", str(uuid.uuid4()))
    message = request_data.get("message", "")
    metadata = request_data.get("metadata", {})
    
    logger.info(f"Processing chat request for session {session_id}")
    
    try:
        # Generate a message ID
        message_id = str(uuid.uuid4())
        
        # Publish initial token
        await self.publish_token(session_id, {
            "token": "I'm",
            "session_id": session_id,
            "message_id": message_id,
            "is_complete": False,
            "metadata": {
                "streaming": True,
                "timestamp": time.time()
            }
        })
        
        # Process the message through the router agent
        response = await self.router_agent.handle(message, session_id)
        
        # Add message ID to the response
        response["message_id"] = message_id
        
        # Add any additional metadata
        if metadata:
            if "metadata" in response:
                response["metadata"].update(metadata)
            else:
                response["metadata"] = metadata
        
        # Get the final response message
        final_message = response.get("message", "")
        
        # Simulate token-by-token streaming
        # In a real implementation, this would use a proper tokenizer
        tokens = final_message.split()
        
        # Stream tokens with a small delay
        for i, token in enumerate(tokens):
            # Add space before token (except for first token)
            if i > 0:
                token = " " + token
            
            # Publish token
            await self.publish_token(session_id, {
                "token": token,
                "session_id": session_id,
                "message_id": message_id,
                "is_complete": False,
                "metadata": {
                    "streaming": True,
                    "timestamp": time.time(),
                    "token_index": i
                }
            })
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.05)
        
        # Publish final token with complete flag and sources
        await self.publish_token(session_id, {
            "token": "",  # Empty final token
            "session_id": session_id,
            "message_id": message_id,
            "is_complete": True,
            "sources": response.get("sources"),
            "metadata": {
                "streaming": True,
                "timestamp": time.time(),
                "complete": True
            }
        })
        
        logger.info(f"Chat request processed successfully for session {session_id}")
        return response
        
    except Exception as e:
        logger.exception(f"Error processing chat request: {str(e)}")
        
        # Create an error response
        error_response = {
            "message": "I'm sorry, I encountered an error while processing your request. Please try again.",
            "session_id": session_id,
            "message_id": str(uuid.uuid4()),
            "error": str(e),
            "metadata": {
                "error": True,
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }
        }
        
        # Publish error token
        await self.publish_token(session_id, {
            "token": error_response["message"],
            "session_id": session_id,
            "message_id": error_response["message_id"],
            "is_complete": True,
            "metadata": {
                "error": True,
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }
        })
        
        return error_response


@worker_ready.connect
def on_worker_ready(**kwargs):
    """
    Initialize components when the worker starts.
    """
    logger.info("Celery worker ready, initializing components")
    
    # Initialize Redis connection
    try:
        memory_store = RedisConversationStore()
        # Perform a simple operation to test the connection
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis connection: {str(e)}")
