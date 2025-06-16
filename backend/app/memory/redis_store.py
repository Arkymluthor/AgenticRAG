import logging
import json
import time
from typing import List, Dict, Any, Optional
import aioredis
from fastapi import Depends

from core.config import settings

logger = logging.getLogger(__name__)


class RedisConversationStore:
    """
    Redis-based implementation of conversation history storage.
    Stores conversation turns as JSON in Redis lists.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisConversationStore, cls).__new__(cls)
            cls._instance.redis = None
            cls._instance.initialized = False
        return cls._instance
    
    async def initialize(self):
        """
        Initialize the Redis connection.
        """
        if self.initialized:
            return
        
        try:
            # Get Redis connection details from settings
            redis_url = settings.REDIS_URL
            
            # Create Redis connection
            self.redis = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            self.initialized = True
            logger.info("Redis conversation store initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {str(e)}")
            raise
    
    async def close(self):
        """
        Close the Redis connection.
        """
        if self.redis:
            await self.redis.close()
            self.redis = None
            self.initialized = False
    
    async def save_turn(self, session_id: str, role: str, content: Dict[str, Any]) -> None:
        """
        Save a conversation turn to Redis.
        
        Args:
            session_id: Session identifier
            role: Role of the message sender ('user' or 'assistant')
            content: Message content and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create a key for the conversation
            conversation_key = f"conversation:{session_id}"
            
            # Create a turn object
            turn = {
                "role": role,
                "content": content,
                "timestamp": time.time()
            }
            
            # Serialize the turn to JSON
            turn_json = json.dumps(turn)
            
            # Add the turn to the conversation list
            await self.redis.rpush(conversation_key, turn_json)
            
            # Set expiration on the conversation (30 days)
            await self.redis.expire(conversation_key, 60 * 60 * 24 * 30)
            
            logger.debug(f"Saved turn for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving conversation turn: {str(e)}")
            raise
    
    async def fetch_history(self, session_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch the last n turns of conversation history.
        
        Args:
            session_id: Session identifier
            n: Number of turns to fetch (default: 5)
            
        Returns:
            List of conversation turns
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create a key for the conversation
            conversation_key = f"conversation:{session_id}"
            
            # Get the length of the list
            list_length = await self.redis.llen(conversation_key)
            
            # Calculate the start index
            start_index = max(0, list_length - n)
            
            # Get the last n turns
            turn_jsons = await self.redis.lrange(conversation_key, start_index, -1)
            
            # Parse the turns
            turns = []
            for turn_json in turn_jsons:
                try:
                    turn = json.loads(turn_json)
                    turns.append(turn)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse turn JSON: {turn_json}")
            
            logger.debug(f"Fetched {len(turns)} turns for session {session_id}")
            return turns
            
        except Exception as e:
            logger.error(f"Error fetching conversation history: {str(e)}")
            return []
    
    async def store_turn(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_message: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a complete conversation turn (user message and assistant response).
        
        Args:
            session_id: Session identifier
            user_message: User's message text
            assistant_message: Assistant's response text
            sources: Optional list of sources used for the response
            metadata: Optional additional metadata
        """
        # Store user message
        user_content = {
            "text": user_message,
            "timestamp": time.time()
        }
        await self.save_turn(session_id, "user", user_content)
        
        # Store assistant message
        assistant_content = {
            "text": assistant_message,
            "sources": sources or [],
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        await self.save_turn(session_id, "assistant", assistant_content)
    
    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the full conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns in a format suitable for agents
        """
        # Fetch raw turns
        raw_turns = await self.fetch_history(session_id, n=100)  # Get up to 100 turns
        
        # Format turns for agent consumption
        formatted_turns = []
        for turn in raw_turns:
            role = turn.get("role")
            content = turn.get("content", {})
            
            if role == "user":
                formatted_turns.append({
                    "user_message": content.get("text", ""),
                    "timestamp": content.get("timestamp")
                })
            elif role == "assistant":
                # Find the previous turn (should be a user message)
                if formatted_turns and "user_message" in formatted_turns[-1]:
                    # Add assistant message to the same turn
                    formatted_turns[-1]["assistant_message"] = content.get("text", "")
                    formatted_turns[-1]["sources"] = content.get("sources", [])
                    formatted_turns[-1]["metadata"] = content.get("metadata", {})
                    formatted_turns[-1]["assistant_timestamp"] = content.get("timestamp")
                else:
                    # No matching user message, create a new turn
                    formatted_turns.append({
                        "assistant_message": content.get("text", ""),
                        "sources": content.get("sources", []),
                        "metadata": content.get("metadata", {}),
                        "timestamp": content.get("timestamp")
                    })
        
        return formatted_turns


# FastAPI dependency
async def get_conversation_store() -> RedisConversationStore:
    """
    FastAPI dependency for the RedisConversationStore singleton.
    """
    store = RedisConversationStore()
    await store.initialize()
    try:
        yield store
    finally:
        # No need to close here as we want to keep the singleton alive
        pass
