import json
import logging
from typing import List, Optional, Dict, Any
import aioredis
from datetime import datetime, timedelta

from app.core.config import settings
from app.schemas.chat import Message

logger = logging.getLogger(__name__)


class RedisStore:
    """
    Redis-based storage for conversation history and other application data.
    """
    def __init__(self):
        """
        Initialize Redis connection.
        """
        self.redis = None
        self.expiry_time = timedelta(days=30)  # Default TTL for conversation data
    
    async def _get_connection(self) -> aioredis.Redis:
        """
        Get or create Redis connection.
        """
        if self.redis is None:
            try:
                connection_kwargs = {
                    "host": settings.REDIS_HOST,
                    "port": settings.REDIS_PORT,
                    "db": settings.REDIS_DB,
                }
                
                if settings.REDIS_PASSWORD:
                    connection_kwargs["password"] = settings.REDIS_PASSWORD
                
                self.redis = await aioredis.create_redis_pool(**connection_kwargs)
                logger.info("Redis connection established")
            except Exception as e:
                logger.exception(f"Failed to connect to Redis: {str(e)}")
                raise
        
        return self.redis
    
    async def close(self):
        """
        Close Redis connection.
        """
        if self.redis is not None:
            self.redis.close()
            await self.redis.wait_closed()
            self.redis = None
            logger.info("Redis connection closed")
    
    async def store_conversation_history(self, conversation_id: str, messages: List[Message]):
        """
        Store conversation history in Redis.
        """
        redis = await self._get_connection()
        
        try:
            # Convert messages to JSON-serializable format
            serialized_messages = []
            for msg in messages:
                # Convert datetime to ISO format string for JSON serialization
                msg_dict = msg.dict()
                if isinstance(msg_dict["timestamp"], datetime):
                    msg_dict["timestamp"] = msg_dict["timestamp"].isoformat()
                serialized_messages.append(msg_dict)
            
            # Store in Redis with expiry
            key = f"conversation:{conversation_id}"
            await redis.set(key, json.dumps(serialized_messages))
            await redis.expire(key, int(self.expiry_time.total_seconds()))
            
            logger.debug(f"Stored {len(messages)} messages for conversation {conversation_id}")
        except Exception as e:
            logger.exception(f"Error storing conversation history: {str(e)}")
            raise
    
    async def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """
        Retrieve conversation history from Redis.
        """
        redis = await self._get_connection()
        
        try:
            key = f"conversation:{conversation_id}"
            data = await redis.get(key)
            
            if not data:
                logger.debug(f"No history found for conversation {conversation_id}")
                return []
            
            # Parse JSON and convert back to Message objects
            messages_data = json.loads(data)
            messages = []
            
            for msg_data in messages_data:
                # Convert ISO timestamp string back to datetime
                if "timestamp" in msg_data and isinstance(msg_data["timestamp"], str):
                    try:
                        msg_data["timestamp"] = datetime.fromisoformat(msg_data["timestamp"].replace("Z", "+00:00"))
                    except ValueError:
                        # If parsing fails, use current time
                        msg_data["timestamp"] = datetime.utcnow()
                
                messages.append(Message(**msg_data))
            
            logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages
        except Exception as e:
            logger.exception(f"Error retrieving conversation history: {str(e)}")
            return []
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation from Redis.
        """
        redis = await self._get_connection()
        
        try:
            key = f"conversation:{conversation_id}"
            deleted = await redis.delete(key)
            success = deleted > 0
            
            if success:
                logger.info(f"Deleted conversation {conversation_id}")
            else:
                logger.warning(f"Conversation {conversation_id} not found for deletion")
            
            return success
        except Exception as e:
            logger.exception(f"Error deleting conversation: {str(e)}")
            return False
