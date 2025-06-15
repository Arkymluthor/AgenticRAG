import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and shared functionality.
    """
    def __init__(self, memory, retriever):
        """
        Initialize the base agent with memory and retrieval components.
        
        Args:
            memory: Memory component for storing conversation history
            retriever: Retrieval component for finding relevant information
        """
        self.memory = memory
        self.retriever = retriever
    
    @abstractmethod
    async def handle(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            message: The user's message text
            session_id: Session identifier for conversation context
            
        Returns:
            Dictionary containing the response and any additional metadata
        """
        pass
    
    async def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for the given session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns
        """
        try:
            return await self.memory.get_history(session_id)
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    async def _store_interaction(
        self, 
        message: str, 
        response: str, 
        session_id: str, 
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store the interaction in memory.
        
        Args:
            message: User message
            response: Agent response
            session_id: Session identifier
            sources: Optional list of sources used for the response
            metadata: Optional additional metadata
        """
        try:
            await self.memory.store_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=response,
                sources=sources,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")
