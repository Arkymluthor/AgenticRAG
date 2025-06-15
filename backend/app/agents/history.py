import logging
import httpx
import json
from typing import Dict, Any, List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings

logger = logging.getLogger(__name__)


class HistoryAgent(BaseAgent):
    """
    Agent specialized in answering questions from conversation history.
    Determines if a question can be answered from previous interactions.
    """
    def __init__(self, memory, retriever):
        """
        Initialize the history agent.
        
        Args:
            memory: Memory component for storing conversation history
            retriever: Retrieval component for finding relevant information
        """
        super().__init__(memory, retriever)
    
    async def handle(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user message by checking if it can be answered from history.
        
        Args:
            message: The user's message
            session_id: Session identifier
            
        Returns:
            Response from conversation history or empty dict if not answerable
        """
        # Get conversation history
        history = await self._get_conversation_history(session_id)
        
        # Try to answer from history
        answer = await self.answer_from_history(message, history)
        
        if answer:
            # Store the interaction
            await self._store_interaction(
                message=message,
                response=answer,
                session_id=session_id,
                metadata={"source": "history"}
            )
            
            # Return the response
            return {
                "message": answer,
                "source": "history",
                "session_id": session_id
            }
        else:
            # Return empty dict to indicate the question couldn't be answered from history
            return {
                "answerable_from_history": False,
                "session_id": session_id
            }
    
    async def answer_from_history(self, message: str, history: List[Dict[str, Any]]) -> Optional[str]:
        """
        Determine if a question can be answered from conversation history.
        
        Args:
            message: The user's message
            history: Conversation history
            
        Returns:
            Answer from history or None if not answerable
        """
        if not history:
            return None
        
        try:
            # Format history for the LLM
            formatted_history = []
            for turn in history:
                if "user_message" in turn:
                    formatted_history.append({"role": "user", "content": turn["user_message"]})
                if "assistant_message" in turn:
                    formatted_history.append({"role": "assistant", "content": turn["assistant_message"]})
            
            # Prepare the prompt
            system_prompt = (
                "You are an AI assistant analyzing conversation history. "
                "Your task is to determine if the user's current question can be answered "
                "using information from the conversation history provided. "
                "If the question can be answered from history, provide the answer. "
                "If the question cannot be answered from history, respond with EXACTLY 'None'."
            )
            
            # Prepare the API request
            url = f"{settings.OPENAI_API_BASE}/openai/deployments/{settings.CHAT_MODEL_DEPLOYMENT}/chat/completions?api-version={settings.OPENAI_API_VERSION}"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": settings.OPENAI_API_KEY,
            }
            
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add conversation history for context
            messages.extend(formatted_history)
            
            # Add the current message with specific instruction
            messages.append({
                "role": "user", 
                "content": f"Based on the conversation history above, can you answer this question: '{message}'? If yes, provide the answer. If no, respond with EXACTLY 'None'."
            })
            
            payload = {
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 500,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
            
            # Check if the answer is "None"
            if answer == "None":
                return None
            else:
                return answer
                
        except Exception as e:
            logger.exception(f"Error checking history for answer: {str(e)}")
            return None  # Default to None on error
