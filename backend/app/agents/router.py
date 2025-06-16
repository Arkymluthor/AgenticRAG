import logging
import httpx
import json
from typing import Dict, Any, List, Optional, Tuple
from fastapi import Depends

from agents.base import BaseAgent
from agents.application import ApplicationAgent
from agents.decision import DecisionAgent
from agents.history import HistoryAgent
from core.config import settings

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """
    Agent that classifies messages and routes them to the appropriate agent.
    """
    def __init__(self, memory, retriever):
        """
        Initialize the router agent.
        
        Args:
            memory: Memory component for storing conversation history
            retriever: Retrieval component for finding relevant information
        """
        super().__init__(memory, retriever)
        
        # Initialize specialized agents
        self.application_agent = ApplicationAgent(memory, retriever)
        self.decision_agent = DecisionAgent(memory, retriever)
        self.history_agent = HistoryAgent(memory, retriever)
    
    async def handle(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user message by routing it to the appropriate agent.
        
        Args:
            message: The user's message
            session_id: Session identifier
            
        Returns:
            Response from the appropriate agent
        """
        try:
            # Get conversation history
            history = await self._get_conversation_history(session_id)
            
            # First, try to answer from history
            history_response = await self.history_agent.handle(message, session_id)
            
            # If the history agent can answer the question, return its response
            if "message" in history_response:
                logger.info(f"Question answered from history: {message}")
                return history_response
            
            # Otherwise, classify the message and route to the appropriate agent
            message_type = await self.classify(message, history)
            
            logger.info(f"Message classified as {message_type}: {message}")
            
            # Route to the appropriate agent
            if message_type == "application":
                return await self.application_agent.handle(message, session_id)
            elif message_type == "decision":
                return await self.decision_agent.handle(message, session_id)
            else:
                # Default to application agent for other types
                return await self.application_agent.handle(message, session_id)
                
        except Exception as e:
            logger.exception(f"Error in RouterAgent.handle: {str(e)}")
            
            # Create a fallback response
            fallback_message = (
                "I'm sorry, I encountered an error while processing your request. "
                "Please try again with more details or contact support if the issue persists."
            )
            
            # Store the interaction
            await self._store_interaction(
                message=message,
                response=fallback_message,
                session_id=session_id,
                metadata={"error": str(e)}
            )
            
            # Return the fallback response
            return {
                "message": fallback_message,
                "error": str(e),
                "session_id": session_id
            }
    
    async def classify(self, message: str, history: List[Dict[str, Any]]) -> str:
        """
        Classify the message into one of the following types:
        - application: Questions about the application
        - decision: Questions about making a decision
        - clarify: Requests for clarification
        - chitchat: General conversation
        
        Args:
            message: The user's message
            history: Conversation history
            
        Returns:
            Message type
        """
        try:
            # Format history for the LLM
            formatted_history = []
            for turn in history[-3:]:  # Use last 3 turns for context
                if "user_message" in turn:
                    formatted_history.append({"role": "user", "content": turn["user_message"]})
                if "assistant_message" in turn:
                    formatted_history.append({"role": "assistant", "content": turn["assistant_message"]})
            
            # Prepare the prompt
            system_prompt = (
                "You are an AI assistant that classifies user messages into one of the following types:\n"
                "- application: Questions about the application, how it works, or its features\n"
                "- decision: Questions about making a decision or choosing between options\n"
                "- clarify: Requests for clarification or more information\n"
                "- chitchat: General conversation, greetings, or small talk\n\n"
                "Respond with EXACTLY one of these types, nothing else."
            )
            
            # Determine which API to use
            use_azure = settings.USE_AZURE_OPENAI
            
            if use_azure:
                # Azure OpenAI API
                url = f"{settings.AZURE_OPENAI_ENDPOINT}/openai/deployments/{settings.AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions?api-version={settings.AZURE_OPENAI_API_VERSION}"
                headers = {
                    "Content-Type": "application/json",
                    "api-key": settings.AZURE_OPENAI_KEY,
                }
            else:
                # OpenAI API
                url = f"{settings.OPENAI_API_BASE}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                }
            
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add conversation history for context
            messages.extend(formatted_history)
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            
            # Common payload
            payload = {
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 10,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            
            # Add model parameter for OpenAI API
            if not use_azure:
                payload["model"] = settings.OPENAI_CHAT_MODEL
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                message_type = result["choices"][0]["message"]["content"].strip().lower()
            
            # Validate the message type
            valid_types = ["application", "decision", "clarify", "chitchat"]
            if message_type not in valid_types:
                # Default to application if the type is not valid
                message_type = "application"
            
            return message_type
                
        except Exception as e:
            logger.exception(f"Error classifying message: {str(e)}")
            # Default to application on error
            return "application"
