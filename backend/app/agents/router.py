import logging
import httpx
import json
from typing import Dict, Any, List, Literal, Optional
from fastapi import Depends

from app.core.config import settings
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """
    Router agent that classifies user messages and routes them to the appropriate agent.
    """
    def __init__(self, memory, retriever, application_agent=None, decision_agent=None, history_agent=None):
        """
        Initialize the router agent with memory, retriever, and specialized agents.
        
        Args:
            memory: Memory component for storing conversation history
            retriever: Retrieval component for finding relevant information
            application_agent: Agent for handling application-specific queries
            decision_agent: Agent for handling decision-making queries
            history_agent: Agent for answering from conversation history
        """
        super().__init__(memory, retriever)
        self.application_agent = application_agent
        self.decision_agent = decision_agent
        self.history_agent = history_agent
    
    async def classify(self, message: str, history: List[Dict[str, Any]]) -> Literal["application", "decision", "clarify", "chitchat"]:
        """
        Classify the user message to determine which agent should handle it.
        
        Args:
            message: The user's message
            history: Conversation history
            
        Returns:
            Classification of the message
        """
        try:
            # Format history for the LLM
            formatted_history = []
            for turn in history[-5:]:  # Use last 5 turns for context
                if "user_message" in turn:
                    formatted_history.append({"role": "user", "content": turn["user_message"]})
                if "assistant_message" in turn:
                    formatted_history.append({"role": "assistant", "content": turn["assistant_message"]})
            
            # Prepare the prompt for classification
            system_prompt = (
                "You are a message classifier for a conversational AI system. "
                "Your task is to classify user messages into one of these categories:\n"
                "- 'application': Questions about the application, its features, or documentation\n"
                "- 'decision': Requests for help with decision-making or recommendations\n"
                "- 'clarify': Messages that are unclear or need clarification\n"
                "- 'chitchat': General conversation, greetings, or small talk\n\n"
                "Respond with ONLY the category name, nothing else."
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
            
            # Add the current message
            messages.append({"role": "user", "content": f"Classify this message: {message}"})
            
            payload = {
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 20,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                classification = result["choices"][0]["message"]["content"].strip().lower()
            
            # Validate and return the classification
            if classification in ["application", "decision", "clarify", "chitchat"]:
                return classification
            else:
                logger.warning(f"Invalid classification: {classification}, defaulting to 'clarify'")
                return "clarify"
                
        except Exception as e:
            logger.exception(f"Error classifying message: {str(e)}")
            return "clarify"  # Default to clarify on error
    
    async def handle(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user message by classifying it and routing to the appropriate agent.
        
        Args:
            message: The user's message
            session_id: Session identifier
            
        Returns:
            Response from the appropriate agent
        """
        # Get conversation history
        history = await self._get_conversation_history(session_id)
        
        # First, try to answer from history
        if self.history_agent:
            history_response = await self.history_agent.answer_from_history(message, history)
            if history_response:
                # If history agent can answer, use that response
                response = {
                    "message": history_response,
                    "source": "history",
                    "session_id": session_id
                }
                
                # Store the interaction
                await self._store_interaction(
                    message=message,
                    response=history_response,
                    session_id=session_id,
                    metadata={"source": "history"}
                )
                
                return response
        
        # If history agent couldn't answer, classify and route
        classification = await self.classify(message, history)
        
        if classification == "application" and self.application_agent:
            # Route to application agent
            result = await self.application_agent.handle(message, session_id)
            result["classification"] = "application"
            return result
            
        elif classification == "decision" and self.decision_agent:
            # Route to decision agent
            result = await self.decision_agent.handle(message, session_id)
            result["classification"] = "decision"
            return result
            
        elif classification == "chitchat":
            # Handle chitchat directly
            chitchat_response = await self._handle_chitchat(message, history)
            
            response = {
                "message": chitchat_response,
                "classification": "chitchat",
                "session_id": session_id
            }
            
            # Store the interaction
            await self._store_interaction(
                message=message,
                response=chitchat_response,
                session_id=session_id,
                metadata={"classification": "chitchat"}
            )
            
            return response
            
        else:  # clarify or fallback
            # Generate a clarifying question
            clarify_response = await self._generate_clarification(message, history)
            
            response = {
                "message": clarify_response,
                "classification": "clarify",
                "session_id": session_id
            }
            
            # Store the interaction
            await self._store_interaction(
                message=message,
                response=clarify_response,
                session_id=session_id,
                metadata={"classification": "clarify"}
            )
            
            return response
    
    async def _handle_chitchat(self, message: str, history: List[Dict[str, Any]]) -> str:
        """
        Handle chitchat messages with a friendly, conversational response.
        
        Args:
            message: The user's message
            history: Conversation history
            
        Returns:
            Chitchat response
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
                "You are a friendly, helpful assistant engaging in casual conversation. "
                "Keep your responses concise, friendly, and conversational. "
                "Don't provide detailed technical information - if the user asks for that, "
                "suggest they ask a specific question about the application instead."
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
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 150,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.exception(f"Error handling chitchat: {str(e)}")
            return "I'm here to help! What would you like to know about our application?"
    
    async def _generate_clarification(self, message: str, history: List[Dict[str, Any]]) -> str:
        """
        Generate a clarifying question when the user's intent is unclear.
        
        Args:
            message: The user's message
            history: Conversation history
            
        Returns:
            Clarifying question
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
                "You are a helpful assistant trying to understand the user's request. "
                "The user's intent is unclear. Generate a clarifying question to better understand "
                "what they're looking for. Focus on determining whether they want information about "
                "the application or help with making a decision."
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
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 150,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.exception(f"Error generating clarification: {str(e)}")
            return "I'm not sure I understand what you're asking. Could you please provide more details or rephrase your question?"


# FastAPI dependency
def get_router_agent(
    memory=Depends(),
    retriever=Depends(),
    application_agent=Depends(),
    decision_agent=Depends(),
    history_agent=Depends()
) -> RouterAgent:
    """
    FastAPI dependency for the RouterAgent.
    """
    return RouterAgent(
        memory=memory,
        retriever=retriever,
        application_agent=application_agent,
        decision_agent=decision_agent,
        history_agent=history_agent
    )
