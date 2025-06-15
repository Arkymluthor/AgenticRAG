import logging
import httpx
import json
from typing import Dict, Any, List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """
    Agent specialized in helping users make decisions.
    Uses retrieval to provide well-justified recommendations.
    """
    # Filter for decision-related documents
    FILTER = "category eq 'decision'"
    
    def __init__(self, memory, retriever):
        """
        Initialize the decision agent.
        
        Args:
            memory: Memory component for storing conversation history
            retriever: Retrieval component for finding relevant information
        """
        super().__init__(memory, retriever)
    
    async def handle(self, message: str, session_id: str) -> Dict[str, Any]:
        """
        Process a user message about making a decision.
        
        Args:
            message: The user's message
            session_id: Session identifier
            
        Returns:
            Response with decision support
        """
        try:
            # Get conversation history
            history = await self._get_conversation_history(session_id)
            
            # Extract keywords from the message
            keywords = await self.retriever.keyword_extractor.extract(message)
            
            # Search for relevant documents with decision filter
            search_results = await self.retriever.search_client.search(
                keywords=keywords,
                filter_expr=self.FILTER,
                k=5
            )
            
            # Format sources for the response
            sources = []
            context_text = ""
            
            if search_results:
                # Format sources for the response
                sources = [
                    {
                        "title": result.get("document_name", "Untitled"),
                        "content_snippet": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                        "document_type": result.get("document_type", "unknown"),
                        "document_year": result.get("document_year", ""),
                        "document_entity": result.get("document_entity", ""),
                        "page": result.get("page", 0),
                        "source_uri": result.get("source_uri", ""),
                        "relevance_score": result.get("score", 0.0)
                    }
                    for result in search_results
                ]
                
                # Create context from search results
                context_pieces = []
                for i, result in enumerate(search_results):
                    content = result.get("content", "")
                    doc_name = result.get("document_name", "Untitled")
                    doc_type = result.get("document_type", "unknown")
                    
                    context_pieces.append(
                        f"[Document {i+1}] {doc_name} (Type: {doc_type})\n{content}"
                    )
                
                context_text = "\n\n".join(context_pieces)
            
            # Format history for the LLM
            formatted_history = []
            for turn in history[-5:]:  # Use last 5 turns for context
                if "user_message" in turn:
                    formatted_history.append({"role": "user", "content": turn["user_message"]})
                if "assistant_message" in turn:
                    formatted_history.append({"role": "assistant", "content": turn["assistant_message"]})
            
            # Prepare the prompt
            system_prompt = (
                "You are an AI assistant specialized in helping users make decisions. "
                "When presented with a decision problem, follow these steps:\n"
                "1. Identify the key decision factors and constraints\n"
                "2. List the available options or generate reasonable alternatives\n"
                "3. Evaluate each option against the decision factors\n"
                "4. Present a recommendation with clear reasoning\n"
                "5. Highlight potential risks or considerations\n\n"
                "Structure your response clearly with headings and bullet points. "
                "Be balanced in your analysis but don't hesitate to make a clear recommendation "
                "when the evidence supports it."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add conversation history for context
            messages.extend(formatted_history)
            
            # Add retrieved context if available
            if context_text:
                messages.append({
                    "role": "system",
                    "content": (
                        "Here is relevant information to help with the decision-making process. "
                        "Use this information to provide well-justified recommendations.\n\n" + context_text
                    )
                })
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            
            # Add decision-specific instructions
            messages.append({
                "role": "system",
                "content": (
                    "For this decision problem, please think step by step:\n"
                    "1. What is the core decision that needs to be made?\n"
                    "2. What factors should be considered in this decision?\n"
                    "3. What are the available options?\n"
                    "4. How does each option perform against the key factors?\n"
                    "5. What is the recommended decision and why?\n\n"
                    "Structure your response with clear headings for each section."
                )
            })
            
            # Prepare the API request
            url = f"{settings.OPENAI_API_BASE}/openai/deployments/{settings.CHAT_MODEL_DEPLOYMENT}/chat/completions?api-version={settings.OPENAI_API_VERSION}"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": settings.OPENAI_API_KEY,
            }
            
            payload = {
                "messages": messages,
                "temperature": 0.5,  # Lower temperature for more consistent reasoning
                "max_tokens": 1000,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
            
            # Store the interaction
            await self._store_interaction(
                message=message,
                response=assistant_message,
                session_id=session_id,
                sources=sources,
                metadata={"num_sources": len(sources)}
            )
            
            # Return the response
            return {
                "message": assistant_message,
                "sources": sources,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.exception(f"Error in DecisionAgent.handle: {str(e)}")
            
            # Create a fallback response
            fallback_message = (
                "I'm sorry, I encountered an error while processing your decision request. "
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
