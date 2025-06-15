from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging

from app.core.config import settings

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    environment: str


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns basic application information and status.
    """
    logger.info("Health check requested")
    
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment="development",  # This would typically be set from an env var
    )
