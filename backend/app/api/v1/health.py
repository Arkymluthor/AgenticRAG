from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

router = APIRouter()


@router.get("/healthz", summary="Liveness probe")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for Kubernetes liveness probe.
    
    Returns:
        Dictionary with status "ok"
    """
    return {"status": "ok"}


@router.post("/ingest", summary="Manual document ingestion")
async def manual_ingest(blob_uri: str) -> Dict[str, Any]:
    """
    Manual kick-off for document ingestion.
    This endpoint is for admin tooling only.
    
    Args:
        blob_uri: URI of the blob to ingest
        
    Returns:
        Dictionary with ingestion status
    """
    try:
        # In a real implementation, this would trigger the DocIngestOrchestrator
        # For now, we just return a success response
        
        return {
            "status": "accepted",
            "message": "Document ingestion started",
            "blob_uri": blob_uri,
            "job_id": "123456"  # In a real implementation, this would be a real job ID
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start document ingestion: {str(e)}"
        )
