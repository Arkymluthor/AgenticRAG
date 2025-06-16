from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from contextlib import asynccontextmanager

from core.config import settings
from core.logging import setup_logging
from api.v1 import health, chat, feedback

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up application")
    
    # Initialize components
    # In a real implementation, this would initialize Redis, etc.
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cleanup resources
    # In a real implementation, this would close connections, etc.


# Create FastAPI application
app = FastAPI(
    title="AgenticRAG API",
    description="API for the AgenticRAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware for logging requests and responses.
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    
    return response


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "Welcome to the AgenticRAG API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
