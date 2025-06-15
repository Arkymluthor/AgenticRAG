from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import chat, feedback, health


def create_app() -> FastAPI:
    """
    FastAPI application factory.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
    )

    # Set up CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix=settings.API_V1_STR)
    app.include_router(chat.router, prefix=settings.API_V1_STR)
    app.include_router(feedback.router, prefix=settings.API_V1_STR)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on application startup.
    """
    # Initialize logging
    from app.core.logging import setup_logging
    setup_logging()
    
    # Initialize any other services (database connections, etc.)
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on application shutdown.
    """
    # Close any connections or clean up resources
    pass
