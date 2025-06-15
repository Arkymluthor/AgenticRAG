import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from app.core.config import settings

try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)
        
        return json.dumps(log_data)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Set up structured JSON logging to stdout and optionally to Azure App Insights.
    """
    log_level = getattr(logging, level or settings.LOG_LEVEL)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler with JSON formatting
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(stdout_handler)
    
    # Add Azure App Insights handler if available and configured
    if AZURE_AVAILABLE and settings.APPINSIGHTS_INSTRUMENTATION_KEY:
        azure_handler = AzureLogHandler(
            connection_string=f"InstrumentationKey={settings.APPINSIGHTS_INSTRUMENTATION_KEY}"
        )
        azure_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(azure_handler)
    
    # Suppress noisy loggers
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log startup message
    logging.info(
        "Logging initialized",
        extra={
            "app_name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "log_level": settings.LOG_LEVEL,
            "app_insights_enabled": AZURE_AVAILABLE and bool(settings.APPINSIGHTS_INSTRUMENTATION_KEY),
        },
    )
