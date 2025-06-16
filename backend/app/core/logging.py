import logging
import sys
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
import traceback

from opencensus.ext.azure.log_exporter import AzureLogHandler
from core.config import settings

# Create logs directory if it doesn't exist
logs_dir = Path(settings.BASE_DIR) / "logs"
logs_dir.mkdir(exist_ok=True)


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def __init__(self, **kwargs):
        self.fmt_dict = kwargs
    
    def format(self, record):
        record_dict = self._prepare_log_dict(record)
        return json.dumps(record_dict)
    
    def _prepare_log_dict(self, record):
        record_dict = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if available
        if record.exc_info:
            record_dict["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, "extra"):
            for key, value in record.extra.items():
                record_dict[key] = value
        
        return record_dict


def setup_logging():
    """
    Configure logging for the application.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=10485760,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(file_handler)
    
    # Add Azure Application Insights handler if configured
    if settings.APPINSIGHTS_CONNECTION_STRING:
        azure_handler = AzureLogHandler(connection_string=settings.APPINSIGHTS_CONNECTION_STRING)
        azure_handler.setLevel(logging.INFO)
        root_logger.addHandler(azure_handler)
    
    # Set log levels for specific loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info("Logging configured")
