import logging
import sys
from typing import Dict, Any
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

from app.core.settings import Settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(settings: Settings) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        settings: Application settings
    """
    # Create custom logger
    logger = logging.getLogger("rag_chatgpt")
    logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(JSONFormatter())
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set logging level for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"rag_chatgpt.{name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to services."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, extra_fields: Dict[str, Any] = None) -> None:
        """Log info message with optional extra fields."""
        self._log_with_extra(logging.INFO, message, extra_fields)
    
    def log_error(self, message: str, extra_fields: Dict[str, Any] = None, exc_info: bool = True) -> None:
        """Log error message with optional extra fields and exception info."""
        self._log_with_extra(logging.ERROR, message, extra_fields, exc_info)
    
    def log_warning(self, message: str, extra_fields: Dict[str, Any] = None) -> None:
        """Log warning message with optional extra fields."""
        self._log_with_extra(logging.WARNING, message, extra_fields)
    
    def log_debug(self, message: str, extra_fields: Dict[str, Any] = None) -> None:
        """Log debug message with optional extra fields."""
        self._log_with_extra(logging.DEBUG, message, extra_fields)
    
    def _log_with_extra(self, level: int, message: str, extra_fields: Dict[str, Any] = None, exc_info: bool = False) -> None:
        """Internal method to log with extra fields."""
        extra = {"extra_fields": extra_fields} if extra_fields else {}
        self.logger.log(level, message, extra=extra, exc_info=exc_info)