"""
Structured logging utility for the adversarial ML assessment pipeline.
Provides console and optional file logging with standard timestamps and level formatting.
"""

import logging
import sys
import os
from typing import Optional

def get_logger(name: str = "adversarial_ml", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger.
        log_file: Optional file path to append logs to.
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if get_logger is called multiple times with same name
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional file handler
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
