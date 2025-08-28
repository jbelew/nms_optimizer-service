import logging
import sys
import os

def setup_logger():
    """
    Sets up the root logger for the application.
    Logs to stdout.
    """
    # Get the root logger
    logger = logging.getLogger()

    # Clear existing handlers to prevent duplicate logs, which can be an issue
    # in environments like Flask or with modules that configure their own logging.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level. Default to INFO.
    # This can be overridden by an environment variable for debugging.
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    # Create a handler that writes to standard output
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and set it for the handler
    # Example format: 2023-10-27 14:45:12,123 - root - INFO - This is a log message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Initial log to confirm setup
    logger.info(f"Logger initialized with level {log_level}")

# Set up the logger when this module is first imported
setup_logger()
