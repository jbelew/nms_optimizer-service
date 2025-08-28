import logging
import sys
import os

def setup_logger():
    """
    Sets up the root logger for the application, but ONLY if no handlers
    are already configured. This allows Gunicorn or other runners to
    define the primary logging configuration.
    """
    # Get the root logger
    logger = logging.getLogger()

    # Only configure handlers if none exist.
    if not logger.hasHandlers():
        # Set the logging level. Default to INFO.
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(log_level)

        # Create a handler that writes to standard output
        handler = logging.StreamHandler(sys.stdout)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)
        logger.info(f"Logger configured by application with level {log_level}")

# Set up the logger when this module is first imported
setup_logger()
