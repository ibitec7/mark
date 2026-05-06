"""Small shared utilities for reviewer-facing analysis scripts."""

import logging
from pathlib import Path

def setup_logger(log_file, log_level=logging.INFO, write_console=True, write_file=True) -> logging.Logger:
    """Create a file/console logger without duplicating handlers on re-entry."""
    logger = logging.getLogger(str(log_file))
    logger.setLevel(log_level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    if write_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if write_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger