"""
Utility functions used across the flag pattern classification pipeline.
"""

import logging
import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple


def setup_logger(name='pipeline', log_file='../log/run.log'):
    """
    Sets up a logger that outputs to stdout only.
    The log file is created by redirecting stdout in the shell/Docker.
    
    Args:
        name: Logger name (default: 'pipeline')
        log_file: Path to log file (unused, kept for compatibility)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only set up if not already configured
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent log propagation to root logger

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler (stdout) - Docker/shell redirection captures this
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def strip_guid(filename: str) -> str:
    """
    Converts 'e2ab0dd4-FILENAME.csv' → 'FILENAME.csv'.
    
    Args:
        filename: Filename potentially with GUID prefix
        
    Returns:
        Filename without GUID prefix
    """
    base = os.path.basename(filename)
    parts = base.split("-", 1)
    if len(parts) == 2:
        return parts[1]
    return base


def ensure_dir(directory: str):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)
