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