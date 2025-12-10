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


def minmax_norm(s: pd.Series) -> pd.Series:
    """
    Min-max normalization of a pandas Series.
    
    Args:
        s: Input series
        
    Returns:
        Normalized series with values in [0, 1]
    """
    vmin = s.min()
    vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - vmin) / (vmax - vmin)


def interpolate_series(orig_x: np.ndarray, y: np.ndarray, tgt_x: np.ndarray) -> np.ndarray:
    """
    Interpolate a series to a new set of x values.
    
    Args:
        orig_x: Original x values
        y: Original y values
        tgt_x: Target x values
        
    Returns:
        Interpolated y values at target x positions
    """
    if len(y) == 0:
        return np.zeros(len(tgt_x), dtype=float)
    if len(y) == 1:
        return np.full(len(tgt_x), float(y[0]))
    return np.interp(tgt_x, orig_x, y)


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
