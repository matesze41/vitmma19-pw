# Utility functions
# Common helper functions used across the project.
import logging
import sys
import os

def setup_logger(name=__name__, log_file='../log/run.log'):
    """
    Sets up a logger that outputs to both console (stdout) and a log file.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file relative to project root (default: 'log/run.log')
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def load_config():
    pass
