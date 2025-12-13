# Inference script
# This script runs the model on new, unseen data.
from utils import setup_logger

logger = setup_logger()

def predict():
    logger.info("Running inference...")

if __name__ == "__main__":
    predict()