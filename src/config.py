# Training configuration parameters
# Paths
BASE_DATA_DIR = "../data"
EXPORT_DIR = f"{BASE_DATA_DIR}/export"
PREPROC_CSV = f"{EXPORT_DIR}/segments_preproc_24.csv"

# Training hyperparameters
SEED = 1
MAX_EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PRIMARY_METRIC = 'pr_auc'