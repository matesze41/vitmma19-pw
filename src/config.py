"""
Configuration settings for the flag pattern classification pipeline.

This module centralizes all hyperparameters, paths, and settings used across
data preprocessing, training, evaluation, and inference.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "export")
SRC_DIR = os.path.join(BASE_DIR, "src")
LOG_DIR = os.path.join(BASE_DIR, "log")

# Data files
SEGMENTS_META_CSV = os.path.join(EXPORT_DIR, "segments_meta.csv")
SEGMENTS_VALUES_CSV = os.path.join(EXPORT_DIR, "segments_values.csv")
SEGMENTS_HDF5 = os.path.join(EXPORT_DIR, "segments.h5")
SEGMENTS_PREPROC_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24.csv")
SEGMENTS_PREPROC_TEST_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24_test.csv")
SEGMENTS_TEST_RAW_CSV = os.path.join(EXPORT_DIR, "segments_test_raw.csv")

# Model checkpoints
CHECKPOINT_DIR = os.path.join(EXPORT_DIR, "checkpoints_v2")
BEST_MODEL_PATH = None  # Will be set after training

# Logging
LOG_FILE = os.path.join(LOG_DIR, "run.log")

# Output directories
OUTPUT_DIR_01 = os.path.join(SRC_DIR, "01_preprocessing")
OUTPUT_DIR_02 = os.path.join(SRC_DIR, "02_training")
OUTPUT_DIR_03 = os.path.join(SRC_DIR, "03_evaluation")
OUTPUT_DIR_04 = os.path.join(SRC_DIR, "04_inference")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
# Train/test split
TEST_SIZE = 0.10
RANDOM_STATE = 11

# Sequence processing
SEQUENCE_LENGTH = 24  # Number of timesteps per segment
WINDOW_SIZE = 5  # Rolling window for volatility features
EPS = 1e-9  # Small constant to avoid division by zero

# Feature engineering
COMPRESSION_RATIO_MIN = 0.2
COMPRESSION_RATIO_MAX = 3.0

# ============================================================================
# MODEL TRAINING
# ============================================================================
# Random seed for reproducibility
SEED = 1

# Model architecture
INPUT_DIM = None  # Will be set based on data
NUM_CLASSES = None  # Will be set based on data
HIDDEN_CHANNELS = 64

# Training hyperparameters
MAX_EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Validation split
VAL_SIZE = 0.2  # 20% of training data for validation

# Learning rate scheduler
LR_SCHEDULER_T_MAX = 20

# Primary metric for model selection
# Options: 'auc_ovo', 'auc_ovr', 'f1', 'accuracy', 'pr_auc'
PRIMARY_METRIC = 'pr_auc'

# Metric configuration
METRIC_CONFIG = {
    'auc_ovo': {
        'name': 'AUC-ROC (OvO)',
        'short': 'ovo',
        'higher_is_better': True,
        'monitor': 'val_auc_ovo',
        'description': 'One-vs-One: evaluates all pairwise class comparisons'
    },
    'auc_ovr': {
        'name': 'AUC-ROC (OvR)',
        'short': 'ovr',
        'higher_is_better': True,
        'monitor': 'val_auc_ovr',
        'description': 'One-vs-Rest: evaluates each class vs all others'
    },
    'f1': {
        'name': 'F1 Score (macro)',
        'short': 'f1',
        'higher_is_better': True,
        'monitor': 'val_f1',
        'description': 'Harmonic mean of precision and recall'
    },
    'accuracy': {
        'name': 'Accuracy',
        'short': 'acc',
        'higher_is_better': True,
        'monitor': 'val_accuracy',
        'description': 'Proportion of correct predictions'
    },
    'pr_auc': {
        'name': 'PR-AUC (macro)',
        'short': 'pr',
        'higher_is_better': True,
        'monitor': 'val_pr_auc',
        'description': 'Precision-Recall curve area (macro): better for imbalanced classes'
    }
}

# ============================================================================
# BASELINE MODEL
# ============================================================================
BASELINE_SLOPE_THRESHOLD = 0.0002
BASELINE_MA_WINDOW = 3

# ============================================================================
# EVALUATION
# ============================================================================
# Figure settings
FIGURE_DPI = 100
FIGURE_FORMAT = 'png'
