# Flag Pattern Classification - Production Pipeline

This directory contains the production-ready Python modules extracted from the Jupyter notebooks. All modules use centralized configuration from `config.py` and share utilities from `utils.py`.

## Module Overview

### Configuration & Utilities

- **`config.py`**: Centralized configuration
  - All file paths (data directories, checkpoints, outputs)
  - Hyperparameters (epochs, batch size, learning rate, etc.)
  - Data preprocessing parameters (sequence length, window size)
  - Metric definitions and primary metric selection
  - Output directories for each module

- **`utils.py`**: Shared utility functions
  - `setup_logger()`: Dual logging (stdout + file at `log/run.log`)
  - `minmax_norm()`: Min-max normalization
  - `interpolate_series()`: 1D interpolation for resampling
  - `strip_guid()`: Remove GUID prefixes from filenames
  - `ensure_dir()`: Safely create directories

### Main Pipeline Modules

#### 1. `01-data-preprocessing.py`

**Purpose**: Extract segments from Label Studio JSON and preprocess into model-ready features.

**Input**:
- Label Studio JSON annotations (from `data/export/`)
- Raw market data CSVs

**Output**:
- `data/export/segments_meta.csv` - Segment metadata
- `data/export/segments_values.csv` - Raw segment values
- `data/export/segments.h5` - Segment data in HDF5 format
- `data/export/segments_preproc_24.csv` - Training set (preprocessed, 24 timesteps)
- `data/export/segments_preproc_24_test.csv` - Test set (preprocessed, 24 timesteps)
- `data/export/segments_test_raw.csv` - Test set (raw values for baseline)

**Key Functions**:
- `extract_segments()`: Parse Label Studio JSON and extract annotated segments
- `process_segment()`: Normalize OHLC, engineer features, resample to 24 steps

**Features Engineered**:
- `open_norm`, `high_norm`, `low_norm`, `close_norm`: Min-max normalized prices
- `vol_close`: Rolling std of close prices (volatility)
- `vol_high_low`: Rolling std of high-low spread
- `compression_ratio`: Rolling mean of high-low spread
- `trend`: Slope over rolling window

**Usage**:
```bash
python 01-data-preprocessing.py
```

---

#### 2. `02-training.py`

**Purpose**: Train the CNN model on preprocessed data.

**Input**:
- `data/export/segments_preproc_24.csv` (from module 01)

**Output**:
- Best model checkpoint (e.g., `data/export/checkpoints_v2/best_model_*.ckpt`)
- `data/export/training_metadata.pkl` - Model metadata for inference
- PyTorch Lightning logs in `lightning_logs/`

**Model Architecture** (`FlagPatternClassifier`):
- **Input**: (batch, 24, 8) - 24 timesteps, 8 features
- **Conv Blocks**:
  - Conv1D(8→64) + BatchNorm + ReLU + Dropout(0.3) + MaxPool
  - Conv1D(64→64) + BatchNorm + ReLU + Dropout(0.3) + MaxPool
  - Conv1D(64→128) + BatchNorm + ReLU + Dropout(0.3) + MaxPool
  - Conv1D(128→128) + BatchNorm + ReLU + Dropout(0.3) + MaxPool
- **Fully Connected**:
  - Flatten → Linear(128) → ReLU → Dropout(0.5)
  - Linear(64) → ReLU → Dropout(0.5)
  - Linear(num_classes)

**Training Configuration**:
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=50)
- Loss: CrossEntropyLoss with class weights
- Epochs: 50 (configurable)
- Batch size: 12
- Train/Val split: 80/20 stratified
- Early stopping: Patience 10 epochs
- Checkpoint: Save best model based on val_pr_auc

**Metrics Logged per Epoch**:
- Accuracy
- F1 Score (macro)
- PR-AUC (macro) ← Primary metric
- AUC-ROC (One-vs-One)
- AUC-ROC (One-vs-Rest)

**Usage**:
```bash
python 02-training.py
```

---

#### 3. `03-evaluation.py`

**Purpose**: Evaluate trained model on test set and compare with baseline.

**Input**:
- Best model checkpoint (from module 02)
- `data/export/training_metadata.pkl`
- `data/export/segments_preproc_24.csv` (for training evaluation)
- `data/export/segments_preproc_24_test.csv` (for test evaluation)
- `data/export/segments_values.csv` (for baseline on training set)
- `data/export/segments_test_raw.csv` (for baseline on test set)

**Output**:
- `src/03_evaluation/evaluation_metrics.txt` - Summary metrics
- `src/03_evaluation/figures/`:
  - `confusion_matrix_training_comparison.png` - CNN vs Baseline on train
  - `confusion_matrix_test_comparison.png` - CNN vs Baseline on test
  - `confusion_matrix_cnn_test.png` - Detailed CNN test confusion matrix
  - `roc_pr_curves_test.png` - ROC and PR curves per class (test set)
  - `training_history.png` - Loss and metric curves over epochs

**Evaluation Metrics**:
- **CNN Model**: Accuracy, F1, PR-AUC, AUC-ROC (OvO), AUC-ROC (OvR)
- **Baseline Model** (heuristic): Accuracy, F1
- **Comparison**: Absolute and percentage improvement

**Usage**:
```bash
python 03-evaluation.py
```

---

#### 4. `04-inference.py`

**Purpose**: Run predictions on new, unseen data.

**Input**:
- Best model checkpoint (from module 02)
- New CSV file with columns: `open`, `high`, `low`, `close` (optionally `segment_id`, `timestamp`, `label`)

**Output**:
- CSV with predictions and confidence scores per class

**Usage**:
```bash
# On labeled data (will compute accuracy)
python 04-inference.py --input data/new_segments.csv --output predictions.csv

# On unlabeled data
python 04-inference.py --input data/unlabeled.csv --output predictions.csv --no-labels
```

**Output Format**:
```csv
segment_id,predicted_label,confidence_bear_flag,confidence_bull_flag,confidence_false_breakout,...
0,bull_flag,0.05,0.82,0.03,...
1,bear_flag,0.78,0.12,0.02,...
```

If labels are present in input, also includes:
- `true_label`: Ground truth label
- `correct`: Boolean indicating correct prediction

---

### Baseline Model

- **`baseline_model.py`**: Heuristic classifier (unchanged from original)
  - Rule-based pattern detection using slope thresholds
  - Used for comparison with CNN model

---

## Complete Pipeline Workflow

```bash
# 1. Preprocess data (extract segments, engineer features, train/test split)
python 01-data-preprocessing.py

# 2. Train CNN model
python 02-training.py

# 3. Evaluate on test set and generate visualizations
python 03-evaluation.py

# 4. Run inference on new data
python 04-inference.py --input path/to/new_data.csv --output predictions.csv
```

---

## Logging

All modules use the unified logging system from `utils.py`:
- **Console output**: All log messages printed to stdout
- **File output**: All log messages saved to `log/run.log`
- Log format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

Each module logs:
- Configuration parameters
- Data loading and shapes
- Processing steps
- Model architecture (training)
- Training progress and metrics (training)
- Evaluation results (evaluation)
- Prediction statistics (inference)

---

## Configuration Customization

Edit `config.py` to customize:

**Paths**:
```python
DATA_DIR = "/work/data"
EXPORT_DIR = "/work/data/export"
CHECKPOINT_DIR = "/work/data/export/checkpoints_v2"
```

**Training Hyperparameters**:
```python
MAX_EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
```

**Data Preprocessing**:
```python
SEQUENCE_LENGTH = 24  # Target number of timesteps
WINDOW_SIZE = 5       # Rolling window for features
TEST_SIZE = 0.1       # Train/test split ratio
```

**Primary Metric**:
```python
PRIMARY_METRIC = "pr_auc"  # Options: "pr_auc", "f1", "accuracy"
```

---

## Dependencies

See `requirements.txt` and `environment.yml` in project root.

Key packages:
- PyTorch + PyTorch Lightning
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

---

## Notes

- Original notebooks remain in `notebook/` directory for reference
- All data files are stored in `data/export/`
- Model checkpoints saved to `data/export/checkpoints_v2/`
- Figures and evaluation results in `src/03_evaluation/`
- Logs written to `log/run.log`
