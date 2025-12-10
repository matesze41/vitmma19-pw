# Project Transformation Summary

## Overview

This document summarizes the complete transformation of the Flag Pattern Classification project from Jupyter notebooks to production-ready Python modules.

**Date**: 2025
**Transformation Type**: Jupyter Notebooks → Python Modules
**Primary Goal**: Create modular, production-ready ML pipeline with comprehensive logging

---

## What Was Transformed

### Source (Jupyter Notebooks)
- `01-data-exploration.ipynb` - Data extraction from Label Studio JSON
- `02-label-analysis.ipynb` - Label distribution analysis
- `03-data-preproc.ipynb` - Feature engineering and preprocessing
- `05-exp.ipynb` - Model training and evaluation

### Target (Python Modules in `src/`)
- `config.py` - Centralized configuration (140+ lines)
- `utils.py` - Shared utilities with dual logging
- `01-data-preprocessing.py` - Complete data pipeline (434 lines)
- `02-training.py` - Model architecture and training (661 lines)
- `03-evaluation.py` - Evaluation and visualization (676 lines)
- `04-inference.py` - Inference on new data (306 lines)
- `baseline_model.py` - Unchanged heuristic baseline (416 lines)

**Total New/Modified Code**: ~2,600+ lines of production Python

---

## Key Improvements

### 1. Comprehensive Logging System

**Requirements Met**:
- ✅ Configuration parameters logged
- ✅ Data shapes and distributions logged
- ✅ Model architecture logged
- ✅ Training progress logged (per epoch)
- ✅ Validation metrics logged
- ✅ Final evaluation results logged

**Implementation**:
```python
# utils.py - Dual output logging
logger = setup_logger(__name__, config.LOG_FILE)
# Logs to both:
#   - stdout (console)
#   - log/run.log (file)
```

### 2. Centralized Configuration

All settings in one place (`config.py`):
- File paths
- Hyperparameters
- Preprocessing parameters
- Metric definitions
- Output directories

**Before**: Hardcoded values scattered across notebooks
**After**: Single source of truth, easy to modify

### 3. Modular Architecture

**Before**: Monolithic notebooks with mixed concerns
**After**: Separated concerns with clear interfaces

```
01-data-preprocessing.py → segments_preproc_24.csv
                         ↓
02-training.py          → best_model.ckpt + metadata.pkl
                         ↓
03-evaluation.py        → metrics + visualizations
                         ↓
04-inference.py         → predictions.csv
```

### 4. Production-Ready Features

- **Error handling**: Robust file I/O and data validation
- **Reproducibility**: Fixed random seeds, deterministic operations
- **Scalability**: Batch processing, GPU support
- **Maintainability**: Clear documentation, type hints
- **Testability**: Modular functions, separated concerns

---

## Detailed Module Specifications

### config.py

**Purpose**: Single configuration file for entire pipeline

**Key Sections**:
1. **Paths** (lines 10-38)
   - Data directories
   - Export paths
   - Checkpoint locations
   - Output directories per module

2. **Data Preprocessing** (lines 40-50)
   - Train/test split: 90/10
   - Sequence length: 24 timesteps
   - Rolling window: 5 periods
   - Random seed: 11

3. **Model Hyperparameters** (lines 52-80)
   - Max epochs: 50
   - Batch size: 12
   - Learning rate: 1e-3
   - Weight decay: 1e-4
   - Dropout: 0.3 (conv), 0.5 (FC)
   - Patience: 10 epochs

4. **Metric Configuration** (lines 82-133)
   - Primary metric: PR-AUC
   - Monitoring mode: max
   - Checkpoint strategy
   - Figure settings (DPI=300)

---

### utils.py

**Purpose**: Shared utility functions across all modules

**Functions**:

1. **`setup_logger(name, log_file)`**
   - Creates logger with dual output (console + file)
   - Format: timestamp, module, level, message
   - File: `log/run.log`

2. **`minmax_norm(arr)`**
   - Min-max normalization to [0, 1]
   - Handles constant arrays (returns zeros)

3. **`interpolate_series(arr, target_len)`**
   - 1D interpolation for resampling
   - Linear interpolation between points
   - Used to standardize sequence lengths

4. **`strip_guid(filename)`**
   - Removes Label Studio GUID prefixes
   - Format: `GUID-actualname.csv` → `actualname.csv`

5. **`ensure_dir(path)`**
   - Safely creates directory if not exists
   - Creates parent directories as needed

---

### 01-data-preprocessing.py

**Purpose**: End-to-end data preparation pipeline

**Input Sources**:
- Label Studio JSON files (`data/export/*.json`)
- Market data CSVs (`data/GUID/SYMBOL_*.csv`)

**Pipeline Steps**:

1. **Extract Segments** (`extract_segments()`)
   - Parse Label Studio JSON annotations
   - Extract time ranges for each labeled segment
   - Load corresponding CSV data
   - Build segments with metadata
   - Output: `segments_meta.csv`, `segments_values.csv`, `segments.h5`

2. **Train/Test Split**
   - Stratified 90/10 split by label
   - Ensures class balance in both sets
   - Saves segment IDs for reproducibility

3. **Feature Engineering** (`process_segment()`)
   - **Normalize**: Min-max scale OHLC to [0, 1]
   - **Volatility**: Rolling std of close (vol_close)
   - **Spread Volatility**: Rolling std of high-low (vol_high_low)
   - **Compression**: Rolling mean of high-low (compression_ratio)
   - **Trend**: Slope over window (trend)

4. **Sequence Standardization**
   - Resample all segments to 24 timesteps
   - Linear interpolation for upsampling/downsampling
   - Pad short sequences by repeating last value

5. **Output Generation**
   - Training set: `segments_preproc_24.csv`
   - Test set: `segments_preproc_24_test.csv`
   - Test raw (for baseline): `segments_test_raw.csv`

**Feature Vector** (8 dimensions):
```
[open_norm, high_norm, low_norm, close_norm, 
 vol_close, vol_high_low, compression_ratio, trend]
```

**Logging**: Comprehensive logging of extraction, splitting, processing stats

---

### 02-training.py

**Purpose**: Train CNN model with PyTorch Lightning

**Components**:

1. **`SegmentDataset`** (PyTorch Dataset)
   - Wraps numpy arrays as tensors
   - Returns (features, label) pairs

2. **`FlagPatternClassifier`** (Lightning Module)
   
   **Architecture**:
   ```
   Input: (batch, 24, 8)
   
   Conv Block 1: Conv1D(8→64, k=3) → BN → ReLU → Drop(0.3) → MaxPool(2)
   Conv Block 2: Conv1D(64→64, k=3) → BN → ReLU → Drop(0.3) → MaxPool(2)
   Conv Block 3: Conv1D(64→128, k=3) → BN → ReLU → Drop(0.3) → MaxPool(2)
   Conv Block 4: Conv1D(128→128, k=3) → BN → ReLU → Drop(0.3) → MaxPool(2)
   
   Flatten
   
   FC1: Linear(128) → ReLU → Drop(0.5)
   FC2: Linear(64) → ReLU → Drop(0.5)
   FC3: Linear(num_classes)
   
   Output: Logits for each class
   ```
   
   **Techniques**:
   - Batch normalization for stable training
   - Dropout for regularization
   - MaxPooling for dimensionality reduction
   - Class weights for imbalanced data
   
   **Loss**: CrossEntropyLoss (with class weights)
   
   **Optimizer**: AdamW (lr=1e-3, wd=1e-4)
   
   **Scheduler**: CosineAnnealingLR (T_max=50)

3. **Training Loop**
   - Data loading with stratified split (80/20 train/val)
   - Class weight calculation (inverse frequency)
   - PyTorch Lightning Trainer with:
     - Early stopping (patience=10)
     - Model checkpointing (save best PR-AUC)
     - Gradient clipping
     - GPU support (if available)
   
4. **Metrics Computation** (per epoch)
   - Accuracy
   - F1 Score (macro average)
   - PR-AUC (macro average) ← Primary
   - AUC-ROC (One-vs-One)
   - AUC-ROC (One-vs-Rest)

**Outputs**:
- Best model checkpoint: `checkpoints_v2/best_model_*.ckpt`
- Training metadata: `training_metadata.pkl` (for inference)
  - Contains: feature_cols, label mappings, num_classes, input_dim, best_model_path

**Logging**: Configuration, architecture, data shapes, epoch metrics, best model info

---

### 03-evaluation.py

**Purpose**: Comprehensive evaluation and visualization

**Evaluation Sets**:
1. Training set (for overfitting check)
2. Test set (primary evaluation)

**Models Compared**:
1. CNN model (deep learning)
2. Baseline model (heuristic rules)

**Metrics Computed**:

**CNN Model**:
- Accuracy
- F1 Score (macro)
- PR-AUC (macro) - per class and averaged
- AUC-ROC (One-vs-One)
- AUC-ROC (One-vs-Rest)
- Confusion matrix
- Per-class precision/recall/F1

**Baseline Model**:
- Accuracy
- F1 Score (macro)
- Confusion matrix

**Visualizations Generated**:

1. **`confusion_matrix_training_comparison.png`**
   - Side-by-side: CNN vs Baseline (training set)
   - Shows PR-AUC and accuracy

2. **`confusion_matrix_test_comparison.png`**
   - Side-by-side: CNN vs Baseline (test set)
   - Primary evaluation figure

3. **`confusion_matrix_cnn_test.png`**
   - Detailed CNN confusion matrix (test set)
   - Includes all metrics

4. **`roc_pr_curves_test.png`**
   - 2 rows × num_classes columns
   - Top row: ROC curves per class
   - Bottom row: PR curves per class
   - Shows AUC/AP values

5. **`training_history.png`** (if available)
   - Loss curves (train/val)
   - Metric curves (train/val)

**Output Files**:
- Figures: `src/03_evaluation/figures/*.png`
- Metrics: `src/03_evaluation/evaluation_metrics.txt`

**Comparison Analysis**:
- Absolute improvement: CNN - Baseline
- Percentage improvement: (CNN/Baseline - 1) × 100%
- Logged for accuracy and F1 score

**Logging**: All metrics, improvements, figure generation

---

### 04-inference.py

**Purpose**: Production inference on new data

**Input Flexibility**:
- **With segment_id**: Processes multiple segments
- **Without segment_id**: Treats as single segment
- **With labels**: Computes evaluation metrics
- **Without labels**: Just predictions and confidence

**Preprocessing**:
- Same pipeline as training: normalize → engineer features → resample to 24
- Uses metadata from training to ensure consistency

**Output Format**:
```csv
segment_id,predicted_label,confidence_bear_flag,confidence_bull_flag,...,true_label,correct
0,bull_flag,0.05,0.82,0.03,...,bull_flag,True
1,bear_flag,0.78,0.12,0.02,...,bear_flag,True
```

**Features**:
- Command-line interface with argparse
- Confidence scores for all classes
- Optional evaluation if labels present
- Prediction distribution logging
- Batch processing with DataLoader

**Usage Examples**:
```bash
# Labeled data
python 04-inference.py --input test.csv --output pred.csv

# Unlabeled data
python 04-inference.py --input new_data.csv --output pred.csv --no-labels
```

**Logging**: Input/output paths, preprocessing stats, predictions, optional metrics

---

## Data Flow Diagram

```
Label Studio JSON + Market CSVs
        ↓
   [01-data-preprocessing.py]
   - Extract segments
   - Train/test split (90/10)
   - Normalize OHLC
   - Engineer features
   - Resample to 24 steps
        ↓
segments_preproc_24.csv (train)
segments_preproc_24_test.csv (test)
        ↓
   [02-training.py]
   - Load preprocessed data
   - Split train/val (80/20)
   - Build CNN model
   - Train with class weights
   - Early stopping
   - Save best model
        ↓
best_model.ckpt + training_metadata.pkl
        ↓
   [03-evaluation.py]
   - Load best model
   - Evaluate on test set
   - Compare with baseline
   - Generate visualizations
   - Save metrics
        ↓
evaluation_metrics.txt + figures/
        ↓
   [04-inference.py]
   - Load best model + metadata
   - Preprocess new data
   - Generate predictions
   - Save with confidence scores
        ↓
predictions.csv
```

---

## Logging Output Structure

All modules log to **both**:
1. **Console (stdout)**: Real-time monitoring
2. **File (`log/run.log`)**: Persistent record

### Log Format
```
2025-01-15 10:30:45,123 - module_name - INFO - Message content
```

### Logged Information by Module

**01-data-preprocessing.py**:
- Number of JSON files found
- Segments extracted per file
- Train/test split sizes
- Label distribution
- Feature engineering progress
- Output file paths

**02-training.py**:
- Configuration summary
- Dataset sizes (train/val)
- Class distribution and weights
- Model architecture
- Device (CPU/GPU)
- Per-epoch metrics (train + val)
- Best model info

**03-evaluation.py**:
- Model loading info
- Evaluation set sizes
- CNN metrics (train + test)
- Baseline metrics (train + test)
- Improvement analysis
- Figure generation
- Classification reports

**04-inference.py**:
- Input/output paths
- Data loading stats
- Preprocessing steps
- Prediction distribution
- Optional evaluation metrics

---

## Testing the Pipeline

### Quick Test (subset of data)

```bash
# 1. Preprocess (should take ~1-2 min)
python src/01-data-preprocessing.py

# Expected output:
# - log/run.log populated
# - data/export/segments_*.csv created
# - Train/test split logged

# 2. Train (may take 10-30 min depending on data size and GPU)
python src/02-training.py

# Expected output:
# - lightning_logs/ created
# - Checkpoint saved
# - training_metadata.pkl created
# - Epoch metrics logged

# 3. Evaluate (should take <1 min)
python src/03-evaluation.py

# Expected output:
# - src/03_evaluation/figures/ populated
# - evaluation_metrics.txt created
# - Comparison results logged

# 4. Inference (should take seconds)
python src/04-inference.py \
  --input data/export/segments_preproc_24_test.csv \
  --output test_predictions.csv

# Expected output:
# - test_predictions.csv created
# - Metrics logged (since input has labels)
```

### Validation Checklist

- [ ] `log/run.log` exists and contains all module logs
- [ ] `data/export/segments_preproc_24.csv` has 8 feature columns
- [ ] `data/export/training_metadata.pkl` loadable with pickle
- [ ] Best model checkpoint saved in `checkpoints_v2/`
- [ ] `src/03_evaluation/figures/` contains 4-5 PNG files
- [ ] `evaluation_metrics.txt` shows improvement over baseline
- [ ] Inference predictions have confidence scores for all classes

---

## Configuration Customization Examples

### Increase Training Epochs

Edit `src/config.py`:
```python
MAX_EPOCHS = 100  # Was 50
```

### Change Batch Size

```python
BATCH_SIZE = 24  # Was 12 (if more GPU memory)
```

### Use Different Sequence Length

```python
SEQUENCE_LENGTH = 48  # Was 24
# Note: Requires re-running preprocessing
```

### Switch Primary Metric to F1

```python
PRIMARY_METRIC = "f1"  # Was "pr_auc"
```

### Adjust Train/Test Split

```python
TEST_SIZE = 0.20  # Was 0.10 (20% test)
```

---

## Performance Expectations

### Typical Results (from notebook experiments)

**CNN Model** (test set):
- Accuracy: ~0.75-0.85
- F1 Score: ~0.70-0.80
- PR-AUC: ~0.75-0.85

**Baseline Model** (test set):
- Accuracy: ~0.60-0.70
- F1 Score: ~0.55-0.65

**Improvement**: CNN typically 10-25% better than baseline

*Note*: Actual performance depends on:
- Data quality and quantity
- Label distribution balance
- Hyperparameter tuning
- Random seed

---

## File Size Estimates

After running full pipeline:

- `segments_preproc_24.csv`: ~5-50 MB (depends on # segments)
- `best_model.ckpt`: ~2-5 MB
- `training_metadata.pkl`: <1 MB
- `log/run.log`: ~100 KB - 1 MB
- Figures (all): ~1-2 MB total

---

## Troubleshooting

### Issue: "No module named 'pytorch_lightning'"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 8  # or 4
```

### Issue: "FileNotFoundError: segments_preproc_24.csv"

**Solution**: Run preprocessing first
```bash
python src/01-data-preprocessing.py
```

### Issue: No improvement during training

**Possible causes**:
1. Class imbalance too severe → Check class weights
2. Learning rate too high/low → Adjust in config
3. Model too simple/complex → Modify architecture
4. Data quality issues → Check preprocessing logs

**Debug**:
```bash
# Check training logs
cat log/run.log | grep "val_pr_auc"
# Look for increasing validation metric
```

### Issue: Poor test performance vs validation

**Likely cause**: Overfitting or distribution shift

**Solutions**:
1. Increase dropout rates
2. Add more data augmentation
3. Reduce model complexity
4. Check test set distribution

---

## Next Steps / Future Enhancements

### Short Term
- [ ] Add data augmentation (time warping, noise injection)
- [ ] Implement cross-validation
- [ ] Add TensorBoard logging
- [ ] Create unit tests

### Medium Term
- [ ] Hyperparameter tuning (Optuna, Ray Tune)
- [ ] Ensemble methods (combine multiple models)
- [ ] Attention mechanisms in model
- [ ] Multi-scale temporal features

### Long Term
- [ ] Real-time inference API (FastAPI/Flask)
- [ ] Model versioning (MLflow, DVC)
- [ ] Automated retraining pipeline
- [ ] A/B testing framework

---

## Maintenance Notes

### When to Retrain
- New labeled data available (monthly/quarterly)
- Performance degradation detected
- Market regime changes
- Feature engineering improvements

### Model Versioning
Currently: Timestamp-based checkpoints
```
best_model_epoch=X_val_pr_auc=Y.ZZZZ.ckpt
```

Recommended: Add version tags
```python
# config.py
MODEL_VERSION = "v1.0.0"
# Use in checkpoint naming
```

### Monitoring
Check logs regularly for:
- Training convergence
- Validation metric trends
- Test performance consistency
- Inference prediction distributions

---

## Credits & References

**Original Work**: Jupyter notebooks in `notebook/`
**Transformation**: Complete modularization with production best practices
**Framework**: PyTorch Lightning for scalable training
**Metrics**: scikit-learn for comprehensive evaluation

---

## Appendix: File Inventory

### Source Code (src/)
| File | Lines | Purpose |
|------|-------|---------|
| config.py | 140 | Configuration |
| utils.py | 80 | Utilities |
| 01-data-preprocessing.py | 434 | Data pipeline |
| 02-training.py | 661 | Model training |
| 03-evaluation.py | 676 | Evaluation |
| 04-inference.py | 306 | Inference |
| baseline_model.py | 416 | Baseline (unchanged) |
| README.md | ~350 | Documentation |

### Data Files (data/export/)
- segments_meta.csv
- segments_values.csv
- segments.h5
- segments_preproc_24.csv
- segments_preproc_24_test.csv
- segments_test_raw.csv
- training_metadata.pkl
- checkpoints_v2/*.ckpt

### Outputs
- log/run.log
- src/03_evaluation/evaluation_metrics.txt
- src/03_evaluation/figures/*.png
- lightning_logs/version_*/

### Original Notebooks (notebook/)
- 01-data-exploration.ipynb
- 02-label-analysis.ipynb
- 03-data-preproc.ipynb
- 04-train.ipynb
- 05-exp.ipynb
- 06-optimize.ipynb
- baseline_model.py

---

**End of Transformation Summary**
