# Inference script
# This script runs the model on new, unseen data.
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils import setup_logger
from train_model import FlagPatternClassifier

logger = setup_logger("model_inference")

BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
EXPORT_DIR = os.path.join(BASE_DATA_DIR, "export")
WINDOW = 5  # rolling window for per-step volatility features
EPS = 1e-9


def minmax_norm(s: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1] range."""
    vmin = s.min()
    vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - vmin) / (vmax - vmin)


def _interp_series(orig_x: np.ndarray, y: np.ndarray, tgt_x: np.ndarray) -> np.ndarray:
    """Interpolate a series to target x values."""
    if len(y) == 0:
        return np.zeros(len(tgt_x), dtype=float)
    if len(y) == 1:
        return np.full(len(tgt_x), float(y[0]))
    return np.interp(tgt_x, orig_x, y)


def preprocess_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing steps to a single segment."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Determine numeric columns available
    num_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    
    if not num_cols:
        logger.warning("No OHLC columns found in segment")
        return None
    
    # Raw spread for compression ratio (per-step)
    raw_high = df["high"].astype(float) if "high" in df.columns else pd.Series(np.zeros(len(df)))
    raw_low = df["low"].astype(float) if "low" in df.columns else pd.Series(np.zeros(len(df)))
    raw_spread = (raw_high - raw_low).to_numpy()
    start_spread = float(raw_spread[0]) if len(raw_spread) else 0.0
    comp_series = (raw_spread / max(abs(start_spread), EPS)) if len(raw_spread) else np.array([], dtype=float)

    # Normalize OHLC per segment on raw values
    norm = {}
    for c in num_cols:
        norm[f"{c}_norm"] = minmax_norm(df[c].astype(float))
    norm_df = pd.DataFrame(norm)

    # Resample normalized columns and compression ratio to 24 steps
    L = len(df)
    orig_x = np.arange(L, dtype=float) if L > 1 else np.array([0.0])
    tgt_x = np.linspace(0.0, max(0.0, (L - 1.0)), 24)

    resampled = {}
    for col in norm_df.columns:
        y = norm_df[col].astype(float).to_numpy()
        resampled[col] = _interp_series(orig_x, y, tgt_x)

    comp_resampled = _interp_series(orig_x, comp_series, tgt_x)

    out = pd.DataFrame(resampled)

    # Per-datapoint engineered features on resampled normalized series
    close_n = out["close_norm"] if "close_norm" in out.columns else pd.Series(np.zeros(24))
    high_n = out["high_norm"] if "high_norm" in out.columns else pd.Series(np.zeros(24))
    low_n = out["low_norm"] if "low_norm" in out.columns else pd.Series(np.zeros(24))
    open_n = out["open_norm"] if "open_norm" in out.columns else pd.Series(np.zeros(24))

    # Rolling volatility features (per point)
    vol_close = pd.Series(close_n).rolling(WINDOW, min_periods=1).std().fillna(0.0).to_numpy()
    spread_n = (pd.Series(high_n) - pd.Series(low_n)).to_numpy()
    vol_high_low = pd.Series(spread_n).rolling(WINDOW, min_periods=1).mean().fillna(0.0).to_numpy()

    # Per-step compression ratio (from raw spread, resampled) with clipping to [0.2, 3.0]
    compression_ratio = np.clip(comp_resampled, 0.2, 3.0)

    # Per-step trend
    trend = (pd.Series(open_n) - pd.Series(close_n)).to_numpy()

    # Assemble output with engineered features
    out["seq_pos"] = np.arange(1, 25, dtype=int)  # 1..24
    out["vol_close"] = vol_close
    out["vol_high_low"] = vol_high_low
    out["compression_ratio"] = compression_ratio
    out["trend"] = trend
    return out


class SegmentDataset(Dataset):
    """Dataset for inference."""
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def load_and_preprocess_csv(csv_path: str, feature_cols: list) -> tuple:
    """
    Load a CSV file and preprocess it.
    Returns (preprocessed_features, filename) or (None, filename) if preprocessing fails.
    """
    filename = os.path.basename(csv_path)
    logger.info(f"Loading CSV: {filename}")
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        
        if "timestamp" not in df.columns:
            logger.warning(f"No timestamp column found in {filename}, skipping")
            return None, filename
        
        logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Preprocess the segment
        processed = preprocess_segment(df)
        
        if processed is None:
            logger.warning(f"Preprocessing failed for {filename}")
            return None, filename
        
        # Extract features in the correct order
        feat = processed[feature_cols].to_numpy(dtype=np.float32)
        
        # Pad to 24 if necessary
        if feat.shape[0] < 24:
            pad = np.repeat(feat[-1:], 24 - feat.shape[0], axis=0)
            feat = np.concatenate([feat, pad])
        else:
            feat = feat[:24]
        
        logger.info(f"Preprocessed {filename}: shape {feat.shape}")
        return feat, filename
        
    except Exception as e:
        logger.error(f"Error loading/preprocessing {csv_path}: {e}")
        return None, filename


def predict(csv_path=None):
    logger.info("Starting inference pipeline...")
    
    # csv_path is required
    if csv_path is None:
        logger.error("CSV path is required")
        return
    
    csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        logger.error(f"Path not found: {csv_path}")
        return
    
    if not os.path.isdir(csv_path):
        logger.error(f"Expected a directory, got a file: {csv_path}")
        return
    
    logger.info(f"Processing directory: {csv_path}")
    
    # Load metadata and model
    metadata_path = os.path.join(EXPORT_DIR, "eval_metadata.pt")
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return
    
    meta = torch.load(metadata_path, weights_only=False)
    
    checkpoint_path = meta["checkpoint_path"]
    num_classes = meta["num_classes"]
    input_dim = meta["input_dim"]
    class_weights = np.array(meta["class_weights"])
    label_to_idx = meta["label_to_idx"]
    idx_to_label = meta["idx_to_label"]
    feature_cols = meta["feature_cols"]
    
    logger.info(f"Loaded metadata from: {metadata_path}")
    logger.info(f"Model classes: {list(idx_to_label.values())}")
    logger.info(f"Feature columns: {feature_cols}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model = FlagPatternClassifier.load_from_checkpoint(
            checkpoint_path,
            input_dim=input_dim,
            num_classes=num_classes,
            class_weights=class_weights,
            weights_only=False
        ).to(device)
        model.eval()
        model.freeze()
        logger.info(f"Loaded model from: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Find all CSV files in directory
    csv_files = sorted(glob.glob(os.path.join(csv_path, "**/*.csv"), recursive=True))
    # Filter out export/data files that aren't raw data
    csv_files = [f for f in csv_files if "export" not in f and "segments" not in f]
    
    if not csv_files:
        logger.error(f"No CSV files found in {csv_path}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    all_predictions = []
    segments_data = []
    
    for csv_file in csv_files:
        feat, filename = load_and_preprocess_csv(csv_file, feature_cols)
        
        if feat is None:
            continue
        
        segments_data.append((feat, filename))
    
    if not segments_data:
        logger.error("No valid segments to process")
        return
    
    logger.info(f"Processing {len(segments_data)} segments")
    
    # Stack features and create dataset
    X = np.stack([feat for feat, _ in segments_data])
    filenames = [filename for _, filename in segments_data]
    
    dataset = SegmentDataset(X)
    loader = DataLoader(dataset, batch_size=12, shuffle=False)
    
    # Run inference
    logger.info("Running inference on preprocessed data...")
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Convert predictions to label names
    predicted_labels = np.vectorize(idx_to_label.get)(all_preds)
    
    # Create results dataframe
    results = pd.DataFrame({
        "filename": filenames,
        "predicted_label": predicted_labels,
        "predicted_class_idx": all_preds,
    })
    
    # Add probability columns for each class
    for i, label in idx_to_label.items():
        results[f"prob_{label}"] = all_probs[:, i]
    
    logger.info(f"\nPrediction Results:")
    logger.info(f"Total segments: {len(results)}")
    logger.info(f"\nLabel distribution:")
    logger.info(f"{results['predicted_label'].value_counts().to_string()}")
    
    # Save predictions
    predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_csv = os.path.join(predictions_dir, "predictions.csv")
    results.to_csv(predictions_csv, index=False)
    logger.info(f"\nSaved predictions to: {predictions_csv}")
    
    # Print results summary
    logger.info("\nDetailed Predictions:")
    logger.info(f"\n{results.to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on all CSV files in a directory")
    parser.add_argument(
        "csv_path",
        help="Path to directory containing CSV files"
    )
    
    args = parser.parse_args()
    predict(args.csv_path)