"""
Inference script for flag pattern classification.

This script:
1. Loads the best trained model
2. Processes new CSV data files (with or without labels)
3. Generates predictions with confidence scores
4. Saves results to CSV

Usage:
    # On labeled data (will also show accuracy if labels present)
    python 04-inference.py --input data/new_segments.csv --output predictions.csv
    
    # On unlabeled data
    python 04-inference.py --input data/unlabeled_segments.csv --output predictions.csv --no-labels
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report

import config
from utils import setup_logger, ensure_dir, minmax_norm, interpolate_series


# Import model from training script
import importlib.util
spec = importlib.util.spec_from_file_location("training", os.path.join(os.path.dirname(__file__), "02-training.py"))
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

FlagPatternClassifier = training_module.FlagPatternClassifier


class InferenceDataset(Dataset):
    """Dataset for inference (no labels required)."""
    
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


def preprocess_segment(df_segment, feature_cols, sequence_length, window_size=5):
    """
    Preprocess a single segment: normalize and engineer features.
    
    Args:
        df_segment: DataFrame with columns [timestamp, open, high, low, close, volume]
        feature_cols: List of feature column names to extract
        sequence_length: Target sequence length
        window_size: Rolling window size for feature engineering
        
    Returns:
        Feature array of shape (sequence_length, num_features)
    """
    df = df_segment.copy()
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Normalize OHLC
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[f'{col}_norm'] = minmax_norm(df[col].values)
    
    # Engineer features
    if 'close_norm' in df.columns and 'high_norm' in df.columns and 'low_norm' in df.columns:
        df['vol_close'] = df['close_norm'].rolling(window=window_size, min_periods=1).std()
        df['vol_high_low'] = (df['high_norm'] - df['low_norm']).rolling(window=window_size, min_periods=1).std()
        df['compression_ratio'] = (df['high_norm'] - df['low_norm']).rolling(window=window_size, min_periods=1).mean()
        df['trend'] = df['close_norm'].rolling(window=window_size, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0, raw=False
        )
    
    # Fill NaN values
    df = df.fillna(0.0)
    
    # Extract features
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    
    # Resample to target sequence length
    if feat.shape[0] != sequence_length:
        resampled = np.zeros((sequence_length, feat.shape[1]), dtype=np.float32)
        for i in range(feat.shape[1]):
            resampled[:, i] = interpolate_series(feat[:, i], sequence_length)
        feat = resampled
    
    return feat


def load_model_and_metadata(logger):
    """
    Load the trained model and metadata.
    
    Returns:
        Tuple of (model, metadata, device)
    """
    # Load training metadata
    metadata_path = os.path.join(config.EXPORT_DIR, 'training_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    best_model_path = metadata['best_model_path']
    num_classes = metadata['num_classes']
    input_dim = metadata['input_dim']
    idx_to_label = metadata['idx_to_label']
    feature_cols = metadata['feature_cols']
    
    logger.info(f"Loaded metadata from: {metadata_path}")
    logger.info(f"Model checkpoint: {best_model_path}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Features: {feature_cols}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    # We don't need class weights for inference
    model = FlagPatternClassifier.load_from_checkpoint(
        best_model_path,
        input_dim=input_dim,
        num_classes=num_classes,
        class_weights=None,
        map_location=device
    )
    model.eval()
    model.freeze()
    model = model.to(device)
    
    logger.info("Model loaded successfully")
    
    return model, metadata, device


def predict_on_loader(loader, model, device, idx_to_label):
    """
    Generate predictions for a data loader.
    
    Args:
        loader: DataLoader instance
        model: Trained model
        device: torch device
        idx_to_label: Dictionary mapping class indices to labels
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.append(preds)
            all_probs.append(probs)
    
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    
    # Convert indices to labels
    pred_labels = np.vectorize(idx_to_label.get)(predictions)
    
    return pred_labels, probabilities


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description='Run inference on new segments')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file with segments')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save predictions CSV')
    parser.add_argument('--no-labels', action='store_true',
                       help='Input data does not have labels')
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(__name__, config.LOG_FILE)
    
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Has labels: {not args.no_labels}")
    
    # Load model and metadata
    logger.info("\n" + "=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)
    
    model, metadata, device = load_model_and_metadata(logger)
    
    feature_cols = metadata['feature_cols']
    idx_to_label = metadata['idx_to_label']
    label_to_idx = metadata['label_to_idx']
    num_classes = metadata['num_classes']
    
    # Load input data
    logger.info("\n" + "=" * 80)
    logger.info("LOADING INPUT DATA")
    logger.info("=" * 80)
    
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Process segments
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING SEGMENTS")
    logger.info("=" * 80)
    
    if 'segment_id' in df.columns:
        # Data is already segmented
        df = df.sort_values(['segment_id', 'seq_pos'] if 'seq_pos' in df.columns else 'segment_id')
        segments = []
        segment_ids = []
        true_labels = [] if not args.no_labels and 'label' in df.columns else None
        
        for seg_id, g in df.groupby('segment_id', sort=True):
            if 'seq_pos' in g.columns:
                g = g.sort_values('seq_pos')
            
            feat = preprocess_segment(g, feature_cols, config.SEQUENCE_LENGTH, config.WINDOW_SIZE)
            segments.append(feat)
            segment_ids.append(seg_id)
            
            if true_labels is not None:
                true_labels.append(g['label'].iloc[0])
        
        logger.info(f"Processed {len(segments)} segments")
    else:
        # Treat entire dataframe as one segment
        feat = preprocess_segment(df, feature_cols, config.SEQUENCE_LENGTH, config.WINDOW_SIZE)
        segments = [feat]
        segment_ids = [0]
        true_labels = None
        logger.info("Processed 1 segment (entire input)")
    
    # Create dataset and loader
    X = np.stack(segments, axis=0)
    dataset = InferenceDataset(X)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Generate predictions
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PREDICTIONS")
    logger.info("=" * 80)
    
    pred_labels, probabilities = predict_on_loader(loader, model, device, idx_to_label)
    
    logger.info(f"Generated predictions for {len(pred_labels)} segments")
    
    # Create results dataframe
    results = pd.DataFrame({
        'segment_id': segment_ids,
        'predicted_label': pred_labels,
    })
    
    # Add confidence scores for each class
    for i in range(num_classes):
        class_label = idx_to_label[i]
        results[f'confidence_{class_label}'] = probabilities[:, i]
    
    # Add true labels if available
    if true_labels is not None:
        results['true_label'] = true_labels
        results['correct'] = (results['predicted_label'] == results['true_label'])
        
        # Calculate metrics
        y_true_idx = np.vectorize(label_to_idx.get)(true_labels)
        y_pred_idx = np.vectorize(label_to_idx.get)(pred_labels)
        
        acc = accuracy_score(y_true_idx, y_pred_idx)
        f1 = f1_score(y_true_idx, y_pred_idx, average='macro')
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 80)
        logger.info(f"Accuracy:         {acc:.4f}")
        logger.info(f"F1 Score (macro): {f1:.4f}")
        
        # Log classification report
        logger.info("\nClassification Report:")
        report = classification_report(true_labels, pred_labels,
                                       target_names=[str(lbl) for lbl in sorted(set(true_labels))])
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    
    # Save results
    ensure_dir(os.path.dirname(args.output))
    results.to_csv(args.output, index=False)
    logger.info(f"\nSaved predictions to: {args.output}")
    
    # Log prediction distribution
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION DISTRIBUTION")
    logger.info("=" * 80)
    pred_dist = pd.Series(pred_labels).value_counts()
    for label, count in pred_dist.items():
        pct = count / len(pred_labels) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE COMPLETED SUCCESSFULLY")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

