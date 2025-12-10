"""
Data preprocessing pipeline for flag pattern classification.

This script:
1. Loads raw JSON annotations and CSV time series data
2. Extracts labeled segments from annotations
3. Creates train/test split (90/10)
4. Normalizes and engineers features
5. Resamples to fixed sequence length (24 steps)
6. Saves preprocessed data for training and evaluation

Usage:
    python 01-data-preprocessing.py
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import h5py
from glob import glob
from typing import List, Dict, Any
from datetime import datetime

import config
from utils import setup_logger, minmax_norm, interpolate_series, strip_guid, ensure_dir


def extract_segments(json_entry: Dict[str, Any], csv_directory: str, logger) -> List[Dict[str, Any]]:
    """
    Extract all labeled time segments from a JSON annotation entry.
    
    Args:
        json_entry: Single annotation entry from Label Studio JSON
        csv_directory: Directory containing CSV files
        logger: Logger instance
        
    Returns:
        List of segment dictionaries with metadata and data
    """
    # Check if file_upload key exists
    if "file_upload" not in json_entry:
        logger.warning("Entry missing 'file_upload' key, skipping")
        return []
    
    guid_name = json_entry["file_upload"]
    
    # Handle None or empty file_upload
    if guid_name is None or guid_name == "":
        logger.warning("file_upload is None or empty, skipping")
        return []
    
    clean_name = strip_guid(guid_name)
    logger.debug(f"Processing file_upload={guid_name}, clean_name={clean_name}")

    # Try loading the CSV from several likely locations
    candidate_dirs = [
        csv_directory,
        os.path.dirname(csv_directory),
        config.DATA_DIR,
    ]

    full_csv_path = None
    for d in candidate_dirs:
        if not d:
            continue
        candidates = [
            os.path.join(d, guid_name),
            os.path.join(d, clean_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                full_csv_path = path
                break
        if full_csv_path is not None:
            break

    if full_csv_path is None:
        logger.warning(f"Cannot find CSV for annotation: {guid_name}")
        return []

    logger.debug(f"Using CSV: {full_csv_path}")

    try:
        df_csv = pd.read_csv(full_csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV {full_csv_path}: {e}")
        return []

    # Normalize column names to lowercase
    df_csv.columns = df_csv.columns.str.lower()
    
    logger.debug(f"CSV loaded with {len(df_csv)} rows and columns {list(df_csv.columns)}")

    # Convert timestamp column to string if it's numeric (Unix timestamp)
    if "timestamp" in df_csv.columns and pd.api.types.is_numeric_dtype(df_csv["timestamp"]):
        logger.debug("Converting numeric timestamps to datetime strings")
        try:
            # Try Unix timestamp conversion (seconds)
            df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], unit='s').dt.strftime('%Y-%m-%d %H:%M')
        except:
            try:
                # Try Unix timestamp conversion (milliseconds)
                df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], unit='ms').dt.strftime('%Y-%m-%d %H:%M')
            except Exception as e:
                logger.warning(f"Could not convert timestamps: {e}")
                return []

    segments = []
    
    # Check if annotations exist
    if "annotations" not in json_entry or not json_entry["annotations"]:
        logger.warning(f"No annotations found for {guid_name}")
        return []
    
    annotation = json_entry["annotations"][0]

    if not annotation["result"]:
        logger.debug(f"No labeled intervals in this entry for {guid_name}")

    for idx, result in enumerate(annotation["result"]):
        val = result["value"]

        start = val["start"]
        end = val["end"]
        label = val["timeserieslabels"][0]

        # Check if timestamp column exists for filtering
        if "timestamp" in df_csv.columns:
            seg_df = df_csv[(df_csv["timestamp"] >= start) & (df_csv["timestamp"] <= end)].copy()
        else:
            logger.warning(f"No timestamp column found in {guid_name}, skipping segment")
            continue

        seg_df.reset_index(drop=True, inplace=True)
        logger.debug(f"Segment #{idx} label={label}, start={start}, end={end}, length={len(seg_df)}")

        segments.append({
            "csv_file": clean_name,
            "start": start,
            "end": end,
            "label": label,
            "length": len(seg_df),
            "segment_data": seg_df
        })

    logger.debug(f"Total segments from {guid_name}: {len(segments)}")
    return segments


def process_segment(g: pd.DataFrame, sequence_length: int = 24, window: int = 5, eps: float = 1e-9) -> pd.DataFrame:
    """
    Process a single segment: normalize, resample, and engineer features.
    
    Args:
        g: Segment dataframe
        sequence_length: Target sequence length (default: 24)
        window: Rolling window size (default: 5)
        eps: Small constant to avoid division by zero
        
    Returns:
        Processed segment dataframe with normalized features
    """
    g = g.sort_values("seq_index").reset_index(drop=True)

    # Identify numeric columns (OHLC)
    num_cols = [c for c in ["open", "high", "low", "close"] if c in g.columns]
    
    # Raw spread for compression ratio (per-step)
    raw_high = g["high"].astype(float) if "high" in g.columns else pd.Series(np.zeros(len(g)))
    raw_low = g["low"].astype(float) if "low" in g.columns else pd.Series(np.zeros(len(g)))
    raw_spread = (raw_high - raw_low).to_numpy()
    start_spread = float(raw_spread[0]) if len(raw_spread) else 0.0
    comp_series = (raw_spread / max(abs(start_spread), eps)) if len(raw_spread) else np.array([], dtype=float)

    # Normalize OHLC per segment on raw values
    norm = {}
    for c in num_cols:
        norm[f"{c}_norm"] = minmax_norm(g[c].astype(float))
    norm_df = pd.DataFrame(norm)

    # Resample normalized columns and compression ratio to sequence_length steps
    L = len(g)
    orig_x = np.arange(L, dtype=float) if L > 1 else np.array([0.0])
    tgt_x = np.linspace(0.0, max(0.0, (L - 1.0)), sequence_length)

    resampled = {}
    for col in norm_df.columns:
        y = norm_df[col].astype(float).to_numpy()
        resampled[col] = interpolate_series(orig_x, y, tgt_x)

    comp_resampled = interpolate_series(orig_x, comp_series, tgt_x)

    out = pd.DataFrame(resampled)

    # Per-datapoint engineered features on resampled normalized series
    close_n = out["close_norm"] if "close_norm" in out.columns else pd.Series(np.zeros(sequence_length))
    high_n = out["high_norm"] if "high_norm" in out.columns else pd.Series(np.zeros(sequence_length))
    low_n = out["low_norm"] if "low_norm" in out.columns else pd.Series(np.zeros(sequence_length))
    open_n = out["open_norm"] if "open_norm" in out.columns else pd.Series(np.zeros(sequence_length))

    # Rolling volatility features (per point)
    vol_close = pd.Series(close_n).rolling(window, min_periods=1).std().fillna(0.0).to_numpy()
    spread_n = (pd.Series(high_n) - pd.Series(low_n)).to_numpy()
    vol_high_low = pd.Series(spread_n).rolling(window, min_periods=1).mean().fillna(0.0).to_numpy()

    # Per-step compression ratio (from raw spread, resampled) with clipping
    compression_ratio = np.clip(comp_resampled, config.COMPRESSION_RATIO_MIN, config.COMPRESSION_RATIO_MAX)

    # Per-step trend
    trend = (pd.Series(open_n) - pd.Series(close_n)).to_numpy()

    # Assemble output with metadata
    out.insert(0, "segment_id", int(g["segment_id"].iloc[0]))
    out.insert(1, "label", g["label"].iloc[0])
    out.insert(2, "csv_file", g["csv_file"].iloc[0])
    out["seq_pos"] = np.arange(1, sequence_length + 1, dtype=int)
    out["vol_close"] = vol_close
    out["vol_high_low"] = vol_high_low
    out["compression_ratio"] = compression_ratio
    out["trend"] = trend
    
    return out


def main():
    """Main preprocessing pipeline."""
    # Setup logger
    logger = setup_logger(__name__, config.LOG_FILE)
    
    logger.info("=" * 80)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Create output directory
    ensure_dir(config.OUTPUT_DIR_01)
    ensure_dir(config.EXPORT_DIR)
    
    # Find all JSON annotation files
    json_files = glob(os.path.join(config.DATA_DIR, "**/*.json"), recursive=True)
    logger.info(f"Found {len(json_files)} JSON annotation files")
    
    # Extract all segments from annotations
    all_segments = []
    
    for i, json_path in enumerate(json_files):
        logger.info(f"Processing JSON {i+1}/{len(json_files)}: {json_path}")
        
        folder = os.path.dirname(json_path)
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            logger.info(f"  Loaded {len(entries)} annotation entries")
        except Exception as e:
            logger.error(f"  Error reading JSON: {e}")
            continue
        
        before_file = len(all_segments)
        for j, entry in enumerate(entries):
            segs = extract_segments(entry, folder, logger)
            all_segments.extend(segs)
        after_file = len(all_segments)
        logger.info(f"  Added {after_file - before_file} segments, total: {after_file}")
    
    logger.info(f"\nTotal labeled segments extracted: {len(all_segments)}")
    
    # Create metadata dataframe
    df_meta = pd.DataFrame([{
        "csv_file": seg["csv_file"],
        "label": seg["label"],
        "start": seg["start"],
        "end": seg["end"],
        "length": seg["length"]
    } for seg in all_segments])
    
    logger.info(f"\nLabel distribution:")
    label_counts = df_meta["label"].value_counts()
    for label, count in label_counts.items():
        logger.info(f"  {label}: {count}")
    
    # Save metadata
    df_meta.to_csv(config.SEGMENTS_META_CSV, index=False)
    logger.info(f"\nSaved metadata: {config.SEGMENTS_META_CSV}")
    
    # Create combined values CSV with sequential index
    logger.info("\nCreating combined segments CSV...")
    rows = []
    for i, seg in enumerate(all_segments):
        seg_df = seg["segment_data"].copy()
        seq_index = np.arange(len(seg_df), dtype=np.int64)
        
        # Drop timestamp if present
        if "timestamp" in seg_df.columns:
            seg_df = seg_df.drop(columns=["timestamp"])
        
        # Coerce to numeric
        seg_df = seg_df.apply(pd.to_numeric, errors="coerce")
        
        # Attach metadata and seq index
        seg_df.insert(0, "seq_index", seq_index)
        seg_df.insert(0, "csv_file", seg["csv_file"])
        seg_df.insert(0, "label", seg["label"])
        seg_df.insert(0, "segment_id", i)
        rows.append(seg_df)
    
    if rows:
        combined_df = pd.concat(rows, axis=0, ignore_index=True)
        combined_df = combined_df.sort_values(["segment_id", "seq_index"], kind="mergesort").reset_index(drop=True)
        combined_df.to_csv(config.SEGMENTS_VALUES_CSV, index=False)
        logger.info(f"Saved combined segments: {config.SEGMENTS_VALUES_CSV}")
    
    # Save to HDF5 format
    logger.info("\nSaving segments to HDF5...")
    with h5py.File(config.SEGMENTS_HDF5, "w") as h5:
        grp = h5.create_group("segments")
        for i, seg in enumerate(all_segments):
            sgrp = grp.create_group(str(i))
            sgrp.attrs["label"] = seg["label"]
            sgrp.attrs["csv_file"] = seg["csv_file"]
            sgrp.attrs["start"] = seg["start"]
            sgrp.attrs["end"] = seg["end"]
            sgrp.attrs["length"] = seg["length"]
            
            seg_df = seg["segment_data"].copy()
            cols = seg_df.columns.tolist()
            
            if "timestamp" in seg_df.columns:
                seq_index = np.arange(len(seg_df), dtype=np.int64)
                sgrp.create_dataset("seq_index", data=seq_index)
                seg_df = seg_df.drop(columns=["timestamp"])
                cols = seg_df.columns.tolist()
            
            sgrp.create_dataset("columns", data=[c.encode("utf-8") for c in cols])
            num_df = seg_df.apply(pd.to_numeric, errors="coerce")
            values = num_df.to_numpy(dtype=np.float64)
            sgrp.create_dataset("values", data=values)
    
    logger.info(f"Saved HDF5 segments: {config.SEGMENTS_HDF5}")
    
    # Load combined values for preprocessing
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN/TEST SPLIT AND PREPROCESSING")
    logger.info("=" * 80)
    
    df_values = pd.read_csv(config.SEGMENTS_VALUES_CSV)
    df_values.columns = df_values.columns.str.lower()
    
    logger.info(f"Loaded segments: {df_values.shape}")
    
    # Create train/test split at segment level
    from sklearn.model_selection import train_test_split
    
    unique_segments = df_values['segment_id'].unique()
    logger.info(f"Total unique segments: {len(unique_segments)}")
    
    # Get labels for stratification
    segment_labels = df_values.groupby('segment_id')['label'].first()
    
    train_seg_ids, test_seg_ids = train_test_split(
        unique_segments,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=segment_labels
    )
    
    logger.info(f"Train segments: {len(train_seg_ids)} ({len(train_seg_ids)/len(unique_segments)*100:.1f}%)")
    logger.info(f"Test segments: {len(test_seg_ids)} ({len(test_seg_ids)/len(unique_segments)*100:.1f}%)")
    
    # Split the data
    df_values_train = df_values[df_values['segment_id'].isin(train_seg_ids)].copy()
    df_values_test = df_values[df_values['segment_id'].isin(test_seg_ids)].copy()
    
    logger.info(f"\nTrain data shape: {df_values_train.shape}")
    logger.info(f"Test data shape: {df_values_test.shape}")
    
    # Log label distributions
    logger.info("\nTrain label distribution:")
    train_label_dist = df_values_train.groupby('segment_id')['label'].first().value_counts()
    for label, count in train_label_dist.items():
        logger.info(f"  {label}: {count}")
    
    logger.info("\nTest label distribution:")
    test_label_dist = df_values_test.groupby('segment_id')['label'].first().value_counts()
    for label, count in test_label_dist.items():
        logger.info(f"  {label}: {count}")
    
    # Save test set raw data for baseline model evaluation
    df_values_test.to_csv(config.SEGMENTS_TEST_RAW_CSV, index=False)
    logger.info(f"\nSaved test raw data: {config.SEGMENTS_TEST_RAW_CSV}")
    
    # Process segments (normalize and engineer features)
    logger.info("\nProcessing training segments...")
    groups_train = df_values_train.groupby("segment_id", sort=True)
    processed_list_train = [
        process_segment(g.copy(), config.SEQUENCE_LENGTH, config.WINDOW_SIZE, config.EPS)
        for _, g in groups_train
    ]
    df_proc_train = pd.concat(processed_list_train, axis=0, ignore_index=True)
    
    logger.info("Processing test segments...")
    groups_test = df_values_test.groupby("segment_id", sort=True)
    processed_list_test = [
        process_segment(g.copy(), config.SEQUENCE_LENGTH, config.WINDOW_SIZE, config.EPS)
        for _, g in groups_test
    ]
    df_proc_test = pd.concat(processed_list_test, axis=0, ignore_index=True)
    
    # Keep only relevant columns
    num_cols = [c for c in ["open", "high", "low", "close"] if c in df_values_train.columns]
    keep_cols = (
        ["segment_id", "label", "csv_file", "seq_pos"] +
        [f"{c}_norm" for c in num_cols] +
        ["vol_close", "vol_high_low", "compression_ratio", "trend"]
    )
    keep_cols = [c for c in keep_cols if c in df_proc_train.columns]
    
    df_proc_train = df_proc_train[keep_cols]
    df_proc_test = df_proc_test[keep_cols]
    
    logger.info(f"\nProcessed train shape: {df_proc_train.shape}")
    logger.info(f"Processed test shape: {df_proc_test.shape}")
    logger.info(f"Features: {[c for c in keep_cols if c not in ['segment_id', 'label', 'csv_file', 'seq_pos']]}")
    
    # Save preprocessed datasets
    df_proc_train_sorted = df_proc_train.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)
    df_proc_train_sorted.to_csv(config.SEGMENTS_PREPROC_CSV, index=False)
    logger.info(f"\nSaved training data: {config.SEGMENTS_PREPROC_CSV}")
    logger.info(f"  Rows: {len(df_proc_train_sorted)}, Segments: {df_proc_train_sorted['segment_id'].nunique()}")
    
    df_proc_test_sorted = df_proc_test.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)
    df_proc_test_sorted.to_csv(config.SEGMENTS_PREPROC_TEST_CSV, index=False)
    logger.info(f"\nSaved test data: {config.SEGMENTS_PREPROC_TEST_CSV}")
    logger.info(f"  Rows: {len(df_proc_test_sorted)}, Segments: {df_proc_test_sorted['segment_id'].nunique()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
