# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
from utils import setup_logger
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import pickle
import h5py
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd

logger = setup_logger()
BASE_DATA_DIR = os.path.abspath("../data")
EXPORT_DIR = os.path.join(BASE_DATA_DIR, "export")
VALUES_CSV = os.path.join(EXPORT_DIR, "segments_values.csv")
META_CSV = os.path.join(EXPORT_DIR, "segments_meta.csv")
TEST_RAW_CSV = os.path.join(EXPORT_DIR, "segments_test_raw.csv")
PREPROC_TEST_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24_test.csv")
WINDOW = 5  # rolling window for per-step volatility features
EPS = 1e-9

def minmax_norm(s: pd.Series) -> pd.Series:
    vmin = s.min()
    vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - vmin) / (vmax - vmin)

def _interp_series(orig_x: np.ndarray, y: np.ndarray, tgt_x: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(len(tgt_x), dtype=float)
    if len(y) == 1:
        return np.full(len(tgt_x), float(y[0]))
    return np.interp(tgt_x, orig_x, y)

def process_segment(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("seq_index").reset_index(drop=True)
    
    # Determine numeric columns available
    num_cols = [c for c in ["open","high","low","close"] if c in g.columns]

    # Raw spread for compression ratio (per-step)
    raw_high = g["high"].astype(float) if "high" in g.columns else pd.Series(np.zeros(len(g)))
    raw_low  = g["low"].astype(float)  if "low" in g.columns  else pd.Series(np.zeros(len(g)))
    raw_spread = (raw_high - raw_low).to_numpy()
    start_spread = float(raw_spread[0]) if len(raw_spread) else 0.0
    comp_series = (raw_spread / max(abs(start_spread), EPS)) if len(raw_spread) else np.array([], dtype=float)

    # Normalize OHLC per segment on raw values
    norm = {}
    for c in num_cols:
        norm[f"{c}_norm"] = minmax_norm(g[c].astype(float))
    norm_df = pd.DataFrame(norm)

    # Resample normalized columns and compression ratio to 24 steps
    L = len(g)
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
    high_n  = out["high_norm"]  if "high_norm"  in out.columns else pd.Series(np.zeros(24))
    low_n   = out["low_norm"]   if "low_norm"   in out.columns else pd.Series(np.zeros(24))
    open_n  = out["open_norm"]  if "open_norm"  in out.columns else pd.Series(np.zeros(24))

    # Rolling volatility features (per point)
    vol_close = pd.Series(close_n).rolling(WINDOW, min_periods=1).std().fillna(0.0).to_numpy()
    spread_n = (pd.Series(high_n) - pd.Series(low_n)).to_numpy()
    vol_high_low = pd.Series(spread_n).rolling(WINDOW, min_periods=1).mean().fillna(0.0).to_numpy()

    # Per-step compression ratio (from raw spread, resampled) with clipping to [0.2, 3.0]
    compression_ratio = np.clip(comp_resampled, 0.2, 3.0)

    # Per-step trend
    trend = (pd.Series(open_n) - pd.Series(close_n)).to_numpy()

    # Assemble output with metadata
    out.insert(0, "segment_id", int(g["segment_id"].iloc[0]))
    out.insert(1, "label", g["label"].iloc[0])
    out.insert(2, "csv_file", g["csv_file"].iloc[0])
    out["seq_pos"] = np.arange(1, 25, dtype=int)  # 1..24
    out["vol_close"] = vol_close
    out["vol_high_low"] = vol_high_low
    out["compression_ratio"] = compression_ratio
    out["trend"] = trend
    return out

# Combined CSV helper function
def build_combined_csv(segments, segment_indices, split_name=""):
    """Build combined CSV from segments with given indices"""
    rows = []
    for new_id, orig_idx in enumerate(segment_indices):
        seg = segments[orig_idx]
        seg_df = seg["segment_data"].copy()
        # Build sequential index regardless of timestamp presence
        seq_index = np.arange(len(seg_df), dtype=np.int64)
        # Drop timestamp if present
        if "timestamp" in seg_df.columns:
            seg_df = seg_df.drop(columns=["timestamp"])  # numeric columns remain
        # Coerce to numeric
        seg_df = seg_df.apply(pd.to_numeric, errors="coerce")
        # Attach metadata and seq index
        seg_df.insert(0, "seq_index", seq_index)
        seg_df.insert(0, "csv_file", seg["csv_file"])
        seg_df.insert(0, "label", seg["label"])
        seg_df.insert(0, "segment_id", new_id)
        seg_df.insert(0, "original_id", orig_idx)
        rows.append(seg_df)
    
    if rows:
        combined = pd.concat(rows, axis=0, ignore_index=True)
        combined = combined.sort_values(["segment_id", "seq_index"], kind="mergesort").reset_index(drop=True)
        return combined
    return None

def extract_segments(json_entry, csv_directory):
    """
    Extracts all labeled time segments from the annotation entry.
    Loads the corresponding CSV and slices the time range.
    Returns a list of segment dicts.
    """

    # Check if file_upload key exists
    if "file_upload" not in json_entry:
        print("[extract_segments] WARNING: Entry missing 'file_upload' key, skipping")
        return []
    
    guid_name = json_entry["file_upload"]  # e.g., e2ab0dd4-GSPC_...
    
    # Handle None or empty file_upload
    if guid_name is None or guid_name == "":
        print("[extract_segments] WARNING: file_upload is None or empty, skipping")
        return []
    
    clean_name = strip_guid(guid_name)
    print(f"[extract_segments] file_upload={guid_name}, clean_name={clean_name}")

    # Try loading the CSV from several likely locations:
    #  - the JSON's folder (csv_directory)
    #  - the parent of that folder (handles e.g. J2QIYD/Labels vs J2QIYD/)
    #  - the global BASE_DATA_DIR
    candidate_dirs = [
        csv_directory,
        os.path.dirname(csv_directory),
        BASE_DATA_DIR,
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
        print("[extract_segments] WARNING: Cannot find CSV for annotation:", guid_name)
        return []

    print(f"[extract_segments] Using CSV: {full_csv_path}")

    try:
        df_csv = pd.read_csv(full_csv_path)
    except Exception as e:
        print("[extract_segments] ERROR reading CSV:", full_csv_path, e)
        return []

    # Normalize column names to lowercase for consistent access
    df_csv.columns = df_csv.columns.str.lower()
    
    # Normalize OHLC column names so we always have 'open', 'high', 'low', 'close'
    print(f"[extract_segments] CSV loaded with {len(df_csv)} rows and columns {list(df_csv.columns)}")

    # Convert timestamp column to string if it's numeric (Unix timestamp)
    if "timestamp" in df_csv.columns and pd.api.types.is_numeric_dtype(df_csv["timestamp"]):
        print(f"[extract_segments] Converting numeric timestamps to datetime strings")
        try:
            # Try Unix timestamp conversion (seconds)
            df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], unit='s').dt.strftime('%Y-%m-%d %H:%M')
        except:
            try:
                # Try Unix timestamp conversion (milliseconds)
                df_csv["timestamp"] = pd.to_datetime(df_csv["timestamp"], unit='ms').dt.strftime('%Y-%m-%d %H:%M')
            except Exception as e:
                print(f"[extract_segments] WARNING: Could not convert timestamps: {e}")
                return []

    segments = []
    
    # Check if annotations exist
    if "annotations" not in json_entry or not json_entry["annotations"]:
        print(f"[extract_segments] WARNING: No annotations found for {guid_name}")
        return []
    
    annotation = json_entry["annotations"][0]

    if not annotation["result"]:
        print(f"[extract_segments] No labeled intervals in this entry for {guid_name}")

    for idx, result in enumerate(annotation["result"]):
        val = result["value"]

        start = val["start"]
        end = val["end"]
        label = val["timeserieslabels"][0]

        # Check if timestamp column exists for filtering
        if "timestamp" in df_csv.columns:
            seg_df = df_csv[(df_csv["timestamp"] >= start) & (df_csv["timestamp"] <= end)].copy()
        else:
            print(f"[extract_segments] WARNING: No timestamp column found in {guid_name}, skipping segment")
            continue

        seg_df.reset_index(drop=True, inplace=True)
        print(f"[extract_segments]  segment #{idx} label={label}, start={start}, end={end}, length={len(seg_df)}")

        segments.append({
            "csv_file": clean_name,
            "start": start,
            "end": end,
            "label": label,
            "length": len(seg_df),
            "segment_data": seg_df
        })

    print(f"[extract_segments] Total segments from {guid_name}: {len(segments)}")
    return segments

def strip_guid(filename):
        """
        Converts 'e2ab0dd4-FILENAME.csv' → 'FILENAME.csv'.
        """
        base = os.path.basename(filename)
        parts = base.split("-", 1)
        if len(parts) == 2:
            print("Stripped GUID from", filename, "to", parts[1])
            return parts[1]
        print("Warning: could not strip GUID from", filename)
        return base

def preprocess():
    plt.style.use("ggplot")

    print("Using base data directory:", BASE_DATA_DIR)

    # %%
    json_files = glob(os.path.join(BASE_DATA_DIR, "**/*.json"), recursive=True)

    print("Found JSON files:")
    for f in json_files:
        print("  ", f)

    print("\nTotal JSON files:", len(json_files))
    print("=" * 80)
    print("JSON FILE DIAGNOSTIC REPORT")
    print("=" * 80)

    json_files = glob(os.path.join(BASE_DATA_DIR, "**/*.json"), recursive=True)

    for json_path in json_files:
        folder = os.path.dirname(json_path)
        folder_name = os.path.basename(folder)
        
        print(f"\n📁 Folder: {folder_name}")
        print(f"   JSON: {os.path.basename(json_path)}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            
            # Count entries with actual annotations
            entries_with_labels = 0
            total_segments = 0
            missing_csvs = []
            
            for entry in entries:
                file_upload = entry.get("file_upload", "")
                annotations = entry.get("annotations", [])
                
                if annotations and annotations[0].get("result"):
                    entries_with_labels += 1
                    total_segments += len(annotations[0]["result"])
                
                # Check if CSV exists
                clean_name = file_upload.split("-", 1)[1] if "-" in file_upload else file_upload
                candidates = [
                    os.path.join(folder, file_upload),
                    os.path.join(folder, clean_name),
                ]
                if not any(os.path.exists(c) for c in candidates):
                    missing_csvs.append(file_upload)
            
            print(f"   Total entries: {len(entries)}")
            print(f"   ✅ Entries with labeled segments: {entries_with_labels}")
            print(f"   📊 Total labeled segments: {total_segments}")
            
            if missing_csvs:
                print(f"   ⚠️  Missing CSV files: {len(missing_csvs)}")
                for csv in missing_csvs[:3]:  # Show first 3
                    print(f"      - {csv}")
                if len(missing_csvs) > 3:
                    print(f"      ... and {len(missing_csvs) - 3} more")
            else:
                print(f"   ✅ All CSV files found")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    print("\n" + "=" * 80)
    print("AVAILABLE CSV FILES PER DIRECTORY:")
    print("=" * 80)

    for folder_path in [f.path for f in os.scandir(BASE_DATA_DIR) if f.is_dir() and f.name != 'export']:
        folder_name = os.path.basename(folder_path)
        csv_files = glob(os.path.join(folder_path, "*.csv"))
        print(f"\n📁 {folder_name}: {len(csv_files)} CSV files")
        if len(csv_files) <= 5:
            for csv in csv_files:
                print(f"   - {os.path.basename(csv)}")
        else:
            for csv in csv_files[:3]:
                print(f"   - {os.path.basename(csv)}")
            print(f"   ... and {len(csv_files) - 3} more")

    all_segments = []

    print(f"[main] Found {len(json_files)} JSON files to process")

    for i, json_path in enumerate(json_files):
        print(f"\n[main] ({i+1}/{len(json_files)}) Processing JSON:", json_path)

        folder = os.path.dirname(json_path)
        print(f"[main]  CSV search base directory: {folder}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            print(f"[main]  Loaded {len(entries)} annotation entries from JSON")
        except Exception as e:
            print("[main] ERROR reading JSON:", json_path, e)
            continue

        before_file = len(all_segments)
        for j, entry in enumerate(entries):
            print(f"[main]   Entry {j+1}/{len(entries)} for file_upload={entry.get('file_upload')}")
            segs = extract_segments(entry, folder)
            print(f"[main]   -> Extracted {len(segs)} segments from this entry")
            all_segments.extend(segs)
        after_file = len(all_segments)
        print(f"[main] Finished {json_path}: added {after_file - before_file} segments, total so far {after_file}")

    print("\n[main] Total labeled segments loaded:", len(all_segments))
    df_meta = pd.DataFrame([{
        "csv_file": seg["csv_file"],
        "label": seg["label"],
        "start": seg["start"],
        "end": seg["end"],
        "length": seg["length"]
    } for seg in all_segments])

    df_meta["label"].value_counts()
    lengths = df_meta["length"].values

    print("Min length:", lengths.min())
    print("Max length:", lengths.max())
    print("Mean length:", lengths.mean())
    print("Median length:", np.median(lengths))
    sample = all_segments[0]
    seg_df = sample["segment_data"]

    print("seg_df columns:", seg_df.columns)
    print(seg_df.head())

    dataset = []

    for seg in all_segments:
        dataset.append({
            "label": seg["label"],
            "csv_file": seg["csv_file"],
            "data": seg["segment_data"]
        })

    len(dataset)
    print("Total labeled segments:", len(dataset))
    print("Unique labels:", df_meta["label"].unique())
    df_meta.describe()


    # Get segment indices and labels
    segment_indices = np.arange(len(all_segments))
    segment_labels = [seg["label"] for seg in all_segments]

    # Split with stratification to maintain label distribution
    train_indices, test_indices = train_test_split(
        segment_indices,
        test_size=0.10,
        random_state=42,
        stratify=segment_labels
    )

    print("="*80)
    print("STRATIFIED TRAIN/TEST SPLIT")
    print("="*80)
    print(f"Total segments: {len(all_segments)}")
    print(f"Training segments: {len(train_indices)} ({len(train_indices)/len(all_segments)*100:.1f}%)")
    print(f"Test segments: {len(test_indices)} ({len(test_indices)/len(all_segments)*100:.1f}%)")

    # Verify stratification
    train_label_dist = pd.Series([all_segments[i]["label"] for i in train_indices]).value_counts(normalize=True).sort_index()
    test_label_dist = pd.Series([all_segments[i]["label"] for i in test_indices]).value_counts(normalize=True).sort_index()

    print("\nLabel distribution (proportions):")
    comparison_df = pd.DataFrame({
        'Train': train_label_dist,
        'Test': test_label_dist,
        'Overall': df_meta["label"].value_counts(normalize=True).sort_index()
    })
    print(comparison_df)
    print("\n" + "="*80)

    test_labels = [all_segments[i]["label"] for i in test_indices]
    test_label_counts = pd.Series(test_labels).value_counts()

    print("Test set label counts:")
    print(test_label_counts)
    print(f"\nTest set total: {len(test_labels)} segments")

    EXPORT_DIR = os.path.join(BASE_DATA_DIR, "export")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print("Exporting data to:", EXPORT_DIR)

    # Create train and test segment lists
    train_segments = [all_segments[i] for i in train_indices]
    test_segments = [all_segments[i] for i in test_indices]

    # 1) Save metadata summary (overall, train, test)
    meta_csv_path = os.path.join(EXPORT_DIR, "segments_meta.csv")
    df_meta.to_csv(meta_csv_path, index=False)
    print("Saved metadata CSV:", meta_csv_path)

    # Save train metadata
    df_meta_train = df_meta.iloc[train_indices].reset_index(drop=True)
    meta_train_path = os.path.join(EXPORT_DIR, "segments_meta_train.csv")
    df_meta_train.to_csv(meta_train_path, index=False)
    print("Saved train metadata CSV:", meta_train_path)

    # Save test metadata
    df_meta_test = df_meta.iloc[test_indices].reset_index(drop=True)
    meta_test_path = os.path.join(EXPORT_DIR, "segments_meta_test.csv")
    df_meta_test.to_csv(meta_test_path, index=False)
    print("Saved test metadata CSV:", meta_test_path)

    # 2) Save full segments to HDF5 (efficient, reloadable)
    h5_path = os.path.join(EXPORT_DIR, "segments.h5")
    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group("segments")
        for i, seg in enumerate(all_segments):
            sgrp = grp.create_group(str(i))
            # Store attributes
            sgrp.attrs["label"] = seg["label"]
            sgrp.attrs["csv_file"] = seg["csv_file"]
            sgrp.attrs["start"] = seg["start"]
            sgrp.attrs["end"] = seg["end"]
            sgrp.attrs["length"] = seg["length"]
            # Store the segment dataframe with numeric values only
            seg_df = seg["segment_data"].copy()
            cols = seg_df.columns.tolist()
            # If timestamp exists, replace it with a sequential integer index
            if "timestamp" in seg_df.columns:
                seq_index = np.arange(len(seg_df), dtype=np.int64)
                sgrp.create_dataset("seq_index", data=seq_index)
                seg_df = seg_df.drop(columns=["timestamp"])  # drop original timestamp
            
            # Coerce to numeric
            seg_df = seg_df.apply(pd.to_numeric, errors="coerce")
            # Store values
            for col in seg_df.columns:
                sgrp.create_dataset(col, data=seg_df[col].values)

    print("Saved segments to HDF5:", h5_path)

    # Save combined CSV for all segments
    combined_csv_path = os.path.join(EXPORT_DIR, "segments_values.csv")
    combined_df = build_combined_csv(all_segments, segment_indices)
    if combined_df is not None:
        combined_df.to_csv(combined_csv_path, index=False)
        print("Saved combined segments CSV:", combined_csv_path)

    # Save train split
    combined_train_path = os.path.join(EXPORT_DIR, "segments_values_train.csv")
    combined_train_df = build_combined_csv(all_segments, train_indices, "train")
    if combined_train_df is not None:
        combined_train_df.to_csv(combined_train_path, index=False)
        print("Saved train segments CSV:", combined_train_path)

    # Save test split
    combined_test_path = os.path.join(EXPORT_DIR, "segments_values_test.csv")
    combined_test_df = build_combined_csv(all_segments, test_indices, "test")
    if combined_test_df is not None:
        combined_test_df.to_csv(combined_test_path, index=False)
        print("Saved test segments CSV:", combined_test_path)

    print("\nSplit summary:")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Train segments: {len(train_indices)} ({len(train_indices)/len(all_segments)*100:.1f}%)")
    print(f"  Test segments: {len(test_indices)} ({len(test_indices)/len(all_segments)*100:.1f}%)")
    
    print("Data Manipulationm begins here")
    
    print("Export dir:", EXPORT_DIR)
    print("Values CSV exists:", os.path.exists(VALUES_CSV))
    print("Meta CSV exists:", os.path.exists(META_CSV))

    df_values = pd.read_csv(VALUES_CSV)
    df_meta = pd.read_csv(META_CSV)

    # Normalize column names to lowercase for consistent access
    df_values.columns = df_values.columns.str.lower()
    df_meta.columns = df_meta.columns.str.lower()

    print("df_values:", df_values.shape)
    print("df_meta:", df_meta.shape)

    # Get unique segment IDs
    unique_segments = df_values['segment_id'].unique()
    print(f"Total segments: {len(unique_segments)}")

    # Split segment IDs into train and test (stratified by label)
    segment_labels = df_values.groupby('segment_id')['label'].first()
    train_seg_ids, test_seg_ids = train_test_split(
        unique_segments, 
        test_size=0.10, 
        random_state=11,
        stratify=segment_labels
    )

    print(f"Train segments: {len(train_seg_ids)}")
    print(f"Test segments: {len(test_seg_ids)}")

    # Split the data
    df_values_train = df_values[df_values['segment_id'].isin(train_seg_ids)].copy()
    df_values_test = df_values[df_values['segment_id'].isin(test_seg_ids)].copy()

    print(f"\nTrain data shape: {df_values_train.shape}")
    print(f"Test data shape: {df_values_test.shape}")

    # Verify label distribution
    print("\nTrain label distribution:")
    print(df_values_train.groupby('segment_id')['label'].first().value_counts())
    print("\nTest label distribution:")
    print(df_values_test.groupby('segment_id')['label'].first().value_counts())

    df_values_test.to_csv(TEST_RAW_CSV, index=False)
    print(f"\nSaved test raw data: {TEST_RAW_CSV}")

    NUM_COLS = [c for c in ["open","high","low","close"] if c in df_values_train.columns]
    assert set(["segment_id","label","csv_file","seq_index"]).issubset(df_values_train.columns), "Required columns missing in values CSV"
    assert len(NUM_COLS) >= 1, "No OHLC columns found"

    # Process training data
    print("Processing training segments...")
    groups_train = df_values_train.groupby("segment_id", sort=True)
    processed_list_train: List[pd.DataFrame] = [process_segment(g.copy()) for _, g in groups_train]
    df_proc_train = pd.concat(processed_list_train, axis=0, ignore_index=True)

    # Process test data
    print("Processing test segments...")
    groups_test = df_values_test.groupby("segment_id", sort=True)
    processed_list_test: List[pd.DataFrame] = [process_segment(g.copy()) for _, g in groups_test]
    df_proc_test = pd.concat(processed_list_test, axis=0, ignore_index=True)

    # Keep normalized OHLC, engineered features, and ids
    keep_cols = (["segment_id","label","csv_file","seq_pos"] +
                [f"{c}_norm" for c in NUM_COLS] +
                ["vol_close","vol_high_low","compression_ratio","trend"]) 
    # Retain only columns that exist (in case some OHLC missing)
    keep_cols = [c for c in keep_cols if c in df_proc_train.columns]

    df_proc_train = df_proc_train[keep_cols]
    df_proc_test = df_proc_test[keep_cols]

    print(f"\nProcessed train shape: {df_proc_train.shape}")
    print(f"Processed test shape: {df_proc_test.shape}")

    # For backward compatibility, keep df_proc as training data
    df_proc = df_proc_train.copy()
    df_proc.head()

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Save training data
    PREPROC_CSV = os.path.join(EXPORT_DIR, "segments_preproc_24.csv")
    _df_train = df_proc_train.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)
    _df_train.to_csv(PREPROC_CSV, index=False)
    print("Saved training data:", PREPROC_CSV)
    print(f"  Rows: {len(_df_train)}, Segments: {_df_train['segment_id'].nunique()}")

    # Save test data
    _df_test = df_proc_test.sort_values(["segment_id", "seq_pos"], kind="mergesort").reset_index(drop=True)
    _df_test.to_csv(PREPROC_TEST_CSV, index=False)
    print("\nSaved test data:", PREPROC_TEST_CSV)
    print(f"  Rows: {len(_df_test)}, Segments: {_df_test['segment_id'].nunique()}")

    print("\nColumns:", list(_df_train.columns))
    print("\nTraining data sample:")
    print(_df_train.sample(min(5, len(_df_train)), random_state=0))
    print("\nTest data sample:")
    print(_df_test.sample(min(5, len(_df_test)), random_state=0))
    
if __name__ == "__main__":
    preprocess()