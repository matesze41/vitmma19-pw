"""
Baseline model for flag pattern classification.

This module implements a simple heuristic-based classifier that predicts
flag pattern types (Bullish/Bearish Pennant, Normal, Wedge) based on
the slope of the consolidation phase moving average.

Usage:
    from baseline_model import BaselineModel
    
    model = BaselineModel()
    prediction = model.predict(df)  # df is a pandas DataFrame with OHLC columns
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any


class BaselineModel:
    """
    Baseline classifier for flag patterns using slope-based heuristics.
    
    The model:
    1. Determines if the pattern is bullish or bearish based on open/close prices
    2. Extracts the consolidation phase (after peak for bullish, after trough for bearish)
    3. Computes a moving average on the consolidation phase
    4. Calculates the slope of the MA trendline
    5. Classifies the pattern based on slope thresholds
    
    Classification rules:
        Bullish:
            slope > +threshold  → Bullish Pennant
            slope < -threshold  → Bullish Wedge
            otherwise           → Bullish Normal
        
        Bearish:
            slope < -threshold  → Bearish Pennant
            slope > +threshold  → Bearish Wedge
            otherwise           → Bearish Normal
    """
    
    def __init__(self, slope_threshold: float = 0.0002, ma_window: int = 3):
        """
        Initialize the baseline model.
        
        Args:
            slope_threshold: Threshold for classifying pennant/wedge vs normal patterns.
            ma_window: Window size for the moving average calculation.
        """
        self.slope_threshold = slope_threshold
        self.ma_window = ma_window
        self.classes = [
            "Bearish Normal", "Bearish Pennant", "Bearish Wedge",
            "Bullish Normal", "Bullish Pennant", "Bullish Wedge"
        ]
    
    def _linear_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the slope of a linear regression line.
        
        Args:
            x: Independent variable (typically time indices).
            y: Dependent variable (typically prices).
            
        Returns:
            Slope of the fitted line, or 0.0 if insufficient data.
        """
        if len(x) < 2:
            return 0.0
        # Remove NaN values
        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 2:
            return 0.0
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        return float(slope)
    
    def _extract_consolidation(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        Extract the consolidation phase from an OHLC segment.
        
        For bullish patterns: consolidation starts after the local maximum (High)
        For bearish patterns: consolidation starts after the local minimum (Low)
        
        Args:
            df: DataFrame with OHLC columns.
            
        Returns:
            Tuple of (is_bullish, consolidation_df)
        """
        open_price = df['Open'].iloc[0]
        close_price = df['Close'].iloc[-1]
        is_bullish = close_price > open_price
        
        if is_bullish:
            peak_idx = df['High'].idxmax()
            start_idx = df.index.get_loc(peak_idx)
        else:
            trough_idx = df['Low'].idxmin()
            start_idx = df.index.get_loc(trough_idx)
        
        # If too short, use the entire segment
        if start_idx + 2 >= len(df):
            return is_bullish, df.copy()
        
        consolidation = df.iloc[start_idx:].copy()
        return is_bullish, consolidation
    
    def _classify_pattern(self, is_bullish: bool, slope: float) -> str:
        """
        Classify the pattern type based on direction and slope.
        
        Args:
            is_bullish: Whether the overall pattern is bullish.
            slope: Slope of the consolidation MA trendline.
            
        Returns:
            Pattern label string.
        """
        if is_bullish:
            if slope > self.slope_threshold:
                return "Bullish Pennant"
            elif slope < -self.slope_threshold:
                return "Bullish Wedge"
            else:
                return "Bullish Normal"
        else:
            if slope < -self.slope_threshold:
                return "Bearish Pennant"
            elif slope > self.slope_threshold:
                return "Bearish Wedge"
            else:
                return "Bearish Normal"
    
    def predict(self, df: pd.DataFrame) -> str:
        """
        Predict the flag pattern type for a single OHLC segment.
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close'].
                The index can be datetime or integer.
                
        Returns:
            Predicted label string (e.g., "Bullish Pennant", "Bearish Normal", etc.)
            
        Raises:
            ValueError: If required columns are missing or data is too short.
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < 3:
            raise ValueError(f"Segment too short: {len(df)} rows (minimum 3 required)")
        
        # Extract consolidation phase
        is_bullish, consolidation = self._extract_consolidation(df)
        
        # Compute moving average on consolidation
        consolidation = consolidation.copy()
        consolidation['MA'] = consolidation['Close'].rolling(
            window=self.ma_window, min_periods=1
        ).mean()
        
        # Calculate slope
        x = np.arange(len(consolidation))
        y = consolidation['MA'].values
        slope = self._linear_slope(x, y)
        
        # Classify
        label = self._classify_pattern(is_bullish, slope)
        return label
    
    def predict_with_details(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict with additional diagnostic information.
        
        Args:
            df: DataFrame with OHLC columns.
            
        Returns:
            Dictionary with keys:
                - 'prediction': The predicted label
                - 'is_bullish': Whether the pattern is bullish
                - 'slope': The calculated slope value
                - 'consolidation_length': Number of bars in consolidation phase
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < 3:
            raise ValueError(f"Segment too short: {len(df)} rows (minimum 3 required)")
        
        is_bullish, consolidation = self._extract_consolidation(df)
        
        consolidation = consolidation.copy()
        consolidation['MA'] = consolidation['Close'].rolling(
            window=self.ma_window, min_periods=1
        ).mean()
        
        x = np.arange(len(consolidation))
        y = consolidation['MA'].values
        slope = self._linear_slope(x, y)
        
        label = self._classify_pattern(is_bullish, slope)
        
        return {
            'prediction': label,
            'is_bullish': is_bullish,
            'slope': slope,
            'consolidation_length': len(consolidation),
            'main_type': 'Bullish' if is_bullish else 'Bearish'
        }
    
    def predict_batch(self, segments: List[pd.DataFrame]) -> List[str]:
        """
        Predict labels for multiple segments.
        
        Args:
            segments: List of DataFrames, each with OHLC columns.
            
        Returns:
            List of predicted labels.
        """
        return [self.predict(seg) for seg in segments]


def load_ohlc_csv(csv_path: str) -> pd.DataFrame:
    """
    Load an OHLC CSV file into a DataFrame.
    
    Automatically detects the datetime column and sets it as index.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        DataFrame with datetime index and OHLC columns.
    """
    df = pd.read_csv(csv_path)
    
    # Try to find the datetime column
    datetime_cols = ["Date", "date", "timestamp", "Timestamp", "Datetime", "datetime"]
    dt_col = None
    for col in datetime_cols:
        if col in df.columns:
            dt_col = col
            break
    
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col)
    
    return df


# Convenience function for direct prediction from CSV
def predict_from_csv(csv_path: str, slope_threshold: float = 0.0002) -> str:
    """
    Load a CSV file and predict the flag pattern type.
    
    Args:
        csv_path: Path to the OHLC CSV file.
        slope_threshold: Threshold for slope-based classification.
        
    Returns:
        Predicted label string.
    """
    df = load_ohlc_csv(csv_path)
    model = BaselineModel(slope_threshold=slope_threshold)
    return model.predict(df)


def predict_from_segments_csv(
    csv_path: str,
    slope_threshold: float = 0.0002,
    segment_id_col: str = "segment_id",
    label_col: str = "label"
) -> pd.DataFrame:
    """
    Load a combined segments CSV (like segments_values.csv from 01-data-exploration)
    and predict for all segments.
    
    The CSV is expected to have columns:
        - segment_id: integer ID grouping rows into segments
        - label: ground truth label (optional, for comparison)
        - Open, High, Low, Close: OHLC price columns
        
    Args:
        csv_path: Path to the combined segments CSV file.
        slope_threshold: Threshold for slope-based classification.
        segment_id_col: Column name for segment IDs.
        label_col: Column name for ground truth labels.
        
    Returns:
        DataFrame with columns: segment_id, gold_label, predicted_label, is_correct
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    
    if segment_id_col not in df.columns:
        raise ValueError(f"Missing segment ID column: {segment_id_col}")
    
    model = BaselineModel(slope_threshold=slope_threshold)
    results = []
    
    for seg_id, seg_df in df.groupby(segment_id_col, sort=True):
        # Sort by seq_index if available
        if 'seq_index' in seg_df.columns:
            seg_df = seg_df.sort_values('seq_index')
        
        # Extract OHLC columns only
        ohlc_df = seg_df[required_cols].reset_index(drop=True)
        
        # Get ground truth label if available
        gold_label = None
        if label_col in seg_df.columns:
            gold_label = seg_df[label_col].iloc[0]
        
        # Predict
        try:
            prediction = model.predict(ohlc_df)
        except ValueError as e:
            print(f"Warning: Segment {seg_id} skipped - {e}")
            continue
        
        results.append({
            'segment_id': seg_id,
            'gold_label': gold_label,
            'predicted_label': prediction,
            'is_correct': gold_label == prediction if gold_label else None
        })
    
    return pd.DataFrame(results)


def evaluate_on_segments_csv(csv_path: str, slope_threshold: float = 0.0002) -> Dict[str, Any]:
    """
    Evaluate the baseline model on a segments CSV and return metrics.
    
    Args:
        csv_path: Path to the combined segments CSV file.
        slope_threshold: Threshold for slope-based classification.
        
    Returns:
        Dictionary with accuracy, per-class metrics, and the predictions DataFrame.
    """
    results_df = predict_from_segments_csv(csv_path, slope_threshold)
    
    # Filter to rows with ground truth
    mask = results_df['gold_label'].notna()
    y_true = results_df.loc[mask, 'gold_label']
    y_pred = results_df.loc[mask, 'predicted_label']
    
    if len(y_true) == 0:
        return {
            'accuracy': None,
            'predictions': results_df,
            'message': 'No ground truth labels available'
        }
    
    accuracy = (y_true == y_pred).mean()
    
    # Per-class accuracy
    per_class = {}
    for label in y_true.unique():
        mask_label = y_true == label
        per_class[label] = (y_true[mask_label] == y_pred[mask_label]).mean()
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class,
        'num_segments': len(results_df),
        'num_correct': int(results_df['is_correct'].sum()),
        'predictions': results_df
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        try:
            result = predict_from_csv(csv_path)
            print(f"Prediction: {result}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python baseline_model.py <csv_path>")
        print("\nOr use as a module:")
        print("  from baseline_model import BaselineModel")
        print("  model = BaselineModel()")
        print("  prediction = model.predict(df)")
