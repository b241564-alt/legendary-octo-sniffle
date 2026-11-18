"""
Data loader and dataset validation module for AQI dataset handling.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_aqi_data.csv"


def load_csv(file_or_buffer: Any) -> pd.DataFrame:
    """
    Loads a CSV file from a file path, file-like object, or Streamlit UploadedFile.
    Tries multiple encodings to handle diverse user datasets gracefully.
    """
    if file_or_buffer is None:
        raise ValueError("No file provided for loading.")

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None

    for enc in encodings:
        try:
            if hasattr(file_or_buffer, "seek"):
                file_or_buffer.seek(0)
            df = pd.read_csv(file_or_buffer, encoding=enc)
            # Strip whitespace from column names
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Failed to parse CSV file with common encodings. Error: {last_err}")


def load_sample_data() -> pd.DataFrame:
    """Loads the bundled sample AQI monitoring dataset."""
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found at {SAMPLE_DATA_PATH}")
    return load_csv(str(SAMPLE_DATA_PATH))


def validate_dataset(df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    """
    Validates the uploaded dataset.
    Returns (is_valid, reason_message).
    """
    if df is None:
        return False, "Dataset is null or could not be loaded."
    if not isinstance(df, pd.DataFrame):
        return False, "Input data is not a valid pandas DataFrame."
    if df.empty:
        return False, "Dataset is completely empty (0 rows)."
    if df.shape[0] < 2:
        return False, "Dataset contains fewer than 2 rows; cannot perform imputation analysis."
    if df.shape[1] < 1:
        return False, "Dataset contains 0 columns."
    return True, "Dataset is valid."


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorizes columns into numerical, datetime-candidate, and categorical/text.
    """
    numerical_cols = []
    datetime_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            numerical_cols.append(col)
            continue

        # Check if it could be a timestamp / date string
        # Sample non-null values
        non_null = series.dropna()
        if len(non_null) > 0 and (
            "date" in col.lower()
            or "time" in col.lower()
            or "timestamp" in col.lower()
            or "hour" in col.lower()
            or "day" in col.lower()
        ):
            try:
                sample = non_null.head(15)
                pd.to_datetime(sample, errors="raise")
                datetime_cols.append(col)
                continue
            except Exception:
                pass

        # Try heuristic datetime parse for first 10 rows if string
        if len(non_null) > 0 and series.dtype == "object":
            try:
                sample = non_null.head(10)
                # Avoid parsing plain numbers or very short strings
                if sample.astype(str).str.len().mean() >= 6:
                    pd.to_datetime(sample, errors="raise")
                    datetime_cols.append(col)
                    continue
            except Exception:
                pass

        categorical_cols.append(col)

    return {
        "numerical": numerical_cols,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
        "all": list(df.columns)
    }


def parse_datetime_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Safely converts a specified column to datetime.
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    return pd.to_datetime(df[column_name], errors="coerce")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes high-level dataset statistics."""
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    col_types = detect_column_types(df)

    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_percentage": round(missing_pct, 2),
        "duplicate_rows": duplicate_rows,
        "numerical_count": len(col_types["numerical"]),
        "datetime_count": len(col_types["datetime"]),
        "categorical_count": len(col_types["categorical"]),
        "column_types": col_types,
    }
