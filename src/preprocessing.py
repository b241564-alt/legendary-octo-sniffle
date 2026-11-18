"""
Data preprocessing module.
Handles duplicate detection, duplicate removal, column filtering,
and chronological timestamp sorting without mutating original DataFrames.
"""

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd


def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detects duplicate rows in the DataFrame.
    Returns count, percentage, and duplicate index list.
    """
    total_rows = len(df)
    dup_mask = df.duplicated(subset=subset, keep="first")
    dup_count = int(dup_mask.sum())
    dup_pct = round((dup_count / total_rows * 100) if total_rows > 0 else 0.0, 2)

    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": dup_pct,
        "has_duplicates": dup_count > 0,
        "duplicate_indices": df[dup_mask].index.tolist(),
        "duplicate_sample": df[dup_mask].head(10) if dup_count > 0 else pd.DataFrame()
    }


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first") -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate rows and returns a brand-new DataFrame copy along with the count of rows removed.
    Does NOT mutate original input DataFrame.
    """
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep).copy()
    rows_removed = len(df) - len(cleaned_df)
    return cleaned_df.reset_index(drop=True), rows_removed


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Filters the DataFrame to include only the specified columns.
    Returns a copy.
    """
    if not columns:
        raise ValueError("At least one column must be selected.")
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Selected columns not present in dataset: {missing_cols}")
    return df[columns].copy()


def sort_by_timestamp(
    df: pd.DataFrame,
    timestamp_col: str,
    ascending: bool = True
) -> Tuple[pd.DataFrame, bool]:
    """
    Attempts to parse timestamp_col into datetimes and sort DataFrame chronologically.
    Returns (sorted_df, success_flag).
    """
    if timestamp_col not in df.columns:
        return df.copy(), False

    df_copy = df.copy()
    try:
        parsed_dt = pd.to_datetime(df_copy[timestamp_col], errors="coerce")
        # If at least some are valid datetimes
        if parsed_dt.notna().sum() > 0:
            df_copy["__parsed_ts__"] = parsed_dt
            sorted_df = df_copy.sort_values(by="__parsed_ts__", ascending=ascending)
            sorted_df = sorted_df.drop(columns=["__parsed_ts__"]).reset_index(drop=True)
            return sorted_df, True
    except Exception:
        pass

    return df.copy(), False
