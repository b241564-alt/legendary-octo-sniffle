"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
from src.data_loader import (
    load_sample_data,
    validate_dataset,
    detect_column_types,
    get_dataset_summary,
)


def test_load_sample_data():
    df = load_sample_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50
    assert "PM2.5" in df.columns
    assert "AQI" in df.columns


def test_validate_dataset():
    # Valid dataframe
    df_valid = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    is_valid, msg = validate_dataset(df_valid)
    assert is_valid is True

    # Empty dataframe
    is_valid, msg = validate_dataset(pd.DataFrame())
    assert is_valid is False
    assert "empty" in msg.lower()

    # None
    is_valid, msg = validate_dataset(None)
    assert is_valid is False

    # Fewer than 2 rows
    df_single = pd.DataFrame({"A": [1]})
    is_valid, msg = validate_dataset(df_single)
    assert is_valid is False
    assert "fewer than 2 rows" in msg.lower()


def test_detect_column_types():
    df = pd.DataFrame({
        "Timestamp": ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
        "PM2.5": [35.5, 42.1, 50.0],
        "Station": ["Station_A", "Station_B", "Station_C"]
    })
    col_types = detect_column_types(df)
    assert "PM2.5" in col_types["numerical"]
    assert "Timestamp" in col_types["datetime"]
    assert "Station" in col_types["categorical"]


def test_get_dataset_summary():
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": [4, None, 6]
    })
    summary = get_dataset_summary(df)
    assert summary["total_rows"] == 3
    assert summary["total_columns"] == 2
    assert summary["total_missing"] == 2
    assert summary["missing_percentage"] == round(2 / 6 * 100, 2)
