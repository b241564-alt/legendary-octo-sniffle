"""
Unit tests for preprocessing module.
"""

import pandas as pd
from src.preprocessing import (
    detect_duplicates,
    remove_duplicates,
    select_columns,
    sort_by_timestamp,
)


def test_duplicate_detection_and_removal():
    df = pd.DataFrame({
        "A": [1, 2, 2, 3],
        "B": ["x", "y", "y", "z"]
    })
    dup_info = detect_duplicates(df)
    assert dup_info["duplicate_count"] == 1
    assert dup_info["has_duplicates"] is True

    cleaned, removed = remove_duplicates(df)
    assert removed == 1
    assert len(cleaned) == 3
    assert len(df) == 4  # Original unchanged!


def test_select_columns():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    selected = select_columns(df, ["A", "C"])
    assert list(selected.columns) == ["A", "C"]
    assert len(df.columns) == 3


def test_sort_by_timestamp():
    df = pd.DataFrame({
        "Timestamp": ["2024-03-01 03:00", "2024-03-01 01:00", "2024-03-01 02:00"],
        "Val": [30, 10, 20]
    })
    sorted_df, ok = sort_by_timestamp(df, "Timestamp", ascending=True)
    assert ok is True
    assert sorted_df.iloc[0]["Val"] == 10
    assert sorted_df.iloc[1]["Val"] == 20
    assert sorted_df.iloc[2]["Val"] == 30
