"""
Unit tests for missing_analysis module.
"""

import pandas as pd
from src.missing_analysis import (
    get_missing_summary,
    get_row_completeness,
    create_missing_bar_chart,
    create_missing_heatmap,
)


def test_missing_summary():
    df = pd.DataFrame({
        "PM2.5": [10.0, None, 30.0, None],
        "PM10": [20.0, 30.0, 40.0, 50.0],
        "City": ["Delhi", "Delhi", None, "Delhi"]
    })
    summary = get_missing_summary(df)
    assert len(summary) == 3
    # PM2.5 has 2 missing
    pm25_row = summary[summary["Column"] == "PM2.5"].iloc[0]
    assert pm25_row["Missing Count"] == 2
    assert pm25_row["Missing (%)"] == 50.0

    # PM10 has 0 missing
    pm10_row = summary[summary["Column"] == "PM10"].iloc[0]
    assert pm10_row["Missing Count"] == 0
    assert pm10_row["Missing (%)"] == 0.0


def test_row_completeness():
    df = pd.DataFrame({
        "A": [1, None, 3, 4],
        "B": [5, 6, None, 8]
    })
    comp = get_row_completeness(df)
    assert comp["total_rows"] == 4
    assert comp["complete_rows"] == 2
    assert comp["incomplete_rows"] == 2
    assert comp["missing_cells"] == 2
    assert comp["cols_with_missing_count"] == 2


def test_missing_chart_and_heatmap():
    df = pd.DataFrame({
        "PM2.5": [10.0, None, 30.0],
        "PM10": [20.0, 30.0, 40.0]
    })
    summary = get_missing_summary(df)
    fig_bar = create_missing_bar_chart(summary)
    assert fig_bar is not None

    fig_heat = create_missing_heatmap(df)
    assert fig_heat is not None
