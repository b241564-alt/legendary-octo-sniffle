"""
Unit tests for imputation module.
"""

import pytest
import pandas as pd
import numpy as np
from src.imputation import (
    apply_forward_fill,
    apply_mean_imputation,
    compare_imputations,
)


def test_forward_fill_basic():
    df = pd.DataFrame({
        "PM2.5": [10.0, np.nan, np.nan, 20.0, np.nan]
    })
    imputed, stats = apply_forward_fill(df, ["PM2.5"])
    assert list(imputed["PM2.5"]) == [10.0, 10.0, 10.0, 20.0, 20.0]
    assert stats["total_initial_missing"] == 3
    assert stats["total_filled"] == 3
    assert stats["total_remaining_missing"] == 0
    assert df["PM2.5"].isna().sum() == 3  # Original unchanged


def test_forward_fill_leading_nans():
    # Leading NaN: ffill CANNOT fill row 0 because there is no prior value
    df = pd.DataFrame({
        "PM2.5": [np.nan, np.nan, 15.0, np.nan, 25.0]
    })
    imputed, stats = apply_forward_fill(df, ["PM2.5"])
    assert pd.isna(imputed["PM2.5"].iloc[0])
    assert pd.isna(imputed["PM2.5"].iloc[1])
    assert imputed["PM2.5"].iloc[2] == 15.0
    assert imputed["PM2.5"].iloc[3] == 15.0
    assert imputed["PM2.5"].iloc[4] == 25.0

    assert stats["total_initial_missing"] == 3
    assert stats["total_filled"] == 1
    assert stats["total_remaining_missing"] == 2
    assert stats["overall_fill_rate"] < 100.0


def test_mean_imputation_basic():
    df = pd.DataFrame({
        "PM2.5": [10.0, np.nan, 30.0]
    })
    imputed, stats = apply_mean_imputation(df, ["PM2.5"])
    # Mean of 10 and 30 is 20.0
    assert list(imputed["PM2.5"]) == [10.0, 20.0, 30.0]
    assert stats["total_initial_missing"] == 1
    assert stats["total_filled"] == 1
    assert stats["total_remaining_missing"] == 0
    assert df["PM2.5"].isna().sum() == 1  # Original unchanged


def test_mean_imputation_non_numeric_error():
    df = pd.DataFrame({
        "City": ["Delhi", np.nan, "Mumbai"]
    })
    with pytest.raises(TypeError):
        apply_mean_imputation(df, ["City"])


def test_mean_imputation_all_nan():
    df = pd.DataFrame({
        "EmptyCol": [np.nan, np.nan, np.nan]
    })
    imputed, stats = apply_mean_imputation(df, ["EmptyCol"])
    assert stats["total_filled"] == 0
    assert stats["total_remaining_missing"] == 3
    assert "EmptyCol" in stats["completely_missing_cols"]


def test_compare_imputations():
    orig_df = pd.DataFrame({
        "PM2.5": [10.0, np.nan, 30.0]
    })
    ffill_df, _ = apply_forward_fill(orig_df, ["PM2.5"])
    mean_df, _ = apply_mean_imputation(orig_df, ["PM2.5"])

    comp = compare_imputations(orig_df, ffill_df, mean_df, ["PM2.5"])
    assert comp["total_initial_missing"] == 1
    assert comp["ffill_total_filled"] == 1
    assert comp["mean_total_filled"] == 1
