"""
Unit tests for evaluation module.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation import run_evaluation_experiment


def test_run_evaluation_experiment():
    # 25 complete rows with 2 numerical features
    np.random.seed(0)
    data = {
        "PM2.5": np.random.uniform(20.0, 100.0, 30),
        "PM10": np.random.uniform(50.0, 200.0, 30)
    }
    df = pd.DataFrame(data)

    results = run_evaluation_experiment(
        df,
        numerical_cols=["PM2.5", "PM10"],
        mask_ratio=0.2,
        random_seed=42
    )

    assert "metrics_table" in results
    assert len(results["metrics_table"]) == 2
    assert results["total_masked_cells"] > 0
    assert results["overall_ffill_rmse"] >= 0.0
    assert results["overall_mean_rmse"] >= 0.0
    assert results["overall_winner"] in ["Forward Fill", "Mean Imputation", "Tie"]


def test_evaluation_not_enough_rows():
    df = pd.DataFrame({
        "PM2.5": [10.0, 20.0, 30.0],
        "PM10": [20.0, 30.0, 40.0]
    })
    with pytest.raises(ValueError) as excinfo:
        run_evaluation_experiment(df, numerical_cols=["PM2.5", "PM10"])
    assert "not enough complete rows" in str(excinfo.value).lower()
