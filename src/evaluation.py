"""
Evaluation module for controlled synthetic missingness benchmarking.
Evaluates Forward Fill vs Mean Imputation against known ground truth
using standard regression metrics (MAE, MSE, RMSE).
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go


def run_evaluation_experiment(
    df: pd.DataFrame,
    numerical_cols: List[str],
    mask_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes a controlled evaluation experiment:
    1. Extracts complete rows for the chosen numerical features as ground truth.
    2. Artificially masks mask_ratio of the values as NaN.
    3. Imputes masked values using Forward Fill and Mean Imputation independently.
    4. Evaluates against original ground truth using MAE, MSE, RMSE.
    
    Returns metrics table, overall score summary, and chart metadata.
    """
    if not numerical_cols:
        raise ValueError("Please select at least one numerical column for evaluation.")

    # Validate numerical types
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numerical.")

    # Filter to complete rows for the selected features
    complete_df = df[numerical_cols].dropna().copy()
    num_complete_rows = len(complete_df)

    if num_complete_rows < 10:
        raise ValueError(
            f"Not enough complete rows ({num_complete_rows} rows found) to run a reliable "
            "evaluation experiment. At least 10 complete rows are required."
        )

    np.random.seed(random_seed)

    # Create artificial mask matrix
    # Ensure at least 1 value per column is masked, but not all
    mask_matrix = np.random.rand(*complete_df.shape) < mask_ratio
    # Guardrail: Ensure at least one masked item per column
    for c_idx in range(complete_df.shape[1]):
        if not mask_matrix[:, c_idx].any():
            random_row = np.random.randint(0, num_complete_rows)
            mask_matrix[random_row, c_idx] = True

    # Guardrail: Do not mask 100% of any column
    for c_idx in range(complete_df.shape[1]):
        if mask_matrix[:, c_idx].all():
            random_row = np.random.randint(0, num_complete_rows)
            mask_matrix[random_row, c_idx] = False

    total_masked_cells = int(mask_matrix.sum())

    # Create corrupted / masked dataset
    corrupted_df = complete_df.copy()
    for col_idx, col in enumerate(numerical_cols):
        corrupted_df.iloc[mask_matrix[:, col_idx], col_idx] = np.nan

    # 1. Apply Forward Fill
    ffill_df = corrupted_df.copy().ffill()
    # Handle leading NaNs if any in simulation by backfilling or column mean so metric calculation doesn't fail
    if ffill_df.isna().sum().sum() > 0:
        ffill_df = ffill_df.bfill()

    # 2. Apply Mean Imputation
    mean_df = corrupted_df.copy()
    for col in numerical_cols:
        col_mean = corrupted_df[col].mean()
        mean_df[col] = mean_df[col].fillna(col_mean)

    # Compute metrics per column
    metrics_list = []
    all_y_true = []
    all_y_pred_ffill = []
    all_y_pred_mean = []

    for col_idx, col in enumerate(numerical_cols):
        mask_col = mask_matrix[:, col_idx]
        y_true = complete_df.iloc[mask_col, col_idx].values
        y_ffill = ffill_df.iloc[mask_col, col_idx].values
        y_mean = mean_df.iloc[mask_col, col_idx].values

        if len(y_true) == 0:
            continue

        mae_ffill = float(mean_absolute_error(y_true, y_ffill))
        mse_ffill = float(mean_squared_error(y_true, y_ffill))
        rmse_ffill = float(math.sqrt(mse_ffill))

        mae_mean = float(mean_absolute_error(y_true, y_mean))
        mse_mean = float(mean_squared_error(y_true, y_mean))
        rmse_mean = float(math.sqrt(mse_mean))

        # Best performer
        better_method = "Forward Fill" if rmse_ffill < rmse_mean else ("Mean Imputation" if rmse_mean < rmse_ffill else "Tie")

        metrics_list.append({
            "Feature": col,
            "Masked Count": int(len(y_true)),
            "FFill MAE": round(mae_ffill, 3),
            "Mean MAE": round(mae_mean, 3),
            "FFill RMSE": round(rmse_ffill, 3),
            "Mean RMSE": round(rmse_mean, 3),
            "FFill MSE": round(mse_ffill, 3),
            "Mean MSE": round(mse_mean, 3),
            "Best Performer": better_method
        })

        all_y_true.extend(y_true)
        all_y_pred_ffill.extend(y_ffill)
        all_y_pred_mean.extend(y_mean)

    overall_mae_ffill = float(mean_absolute_error(all_y_true, all_y_pred_ffill))
    overall_rmse_ffill = float(math.sqrt(mean_squared_error(all_y_true, all_y_pred_ffill)))

    overall_mae_mean = float(mean_absolute_error(all_y_true, all_y_pred_mean))
    overall_rmse_mean = float(math.sqrt(mean_squared_error(all_y_true, all_y_pred_mean)))

    overall_winner = "Forward Fill" if overall_rmse_ffill < overall_rmse_mean else "Mean Imputation"

    metrics_df = pd.DataFrame(metrics_list)

    return {
        "metrics_table": metrics_df,
        "sample_size": num_complete_rows,
        "total_cells_evaluated": int(complete_df.size),
        "total_masked_cells": total_masked_cells,
        "mask_ratio": mask_ratio,
        "overall_winner": overall_winner,
        "overall_ffill_mae": round(overall_mae_ffill, 3),
        "overall_ffill_rmse": round(overall_rmse_ffill, 3),
        "overall_mean_mae": round(overall_mae_mean, 3),
        "overall_mean_rmse": round(overall_rmse_mean, 3),
        "disclaimer": (
            "Academic Experimental Note: These metrics are calculated on an artificially masked subset "
            "of originally complete records where ground truth was known. In real-world data missingness, "
            "the true ground truth is fundamentally unobserved."
        )
    }


def create_evaluation_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive Plotly bar chart comparing RMSE between Forward Fill and Mean Imputation.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Forward Fill RMSE",
        x=metrics_df["Feature"],
        y=metrics_df["FFill RMSE"],
        marker_color="#38BDF8",  # Sky blue
        text=[f"{v:.2f}" for v in metrics_df["FFill RMSE"]],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="Mean Imputation RMSE",
        x=metrics_df["Feature"],
        y=metrics_df["Mean RMSE"],
        marker_color="#A78BFA",  # Purple
        text=[f"{v:.2f}" for v in metrics_df["Mean RMSE"]],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Root Mean Squared Error (RMSE) by Feature (Lower is Better)</b>",
            font=dict(size=18, color="#F8FAFC")
        ),
        barmode="group",
        xaxis=dict(
            title=dict(text="<b>Pollutant / Feature</b>", font=dict(color="#F8FAFC")),
            tickangle=-25,
            tickfont=dict(color="#CBD5E1")
        ),
        yaxis=dict(
            title=dict(text="<b>RMSE (Error Magnitude)</b>", font=dict(color="#F8FAFC")),
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155"
        ),
        legend=dict(
            font=dict(color="#F8FAFC"),
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="#475569",
            borderwidth=1
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=40, r=40, t=60, b=80),
        height=450
    )
    return fig
