"""
Imputation module implementing Forward Fill (FFill) and Mean Imputation.
Provides independent execution, detailed tracking, leading NaN handling,
and comprehensive comparison metrics.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def apply_forward_fill(
    df: pd.DataFrame,
    columns_to_impute: List[str],
    timestamp_col: Optional[str] = None,
    sort_chronologically: bool = False,
    preserve_original_order: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies Forward Fill (ffill / last observation carried forward) on selected columns.
    
    Parameters:
    - df: Source DataFrame (never mutated).
    - columns_to_impute: List of column names to impute.
    - timestamp_col: Optional timestamp column for chronological ordering.
    - sort_chronologically: If True, sorts before ffill.
    - preserve_original_order: If True and sorted, restores original row order after ffill.
    
    Returns:
    - (imputed_df, stats_dict)
    """
    if not columns_to_impute:
        raise ValueError("Please select at least one column to impute.")

    working_df = df.copy()
    original_index = working_df.index

    # Optional chronological sorting
    was_sorted = False
    if sort_chronologically and timestamp_col and timestamp_col in working_df.columns:
        parsed_ts = pd.to_datetime(working_df[timestamp_col], errors="coerce")
        if parsed_ts.notna().sum() > 0:
            working_df["__temp_sort_idx__"] = parsed_ts
            working_df = working_df.sort_values(by="__temp_sort_idx__")
            working_df = working_df.drop(columns=["__temp_sort_idx__"])
            was_sorted = True

    column_stats = []
    total_initial_missing = 0
    total_filled = 0
    total_remaining_missing = 0
    columns_with_leading_nans = []

    for col in columns_to_impute:
        if col not in working_df.columns:
            continue

        initial_missing = int(working_df[col].isna().sum())
        total_initial_missing += initial_missing

        # Detect leading missing values before ffill
        first_valid_idx = working_df[col].first_valid_index()
        leading_nan_count = 0
        if initial_missing > 0:
            if first_valid_idx is None:
                # Entire column is NaN
                leading_nan_count = initial_missing
            else:
                # Count NaNs before the first valid index in current sequence
                pos = working_df.index.get_loc(first_valid_idx)
                leading_nan_count = pos

        # Apply Forward Fill
        working_df[col] = working_df[col].ffill()

        remaining_missing = int(working_df[col].isna().sum())
        filled_count = initial_missing - remaining_missing
        total_filled += filled_count
        total_remaining_missing += remaining_missing

        fill_rate = (filled_count / initial_missing * 100) if initial_missing > 0 else 100.0

        if leading_nan_count > 0:
            columns_with_leading_nans.append({
                "column": col,
                "leading_nans": leading_nan_count
            })

        column_stats.append({
            "Column": col,
            "Initial Missing": initial_missing,
            "Values Filled": filled_count,
            "Remaining Missing": remaining_missing,
            "Fill Rate (%)": round(fill_rate, 2),
            "Leading NaNs": leading_nan_count,
            "Status": "⚠️ Unfilled Leading Values" if remaining_missing > 0 else "✅ Fully Filled"
        })

    # Restore original row order if requested and sorting occurred
    if was_sorted and preserve_original_order:
        working_df = working_df.reindex(original_index)

    stats_df = pd.DataFrame(column_stats)

    overall_stats = {
        "method": "Forward Fill (LOCF)",
        "columns_imputed": columns_to_impute,
        "total_initial_missing": total_initial_missing,
        "total_filled": total_filled,
        "total_remaining_missing": total_remaining_missing,
        "overall_fill_rate": round((total_filled / total_initial_missing * 100) if total_initial_missing > 0 else 100.0, 2),
        "columns_with_leading_nans": columns_with_leading_nans,
        "was_chronologically_sorted": was_sorted,
        "order_preserved": preserve_original_order,
        "summary_table": stats_df
    }

    return working_df, overall_stats


def apply_mean_imputation(
    df: pd.DataFrame,
    columns_to_impute: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies Mean Imputation on selected numerical columns.
    
    Parameters:
    - df: Source DataFrame (never mutated).
    - columns_to_impute: List of numerical columns to impute.
    
    Returns:
    - (imputed_df, stats_dict)
    """
    if not columns_to_impute:
        raise ValueError("Please select at least one numerical column to impute.")

    working_df = df.copy()
    column_stats = []
    total_initial_missing = 0
    total_filled = 0
    total_remaining_missing = 0
    completely_missing_cols = []

    for col in columns_to_impute:
        if col not in working_df.columns:
            continue

        # Strict validation: Only numerical columns permitted
        if not pd.api.types.is_numeric_dtype(working_df[col]):
            raise TypeError(
                f"Mean imputation cannot be applied to non-numerical column '{col}' "
                f"of type {working_df[col].dtype}."
            )

        initial_missing = int(working_df[col].isna().sum())
        total_initial_missing += initial_missing
        non_missing_count = len(working_df) - initial_missing

        if non_missing_count == 0:
            # 100% missing column - mean cannot be computed
            completely_missing_cols.append(col)
            column_stats.append({
                "Column": col,
                "Calculated Mean": "N/A (All NaN)",
                "Initial Missing": initial_missing,
                "Values Filled": 0,
                "Remaining Missing": initial_missing,
                "Fill Rate (%)": 0.0,
                "Status": "❌ Failed: Column has 0 valid values"
            })
            total_remaining_missing += initial_missing
            continue

        # Calculate mean of available observed values
        col_mean = float(working_df[col].mean())
        col_mean_rounded = round(col_mean, 3)

        # Replace missing with calculated mean
        working_df[col] = working_df[col].fillna(col_mean)

        remaining_missing = int(working_df[col].isna().sum())
        filled_count = initial_missing - remaining_missing
        total_filled += filled_count
        total_remaining_missing += remaining_missing

        fill_rate = (filled_count / initial_missing * 100) if initial_missing > 0 else 100.0

        column_stats.append({
            "Column": col,
            "Calculated Mean": col_mean_rounded,
            "Initial Missing": initial_missing,
            "Values Filled": filled_count,
            "Remaining Missing": remaining_missing,
            "Fill Rate (%)": round(fill_rate, 2),
            "Status": "✅ Fully Filled" if remaining_missing == 0 else "⚠️ Partial"
        })

    stats_df = pd.DataFrame(column_stats)

    overall_stats = {
        "method": "Mean Imputation",
        "columns_imputed": columns_to_impute,
        "total_initial_missing": total_initial_missing,
        "total_filled": total_filled,
        "total_remaining_missing": total_remaining_missing,
        "overall_fill_rate": round((total_filled / total_initial_missing * 100) if total_initial_missing > 0 else 100.0, 2),
        "completely_missing_cols": completely_missing_cols,
        "summary_table": stats_df
    }

    return working_df, overall_stats


def compare_imputations(
    original_df: pd.DataFrame,
    ffill_df: pd.DataFrame,
    mean_df: pd.DataFrame,
    columns: List[str]
) -> Dict[str, Any]:
    """
    Compares Forward Fill and Mean Imputation results side-by-side.
    Both were computed independently from the original dataset.
    """
    comparison_rows = []

    for col in columns:
        if col not in original_df.columns:
            continue

        orig_missing = int(original_df[col].isna().sum())
        ffill_missing = int(ffill_df[col].isna().sum()) if col in ffill_df.columns else orig_missing
        mean_missing = int(mean_df[col].isna().sum()) if col in mean_df.columns else orig_missing

        ffill_filled = orig_missing - ffill_missing
        mean_filled = orig_missing - mean_missing

        ffill_rate = round((ffill_filled / orig_missing * 100) if orig_missing > 0 else 100.0, 1)
        mean_rate = round((mean_filled / orig_missing * 100) if orig_missing > 0 else 100.0, 1)

        comparison_rows.append({
            "Column": col,
            "Initial Missing": orig_missing,
            "FFill Filled": ffill_filled,
            "FFill Remaining": ffill_missing,
            "FFill Fill Rate (%)": ffill_rate,
            "Mean Filled": mean_filled,
            "Mean Remaining": mean_missing,
            "Mean Fill Rate (%)": mean_rate
        })

    comp_df = pd.DataFrame(comparison_rows)

    total_orig_missing = comp_df["Initial Missing"].sum() if not comp_df.empty else 0
    total_ffill_filled = comp_df["FFill Filled"].sum() if not comp_df.empty else 0
    total_ffill_remaining = comp_df["FFill Remaining"].sum() if not comp_df.empty else 0
    total_mean_filled = comp_df["Mean Filled"].sum() if not comp_df.empty else 0
    total_mean_remaining = comp_df["Mean Remaining"].sum() if not comp_df.empty else 0

    return {
        "comparison_table": comp_df,
        "total_initial_missing": int(total_orig_missing),
        "ffill_total_filled": int(total_ffill_filled),
        "ffill_total_remaining": int(total_ffill_remaining),
        "mean_total_filled": int(total_mean_filled),
        "mean_total_remaining": int(total_mean_remaining),
        "ffill_fill_rate": round((total_ffill_filled / total_orig_missing * 100) if total_orig_missing > 0 else 100.0, 2),
        "mean_fill_rate": round((total_mean_filled / total_orig_missing * 100) if total_orig_missing > 0 else 100.0, 2),
    }


def create_comparison_bar_chart(comp_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive side-by-side Plotly bar chart comparing values filled
    by Forward Fill vs Mean Imputation across features.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Forward Fill Filled",
        x=comp_df["Column"],
        y=comp_df["FFill Filled"],
        marker_color="#38BDF8",  # Sky blue
        text=comp_df["FFill Filled"],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="Mean Imputation Filled",
        x=comp_df["Column"],
        y=comp_df["Mean Filled"],
        marker_color="#818CF8",  # Indigo
        text=comp_df["Mean Filled"],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Imputation Comparison: Filled Count by Technique</b>",
            font=dict(size=18, color="#F8FAFC")
        ),
        barmode="group",
        xaxis=dict(
            title=dict(text="<b>Feature Column</b>", font=dict(color="#F8FAFC")),
            tickangle=-25,
            tickfont=dict(color="#CBD5E1")
        ),
        yaxis=dict(
            title=dict(text="<b>Number of Values Filled</b>", font=dict(color="#F8FAFC")),
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
