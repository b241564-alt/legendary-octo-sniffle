"""
Missing value analysis and visualization module.
Generates comprehensive missingness metrics, summary tables, Plotly charts,
and Seaborn heatmap visualizations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes missingness summary per column:
    - Column Name
    - Data Type
    - Total Values
    - Non-Missing Count
    - Missing Count
    - Missing Percentage (%)
    """
    total_rows = len(df)
    summary_data = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        non_missing = total_rows - missing_count
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0

        summary_data.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Total Rows": total_rows,
            "Valid Count": non_missing,
            "Missing Count": missing_count,
            "Missing (%)": round(missing_pct, 2),
            "Status": "⚠️ Incomplete" if missing_count > 0 else "✅ Complete"
        })

    summary_df = pd.DataFrame(summary_data)
    # Sort with highest missing values first for immediate inspection
    return summary_df.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)


def get_row_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates row-level completeness metrics:
    - Complete rows (0 missing values)
    - Incomplete rows (>= 1 missing values)
    - Missing cells count and ratio
    """
    total_rows = len(df)
    incomplete_mask = df.isna().any(axis=1)
    incomplete_rows = int(incomplete_mask.sum())
    complete_rows = total_rows - incomplete_rows

    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    complete_cells = total_cells - missing_cells

    cols_with_missing = [c for c in df.columns if df[c].isna().sum() > 0]

    return {
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "complete_rows_pct": round((complete_rows / total_rows * 100) if total_rows > 0 else 0, 2),
        "incomplete_rows": incomplete_rows,
        "incomplete_rows_pct": round((incomplete_rows / total_rows * 100) if total_rows > 0 else 0, 2),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "complete_cells": complete_cells,
        "missing_cells_pct": round((missing_cells / total_cells * 100) if total_cells > 0 else 0, 2),
        "cols_with_missing_count": len(cols_with_missing),
        "cols_with_missing_names": cols_with_missing,
    }


def create_missing_bar_chart(summary_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive Plotly bar chart of missing value percentages per column.
    """
    # Filter to columns that actually have missing values or show all if none
    plot_df = summary_df.copy()

    # Color code bars by severity
    def assign_color(pct):
        if pct == 0:
            return "#10B981"  # Emerald green
        elif pct < 15:
            return "#F59E0B"  # Amber
        elif pct < 40:
            return "#F97316"  # Orange
        else:
            return "#EF4444"  # Red

    colors = [assign_color(pct) for pct in plot_df["Missing (%)"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["Column"],
                y=plot_df["Missing (%)"],
                text=[f"{count} missing ({pct}%)" for count, pct in zip(plot_df["Missing Count"], plot_df["Missing (%)"])],
                textposition="auto",
                marker=dict(
                    color=colors,
                    line=dict(color="#1E293B", width=1.5)
                ),
                hovertemplate="<b>%{x}</b><br>Missing Values: %{text}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Missing Value Percentage by Column</b>",
            font=dict(size=18, color="#F8FAFC")
        ),
        xaxis=dict(
            title=dict(text="<b>Feature Column</b>", font=dict(color="#F8FAFC")),
            tickangle=-30,
            tickfont=dict(color="#CBD5E1")
        ),
        yaxis=dict(
            title=dict(text="<b>Missing Percentage (%)</b>", font=dict(color="#F8FAFC")),
            range=[0, max(100, (plot_df["Missing (%)"].max() if not plot_df.empty else 0) + 10)],
            tickfont=dict(color="#CBD5E1"),
            gridcolor="#334155"
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        margin=dict(l=40, r=40, t=60, b=80),
        height=450
    )
    return fig


def create_missing_heatmap(df: pd.DataFrame, max_rows: int = 150) -> plt.Figure:
    """
    Generates a Seaborn missingness heatmap for the dataset.
    Uses yellow/gold for missing values and dark slate for observed values.
    """
    plot_df = df.head(max_rows) if len(df) > max_rows else df
    missing_matrix = plot_df.isna()

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0F172A")
    ax.set_facecolor("#1E293B")

    # Custom colormap: 0 = observed (#1E293B), 1 = missing (#F59E0B)
    cmap = sns.color_palette(["#1E293B", "#F59E0B"])

    sns.heatmap(
        missing_matrix,
        cmap=cmap,
        cbar=True,
        cbar_kws={
            "ticks": [0.25, 0.75],
            "label": "Data Status (Observed vs Missing)",
            "shrink": 0.8
        },
        yticklabels=False,
        ax=ax
    )

    # Customize colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["Observed", "Missing"])
    cbar.ax.yaxis.label.set_color("#F8FAFC")
    cbar.ax.tick_params(colors="#F8FAFC")

    title_text = f"Missingness Heatmap (First {len(plot_df)} Rows)" if len(df) > max_rows else "Dataset Missingness Heatmap"
    ax.set_title(title_text, fontsize=14, fontweight="bold", color="#F8FAFC", pad=12)
    ax.set_xlabel("Columns", fontsize=11, fontweight="bold", color="#F8FAFC", labelpad=10)
    ax.set_ylabel(f"Sample Records (n={len(plot_df)})", fontsize=11, fontweight="bold", color="#F8FAFC", labelpad=10)
    ax.tick_params(axis="x", colors="#CBD5E1", rotation=40)

    plt.tight_layout()
    return fig
