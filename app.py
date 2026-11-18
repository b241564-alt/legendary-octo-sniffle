"""
AQI Missing Value Imputer – Air Quality Data Preprocessing and Imputation Tool
Main Streamlit Application
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

# Internal module imports
from src.data_loader import (
    load_csv,
    load_sample_data,
    validate_dataset,
    detect_column_types,
    get_dataset_summary,
)
from src.missing_analysis import (
    get_missing_summary,
    get_row_completeness,
    create_missing_bar_chart,
    create_missing_heatmap,
)
from src.preprocessing import (
    detect_duplicates,
    remove_duplicates,
    select_columns,
    sort_by_timestamp,
)
from src.imputation import (
    apply_forward_fill,
    apply_mean_imputation,
    compare_imputations,
    create_comparison_bar_chart,
)
from src.evaluation import (
    run_evaluation_experiment,
    create_evaluation_bar_chart,
)
from src.ui_components import (
    apply_custom_css,
    render_hero,
    render_metric_card,
    render_callout,
    render_footer,
)

# Page configuration
st.set_page_config(
    page_title="AQI Missing Value Imputer",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply unified design system
apply_custom_css()

# Session State Initialization
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "ffill_df" not in st.session_state:
    st.session_state.ffill_df = None
if "ffill_stats" not in st.session_state:
    st.session_state.ffill_stats = None
if "mean_df" not in st.session_state:
    st.session_state.mean_df = None
if "mean_stats" not in st.session_state:
    st.session_state.mean_stats = None
if "final_selected_method" not in st.session_state:
    st.session_state.final_selected_method = "Forward Fill"


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🌫️ **AQI Imputer Studio**")
    st.caption("Data Preprocessing & Imputation System")
    st.divider()

    nav_options = [
        "1. 🏠 Home / Overview",
        "2. 📂 Upload Dataset",
        "3. 🔍 Missing Value Analysis",
        "4. ⚙️ Data Preprocessing",
        "5. ⏩ Forward Fill Imputation",
        "6. 📊 Mean Imputation",
        "7. ⚖️ Comparison Dashboard",
        "8. 🧪 Evaluation Experiment",
        "9. 💾 Download Results",
        "10. ℹ️ About Project",
    ]

    selected_nav = st.radio(
        "Navigation Menu",
        nav_options,
        index=0
    )

    st.divider()
    # Active Dataset Status Indicator
    if st.session_state.raw_df is not None:
        st.success(f"**Loaded:** {st.session_state.dataset_name}")
        st.caption(f"📏 Shape: {st.session_state.raw_df.shape[0]} rows × {st.session_state.raw_df.shape[1]} cols")
    else:
        st.info("ℹ️ No dataset loaded yet.")

    st.caption("Academic ML & Environmental Science")


# -----------------------------------------------------------------------------
# 1. HOME / OVERVIEW
# -----------------------------------------------------------------------------
if selected_nav.startswith("1."):
    render_hero(
        title="AQI Missing Value Imputer",
        subtitle="An Academic Machine Learning & Preprocessing Suite for Air Quality Monitoring Networks",
        badge="Environmental Data Science • ML Preprocessing"
    )

    st.markdown("### 📌 Why Missing Data Imputation in AQI Matters")
    st.write(
        """
        Ambient air quality monitoring relies on continuous sensory networks tracking pollutants like 
        **PM2.5, PM10, NO2, SO2, CO, and O3**. In real-world sensor deployments, missing data frequently arises due to:
        - ⚡ **Sensor power outages & network disconnects**
        - 🛠️ **Sensor drift, degradation, and periodic recalibration cycles**
        - 🌧️ **Adverse atmospheric conditions (extreme condensation or dust clogging)**
        """
    )

    st.markdown("---")
    st.markdown("### 🔄 End-to-End Workflow")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            #### 1. Ingestion & Diagnostics
            - Upload custom CSV or load built-in realistic AQI sensor records.
            - Automated schema, duplicate, and missingness pattern diagnosis.
            - Missingness heatmap and percentage profiling.
            """
        )
    with col2:
        st.markdown(
            """
            #### 2. Imputation Engine
            - **Forward Fill (LOCF)**: Preserves temporal autocorrelation across sequential hours.
            - **Mean Imputation**: Statistical central tendency baseline for numerical series.
            - Independent execution guaranteeing unbiased comparative metrics.
            """
        )
    with col3:
        st.markdown(
            """
            #### 3. Benchmark & Export
            - Controlled masking experiment calculating ground-truth **MAE, MSE, & RMSE**.
            - Comparative analytics dashboards.
            - 1-click download of all imputed and report datasets in standard CSV formats.
            """
        )

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("📥 Load Built-in Sample AQI Dataset", type="primary", use_container_width=True):
            sample_data = load_sample_data()
            st.session_state.raw_df = sample_data
            st.session_state.dataset_name = "sample_aqi_data.csv (Delhi Monitoring Network)"
            st.session_state.processed_df = sample_data.copy()
            # Clear previous imputations
            st.session_state.ffill_df = None
            st.session_state.mean_df = None
            st.success("✅ Sample AQI dataset loaded successfully! Navigate to 'Upload Dataset' or 'Missing Value Analysis' in the sidebar.")
    with btn_col2:
        st.info("💡 You can also head over to the **Upload Dataset** tab to upload your own custom CSV file.")


# -----------------------------------------------------------------------------
# 2. UPLOAD DATASET
# -----------------------------------------------------------------------------
elif selected_nav.startswith("2."):
    st.markdown("## 📂 Upload Air Quality Dataset")
    st.caption("Upload a custom air quality CSV or load our preloaded monitoring dataset.")

    up_col1, up_col2 = st.columns([2, 1])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Select an Air Quality CSV file",
            type=["csv"],
            help="Upload a standard CSV file with pollutant readings and timestamps."
        )

    with up_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Load Sample AQI Dataset", use_container_width=True):
            try:
                sample_data = load_sample_data()
                st.session_state.raw_df = sample_data
                st.session_state.dataset_name = "sample_aqi_data.csv (Delhi Anand Vihar Station)"
                st.session_state.processed_df = sample_data.copy()
                st.session_state.ffill_df = None
                st.session_state.mean_df = None
                st.success("Loaded preloaded multi-pollutant sample dataset!")
            except Exception as e:
                st.error(f"Failed to load sample dataset: {e}")

    if uploaded_file is not None:
        try:
            user_df = load_csv(uploaded_file)
            is_valid, reason = validate_dataset(user_df)
            if not is_valid:
                st.error(f"❌ Invalid Dataset: {reason}")
            else:
                st.session_state.raw_df = user_df
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.processed_df = user_df.copy()
                st.session_state.ffill_df = None
                st.session_state.mean_df = None
                st.success(f"✅ Successfully loaded `{uploaded_file.name}`!")
        except Exception as e:
            st.error(f"❌ Error parsing uploaded CSV: {e}")

    # Display dataset details if available
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        summary = get_dataset_summary(df)

        st.markdown("---")
        st.markdown(f"### 📋 Dataset Overview: `{st.session_state.dataset_name}`")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Total Rows", f"{summary['total_rows']:,}", "Data points", "#0284C7")
        with m2:
            render_metric_card("Total Columns", f"{summary['total_columns']}", "Features / Pollutants", "#38BDF8")
        with m3:
            render_metric_card("Missing Values", f"{summary['total_missing']:,}", f"{summary['missing_percentage']}% of all cells", "#F59E0B")
        with m4:
            render_metric_card("Duplicate Rows", f"{summary['duplicate_rows']}", "Identical records", "#10B981" if summary['duplicate_rows'] == 0 else "#EF4444")

        # Column Type Badges
        col_types = summary["column_types"]
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.markdown(f"**🔢 Numerical Features ({len(col_types['numerical'])}):**")
            st.code(", ".join(col_types["numerical"]) if col_types["numerical"] else "None detected")
        with t_col2:
            st.markdown(f"**⏰ Timestamp / Date Features ({len(col_types['datetime'])}):**")
            st.code(", ".join(col_types["datetime"]) if col_types["datetime"] else "None detected")
        with t_col3:
            st.markdown(f"**🏷️ Categorical Features ({len(col_types['categorical'])}):**")
            st.code(", ".join(col_types["categorical"]) if col_types["categorical"] else "None detected")

        # Data preview
        st.markdown("#### 👁️ First 10 Rows Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        render_callout(
            "Please upload a CSV file above or click 'Load Sample AQI Dataset' to get started.",
            title="No Active Dataset",
            alert_type="info"
        )


# -----------------------------------------------------------------------------
# 3. MISSING VALUE ANALYSIS
# -----------------------------------------------------------------------------
elif selected_nav.startswith("3."):
    st.markdown("## 🔍 Missing Value Analysis")
    st.caption("In-depth statistical diagnostics of missingness patterns across all features.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        completeness = get_row_completeness(df)
        missing_summary_df = get_missing_summary(df)

        # High-level Completeness KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            render_metric_card("Total Missing Cells", f"{completeness['missing_cells']:,}", f"{completeness['missing_cells_pct']}% of total", "#F59E0B")
        with kpi2:
            render_metric_card("Complete Rows", f"{completeness['complete_rows']:,}", f"{completeness['complete_rows_pct']}% of records", "#10B981")
        with kpi3:
            render_metric_card("Incomplete Rows", f"{completeness['incomplete_rows']:,}", f"{completeness['incomplete_rows_pct']}% have ≥1 NaN", "#EF4444")
        with kpi4:
            render_metric_card("Affected Columns", f"{completeness['cols_with_missing_count']}", f"out of {len(df.columns)} features", "#38BDF8")

        st.markdown("---")

        # Bar chart & Heatmap tabs
        viz_tab1, viz_tab2 = st.tabs(["📊 Missing Value Bar Chart", "🌡️ Missingness Heatmap"])

        with viz_tab1:
            st.plotly_chart(create_missing_bar_chart(missing_summary_df), use_container_width=True)

        with viz_tab2:
            st.markdown("##### Missingness Sparsity Matrix (Gold = Missing, Dark Slate = Observed)")
            heatmap_fig = create_missing_heatmap(df)
            st.pyplot(heatmap_fig)

        st.markdown("---")
        st.markdown("### 📋 Missing Value Summary Table")
        st.dataframe(missing_summary_df, use_container_width=True)

        # Contextual explanation
        render_callout(
            "Air quality sensors typically suffer from either **Missing Completely at Random (MCAR)** "
            "(e.g., transient network packet drops) or **Missing at Random / Not at Random (MAR/MNAR)** "
            "(e.g., sensor saturation during extreme pollution spikes or maintenance blackouts). "
            "Identifying whether gaps are isolated single-hour dropouts or multi-hour consecutive blocks "
            "is critical when choosing between Forward Fill and statistical imputation.",
            title="Academic Context: Missingness Mechanisms in Environmental IoT",
            alert_type="info"
        )


# -----------------------------------------------------------------------------
# 4. DATA PREPROCESSING
# -----------------------------------------------------------------------------
elif selected_nav.startswith("4."):
    st.markdown("## ⚙️ Data Preprocessing & Sanitization")
    st.caption("Inspect and handle duplicate records, filter columns, and sort chronologically.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        dup_info = detect_duplicates(df)
        col_types = detect_column_types(df)

        st.markdown("### 1. Duplicate Row Management")
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            render_metric_card(
                "Duplicate Rows Found",
                f"{dup_info['duplicate_count']}",
                f"{dup_info['duplicate_percentage']}% of dataset",
                "#10B981" if dup_info['duplicate_count'] == 0 else "#EF4444"
            )
        with d_col2:
            if dup_info["has_duplicates"]:
                st.warning(f"⚠️ Found {dup_info['duplicate_count']} exact duplicate records in the dataset.")
                if st.button("🗑️ Remove Duplicate Rows", type="primary"):
                    cleaned_df, removed = remove_duplicates(df)
                    st.session_state.processed_df = cleaned_df
                    st.success(f"Successfully removed {removed} duplicate rows! Current rows: {len(cleaned_df):,}")
                    st.rerun()
            else:
                st.success("✅ No duplicate rows detected in the dataset. All records are unique.")

        st.markdown("---")
        st.markdown("### 2. Feature / Column Filtering")
        st.write("Select the subset of features to include in the downstream imputation workflow:")

        all_cols = list(df.columns)
        selected_cols = st.multiselect(
            "Active Columns",
            options=all_cols,
            default=all_cols,
            help="Deselect any irrelevant metadata columns before imputation."
        )

        if st.button("Apply Column Selection"):
            if not selected_cols:
                st.error("You must retain at least one column.")
            else:
                st.session_state.processed_df = select_columns(df, selected_cols)
                st.success(f"Updated active dataset to {len(selected_cols)} columns.")
                st.rerun()

        st.markdown("---")
        st.markdown("### 3. Chronological Timestamp Sorting")
        datetime_cols = col_types["datetime"]

        if datetime_cols:
            ts_col = st.selectbox("Select Timestamp Column for Chronological Ordering", datetime_cols)
            sort_order = st.radio("Sort Order", ["Ascending (Oldest to Newest)", "Descending (Newest to Oldest)"], index=0)

            if st.button("📅 Sort Chronologically"):
                ascending = "Ascending" in sort_order
                sorted_df, ok = sort_by_timestamp(df, ts_col, ascending=ascending)
                if ok:
                    st.session_state.processed_df = sorted_df
                    st.success(f"Dataset successfully sorted chronologically by `{ts_col}`!")
                    st.rerun()
                else:
                    st.error("Failed to parse timestamps for sorting.")
        else:
            st.info("No dedicated timestamp column was automatically recognized. You can still proceed with natural row ordering.")

        st.markdown("---")
        if st.button("🔄 Reset Preprocessing to Original Dataset"):
            st.session_state.processed_df = st.session_state.raw_df.copy()
            st.success("Reset active dataset to the original raw uploaded version.")
            st.rerun()


# -----------------------------------------------------------------------------
# 5. FORWARD FILL IMPUTATION
# -----------------------------------------------------------------------------
elif selected_nav.startswith("5."):
    st.markdown("## ⏩ Forward Fill Imputation (LOCF)")
    st.caption("Last Observation Carried Forward — propagating previous valid sensor observations.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        col_types = detect_column_types(df)
        cols_with_missing = [c for c in df.columns if df[c].isna().sum() > 0]

        # Theoretical Explanation
        render_callout(
            "**Forward Fill** propagates the last known valid sensor measurement forward until a new valid measurement "
            "is observed. In time-series air quality monitoring, atmospheric concentrations (e.g., PM2.5, NO2) exhibit high "
            "**temporal autocorrelation**, making forward fill effective for short transient dropouts (1–2 hours). "
            "<br><br><strong>Edge Case Warning:</strong> If a column has missing values at the very start of the sequence "
            "(leading NaNs), Forward Fill <em>cannot</em> fill them because there is no preceding observation to carry forward.",
            title="Algorithm Theory & Assumptions",
            alert_type="info"
        )

        st.markdown("### Configure Forward Fill")
        c1, c2 = st.columns(2)
        with c1:
            selected_ffill_cols = st.multiselect(
                "Columns to Impute with Forward Fill",
                options=df.columns.tolist(),
                default=cols_with_missing if cols_with_missing else df.columns.tolist(),
                help="Select columns to apply ffill."
            )
        with c2:
            ts_candidates = col_types["datetime"]
            ts_selection = st.selectbox(
                "Optional Timestamp Column (for sorting before ffill)",
                options=["None"] + ts_candidates,
                index=1 if ts_candidates else 0
            )
            sort_chrono = st.checkbox(
                "Ensure Chronological Sorting before Imputation",
                value=True if ts_candidates else False
            )
            preserve_order = st.checkbox(
                "Preserve Original Row Order in Final Result",
                value=True,
                help="If sorted for ffill, restores the original input row order."
            )

        if st.button("🚀 Apply Forward Fill Imputation", type="primary"):
            if not selected_ffill_cols:
                st.error("Please select at least one column to impute.")
            else:
                ts_col_param = ts_selection if ts_selection != "None" else None
                imputed_df, stats = apply_forward_fill(
                    df,
                    columns_to_impute=selected_ffill_cols,
                    timestamp_col=ts_col_param,
                    sort_chronologically=sort_chrono,
                    preserve_original_order=preserve_order
                )
                st.session_state.ffill_df = imputed_df
                st.session_state.ffill_stats = stats
                st.success("✅ Forward Fill completed successfully!")

        # Display results if ffill has been run
        if st.session_state.ffill_stats is not None:
            stats = st.session_state.ffill_stats
            st.markdown("---")
            st.markdown("### 📊 Forward Fill Results Summary")

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                render_metric_card("Initial Missing", f"{stats['total_initial_missing']:,}", "Across selected columns", "#0284C7")
            with s2:
                render_metric_card("Values Filled", f"{stats['total_filled']:,}", f"{stats['overall_fill_rate']}% fill rate", "#10B981")
            with s3:
                render_metric_card("Remaining Missing", f"{stats['total_remaining_missing']:,}", "Leading NaNs", "#EF4444" if stats['total_remaining_missing'] > 0 else "#10B981")
            with s4:
                render_metric_card("Chronologically Sorted", "Yes" if stats['was_chronologically_sorted'] else "No", "Temporal sequence", "#38BDF8")

            # Alert if leading NaNs prevented 100% fill
            if stats["total_remaining_missing"] > 0:
                render_callout(
                    f"Notice: **{stats['total_remaining_missing']} missing values could not be filled** because they occur "
                    "at the beginning of the sequence (leading missing values) where no prior observation exists. "
                    "This is an inherent mathematical property of Forward Fill.",
                    title="Leading Missing Values Detected",
                    alert_type="warning"
                )

            st.markdown("#### Column-by-Column Breakdown")
            st.dataframe(stats["summary_table"], use_container_width=True)

            st.markdown("#### Imputed Dataset Preview (First 10 Rows)")
            st.dataframe(st.session_state.ffill_df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# 6. MEAN IMPUTATION
# -----------------------------------------------------------------------------
elif selected_nav.startswith("6."):
    st.markdown("## 📊 Mean Imputation")
    st.caption("Parametric statistical imputation replacing missing numerical entries with feature means.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        col_types = detect_column_types(df)
        num_cols = col_types["numerical"]
        num_cols_with_missing = [c for c in num_cols if df[c].isna().sum() > 0]

        # Theoretical Explanation
        render_callout(
            "**Mean Imputation** replaces missing values with the arithmetic mean of observed samples: "
            "<br><code>x̄ = (1 / N_obs) * Σ x_i</code><br>"
            "While straightforward and guaranteed to preserve the sample mean, mean imputation artificially "
            "**reduces variance**, weakens feature correlations, and ignores the temporal dynamics of air pollution streams. "
            "<br><br><strong>Restriction:</strong> Mean imputation is strictly restricted to numerical measurements.",
            title="Algorithm Theory & Assumptions",
            alert_type="info"
        )

        if not num_cols:
            st.error("No numerical columns found in the dataset for mean imputation.")
        else:
            selected_mean_cols = st.multiselect(
                "Select Numerical Columns to Impute",
                options=num_cols,
                default=num_cols_with_missing if num_cols_with_missing else num_cols,
                help="Only numeric features can be mean-imputed."
            )

            if st.button("🚀 Apply Mean Imputation", type="primary"):
                if not selected_mean_cols:
                    st.error("Please select at least one numerical column.")
                else:
                    try:
                        imputed_df, stats = apply_mean_imputation(df, selected_mean_cols)
                        st.session_state.mean_df = imputed_df
                        st.session_state.mean_stats = stats
                        st.success("✅ Mean Imputation completed successfully!")
                    except Exception as e:
                        st.error(f"Error during mean imputation: {e}")

            # Display results if mean imputation has been run
            if st.session_state.mean_stats is not None:
                stats = st.session_state.mean_stats
                st.markdown("---")
                st.markdown("### 📊 Mean Imputation Results Summary")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    render_metric_card("Initial Missing", f"{stats['total_initial_missing']:,}", "Across selected columns", "#0284C7")
                with m2:
                    render_metric_card("Values Filled", f"{stats['total_filled']:,}", f"{stats['overall_fill_rate']}% fill rate", "#10B981")
                with m3:
                    render_metric_card("Remaining Missing", f"{stats['total_remaining_missing']:,}", "Unfilled cells", "#EF4444" if stats['total_remaining_missing'] > 0 else "#10B981")
                with m4:
                    render_metric_card("Features Imputed", f"{len(stats['columns_imputed'])}", "Numerical features", "#38BDF8")

                if stats["completely_missing_cols"]:
                    render_callout(
                        f"Columns {stats['completely_missing_cols']} had 0 valid values; mean could not be computed.",
                        title="Completely Missing Features",
                        alert_type="warning"
                    )

                st.markdown("#### Computed Column Means & Breakdown")
                st.dataframe(stats["summary_table"], use_container_width=True)

                st.markdown("#### Imputed Dataset Preview (First 10 Rows)")
                st.dataframe(st.session_state.mean_df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# 7. COMPARISON DASHBOARD
# -----------------------------------------------------------------------------
elif selected_nav.startswith("7."):
    st.markdown("## ⚖️ Imputation Comparison Dashboard")
    st.caption("Rigorous side-by-side comparison of Forward Fill vs Mean Imputation on identical baseline data.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        col_types = detect_column_types(df)
        num_cols = col_types["numerical"]

        # Ensure both imputations have been performed
        if st.session_state.ffill_df is None or st.session_state.mean_df is None:
            st.warning("⚠️ Both Forward Fill and Mean Imputation must be executed first to generate the comparison.")
            if st.button("⚡ Run Both Imputations Automatically on Numerical Features"):
                with st.spinner("Executing independent Forward Fill and Mean Imputation..."):
                    ts_col = col_types["datetime"][0] if col_types["datetime"] else None
                    ffill_res, ffill_s = apply_forward_fill(
                        df,
                        columns_to_impute=num_cols,
                        timestamp_col=ts_col,
                        sort_chronologically=True if ts_col else False,
                        preserve_original_order=True
                    )
                    mean_res, mean_s = apply_mean_imputation(df, columns_to_impute=num_cols)
                    st.session_state.ffill_df = ffill_res
                    st.session_state.ffill_stats = ffill_s
                    st.session_state.mean_df = mean_res
                    st.session_state.mean_stats = mean_s
                    st.success("Both imputation models generated!")
                    st.rerun()
        else:
            comp_results = compare_imputations(
                original_df=df,
                ffill_df=st.session_state.ffill_df,
                mean_df=st.session_state.mean_df,
                columns=num_cols
            )

            # High-level Comparative KPIs
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Initial Missing", f"{comp_results['total_initial_missing']:,}", "Baseline missing cells", "#0284C7")
            with c2:
                render_metric_card("FFill Filled", f"{comp_results['ffill_total_filled']:,}", f"{comp_results['ffill_fill_rate']}% fill rate", "#38BDF8")
            with c3:
                render_metric_card("Mean Filled", f"{comp_results['mean_total_filled']:,}", f"{comp_results['mean_fill_rate']}% fill rate", "#818CF8")
            with c4:
                render_metric_card("Leading NaNs Left", f"{comp_results['ffill_total_remaining']:,}", "Unfillable by FFill", "#EF4444" if comp_results['ffill_total_remaining'] > 0 else "#10B981")

            st.markdown("---")
            # Side-by-side Chart
            st.plotly_chart(create_comparison_bar_chart(comp_results["comparison_table"]), use_container_width=True)

            st.markdown("---")
            st.markdown("### 📋 Side-by-Side Metrics Table")
            st.dataframe(comp_results["comparison_table"], use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔬 Methodological Comparison: Advantages & Limitations")

            a_col1, a_col2 = st.columns(2)
            with a_col1:
                st.markdown(
                    """
                    #### ⏩ Forward Fill (LOCF)
                    **Advantages:**
                    - ✅ Respects temporal continuity and autocorrelation in sensory time series.
                    - ✅ Preserves natural variance and avoids artificial flat lines over short bursts.
                    - ✅ Computationally lightweight and streaming-friendly.
                    
                    **Limitations:**
                    - ❌ Completely unable to fill leading missing values (row 0).
                    - ❌ Propagates outdated readings across extended multi-day sensor dropouts.
                    """
                )
            with a_col2:
                st.markdown(
                    """
                    #### 📊 Mean Imputation
                    **Advantages:**
                    - ✅ Guaranteed to fill 100% of missing cells (as long as ≥1 valid observation exists).
                    - ✅ Mathematically preserves the global sample mean.
                    - ✅ Simple baseline widely supported across classical statistical packages.
                    
                    **Limitations:**
                    - ❌ Artificially suppresses sample variance and alters standard deviation.
                    - ❌ Distorts covariance and correlation between co-pollutants (e.g. PM2.5 and PM10).
                    - ❌ Destroys chronological diurnal patterns (e.g., rush-hour pollution peaks).
                    """
                )


# -----------------------------------------------------------------------------
# 8. EVALUATION EXPERIMENT
# -----------------------------------------------------------------------------
elif selected_nav.startswith("8."):
    st.markdown("## 🧪 Controlled Evaluation Experiment")
    st.caption("Synthetic missingness benchmarking against known ground truth.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        col_types = detect_column_types(df)
        num_cols = col_types["numerical"]

        # Prominent Academic Disclaimer
        render_callout(
            "**Experimental Protocol Disclaimer:** In real-world data science, the true values of missing sensor readings "
            "are inherently unobserved and cannot be directly measured. To scientifically compare the accuracy of Forward Fill "
            "vs Mean Imputation, this experiment selects **originally complete rows**, artificially masks a random sample "
            "of known readings as NaN, applies each algorithm, and computes quantitative error metrics "
            "(**MAE, MSE, and RMSE**) against the known ground truth.",
            title="Scientific Methodology Note",
            alert_type="info"
        )

        st.markdown("### Experiment Parameters")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            eval_cols = st.multiselect(
                "Features to Benchmark",
                options=num_cols,
                default=num_cols[:6] if len(num_cols) >= 6 else num_cols
            )
        with p_col2:
            mask_slider = st.slider(
                "Artificial Masking Ratio",
                min_value=0.05,
                max_value=0.35,
                value=0.15,
                step=0.05,
                help="Fraction of complete values to artificially mask as missing."
            )
        with p_col3:
            rand_seed = st.number_input("Random Seed (for reproducibility)", min_value=1, max_value=9999, value=42)

        if st.button("🚀 Run Controlled Benchmark Experiment", type="primary"):
            if not eval_cols:
                st.error("Please select at least one numerical feature.")
            else:
                with st.spinner("Simulating synthetic missingness and benchmarking algorithms..."):
                    try:
                        eval_results = run_evaluation_experiment(
                            df,
                            numerical_cols=eval_cols,
                            mask_ratio=mask_slider,
                            random_seed=int(rand_seed)
                        )
                        st.session_state.eval_results = eval_results
                        st.success("✅ Evaluation experiment concluded!")
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")

        # Display evaluation metrics if available
        if "eval_results" in st.session_state and st.session_state.eval_results is not None:
            res = st.session_state.eval_results
            st.markdown("---")
            st.markdown(f"### 🏆 Overall Benchmark Winner: **{res['overall_winner']}**")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                render_metric_card("FFill Overall RMSE", f"{res['overall_ffill_rmse']}", "Root Mean Squared Error", "#38BDF8")
            with r2:
                render_metric_card("Mean Overall RMSE", f"{res['overall_mean_rmse']}", "Root Mean Squared Error", "#A78BFA")
            with r3:
                render_metric_card("FFill Overall MAE", f"{res['overall_ffill_mae']}", "Mean Absolute Error", "#38BDF8")
            with r4:
                render_metric_card("Mean Overall MAE", f"{res['overall_mean_mae']}", "Mean Absolute Error", "#A78BFA")

            st.caption(f"Evaluated on {res['sample_size']} complete rows with {res['total_masked_cells']} artificially masked cells ({int(res['mask_ratio']*100)}% mask ratio).")

            st.markdown("---")
            st.plotly_chart(create_evaluation_bar_chart(res["metrics_table"]), use_container_width=True)

            st.markdown("#### Detailed Regression Error Metrics Table (Lower is Better)")
            st.dataframe(res["metrics_table"], use_container_width=True)


# -----------------------------------------------------------------------------
# 9. DOWNLOAD RESULTS
# -----------------------------------------------------------------------------
elif selected_nav.startswith("9."):
    st.markdown("## 💾 Download Cleaned Datasets & Reports")
    st.caption("Export publication-ready CSV files of all processed datasets and analysis summaries.")

    if st.session_state.raw_df is None:
        render_callout("Please load or upload a dataset first in the 'Upload Dataset' tab.", alert_type="warning")
    else:
        df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df

        st.markdown("### 📥 Available Exports")

        d_col1, d_col2 = st.columns(2)

        # 1. Forward Fill CSV
        with d_col1:
            st.markdown("#### 1. Forward Fill Imputed Dataset")
            st.caption("Filename: `aqi_forward_fill.csv`")
            if st.session_state.ffill_df is not None:
                ffill_csv = st.session_state.ffill_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Forward Fill CSV",
                    data=ffill_csv,
                    file_name="aqi_forward_fill.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"Shape: {st.session_state.ffill_df.shape[0]} rows × {st.session_state.ffill_df.shape[1]} cols")
            else:
                st.info("ℹ️ Forward Fill has not been executed yet.")

        # 2. Mean Imputation CSV
        with d_col2:
            st.markdown("#### 2. Mean Imputed Dataset")
            st.caption("Filename: `aqi_mean_imputed.csv`")
            if st.session_state.mean_df is not None:
                mean_csv = st.session_state.mean_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Mean Imputed CSV",
                    data=mean_csv,
                    file_name="aqi_mean_imputed.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"Shape: {st.session_state.mean_df.shape[0]} rows × {st.session_state.mean_df.shape[1]} cols")
            else:
                st.info("ℹ️ Mean Imputation has not been executed yet.")

        st.markdown("---")

        d_col3, d_col4 = st.columns(2)

        # 3. Final Selected Cleaned CSV
        with d_col3:
            st.markdown("#### 3. Final Selected Cleaned Dataset")
            st.caption("Filename: `aqi_cleaned_final.csv`")

            options = []
            if st.session_state.ffill_df is not None:
                options.append("Forward Fill Result")
            if st.session_state.mean_df is not None:
                options.append("Mean Imputation Result")
            options.append("Preprocessed Dataset (Before Imputation)")

            chosen_target = st.selectbox("Select which version to export as final cleaned dataset:", options)

            target_df = None
            if chosen_target == "Forward Fill Result":
                target_df = st.session_state.ffill_df
            elif chosen_target == "Mean Imputation Result":
                target_df = st.session_state.mean_df
            else:
                target_df = df

            final_csv = target_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Final Cleaned CSV ({chosen_target})",
                data=final_csv,
                file_name="aqi_cleaned_final.csv",
                mime="text/csv",
                use_container_width=True
            )

        # 4. Missing Value Analysis Report CSV
        with d_col4:
            st.markdown("#### 4. Missing Value Diagnostic Report")
            st.caption("Filename: `aqi_missing_value_report.csv`")
            report_df = get_missing_summary(df)
            report_csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Missingness Report CSV",
                data=report_csv,
                file_name="aqi_missing_value_report.csv",
                mime="text/csv",
                use_container_width=True
            )


# -----------------------------------------------------------------------------
# 10. ABOUT PROJECT
# -----------------------------------------------------------------------------
elif selected_nav.startswith("10."):
    st.markdown("## ℹ️ About AQI Missing Value Imputer")
    st.caption("Academic and Engineering Documentation for College Demonstration")

    st.markdown("### 🎓 Academic Project Profile")
    st.write(
        """
        - **Project Name:** AQI Missing Value Imputer – Air Quality Data Preprocessing and Imputation Tool
        - **Domain:** Environmental Informatics, Machine Learning Data Preprocessing, Time-Series Analysis
        - **Primary Objective:** Provide automated, robust, and mathematically sound imputation of sensory missing values
          in air quality datasets, comparing temporal autocorrelation models with central tendency statistics.
        """
    )

    st.markdown("---")
    st.markdown("### 📐 Mathematical Formulations")

    st.markdown("#### 1. Forward Fill (Last Observation Carried Forward - LOCF)")
    st.latex(r"x_t = x_{t-1} \quad \text{if } x_t \text{ is missing and } x_{t-1} \text{ is observed}")
    st.write(
        "For a sequence of missing values, Forward Fill maintains the latest valid measurement. "
        "If `x_0` is unobserved, Forward Fill yields `NaN` due to the absence of preceding observations."
    )

    st.markdown("#### 2. Mean Imputation")
    st.latex(r"\bar{x} = \frac{1}{N_{obs}} \sum_{i \in \text{Obs}} x_i \implies x_j = \bar{x} \quad \forall j \in \text{Miss}")
    st.write("Preserves the sample mean while decreasing variance: `Var(X_imputed) < Var(X_true)`.")

    st.markdown("#### 3. Evaluation Error Metrics")
    st.latex(r"\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|")
    st.latex(r"\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2")
    st.latex(r"\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}")

    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack & Libraries")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            """
            - **Python 3.13+**
            - **Streamlit**: Interactive web dashboard
            - **Pandas**: High-performance data manipulation
            """
        )
    with t2:
        st.markdown(
            """
            - **NumPy**: Numerical vector operations
            - **Plotly**: Interactive responsive charts
            - **Seaborn & Matplotlib**: Statistical sparsity heatmaps
            """
        )
    with t3:
        st.markdown(
            """
            - **Scikit-learn**: Regression evaluation metrics
            - **Pytest**: Modular automated unit testing
            """
        )

# Render unified footer
render_footer()
