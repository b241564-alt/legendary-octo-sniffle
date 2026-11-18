# 🌫️ AQI Missing Value Imputer – Air Quality Data Preprocessing & Imputation Tool

An end-to-end Machine Learning preprocessing and statistical imputation web application designed for environmental sensor networks and air quality index (AQI) time-series monitoring.

![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-darkblue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-purple.svg)
![Tests](https://img.shields.io/badge/Tests-18%2F18%20Passing-brightgreen.svg)

---

## 📌 1. Project Overview & Problem Statement

Continuous Ambient Air Quality Monitoring Systems (CAAQMS) record critical criteria pollutants:
- **$\text{PM}_{2.5}$ & $\text{PM}_{10}$** (Particulate Matter)
- **$\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$** (Trace Gases)
- Meteorological covariates (**Temperature**, **Relative Humidity**)

In real-world telemetry networks, missing data is unavoidable due to:
1. **Sensor drift, degradation, and periodic zero/span calibration blackouts**
2. **Transient power failures and transmission packet dropouts**
3. **Adverse meteorological extremes (condensation, inlet clogging)**

Most downstream machine learning models (e.g., Random Forests, XGBoost, LSTMs) cannot operate on datasets with `NaN` values. Naive deletion (listwise dropping) discards valuable temporal sequences and induces bias.

**AQI Missing Value Imputer** provides a rigorous, interactive, and transparent pipeline to diagnose missingness mechanisms, apply independent imputation strategies, conduct controlled synthetic benchmarking against ground truth, and export publication-ready datasets.

---

## 🏗️ 2. Architectural Diagram

```
+-------------------------------------------------------------+
|                      User Ingestion                         |
|   (Custom CSV Upload OR Bundled Delhi CAAQMS Dataset)       |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              Schema Validation & Diagnostics                |
|  - Dtype Identification (Numerical, Datetime, Categorical)  |
|  - Missing Count & Sparsity Ratios                          |
|  - Interactive Heatmap (Seaborn) & Bar Charts (Plotly)      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   Data Preprocessing                        |
|  - Exact Duplicate Row Detection & Optional Removal         |
|  - Feature Selection & Subsetting                           |
|  - Chronological Sorting by Timestamp                       |
+---------------+-----------------------------+---------------+
                |                             |
                v                             v
+-------------------------------+ +-------------------------------+
|      Forward Fill (LOCF)      | |        Mean Imputation        |
| - Last Observation Carried    | | - Parametric Central Tendency |
|   Forward across time series  | | - Arithmetic sample mean fill |
| - Leading NaN edge handling   | | - Variance preservation check |
+---------------+---------------+ +---------------+---------------+
                |                                 |
                +---------------+-----------------+
                                |
                                v
+-------------------------------------------------------------+
|               Independent Comparison Engine                 |
|  - Values filled vs remaining missing side-by-side          |
|  - Autocorrelation vs Variance trade-off analysis           |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             Controlled Evaluation Benchmark                 |
|  - Controlled masking on complete ground-truth records      |
|  - Error quantification: MAE, MSE, and RMSE                 |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Export Engine                           |
|  - `aqi_forward_fill.csv`                                   |
|  - `aqi_mean_imputed.csv`                                   |
|  - `aqi_cleaned_final.csv`                                  |
|  - `aqi_missing_value_report.csv`                           |
+-------------------------------------------------------------+
```

---

## 🔬 3. Imputation Methodologies

### A. Forward Fill (Last Observation Carried Forward - LOCF)
- **Concept:** Propagates the previous valid sensor reading forward until a subsequent valid observation occurs:
  $$x_t = x_{t-1} \quad \text{if } x_t \text{ is missing and } x_{t-1} \text{ is observed}$$
- **Strengths:** Exploits the strong short-term temporal autocorrelation inherent in atmospheric concentrations. Does not artificially drag variance toward a static center.
- **Critical Edge Case:** If missing values exist at row 0 (the beginning of the time series), Forward Fill **cannot** impute them because no historical baseline exists. Our application detects and transparently reports these leading missing values rather than falsely claiming 100% completion.

### B. Mean Imputation
- **Concept:** Replaces missing entries with the arithmetic mean of observed samples:
  $$\bar{x} = \frac{1}{N_{\text{obs}}} \sum_{i \in \text{Obs}} x_i \implies x_j = \bar{x} \quad \forall j \in \text{Miss}$$
- **Strengths:** Guaranteed to impute 100% of missing cells (provided at least one valid observation exists). Exactly preserves the global sample mean.
- **Limitations:** Artificially suppresses variance ($\text{Var}(X_{\text{imp}}) < \text{Var}(X_{\text{true}})$), attenuates covariance between correlated pollutants (e.g. $\text{PM}_{2.5}$ and $\text{PM}_{10}$), and flattens diurnal peaks.

---

## 🧪 4. Controlled Benchmark Experiment

In real-world deployments, true values of missing sensor readings are unobservable. To benchmark algorithms scientifically:
1. Extract rows that are originally complete across selected features (ground truth).
2. Artificially mask a controlled fraction (e.g., 15%) as `NaN`.
3. Apply Forward Fill and Mean Imputation independently.
4. Calculate standard regression loss metrics against the hidden ground truth:
   - **Mean Absolute Error (MAE):** $\frac{1}{N} \sum |y_i - \hat{y}_i|$
   - **Mean Squared Error (MSE):** $\frac{1}{N} \sum (y_i - \hat{y}_i)^2$
   - **Root Mean Squared Error (RMSE):** $\sqrt{\text{MSE}}$

---

## 💻 5. Local Setup & Execution

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- `pip` package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/aqi-missing-value-imputer.git
   cd aqi-missing-value-imputer
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the automated test suite:**
   ```bash
   pytest tests -v
   ```

5. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ 6. Online Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and click **"Create app"**.
4. Select the repository, set Branch to `main` (or `master`), and Main file path to `app.py`.
5. Click **"Deploy"**. The app will build automatically using `requirements.txt`.

### Option 2: Render
1. Create a free account at [render.com](https://render.com).
2. Click **New +** > **Web Service** and connect your GitHub repository.
3. Configure:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **Deploy Web Service**.

---

## 📂 7. Project Structure

```
aqi-missing-value-imputer/
├── app.py                     # Streamlit application entry point & page router
├── requirements.txt           # Production dependencies
├── .gitignore                 # Standard Python & environment exclusions
├── README.md                  # Academic documentation & deployment manual
├── .streamlit/
│   └── config.toml            # UI theme parameters & server configurations
├── data/
│   └── sample_aqi_data.csv    # 120-hour multi-pollutant dataset (Delhi CAAQMS)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV parsing, encodings, and schema detection
│   ├── missing_analysis.py    # Missingness statistics, Plotly bar, Seaborn heatmap
│   ├── preprocessing.py       # Duplicate handling, column selection, time sorting
│   ├── imputation.py          # Forward fill & mean imputation algorithms
│   ├── evaluation.py          # Synthetic missingness simulation & metrics (MAE, RMSE)
│   └── ui_components.py       # Modern CSS design system, KPI cards, alerts
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_missing_analysis.py
    ├── test_preprocessing.py
    ├── test_imputation.py
    └── test_evaluation.py
```

---

## 📄 8. License & Academic Citation

Developed for academic demonstration and machine learning research in environmental data preprocessing. Free to use and extend under the MIT License.
