# Store Demand Forecasting & Inventory Optimization Hub

This project provides an end-to-end data platform for supply chain management. It transforms raw sales data from the Kaggle "Store Item Demand Forecasting" competition into a predictive, interactive decision-support tool.

The pipeline ingests historical data, trains a time-series machine learning model (XGBoost) to generate a 90-day forecast, and then feeds this forecast into a multi-page Streamlit application. This dashboard is designed for supply chain planners to monitor forecast accuracy, manage inventory levels, and make data-driven replenishment decisions.

## 📋 Table of Contents

* [Core Features](#-core-features)
* [Business Problem & Solution](#-business-problem--solution)
* [Technical Stack](#-technical-stack)
* [Project Structure](#-project-structure)
* [Installation & Quickstart](#-installation--quickstart)
* [In-Depth: The Data Pipeline](#-in-depth-the-data-pipeline)
* [In-Depth: The Streamlit Dashboard](#-in-depth-the-streamlit-dashboard)
    * [🏠 Home: Operations Dashboard](#-home-operations-dashboard)
    * [📈 Daily Operations Monitor](#-daily-operations-monitor)
    * [🚚 Replenishment & POs](#-replenishment--pos)
    * [📊 Supplier Performance (OTIF)](#-supplier-performance-otif)
    * [🌐 Global Trends (ABC-XYZ)](#-global-trends-abc-xyz)
* [CSV Upload: File Format Requirements](#-csv-upload-file-format-requirements)
* [Additional Components](#-additional-components)

## ✨ Core Features

* **End-to-End ML Pipeline:** A reproducible pipeline that cleans data, engineers time-series features (lags, rolling averages), and trains an XGBoost model.
* **Inventory Metric Calculation:** Automatically calculates core inventory metrics like **Safety Stock (SS)** and **Reorder Point (ROP)** from the forecast data.
* **Interactive POH Projection:** The app's core feature—a "Projected On-Hand" (POH) chart that visualizes future inventory, showing the impact of demand and incoming supply.
* **"What-If" PO Simulator:** A decision-support tool that allows planners to simulate new purchase orders and see their immediate impact on the future inventory projection.
* **Anomaly Detection:** A monitoring page that tracks `Actual Sales` vs. `Forecast` and flags any significant deviations from the 95% confidence interval.
* **Portfolio Classification:** An ABC-XYZ analysis page that automatically classifies the entire product portfolio by value (ABC) and volatility (XYZ).

## 🎯 Business Problem & Solution

* **Problem:** Traditional inventory management often relies on static spreadsheets, simple averages, and "gut feel." This leads to reactive decision-making, resulting in costly **stockouts** (lost sales) or **overstocks** (high capital cost).
* **Solution:** This platform moves the process from reactive to **proactive**. By leveraging a machine-learning forecast, it provides a more accurate, forward-looking view of demand. The interactive dashboard allows planners to spot risks (like future stockouts) weeks in advance and use the "What-If" simulator to test their decisions before committing to them.

## 🛠️ Technical Stack

* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Web Dashboard:** Streamlit
* **Data Visualization:** Plotly, Plotly Express

## 📂 Project Structure

```
.
├── app/
│   └── streamlit_app.py        # The all-in-one Streamlit application
├── artifacts/
│   ├── forecast.csv            # The final output of the pipeline (ML forecast + inventory stats)
│   └── model_xgb.pkl           # The trained XGBoost model
├── data/
│   ├── raw/                    # Original Kaggle data (train.csv, test.csv)
│   └── processed/              # Cleaned, merged, and intermediate data
├── notebooks/                  # Jupyter notebooks for exploration (EDA)
├── powerbi/
│   └── measures.dax            # Sample DAX measures for a Power BI dashboard
├── sql/
│   └── schema.sql              # A sample star-schema for a data warehouse
├── src/
│   ├── etl.py                  # Cleans and prepares raw data
│   ├── features.py             # Creates time-series features (lags, rolling)
│   ├── train.py                # Trains the XGBoost model
│   ├── forecast.py             # Generates the final forecast.csv
│   └── utils.py                # Utility functions
├── config.yaml                 # Configuration file for the pipeline
└── requirements.txt            # Python dependencies
```

## 🚀 Installation & Quickstart

1.  **Clone the repository:**
    ```bash
    git clone https://your-repo-url/scm-operations-hub.git
    cd scm-operations-hub
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # on Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place raw data:**
    Download `train.csv` and `test.csv` from the [Kaggle Competition](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) and place them in the `data/raw/` directory.

5.  **Run the ML pipeline:**
    This runs all the scripts in order to generate the `artifacts/forecast.csv` file.
    ```bash
    python src/etl.py
    python src/features.py
    python src/train.py
    python src/forecast.py
    ```
6.  **Launch the Streamlit app:**
    ```bash
    streamlit run app/streamlit_app.py
    ```

## 🧠 In-Depth: The Data Pipeline

The core of this project is the `src/` pipeline, which creates the `artifacts/forecast.csv` file used by the Streamlit app.

1.  **`src/etl.py`:**
    * Loads `train.csv` and `test.csv` from `data/raw/`.
    * Cleans data, handles missing values, and merges the files.
    * Calculates the two key historical metrics needed for inventory: `avg_daily` (average sales) and `std_daily` (standard deviation of sales).
    * Saves the processed data to `data/processed/`.

2.  **`src/features.py`:**
    * This is the core of the machine learning model.
    * It creates time-aware features (feature engineering) for the model to learn from.
    * **Lag Features:** Sales from 7, 14, and 30 days ago.
    * **Rolling Window Features:** 7-day and 30-day rolling sales averages and standard deviations.
    * **Date Features:** Day of week, month, quarter, and year.

3.  **`src/train.py`:**
    * Loads the processed data with all the new features.
    * Trains an `XGBRegressor` (XGBoost) model on this data.
    * Uses a time-series-aware validation strategy to evaluate the model.
    * Saves the final, trained model object as `artifacts/model_xgb.pkl`.

4.  **`src/forecast.py`:**
    * Loads the trained `model_xgb.pkl`.
    * Generates a 90-day future date range.
    * Creates the same lag/rolling/date features for this future period.
    * Runs `model.predict()` on the future features to get the 90-day forecast.
    * Merges this forecast with the `avg_daily` and `std_daily` stats from `etl.py`.
    * Saves the final, complete file to `artifacts/forecast.csv`.

## 🖥️ In-Depth: The Streamlit Dashboard

The Streamlit app is the user-facing tool for planners. It reads `artifacts/forecast.csv` (or an uploaded file) and provides several pages for analysis.

### 🏠 Home: Operations Dashboard

This is the "Mission Control" landing page. It gives a high-level summary of the entire operation's health.
* **At-a-Glance Metrics:** KPIs for **"Items to Reorder"** (SKUs below their reorder point), **"New Anomalies Today"**, **"Total Inventory Value"**, and the **"Worst Performing Supplier"** (by OTIF %).
* **Priority Action Lists:** Two tables that show the *specific* items that need immediate attention:
    1.  Items Below Reorder Point
    2.  Anomalies Detected Today

### 📈 Daily Operations Monitor

This page is for demand planners to validate the forecast's performance against reality.
* **Anomaly Detection Chart:** A Plotly chart that plots `Actual Sales` vs. `Forecast`. It also plots a 95% confidence interval (calculated from `std_daily`). Any "actual" sales that fall outside this band are flagged as red 'X' anomalies, indicating a significant, unexpected event.
* **Forecast Performance KPIs:**
    * **MAPE (Mean Absolute Percentage Error):** Measures the average *magnitude* of forecast errors. Lower is better.
    * **Forecast Bias:** Measures the *direction* of the error. A positive bias means the model consistently under-forecasts (sales are higher than predicted). A negative bias means it over-forecasts.

### 🚚 Replenishment & POs

This is the most powerful planning tool in the app. It projects future inventory and lets planners test their decisions.
* **Projected On-Hand (POH) Chart:** This is the core of the page. It calculates POH for the next 90 days using the formula:
    `POH = (Yesterday's POH) - (Forecasted Demand) + (Arriving POs)`
* **Inventory Policy Lines:** The chart plots the **Safety Stock** and **Reorder Point** as horizontal lines. This allows a planner to instantly see *when* the POH is projected to drop into the "danger zone" (below safety stock).
* **"What-If" PO Simulation:** An expandable section with a form where a planner can enter a **Quantity** and **Due Date** for a *new, simulated* PO. Clicking "Run Simulation" instantly adds this simulated PO to the POH chart, showing the planner exactly how their action will fix a future stockout.
* **Automated Actions:** If a stockout is projected, the app provides a table of "POs to Expedite," recommending which existing POs could be pulled in to solve the problem.

### 📊 Supplier Performance (OTIF)

This page serves as a scorecard for procurement and supplier management.
* **OTIF % (On-Time, In-Full):** This is the key metric for supplier reliability.
* **OTIF Trend Chart:** A line chart showing the OTIF performance of all suppliers over the past 52 weeks.
* **Current Supplier Standings:** A bar chart and data table that ranks suppliers by their most recent weekly performance, making it easy to see who is performing well and who is not.

### 🌐 Global Trends (ABC-XYZ)

This is a strategic page for inventory managers to set policy for the entire product portfolio.
* **ABC-XYZ Classification:** This tool automatically segments all products based on two dimensions:
    1.  **ABC Analysis (Value):** "A" items are high-value (top 80% of sales), "B" items are medium-value (next 15%), "C" items are low-value (bottom 5%).
    2.  **XYZ Analysis (Volatility):** "X" items have stable, predictable demand (low volatility), "Y" items have moderate volatility, "Z" items have highly volatile, unpredictable demand.
* **Classification Matrix:** A heatmap that shows the count of items in each segment (e.g., **AX** = high-value, stable; **CZ** = low-value, volatile).
* **Drill-Down Table:** An interactive table that allows the user to select a class (like "AX") and see a full list of all items that fall into that category.

## ⚠️ CSV Upload: File Format Requirements

The app includes a sidebar file uploader to use your own data. For the app to function, your CSV file **must** contain the following columns with these exact names and data types:

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`date`** | Date/Datetime | The date of the forecast. | `2025-11-11` |
| **`store`** | String or Int | The unique identifier for the store. | `1` |
| **`item`** | String or Int | The unique identifier for the item. | `1` |
| **`avg_daily`** | Numeric | The historical average daily sales for this store/item. | `20.5` |
| **`std_daily`** | Numeric | The historical standard deviation of daily sales. | `5.2` |
| **`forecast`** | Numeric | The forecasted sales number for this date. | `22` |

**Note:** The app will fail to load and will display a specific error message if any of these columns are missing or incorrectly named.

## 📦 Additional Components

* **`sql/schema.sql`:** A sample SQL (PostgreSQL dialect) script to create a star-schema data warehouse. This shows how you would store this data in a production database for BI analysis.
* **`powerbi/measures.dax`:** A file containing ready-to-use DAX measures (e.g., `Forecast MTD`, `Sales YTD`, `POH`) that can be pasted into a Power BI or Excel data model to analyze the same data.