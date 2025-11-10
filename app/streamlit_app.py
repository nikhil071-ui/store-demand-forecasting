import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
# import altair as alt # Altair wasn't used, so I commented it out.

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="SCM Operations Hub",
    page_icon="📦", # Added an icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Loading & Mock Data Generation ---
@st.cache_data
def load_data(filepath):
    """Loads forecast data and creates a rich mock dataset."""
    
    # ==================================================================
    #
    # ❗️❗️❗️ START OF THE FIX ❗️❗️❗️
    #
    # We will separate file reading from validation.
    # The error "Missing column... 'date'" means the uploaded file 
    # doesn't match the app's required format.
    
    # --- Step 1: Read the file, handling only encoding errors ---
    try:
        # Try standard UTF-8 first
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        st.warning("File is not 'UTF-8' encoded. Trying 'latin1' encoding...")
        try:
            # Try 'latin1' (common for European languages / older systems)
            df = pd.read_csv(filepath, encoding='latin1')
        except Exception as e:
            st.error(f"🚨 Failed to read file with both 'utf-8' and 'latin1' encodings. Error: {e}")
            return None, None, None, None
    except FileNotFoundError:
        st.error(f"⚠️ Default forecast file not found at '{filepath}'. Please upload a file.", icon="🚨")
        return None, None, None, None
    except Exception as e:
        # Catch any other unexpected read errors
        st.error(f"🚨 An error occurred while reading the file: {e}")
        return None, None, None, None
        
    # --- Step 2: Validate the file's schema (column names) ---
    REQUIRED_COLUMNS = ['date', 'store', 'item', 'avg_daily', 'std_daily', 'forecast']
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        st.error(f"🚨 The uploaded file is invalid. It's missing required columns: **{', '.join(missing_cols)}**.")
        st.info("Please upload a CSV file with the correct header columns.")
        return None, None, None, None

    # --- Step 3: Parse dates (now that we know the 'date' column exists) ---
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        st.error(f"🚨 Could not parse the 'date' column. Please ensure it's in a standard date format. Error: {e}")
        return None, None, None, None
        
    # ❗️❗️❗️ END OF THE FIX ❗️❗️❗️
    # --- END OF FIX ---


    # --- Master SKU Table (KPI Summary) ---
    # We can now safely run this, knowing the columns exist.
    kpi_summary = df.groupby(['store', 'item']).agg(
        avg_daily=('avg_daily', 'first'),
        std_daily=('std_daily', 'first')
    ).reset_index()

    # Add mock financials, inventory, and supplier
    np.random.seed(42)
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C']
    kpi_summary['supplier'] = np.random.choice(suppliers, len(kpi_summary))
    kpi_summary['unit_cost'] = np.random.uniform(10, 200, len(kpi_summary))
    # Lead time (days)
    kpi_summary['lead_time'] = np.random.choice([7, 10, 14], len(kpi_summary)) 
    kpi_summary['safety_stock'] = (norm.ppf(0.95) * kpi_summary['std_daily'] * np.sqrt(kpi_summary['lead_time'])).round(0)
    kpi_summary['reorder_point'] = (kpi_summary['avg_daily'] * kpi_summary['lead_time']) + kpi_summary['safety_stock']
    kpi_summary['current_inventory'] = kpi_summary['reorder_point'] + np.random.randint(-100, 200, len(kpi_summary))
    kpi_summary['inventory_value'] = kpi_summary['current_inventory'] * kpi_summary['unit_cost']

    # --- Mock Open Purchase Orders Table ---
    po_list = []
    for i, row in kpi_summary.iterrows():
        if np.random.rand() > 0.7: # 30% chance of an open PO
            po_list.append({
                'store': row['store'],
                'item': row['item'],
                'po_id': f"PO_00{np.random.randint(1000, 9999)}",
                'qty': np.random.randint(100, 500),
                'due_date': pd.Timestamp.now() + pd.Timedelta(days=np.random.randint(2, 14))
            })
    open_pos_df = pd.DataFrame(po_list)

    # --- Mock Supplier Performance (OTIF) Table ---
    otif_list = []
    for supplier in suppliers:
        for i in range(52): # 52 weeks of data
            date = pd.Timestamp.now() - pd.Timedelta(weeks=i)
            ot = np.random.uniform(0.85, 0.98)
            if_ = np.random.uniform(0.90, 0.99)
            otif_list.append({
                'supplier': supplier,
                'week': date,
                'On-Time %': ot,
                'In-Full %': if_,
                'OTIF %': ot * if_
            })
    supplier_perf_df = pd.DataFrame(otif_list)
    
    # --- Mock Actual Sales (for Anomaly Detection) ---
    
    # ==================================================================
    #
    # We must explicitly define which columns to merge from kpi_summary
    # to avoid the 'std_daily_x' / 'std_daily_y' naming conflict.
    
    # 1. Get the columns from the original df
    original_df_cols = df.columns
    
    # 2. Find which columns in kpi_summary are NEW (i.e., not in the original df)
    new_kpi_cols = kpi_summary.columns.difference(original_df_cols).tolist()
    
    # 3. Create the final list of columns to merge (the keys + the new columns)
    merge_cols = ['store', 'item'] + new_kpi_cols

    # 4. Join forecast data with master data (merging ONLY the new columns)
    df = df.merge(
        kpi_summary[merge_cols], 
        on=['store', 'item'], 
        how='left'
    )
    # ==================================================================

    # Simulate actuals based on forecast +/- noise
    # Add a check for NaNs in std_daily which can happen if kpi_summary merge fails or data is bad
    df['std_daily'] = df['std_daily'].fillna(0)
    noise = np.random.normal(0, df['std_daily'] * 1.5, len(df))
    df['actual_sales'] = (df['forecast'] + noise).clip(lower=0).astype(int)
    
    # Calculate anomaly bounds (95% CI)
    df['ci_lower'] = (df['forecast'] - 1.96 * df['std_daily']).clip(lower=0)
    df['ci_upper'] = df['forecast'] + 1.96 * df['std_daily']
    
    # Flag anomalies
    df['anomaly'] = (df['actual_sales'] < df['ci_lower']) | (df['actual_sales'] > df['ci_upper'])
    df['anomaly_date'] = df.apply(lambda row: row['date'] if row['anomaly'] else pd.NaT, axis=1)

    return df, open_pos_df, supplier_perf_df, kpi_summary

# --- 3. Session State & Sidebar ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

with st.sidebar:
    st.title("📦 SCM Operations Hub")
    
    # --- NEW FEATURE: File Uploader ---
    uploaded_file = st.file_uploader("Upload Your Forecast CSV", type="csv")
    
    st.divider()

    st.session_state.page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Daily Operations Monitor", "🚚 Replenishment & POs", "📊 Supplier Performance (OTIF)", "🌐 Global Trends (ABC-XYZ)"],
        key="page_radio"
    )
    
    st.divider()
    st.info("This app demonstrates advanced SCM concepts like POH, anomaly detection, and ABC-XYZ analysis.")


# --- Load Data (Conditionally) ---
data_source = uploaded_file if uploaded_file else Path("artifacts/forecast.csv")
df, open_pos_df, supplier_perf_df, kpi_summary = load_data(data_source)

if df is None:
    st.stop()


# --- 4. Page Routing ---

# == PAGE 0: 🏠 HOME (NEW!) ==
if st.session_state.page == "🏠 Home":
    st.title("🏠 Home: Operations Dashboard")
    st.markdown("A high-level overview of your supply chain's health.")

    # --- Calculate Global KPIs ---
    today = df['date'].max()
    
    # KPI 1: Items needing reorder
    items_to_reorder = kpi_summary[kpi_summary['current_inventory'] <= kpi_summary['reorder_point']].shape[0]
    total_items = kpi_summary.shape[0]

    # KPI 2: New anomalies today
    new_anomalies = df[df['anomaly_date'] == today]['anomaly'].sum()
    
    # KPI 3: Total Inventory Value
    total_value = kpi_summary['inventory_value'].sum()

    # KPI 4: Worst Supplier
    current_perf = supplier_perf_df.sort_values('week', ascending=False).groupby('supplier').first()
    worst_supplier = current_perf.sort_values('OTIF %').iloc[0]

    # --- Display KPIs in Columns ---
    st.subheader("At-a-Glance Metrics")
    cols = st.columns(4)
    with cols[0]:
        with st.container(border=True):
            st.metric(
                label="Items to Reorder",
                value=f"{items_to_reorder}",
                delta=f"out of {total_items} total",
                delta_color="inverse"
            )
    with cols[1]:
        with st.container(border=True):
            st.metric(
                label="New Anomalies Today",
                value=f"{new_anomalies}",
                help="Sales that were significantly higher or lower than forecasted."
            )
    with cols[2]:
        with st.container(border=True):
            st.metric(
                label="Total Inventory Value",
                value=f"${total_value:,.0f}"
            )
    with cols[3]:
        with st.container(border=True):
            st.metric(
                label="Worst Supplier (OTIF)",
                value=f"{worst_supplier.name}",
                delta=f"{worst_supplier['OTIF %']:.1%}",
                delta_color="inverse"
            )
    
    st.divider()
    
    # --- Quick-Access Tables ---
    st.subheader("Priority Action Lists")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### ⚠️ Items Below Reorder Point")
        reorder_df = kpi_summary[kpi_summary['current_inventory'] <= kpi_summary['reorder_point']]
        st.dataframe(reorder_df[['store', 'item', 'current_inventory', 'reorder_point', 'supplier']], use_container_width=True)

    with c2:
        st.markdown("#### 🚨 Anomalies Detected Today")
        anomaly_df = df[df['anomaly_date'] == today][['store', 'item', 'forecast', 'actual_sales']]
        st.dataframe(anomaly_df, use_container_width=True)


# == PAGE 1: DAILY OPERATIONS MONITOR (ANOMALY DETECTION) ==
elif st.session_state.page == "📈 Daily Operations Monitor":
    st.title("📈 Daily Operations Monitor")
    st.markdown("Detecting real-time deviations between forecasted and actual sales.")

    # --- UI ENHANCEMENT: Filters in a container ---
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            store_list = df["store"].unique()
            st.session_state.store = st.selectbox("Select Store", store_list, index=0)
        with c2:
            item_list = df[df.store == st.session_state.store]["item"].unique()
            # Add a check for empty item list
            if len(item_list) > 0:
                st.session_state.item = st.selectbox("Select Item", item_list, index=0)
            else:
                st.session_state.item = None
                st.warning("No items found for this store.")
    
    if st.session_state.item: # Only proceed if an item is selected
        st.header(f"Monitor: {st.session_state.item} (Store {st.session_state.store})")
        
        # --- FIX for nan%/Empty Chart ---
        # Filter for the selected SKU
        sku_df_full = df[
            (df.store == st.session_state.store) &
            (df.item == st.session_state.item)
        ].sort_values("date")
        
        # Instead of "last 30 days", take the *first* 30 days from the data file
        # This makes the demo robust even if the data is all in the future
        sku_df = sku_df_full.head(30)
        
        st.subheader("Forecast vs. Actuals (First 30 Days in Data)")
        
        # --- "Hire Me" Chart: Anomaly Detection ---
        fig = go.Figure()
        # Confidence Interval (CI)
        fig.add_trace(go.Scatter(
            x=sku_df['date'], y=sku_df['ci_upper'],
            fill=None, mode='lines', line=dict(color='rgba(0,100,80,0.2)'),
            name='Upper 95% CI'
        ))
        fig.add_trace(go.Scatter(
            x=sku_df['date'], y=sku_df['ci_lower'],
            fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'),
            name='Lower 95% CI'
        ))
        # Forecast Line
        fig.add_trace(go.Scatter(
            x=sku_df['date'], y=sku_df['forecast'],
            mode='lines', line=dict(color='blue', dash='dash'), name='Forecast'
        ))
        # Actual Sales Line
        fig.add_trace(go.Scatter(
            x=sku_df['date'], y=sku_df['actual_sales'],
            mode='lines+markers', line=dict(color='black'), name='Actual Sales'
        ))
        # Anomalies
        anomaly_df = sku_df[sku_df['anomaly'] == True]
        fig.add_trace(go.Scatter(
            x=anomaly_df['date'], y=anomaly_df['actual_sales'],
            mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Anomaly'
        ))
        
        fig.update_layout(
            title="Real-Time Anomaly Detection",
            xaxis_title="Date", yaxis_title="Units Sold"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Forecast Accuracy KPIs ---
        st.subheader("Forecast Performance")
        
        # Calculate MAPE
        sku_df['abs_error'] = (sku_df['actual_sales'] - sku_df['forecast']).abs()
        # Avoid division by zero if actual_sales is 0
        safe_actuals = sku_df[sku_df['actual_sales'] > 0]['actual_sales']
        safe_errors = sku_df[sku_df['actual_sales'] > 0]['abs_error']
        
        if len(safe_actuals) > 0:
            mape = (safe_errors / safe_actuals).replace(np.inf, 0).mean() * 100
            bias_sum_actual = sku_df['actual_sales'].sum()
            if bias_sum_actual > 0:
                 bias = (sku_df['actual_sales'].sum() - sku_df['forecast'].sum()) / bias_sum_actual * 100
            else:
                bias = 0.0 # Avoid division by zero if all actuals are 0
        else:
            mape = 0.0 # No positive actuals to calculate MAPE
            bias = 0.0 # No actuals to calculate bias
        

        # --- UI ENHANCEMENT: KPIs in a container ---
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("MAPE (30-day)", f"{mape:.1f}%",
                        help="Mean Absolute Percentage Error. Lower is better.")
            c2.metric("Bias (30-day)", f"{bias:.1f}%",
                        help="Tendency to over- (negative) or under- (positive) forecast.")
            c3.metric("Anomalies (30-day)", f"{len(anomaly_df)}")

        # --- NEW FEATURE: Anomaly Drill-Down Table ---
        with st.expander("View Anomaly Details"):
            if anomaly_df.empty:
                st.info("No anomalies detected in this period.")
            else:
                st.dataframe(anomaly_df[['date', 'forecast', 'actual_sales', 'ci_lower', 'ci_upper']], use_container_width=True)

# == PAGE 2: REPLENISHMENT & POs (PLANNER'S WORKBENCH) ==
elif st.session_state.page == "🚚 Replenishment & POs":
    st.title("🚚 Replenishment Workbench")
    st.markdown("Projecting future inventory and simulating replenishment actions.")

    # --- UI ENHANCEMENT: Filters in a container ---
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            store_list = df["store"].unique()
            st.session_state.store = st.selectbox("Select Store", store_list, index=0)
        with c2:
            item_list = df[df.store == st.session_state.store]["item"].unique()
            if len(item_list) > 0:
                st.session_state.item = st.selectbox("Select Item", item_list, index=0)
            else:
                st.session_state.item = None
                st.warning("No items found for this store.")
                
    if st.session_state.item: # Only proceed if an item is selected
        st.header(f"Projection: {st.session_state.item} (Store {st.session_state.store})")

        # Get SKU master data
        sku_master = kpi_summary[
            (kpi_summary.store == st.session_state.store) & (kpi_summary.item == st.session_state.item)
        ].iloc[0]

        # --- FIX for TypeError ---
        # Get today's date at midnight (as a full timestamp)
        today_start = pd.Timestamp.now().normalize()
    
        # Get future forecast (next 90 days)
        forecast_df = df[
            (df.store == st.session_state.store) &
            (df.item == st.session_state.item) &
            (df.date >= today_start) # <-- Use normalized timestamp for comparison
        ].sort_values("date").head(90).reset_index() # Must reset index
        
        # Get open POs for this SKU
        sku_pos = open_pos_df[
            (open_pos_df.store == st.session_state.store) &
            (open_pos_df.item == st.session_state.item)
        ]
        
        # --- NEW FEATURE: "What-If" PO Simulation ---
        sim_pos_df = pd.DataFrame()
        with st.expander("🔬 Simulate New PO"):
            with st.form("po_sim_form"):
                st.markdown("Test how a new PO would affect your inventory.")
                sim_qty = st.number_input("PO Quantity", min_value=0, value=100)
                sim_date = st.date_input("PO Due Date", value=pd.Timestamp.now() + pd.Timedelta(days=int(sku_master['lead_time'])))
                
                submitted = st.form_submit_button("Run Simulation")
                
                if submitted:
                    new_po = {
                        'store': st.session_state.store,
                        'item': st.session_state.item,
                        'po_id': "SIM_PO_001",
                        'qty': sim_qty,
                        'due_date': pd.to_datetime(sim_date)
                    }
                    sim_pos_df = pd.DataFrame([new_po])

        # Combine real POs with simulated POs
        all_pos_for_calc = pd.concat([sku_pos, sim_pos_df])

        # --- "Hire Me" Feature: Calculate Projected On-Hand (POH) ---
        poh = []
        current_inv = sku_master['current_inventory']
        
        for i, row in forecast_df.iterrows():
            date = row['date'] # This is a pd.Timestamp (normalized from CSV)
            
            # --- FIX for TypeError ---
            # Check for PO arrivals (from combined real + sim POs)
            # We normalize the 'due_date' column to compare against the normalized 'date'
            arriving_qty = all_pos_for_calc[
                all_pos_for_calc['due_date'].dt.normalize() == date.normalize()
            ]['qty'].sum()
            
            # POH = Yesterday's POH + POs Arriving - Forecasted Demand
            if i == 0:
                poh_today = current_inv + arriving_qty - row['forecast']
            else:
                poh_today = poh[-1] + arriving_qty - row['forecast']
            
            poh.append(poh_today)
            
        forecast_df['poh'] = poh

        # --- "Hire Me" Chart: POH Gantt Chart ---
        st.subheader("Projected On-Hand (POH) Inventory")
        
        fig = go.Figure()

        # POH Line
        fig.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['poh'],
            mode='lines', name='Projected On-Hand (POH)', line=dict(color='blue', width=3)
        ))
        
        # Safety Stock & Reorder Point Lines
        fig.add_hline(y=sku_master['safety_stock'], line=dict(color="red", dash="dot"), name="Safety Stock")
        fig.add_hline(y=sku_master['reorder_point'], line=dict(color="orange", dash="dot"), name="Reorder Point")

        # Add POs as vertical "supply" events
        for i, po in all_pos_for_calc.iterrows():
            is_simulated = po['po_id'].startswith("SIM")
            fig.add_trace(go.Scatter(
                x=[po['due_date'], po['due_date']],
                y=[0, forecast_df['poh'].max() if not forecast_df['poh'].empty else 100], # Handle empty POH
                mode='lines',
                line=dict(color='green' if not is_simulated else 'purple', dash='dash'),
                name=f"{'SIM: ' if is_simulated else 'PO:'} {po['qty']} units"
            ))

        fig.update_layout(
            title="Inventory Projection (90 Days)",
            xaxis_title="Date", yaxis_title="Units",
            yaxis_range=[0, (forecast_df['poh'].max() * 1.2) if not forecast_df['poh'].empty else 100] # Start y-axis at 0
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Actionable Tables ---
        st.subheader("Automated Replenishment Actions")
        
        # Find POH breaches
        breach_df = forecast_df[forecast_df['poh'] <= sku_master['safety_stock']]
        
        if breach_df.empty:
            st.success("✅ **Inventory is Stable.** No projected stockouts in the next 90 days.")
        else:
            first_breach_date = breach_df['date'].min()
            st.error(f"🚨 **Stockout Risk!** Projected to breach safety stock on **{first_breach_date.strftime('%Y-%m-%d')}**.", icon="🚨")
            
            st.markdown("#### POs to Expedite:")
            # Find POs due *after* the first breach
            expedite_pos = sku_pos[sku_pos['due_date'] > first_breach_date]
            if expedite_pos.empty:
                st.info("No open POs due after the breach date are available to expedite.")
            else:
                st.dataframe(expedite_pos)

# == PAGE 3: SUPPLIER PERFORMANCE (OTIF) ==
elif st.session_state.page == "📊 Supplier Performance (OTIF)":
    st.title("📊 Supplier Performance Scorecard")
    st.markdown("Tracking **On-Time, In-Full (OTIF)** performance by supplier. This is critical for understanding reliability and setting correct safety stock levels.")
    
    # --- OTIF Trend Chart ---
    st.subheader("OTIF Performance Over Time")
    
    fig = px.line(
        supplier_perf_df,
        x='week',
        y='OTIF %',
        color='supplier',
        title="Supplier OTIF % (Weekly Trend)"
    )
    fig.update_yaxes(range=[0.7, 1.0], tickformat=".0%") # OTIF is usually high
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Current Performance ---
    st.subheader("Current Supplier Standings")
    
    current_perf = supplier_perf_df.sort_values('week', ascending=False).groupby('supplier').first().reset_index()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Current OTIF % (Latest Week)")
        fig_bar = px.bar(
            current_perf,
            x='supplier',
            y='OTIF %',
            color='supplier',
            range_y=[0.7, 1.0],
            text='OTIF %'
        )
        fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.markdown("#### Performance Breakdown")
        # Format the dataframe for better display
        formatted_perf = current_perf[['supplier', 'On-Time %', 'In-Full %', 'OTIF %']].copy()
        for col in ['On-Time %', 'In-Full %', 'OTIF %']:
            formatted_perf[col] = formatted_perf[col].map('{:.1%}'.format)
        st.dataframe(formatted_perf, hide_index=True, use_container_width=True)

# == PAGE 4: GLOBAL TRENDS (ABC-XYZ) ==
elif st.session_state.page == "🌐 Global Trends (ABC-XYZ)":
    st.title("🌐 Global Trends & ABC-XYZ Analysis")
    st.markdown("Classifying all items by **Value (ABC)** and **Volatility (XYZ)** to define inventory policy.")
    
    # --- "Hire Me" Feature: ABC-XYZ Analysis ---
    # This calculation should use the global kpi_summary
    
    # ABC (Value) - Using Inventory Value for a better ABC
    abc_df = kpi_summary.sort_values(by='inventory_value', ascending=False)
    abc_df['cum_perc'] = abc_df['inventory_value'].cumsum() / abc_df['inventory_value'].sum()
    abc_df['abc_class'] = 'C'
    abc_df.loc[abc_df['cum_perc'] <= 0.80, 'abc_class'] = 'A'
    abc_df.loc[(abc_df['cum_perc'] > 0.80) & (abc_df['cum_perc'] <= 0.95), 'abc_class'] = 'B'
    
    # XYZ (Volatility)
    # Add a small epsilon to avoid division by zero if avg_daily is 0
    abc_df['cov'] = abc_df['std_daily'] / (abc_df['avg_daily'] + 1e-6) # Coeff. of Variation
    abc_df['xyz_class'] = 'Z' # High volatility
    abc_df.loc[abc_df['cov'] <= 0.25, 'xyz_class'] = 'X' # Low volatility
    abc_df.loc[(abc_df['cov'] > 0.25) & (abc_df['cov'] <= 0.5), 'xyz_class'] = 'Y'
    abc_df['abc_xyz_class'] = abc_df['abc_class'] + abc_df['xyz_class']

    # --- ABC-XYZ Heatmap ---
    st.subheader("ABC-XYZ Classification Matrix")
    
    abc_pivot = abc_df.groupby(['abc_class', 'xyz_class'])['item'].count().unstack().fillna(0)
    
    # ==================================================================
    #
    # The syntax error was here. 'C' was missing its opening quote.
    
    fig_abc = px.imshow(
        abc_pivot.reindex(index=['A', 'B', 'C'], columns=['X', 'Y', 'Z']), # <-- FIXED
        text_auto=True,
        color_continuous_scale='Blues',
        title="ABC-XYZ Classification (Count of Items)"
    )
    
    # ==================================================================
    
    st.plotly_chart(fig_abc, use_container_width=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
        ### Inventory Policy:
        - **AX:** High value, stable demand.
        - **AZ:** High value, volatile demand.
        - **CX:** Low value, stable demand.
        - **CZ:** Low value, volatile demand.
        """)
    
    # --- NEW FEATURE: ABC-XYZ Drill-Down ---
    with c2:
        with st.container(border=True):
            st.markdown("#### 🕵️‍♀️ Drill-Down into a Class")
            
            # Create a list of all classes
            classes = [f"{a}{x}" for a in ['A', 'B', 'C'] for x in ['X', 'Y', 'Z']]
            selected_class = st.selectbox("Select a class to inspect", classes)
            
            drill_df = abc_df[abc_df['abc_xyz_class'] == selected_class]
            st.dataframe(
                drill_df[['store', 'item', 'inventory_value', 'cov', 'abc_xyz_class']],
                height=200,
                use_container_width=True
            )