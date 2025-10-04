import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from kneed import KneeLocator
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Pharma Corp Analytics Dashboard")
st.title("💊 Pharma Corp Strategic Insights Dashboard")
st.markdown("---")

# --- 1. Data Loading from User Uploaded CSVs ---

@st.cache_data
def load_customer_data():
    """Loads customer data from Customer.csv."""
    try:
        # Assuming Customer.csv is the uploaded file name
        df = pd.read_csv('Customer.csv')
        st.info("Customer data loaded from Customer.csv.")
        return df
    except FileNotFoundError:
        st.error("Error: Customer.csv not found. Please ensure the file is correctly uploaded.")
        return pd.DataFrame()

@st.cache_data
def load_sales_data():
    """Loads sales data from Sales.csv and ensures the date column is correctly parsed."""
    try:
        # Assuming Sales.csv is the uploaded file name
        df = pd.read_csv('Sales.csv')
        
        # --- Date Parsing Logic (Crucial for Time Series) ---
        # Assuming the date column is the first column or explicitly named 'Month'/'Date'
        date_col = None
        if 'Month' in df.columns:
            date_col = 'Month'
        elif 'Date' in df.columns:
            date_col = 'Date'
        elif len(df.columns) > 0:
            # Assume first column is the date column if standard names are missing
            date_col = df.columns[0]
            
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            # Rename the date column to 'Month' for consistency with the rest of the script
            df = df.rename(columns={date_col: 'Month'}) 
            st.info("Sales data loaded from Sales.csv and date column successfully parsed.")
            return df
        else:
            st.error("Sales.csv seems to be empty or missing a recognizable date column.")
            return pd.DataFrame()
            
    except FileNotFoundError:
        st.error("Error: Sales.csv not found. Please ensure the file is correctly uploaded.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error parsing Sales data: {e}. Check if the date column is in a valid format.")
        return pd.DataFrame()

# Load the dataframes
customer_df = load_customer_data()
sales_df = load_sales_data()

# Conditional exit if data loading fails
if customer_df.empty or sales_df.empty:
    st.stop()

# --- 2. Customer Segmentation Analysis ---
st.header("1. Customer Segmentation Analysis")

# --- Preprocessing ---
# Ensure these features exist in the uploaded Customer.csv
features = ['IncomeLevel', 'PurchaseFrequency', 'RecencyDays', 'LifetimeSpend', 'SatisfactionScore', 'Region']

# Check if required columns are present
missing_cols = [col for col in features if col not in customer_df.columns]
if missing_cols:
    st.error(f"Customer data is missing required columns: {', '.join(missing_cols)}. Please check your 'Customer.csv'.")
    st.stop()

df_selected = customer_df[features]
df_encoded = pd.get_dummies(df_selected, drop_first=True)
num_cols = ['PurchaseFrequency', 'RecencyDays', 'LifetimeSpend', 'SatisfactionScore']
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
x_scaler = df_encoded.values

# PCA to 2 components
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(x_scaler)

# PCA DataFrame for plotting
pca_df = pd.DataFrame({'PCA1': x_pca[:, 0], 'PCA2': x_pca[:, 1]})

# --- K-Value Determination ---
st.subheader("1.1 Optimal Cluster ($k$) Determination")

col1, col2 = st.columns(2)

# Elbow Method (Strictly following provided logic)
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(2, 11), wcss, curve='convex', direction='decreasing')
elbow_k = knee.elbow if knee.elbow else 4 # Default to 4

with col1:
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
    ax_elbow.plot(range(2, 11), wcss, marker='o', color='#3498db')
    ax_elbow.axvline(x=elbow_k, color='red', linestyle='--', label=f'Optimal k = {elbow_k}')
    ax_elbow.set_xlabel('Number of clusters (k)')
    ax_elbow.set_ylabel('WCSS (Inertia)')
    ax_elbow.set_title('Elbow Method (WCSS)')
    ax_elbow.legend()
    st.pyplot(fig_elbow)
    st.markdown(f"**Optimal clusters (Elbow/Knee method):** $k = {elbow_k}$")

# Silhouette Method (Strictly following provided logic)
sil_scores = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(x_pca)
    score = silhouette_score(x_pca, labels)
    sil_scores.append(score)
    # Print statement simulating the original code's output
    if k == 4: # Highlight the value from the original request
        st.markdown(f"*Clusters={k}, Silhouette Score={score:.3f}* (Used for analysis)")
    else:
        st.markdown(f"Clusters={k}, Silhouette Score={score:.3f}")

best_k_silhouette = k_values[np.argmax(sil_scores)]

with col2:
    fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
    ax_sil.plot(k_values, sil_scores, marker='o', color='#2ecc71')
    ax_sil.axvline(x=best_k_silhouette, color='red', linestyle='--', label=f'Optimal k = {best_k_silhouette}')
    ax_sil.set_xlabel('Number of clusters (k)')
    ax_sil.set_ylabel('Silhouette Score')
    ax_sil.set_title('Silhouette Method')
    ax_sil.legend()
    st.pyplot(fig_sil)
    st.markdown(f"**Optimal clusters (Silhouette method):** $k = {best_k_silhouette}$")

# --- K-Means Clustering and Profiling (Using k=4 as specified) ---
final_k = 4 
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init='auto')
labels_final = kmeans_final.fit_predict(x_pca)
customer_df['Cluster'] = labels_final
pca_df['Cluster'] = labels_final

st.subheader(f"1.2 Customer Segments Visualization (K-Means with $k = {final_k}$)")

fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=pca_df, 
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    palette='Set2', 
    s=80, 
    edgecolor='white',
    ax=ax_cluster
)
ax_cluster.set_title(f'Customer Segments (k = {final_k})', fontsize=14)
ax_cluster.set_xlabel('PCA Component 1')
ax_cluster.set_ylabel('PCA Component 2')
ax_cluster.legend(title='Cluster')
st.pyplot(fig_cluster)

# Cluster Profiling (Strictly following provided logic)
st.subheader("1.3 Cluster Profile Table and Best Segment Identification")
cluster_profile = customer_df.groupby('Cluster').agg({
    'PurchaseFrequency': 'mean',
    'RecencyDays': 'mean',
    'LifetimeSpend': 'mean',
    'SatisfactionScore': 'mean',
    'IncomeLevel': lambda x: x.mode()[0],
    'Region': lambda x: x.mode()[0],
    'Cluster': 'count'
}).rename(columns={'Cluster': 'CustomerCount'}).reset_index()

# Formatting for display
cluster_profile = cluster_profile.sort_values(by=['LifetimeSpend', 'PurchaseFrequency'], ascending=[False, False])
cluster_profile['LifetimeSpend'] = cluster_profile['LifetimeSpend'].round(2)
cluster_profile['PurchaseFrequency'] = cluster_profile['PurchaseFrequency'].round(1)
cluster_profile['RecencyDays'] = cluster_profile['RecencyDays'].round(1)
cluster_profile['SatisfactionScore'] = cluster_profile['SatisfactionScore'].round(1)

st.dataframe(cluster_profile, hide_index=True)

# Determine Best Segment
if not cluster_profile.empty:
    best_segment_row = cluster_profile.iloc[0]
    st.success(f"**Conclusion: Best Customer Segment**")
    st.markdown(f"""
    The **Best Customer Segment** for Pharma Corp is **Cluster {int(best_segment_row['Cluster'])}**.

    This segment exhibits the highest average **Lifetime Spend** ($\${best_segment_row['LifetimeSpend']}$) and is typically characterized by a common **Income Level** of **{best_segment_row['IncomeLevel']}**.
    """)


# --- 3. Time Series Forecasting Analysis ---
st.header("2. Sales Forecasting and Best Quarter Identification")

# Prepare sales data for time series analysis
# Ensure 'Sales' column exists and 'Month' is the datetime column
if 'Sales' not in sales_df.columns or 'Month' not in sales_df.columns:
    st.error("Sales data is missing 'Sales' or 'Month' column. Please check your 'Sales.csv'.")
    st.stop()
    
df = sales_df.set_index('Month')

# --- 3.1 Stationarity Check (ADF Test) ---
st.subheader("2.1 Stationarity Check (Augmented Dickey-Fuller Test)")

def ad_test(series, name):
    """Performs and formats the ADF test."""
    result = adfuller(series.dropna())
    adf_result = {
        'ADE Statistics': f'{result[0]:.3f}',
        'p-value': f'{result[1]:.3f}',
        'Critical Value (5%)': f'{result[4]["5%"]:.3f}',
        'Conclusion': 'Data is stationary (p-value <= 0.05)' if result[1] <= 0.05 else 'Data is NOT stationary (p-value > 0.05)'
    }
    return adf_result

adf_original = ad_test(df['Sales'], 'Original Series')
df['F-diff'] = df['Sales'] - df['Sales'].shift(1) # First Differencing
adf_fdiff = ad_test(df['F-diff'], 'First Difference')
df['12-diff'] = df['Sales'] - df['Sales'].shift(12) # Seasonal Differencing
adf_12diff = ad_test(df['12-diff'], '12th Difference')

adf_summary = pd.DataFrame({
    'Original Sales': adf_original,
    'First Difference': adf_fdiff,
    '12th Difference': adf_12diff,
}).T

st.dataframe(adf_summary)


# --- 3.2 ACF and PACF Plots ---
st.subheader("2.2 Autocorrelation and Partial Autocorrelation (ACF/PACF) Plots")

col_acf1, col_pacf1 = st.columns(2)
with col_acf1:
    fig_acf1, ax_acf1 = plt.subplots(figsize=(6, 4))
    plot_acf(df['F-diff'].dropna(), ax=ax_acf1, title='ACF for First Difference (d=1)')
    st.pyplot(fig_acf1)
with col_pacf1:
    fig_pacf1, ax_pacf1 = plt.subplots(figsize=(6, 4))
    plot_pacf(df['F-diff'].dropna(), ax=ax_pacf1, title='PACF for First Difference (d=1)')
    st.pyplot(fig_pacf1)

col_acf12, col_pacf12 = st.columns(2)
with col_acf12:
    fig_acf12, ax_acf12 = plt.subplots(figsize=(6, 4))
    plot_acf(df['12-diff'].dropna(), ax=ax_acf12, title='ACF for 12th Difference (D=1)')
    st.pyplot(fig_acf12)
with col_pacf12:
    fig_pacf12, ax_pacf12 = plt.subplots(figsize=(6, 4))
    plot_pacf(df['12-diff'].dropna(), ax=ax_pacf12, title='PACF for 12th Difference (D=1)')
    st.pyplot(fig_pacf12)


# --- 3.3 Model Training and Evaluation ---
st.subheader("2.3 Model Training and Performance")

# Determine split based on the loaded data's index years
all_years = df.index.year.unique()
if len(all_years) < 3:
    st.error("Not enough historical data (less than 3 years) to perform a proper 2-year train/test split.")
    st.stop()
    
train_end_year = all_years[-3]
test_start_year = all_years[-2]
test_end_year = all_years[-1]

train_data = df.loc[:str(train_end_year)]
test_data = df.loc[str(test_start_year):str(test_end_year)]

forecast_steps = len(test_data)
st.info(f"Training Period: {train_data.index.min().strftime('%Y-%m')} to {train_data.index.max().strftime('%Y-%m')} | Testing Period: {test_data.index.min().strftime('%Y-%m')} to {test_data.index.max().strftime('%Y-%m')} ({forecast_steps} months)")

@st.cache_data(show_spinner="Training ARIMA (1, 1, 1)...")
def train_arima(data, order):
    model = SARIMAX(data['Sales'], order=order)
    model_fit = model.fit(disp=False)
    # Ensure prediction steps match test data length
    forecast = model_fit.get_forecast(steps=len(test_data))
    predicted_values = forecast.predicted_mean
    
    # Calculate metrics using test data and predictions
    mae = mean_absolute_error(test_data['Sales'], predicted_values)
    rmse = np.sqrt(mean_squared_error(test_data['Sales'], predicted_values))
    mape = np.mean(np.abs((test_data['Sales'] - predicted_values) / test_data['Sales'])) * 100
    
    return model_fit, predicted_values, mae, rmse, mape

@st.cache_data(show_spinner="Training SARIMA (1, 1, 1)x(1, 1, 1, 12)...")
def train_sarima(data, order, seasonal_order):
    model = SARIMAX(data['Sales'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    # Ensure prediction steps match test data length
    forecast = model_fit.get_forecast(steps=len(test_data))
    predicted_values = forecast.predicted_mean
    
    # Calculate metrics using test data and predictions
    mae = mean_absolute_error(test_data['Sales'], predicted_values)
    rmse = np.sqrt(mean_squared_error(test_data['Sales'], predicted_values))
    mape = np.mean(np.abs((test_data['Sales'] - predicted_values) / test_data['Sales'])) * 100
    
    return model_fit, predicted_values, mae, rmse, mape

# ARIMA Model (Non-Seasonal)
arima_order = (1, 1, 1)
arima_results, arima_predicted, arima_mae, arima_rmse, arima_mape = train_arima(train_data, arima_order)

# SARIMA Model (Seasonal)
sarima_order = (1, 1, 1)
sarima_seasonal = (1, 1, 1, 12)
sarima_results, sarima_predicted, sarima_mae, sarima_rmse, sarima_mape = train_sarima(train_data, sarima_order, sarima_seasonal)


# Performance Table
performance_data = {
    'Model': ['ARIMA (1,1,1)', 'SARIMA (1,1,1)x(1,1,1,12)'],
    'MAE': [f'{arima_mae:.2f}', f'{sarima_mae:.2f}'],
    'RMSE': [f'{arima_rmse:.2f}', f'{sarima_rmse:.2f}'],
    'MAPE (%)': [f'{arima_mape:.2f}%', f'{sarima_mape:.2f}%']
}
performance_df = pd.DataFrame(performance_data)

st.subheader("Model Performance Comparison (Test Period)")
st.dataframe(performance_df, hide_index=True)

# Model Summaries
col_summary1, col_summary2 = st.columns(2)

with col_summary1:
    st.subheader("ARIMA (1,1,1) Summary")
    st.text(arima_results.summary().as_text())

with col_summary2:
    st.subheader("SARIMA (1,1,1)x(1,1,1,12) Summary")
    st.text(sarima_results.summary().as_text())


# --- 3.4 Final Forecast and Best Quarter ---
st.subheader("2.4 Final Sales Forecast and Best Quarter")

# Use SARIMA for final forecast
final_model = SARIMAX(df['Sales'],
                       order=sarima_order,
                       seasonal_order=sarima_seasonal,
                       enforce_stationarity=False,
                       enforce_invertibility=False)

final_results = final_model.fit(disp=False)
# Forecast for 16 steps (4 quarters, or 1 year + 4 months)
forecast_obj = final_results.get_forecast(steps=16)
final_forecast_values = forecast_obj.predicted_mean

# --- Best Quarter Calculation (16 steps = Jan next year to Apr the year after) ---
forecast_df = pd.DataFrame({'Sales': final_forecast_values})
forecast_df['Quarter'] = forecast_df.index.to_period('Q')
quarterly_sales = forecast_df.groupby('Quarter')['Sales'].sum().reset_index()
quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)

# Focus on the first four full quarters (The next full year's forecast)
full_quarters = quarterly_sales.head(4) 
best_quarter_row = full_quarters.loc[full_quarters['Sales'].idxmax()]
best_quarter_label = best_quarter_row['Quarter']
best_quarter_sales = best_quarter_row['Sales']

# Forecast Plot
fig_forecast, ax_forecast = plt.subplots(figsize=(10, 6))
ax_forecast.plot(df['Sales'], label='Historical Sales', color='#3498db')
ax_forecast.plot(final_forecast_values, label='16-Month SARIMA Forecast', color='#e74c3c', linestyle='--')
ax_forecast.set_title('Historical Sales and 16-Month SARIMA Forecast')
ax_forecast.set_xlabel('Date')
ax_forecast.set_ylabel('Sales Value')
ax_forecast.legend()
ax_forecast.grid(True)
st.pyplot(fig_forecast)

# Best Quarter Conclusion
st.success(f"**Conclusion: Best Quarter**")
st.markdown(f"""
Based on the SARIMA forecast for the next full year:
The **Best Quarter** is projected to be **{best_quarter_label}** with total estimated sales of **$\${best_quarter_sales:,.2f}$**.
""")

st.markdown("---")
st.markdown("### 📈 Strategic Summary for Pharma Corp")
st.markdown(f"""
1.  **Customer Focus:** Prioritize efforts on the identified high-value segment (based on the cluster profile table above).
2.  **Sales Strategy:** Prepare resources and inventory for the upcoming **{best_quarter_label}** to capitalize on the projected peak sales period.
""")
