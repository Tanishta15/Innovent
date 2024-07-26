import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Step 1: Load and Prepare Data
# Load the dataset
df = pd.read_csv('/Users/tanishta/Desktop/Work/Python/Innovent/TATAI2/small_parts_revenue_dataset.csv')

# Inspect the dataset
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Parse dates and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Aggregate sales data by parts and time periods (monthly)
monthly_sales = df.groupby(['Part_ID', pd.Grouper(freq='M')])['Sales'].sum().reset_index()
print(monthly_sales.head())

# Step 2: Seasonal Decomposition
def decompose_time_series(part_id):
    part_sales = monthly_sales[monthly_sales['Part_ID'] == part_id].set_index('Date')['Sales']
    decomposition = seasonal_decompose(part_sales, model='additive')
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition for Part {part_id}')
    plt.show()

decompose_time_series('A1')  # Example part ID, replace with actual part ID

# Step 3: Forecasting Model
def forecast_demand(part_id):
    part_sales = monthly_sales[monthly_sales['Part_ID'] == part_id].set_index('Date')['Sales']
    
    # Fit ARIMA model
    arima_model = ARIMA(part_sales, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=12)
    
    # Fit SARIMA model
    sarima_model = SARIMAX(part_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit()
    sarima_forecast = sarima_fit.forecast(steps=12)
    
    # Plot forecasts
    plt.plot(part_sales, label='Historical Sales')
    plt.plot(arima_forecast, label='ARIMA Forecast', color='red')
    plt.plot(sarima_forecast, label='SARIMA Forecast', color='green')
    plt.legend()
    plt.title(f'Demand Forecast for Part {part_id}')
    plt.show()

forecast_demand('A1')  # Example part ID, replace with actual part ID

# Step 4: Identifying Seasonal Parts
def identify_seasonal_parts(df, threshold=0.1):
    seasonal_parts = []
    for part_id in df['Part_ID'].unique():
        part_sales = df[df['Part_ID'] == part_id].resample('M').sum()['Sales']
        decomposition = seasonal_decompose(part_sales, model='additive')
        if decomposition.seasonal.std() > threshold:
            seasonal_parts.append(part_id)
    return seasonal_parts

seasonal_parts = identify_seasonal_parts(df)
print("Seasonal Parts: ", seasonal_parts)
