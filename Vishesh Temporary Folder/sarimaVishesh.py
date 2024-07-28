import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the CSV file
data = pd.read_csv('/Users/visheshgoyal/Innovent/supply_chain_optimization_dataset.csv')

# Assuming 'date' is a column and 'product_id' and 'quantity_sold' are columns
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Get unique product IDs
product_ids = data['Product_ID'].unique()

# Dictionary to hold results
forecast_results = {}

# Iterate over each product ID
for product_id in product_ids:
    # Filter data for the product
    product_data = data[data['Product_ID'] == product_id]['Demand']
    
    # Split the data into training and testing sets
    train_size = int(len(product_data) * 0.8)
    train, test = product_data[:train_size], product_data[train_size:]
    
    # Fit the SARIMAX model
    model = SARIMAX(train, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    
    # Forecasting
    forecast = model_fit.forecast(steps=len(test))
    forecast_results[product_id] = forecast
    
    # Evaluate the forecast
    mse = mean_squared_error(test, forecast)
    print(f'Product ID: {product_id}, MSE: {mse}')
    
    # Inventory and supplier management calculations can be added here

print(forecast_results)

# referenced from : https://chatgpt.com/share/57c21c7d-b00e-40e8-a1bd-b46a272d4736