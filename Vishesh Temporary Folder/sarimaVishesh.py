import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the CSV file
data = pd.read_csv('/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/Edited Dataset.csv')

# Assuming 'date' is a column and 'product_id' and 'quantity_sold' are columns
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Get unique product IDs
product_ids = data['Product_ID'].unique()

# Dictionary to hold results
forecast_results = {}
mse_results = {}

# Define parameters for EOQ
annual_demand_estimate = 365  # Estimate annual demand (can be adjusted based on historical data)
holding_cost_per_unit_per_year = 10  # Example value, should be defined based on actual costs

# Iterate over each product ID
for product_id in product_ids:
    # Filter data for the 
    product_data = data[data['Product_ID'] == product_id]['quantity_sold']
    
    # Few Params
    reorder_point = data[data['Product_ID'] == product_id]['reorder_point'].values[0]
    supplier_lead_time = data[data['Product_ID'] == product_id]['Supplier_Lead_Time'].values[0]
    starting_inventory = data[data['Product_ID'] == product_id]['Inventory_Level'].values[0]
    transportation_cost = data[data['Product_ID'] == product_id]['Transportation_Cost'].values[0]
    order_cost = transportation_cost
    eoq = np.sqrt((2 * annual_demand_estimate * order_cost) / holding_cost_per_unit_per_year)

    # Split the data into training and testing sets
    train_size = int(len(product_data) * 0.8)
    train, test = product_data[:train_size], product_data[train_size:]
    
    # Fit the SARIMAX model
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)
    
    # Forecasting
    forecast = model_fit.forecast(steps=len(test))
    forecast_results[product_id] = forecast

    # Inventory Management
    inventory_level = starting_inventory
    for day, demand in enumerate(forecast, start=1):
        print(f"Day {day} forecast for Product {product_id}: {demand}")
        inventory_level -= demand  # Reduce inventory by forecasted demand
        print(f"Inventory level: {inventory_level}")
        
        # Check if inventory falls below reorder point
        if inventory_level <= reorder_point:
            inventory_level += eoq
            print(f"Reorder triggered for Product {product_id}")
            print(f"New inventory level after reorder: {inventory_level}")
    
    # Evaluate the forecast
    mse = mean_squared_error(test, forecast)
    mse_results[product_id] = mse
    print(f'Product ID: {product_id}, MSE: {mse}')

    # Plotting the forecast against actual values
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title(f'Forecast vs Actual for Product ID {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.show()

print(forecast_results)
print(mse_results)

# referenced from : https://chatgpt.com/share/57c21c7d-b00e-40e8-a1bd-b46a272d4736