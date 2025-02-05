import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import urllib
from PIL import Image
import re
import csv
from datetime import datetime, timedelta

# Define the path to the results CSV file
results_path = '/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/results.csv'

# Create the results CSV file with headers if it doesn't exist
headers = ['Date', 'Product_ID', 'Inventory_Level', 'EOQ', 'Supplier_Lead_Time']
with open(results_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

# Load the CSV file
data = pd.read_csv('/Users/visheshgoyal/Innovent/ExpandedDataset.csv')

# Assuming 'Date' is a column and 'Product_ID' and 'quantity_sold' are columns
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, dayfirst=True)
data.set_index('Date', inplace=True)

# Get unique product IDs
product_ids = data['Product_ID'].unique()

# Dictionary to hold results
forecast_results = {}
mse_results = {}

# Define parameters for EOQ
annual_demand_estimate = 365  # Estimate annual demand (can be adjusted based on historical data)
holding_cost_per_unit_per_year = 2  # Example value, should be defined based on actual costs

# Iterate over each product ID
for product_id in product_ids:
    # Filter data for the product
    product_data = data[data['Product_ID'] == product_id]['quantity_sold']
    
    # Few Params
    reorder_point = data[data['Product_ID'] == product_id]['reorder_point'].values[0]
    supplier_lead_time = data[data['Product_ID'] == product_id]['Supplier_Lead_Time'].values[0]
    starting_inventory = data[data['Product_ID'] == product_id]['Inventory_Level'].values[0]
    transportation_cost = data[data['Product_ID'] == product_id]['Transportation_Cost'].values[0]
    order_cost = transportation_cost
    safety_stock = data[data['Product_ID'] == product_id]['safety_stock'].values[0]

    # EOQ
    eoq = np.sqrt((2 * annual_demand_estimate * order_cost) / holding_cost_per_unit_per_year)
    eoq = round(eoq)
    print(f'EOQ for Product {product_id}: {eoq}')
    print(f'Safety Stock for Product {product_id}: {safety_stock}')

    # Map data
    city_of_production = data[data['Product_ID'] == product_id]['City_of_Production'].values[0]
    city_of_plant = 'Ghaziabad'
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={city_of_production}&destination={city_of_plant}&key=AIzaSyBA57ryMyTD27c-UkdDy-TfWNtTwJ6bC34"
    response = requests.get(url)
    routedata = response.json()
    if routedata['status'] == 'OK':
        route = routedata['routes'][0]['legs'][0]
        distance = route['distance']['text']
        duration = route['duration']['text']
        print(f"Distance from {city_of_production} to {city_of_plant}: {distance}")
        print(f"Estimated travel time: {duration}")
        polyline = routedata['routes'][0]['overview_polyline']['points']
        static_map_url = (
            f'https://maps.googleapis.com/maps/api/staticmap?size=600x400&maptype=roadmap'
            f'&path=enc:{urllib.parse.quote(polyline)}'
            f'&key=AIzaSyBA57ryMyTD27c-UkdDy-TfWNtTwJ6bC34'
        )

        response = requests.get(static_map_url)
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.show()
        with open(f'/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/Map{product_id}.png', 'wb') as file:
            file.write(response.content)

        # Extract the textual instructions and save them to a file
        with open(f'/Users/visheshgoyal/Innovent/Vishesh Temporary Folder/directions{product_id}.txt', 'w') as file:
            for step in routedata['routes'][0]['legs'][0]['steps']:
                # Extract the plain text instruction
                instruction = step['html_instructions']
                # Strip HTML tags
                clean_instruction = re.sub('<.*?>', '', instruction)
                # Write to file
                file.write(clean_instruction + '\n')
    else:
        print("Error:", routedata['status'])

    # Split the data into training and testing sets
    train_size = int(len(product_data) * 0.8)
    train, test = product_data[:train_size], product_data[train_size:]
    
    # Fit the SARIMAX model
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)
    
    # Forecasting
    forecast = model_fit.forecast(steps=len(test))
    forecast_results[product_id] = forecast

    date_for_forecast = test.index[0].strftime('%d-%m-%Y')
    date_for_forecast = datetime.strptime(date_for_forecast, '%d-%m-%Y')

    # Inventory Management
    inventory_level = starting_inventory
    for day, demand in enumerate(forecast, start=1):
        date_for_forecast = date_for_forecast - timedelta(days=day-1)
        print(f"Day {day} forecast for Product {product_id}: {demand}")
        inventory_level -= demand  # Reduce inventory by forecasted demand
        print(f"Inventory level: {inventory_level}")
        
        # Check if inventory falls below reorder point
        if inventory_level <= reorder_point:
            date_for_forecast = test.index[0].strftime('%d-%m-%Y')
            inventory_level = round(inventory_level)
            # Save results to CSV
            with open(results_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([date_for_forecast, product_id, inventory_level, eoq, supplier_lead_time])
            break
    
    # Evaluate the forecast
    if test.isna().sum() == 0 and forecast.isna().sum() == 0:
        mse = mean_squared_error(test, forecast)
        mse_results[product_id] = mse
        print(f'Product ID: {product_id}, MSE: {mse}')

    # Risk Management
    risk_data = data.groupby('Product_ID')['Supply_Chain_Risk'].mean()
    risk_data.plot(kind='bar', title='Average Supply Chain Risk by Product')
    plt.xlabel('Product_ID')
    plt.ylabel('Risk')
    plt.show()

    # Plotting the forecast against actual values
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title(f'Forecast vs Actual for Product ID {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.show()

# Save the forecast results to CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
forecast_results_df = pd.DataFrame.from_dict(forecast_results, orient='index').transpose()
forecast_results_df.to_csv(f'forecast_results_.csv', index=False)

# Save the MSE results to CSV
mse_results_df = pd.DataFrame(list(mse_results.items()), columns=['Product_ID', 'MSE'])
mse_results_df.to_csv(f'mse_results_.csv', index=False)

print(forecast_results)
print(mse_results)
