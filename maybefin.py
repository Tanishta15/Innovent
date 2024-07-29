import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import folium
import polyline

# Load the CSV file
data = pd.read_csv('/Users/tanishta/Desktop/Python/Innovent/Innovent/Edited Dataset.csv')

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
holding_cost_per_unit_per_year = 2  # Example value, should be defined based on actual costs

# Create a base map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center of India

# Iterate over each product ID
for product_id in product_ids:
    # Filter data for the product
    product_data = data[data['Product_ID'] == product_id]['quantity_sold'].dropna()

    if len(product_data) == 0:
        print(f"No data available for Product {product_id}")
        continue

    # Few Params
    reorder_point = data[data['Product_ID'] == product_id]['reorder_point'].values[0]
    supplier_lead_time = data[data['Product_ID'] == product_id]['Supplier_Lead_Time'].values[0]
    starting_inventory = data[data['Product_ID'] == product_id]['Inventory_Level'].values[0]
    transportation_cost = data[data['Product_ID'] == product_id]['Transportation_Cost'].values[0]
    order_cost = transportation_cost
    safety_stock = data[data['Product_ID'] == product_id]['safety_stock'].values[0]

    # EOQ
    eoq = np.sqrt((2 * annual_demand_estimate * order_cost) / holding_cost_per_unit_per_year)
    print(f'EOQ for Product {product_id}: {eoq}')
    print(f'Safety Stock for Product {product_id}: {safety_stock}')

    # Map data
    city_of_production = data[data['Product_ID'] == product_id]['City_of_Production'].values[0]
    city_of_plant = 'Ghaziabad'
    
    # Fetch route from Google Maps API
    api_key = 'YOUR_API_KEY'  # Make sure to replace with your actual API key
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={city_of_production}&destination={city_of_plant}&key={api_key}"
    response = requests.get(url)
    routedata = response.json()
    
    if routedata['status'] == 'OK':
        route = routedata['routes'][0]['overview_polyline']['points']
        
        try:
            # Decode the polyline
            decoded_points = polyline.decode(route)
            
            # Add route to the map
            folium.PolyLine(decoded_points, color='green', weight=5, opacity=0.7).add_to(m)
            
            # Add markers for the cities
            folium.Marker([18.5204, 73.8567], popup='Pune', icon=folium.Icon(color='blue')).add_to(m)
            folium.Marker([28.6692, 77.4538], popup='Ghaziabad', icon=folium.Icon(color='red')).add_to(m)
            
            distance = routedata['routes'][0]['legs'][0]['distance']['text']
            duration = routedata['routes'][0]['legs'][0]['duration']['text']
            print(f"Distance from {city_of_production} to {city_of_plant}: {distance}")
            print(f"Estimated travel time: {duration}")
        except Exception as e:
            print("Error decoding polyline:", e)
    else:
        print("Error in API request:", routedata.get('status', 'Unknown error'))

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

print(forecast_results)
print(mse_results)

# Save the map to an HTML file
m.save("pune_ghaziabad_route_map.html1")