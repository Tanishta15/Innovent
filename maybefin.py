import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model

# Create a small sample dataset
data = {
    'Date': pd.date_range(start='2020-01-01', periods=24, freq='M').tolist() * 3,
    'Product_ID': ['A1']*24 + ['B1']*24 + ['C1']*24,
    'Demand': [50, 60, 65, 70, 80, 90, 55, 75, 85, 95, 105, 110, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130] * 3,
    'Supplier_Lead_Time': [2, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 2, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2, 3] * 3,
    'Supply_Chain_Risk': [0.1, 0.2, 0.15, 0.2, 0.25, 0.2, 0.1, 0.2, 0.25, 0.2, 0.1, 0.2, 0.1, 0.2, 0.15, 0.2, 0.25, 0.2, 0.1, 0.2, 0.25, 0.2, 0.1, 0.2] * 3
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('small_parts_revenue_dataset.csv', index=False)

# Load the dataset
df = pd.read_csv('small_parts_revenue_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Inspect the dataset
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Improved Accuracy in Demand Forecasting
def demand_forecasting(product_id):
    product_demand = df[df['Product_ID'] == product_id]['Demand']
    model = ARIMA(product_demand, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    plt.figure(figsize=(10, 5))
    plt.plot(product_demand, label='Historical Demand')
    plt.plot(forecast, label='Forecasted Demand', color='red')
    plt.legend()
    plt.title(f'Demand Forecasting for Product {product_id}')
    plt.show()

demand_forecasting('A1')

# Reduced Inventory Holding Costs and Minimized Stockouts
def calculate_eoq_and_safety_stock(product_id):
    product_demand = df[df['Product_ID'] == product_id]['Demand']
    annual_demand = product_demand.sum()
    ordering_cost = 100  # Cost per order
    holding_cost = 2  # Holding cost per unit per year
    EOQ = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    
    lead_time = df[df['Product_ID'] == product_id]['Supplier_Lead_Time'].mean()
    demand_std = product_demand.std()
    safety_stock = lead_time * demand_std
    
    print(f'EOQ for Product {product_id}: {EOQ}')
    print(f'Safety Stock for Product {product_id}: {safety_stock}')

calculate_eoq_and_safety_stock('A1')

# Enhanced Supplier Performance and Collaboration
def supplier_performance():
    performance = df.groupby('Product_ID')['Supplier_Lead_Time'].std()
    print('Supplier Performance (Lead Time Variability):')
    print(performance)

supplier_performance()

# Optimized Transportation Routes and Reduced Logistics Costs
def route_optimization():
    def create_data_model():
        data = {}
        data['distance_matrix'] = [
            [0, 1600, 1700, 1300,1400,1800,1500], 

            [1600,0,1400,1700,700,800,200], 

            [1700,1400,0,200,1300,1500,1300], 

            [1300,1700,200,0,1500,1600,1600], 

            [1400,700,1300,1500,0,500,100], 

	        [1800,800,1500,1600,500,0,600], 

            [1500,200,1300,1600,100,600,0],
        ]
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print('Objective: {}'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route for vehicle 0:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)

route_optimization()

# Increased Production Efficiency and Flexibility
def production_scheduling():
    jobs_data = [
        [(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)]
    ]
    model = cp_model.CpModel()
    num_machines = 3
    num_jobs = len(jobs_data)
    all_machines = range(num_machines)
    all_jobs = range(num_jobs)
    all_tasks = {}
    for job in all_jobs:
        for task_id, task in enumerate(jobs_data[job]):
            machine, duration = task
            suffix = '_%i_%i' % (job, task_id)
            all_tasks[job, task_id] = model.NewIntervalVar(
                model.NewIntVar(0, 20, 'start' + suffix),
                model.NewIntVar(duration, duration, 'duration' + suffix),
                model.NewIntVar(0, 20, 'end' + suffix), 'interval' + suffix)

    for machine in all_machines:
        machine_tasks = []
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                if task[0] == machine:
                    machine_tasks.append(all_tasks[job, task_id])
        model.AddNoOverlap(machine_tasks)

    obj_var = model.NewIntVar(0, 1000, 'makespan')
    model.AddMaxEquality(obj_var, [all_tasks[job, len(jobs_data[job])-1].EndExpr() for job in all_jobs])
    model.Minimize(obj_var)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        for job in all_jobs:
            for task_id, task in enumerate(jobs_data[job]):
                machine, duration = task
                start = solver.Value(all_tasks[job, task_id].StartExpr())
                end = solver.Value(all_tasks[job, task_id].EndExpr())
                print('Job %i Task %i: Start %i End %i' % (job, task_id, start, end))

production_scheduling()

# Mitigated Supply Chain Risks and Improved Resilience
def risk_assessment():
    risk_data = df.groupby('Product_ID')['Supply_Chain_Risk'].mean()
    risk_data.plot(kind='bar', title='Average Supply Chain Risk by Product')
    plt.xlabel('Product_ID')
    plt.ylabel('Risk')
    plt.show()

risk_assessment()
