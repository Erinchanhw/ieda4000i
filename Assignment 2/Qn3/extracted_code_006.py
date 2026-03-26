"""
Extracted Code from inference.py - Instance 006
Warehouse location problem
"""

from coptpy import *

env = Envr()
model = env.createModel("facility_location")

# Potential warehouse locations and customers
warehouses = range(3)
customers = range(5)

# Fixed costs for opening warehouses
fixed_cost = [1000, 800, 1200]

# Transportation costs from warehouse i to customer j
transport_cost = [
    [40, 50, 45, 55, 60],
    [45, 40, 50, 45, 55],
    [55, 45, 40, 50, 45]
]

# Demand of each customer
demand = [100, 80, 120, 90, 110]

# Variables
y = [model.addVar(vtype=COPT.BINARY, name=f"open_{i}") for i in warehouses]
x = [[model.addVar(lb=0, name=f"ship_{i}_{j}") for j in customers] for i in warehouses]

# Objective: Minimize fixed + transportation costs
obj = sum(fixed_cost[i] * y[i] for i in warehouses) + \
      sum(transport_cost[i][j] * x[i][j] for i in warehouses for j in customers)
model.setObjective(obj, sense=COPT.MINIMIZE)

# Demand constraints
for j in customers:
    model.addConstr(sum(x[i][j] for i in warehouses) >= demand[j], f"demand_{j}")

# Capacity and linking constraints
for i in warehouses:
    model.addConstr(sum(x[i][j] for j in customers) <= 500 * y[i], f"capacity_{i}")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Total cost: ${model.objval:.2f}")
    for i in warehouses:
        if y[i].x > 0.5:
            print(f"Open warehouse {i} (cost: ${fixed_cost[i]})")
            for j in customers:
                if x[i][j].x > 0:
                    print(f"  Ship to customer {j}: {x[i][j].x:.0f} units")
