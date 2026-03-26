"""
Extracted Code from inference.py - Instance 003
Production planning with setup costs
"""

from coptpy import *

env = Envr()
model = env.createModel("production_planning")

# Parameters
months = range(4)
demand = [100, 150, 200, 180]
production_cost = [2, 2.5, 3, 2.8]
setup_cost = [50, 60, 70, 55]
holding_cost = 0.5
capacity = 250

# Variables
x = [model.addVar(lb=0, ub=capacity, name=f"prod_{t}") for t in months]
y = [model.addVar(vtype=COPT.BINARY, name=f"setup_{t}") for t in months]
inventory = [model.addVar(lb=0, name=f"inv_{t}") for t in months]

# Objective
obj = sum(production_cost[t] * x[t] + setup_cost[t] * y[t] + holding_cost * inventory[t] 
          for t in months)
model.setObjective(obj, sense=COPT.MINIMIZE)

# Constraints
# Inventory balance
model.addConstr(inventory[0] == x[0] - demand[0])
for t in range(1, len(months)):
    model.addConstr(inventory[t] == inventory[t-1] + x[t] - demand[t])

# Capacity and setup constraints
for t in months:
    model.addConstr(x[t] <= capacity * y[t], f"capacity_{t}")

# Non-negative inventory at end
model.addConstr(inventory[-1] >= 0)

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Total cost: {model.objval:.2f}")
    for t in months:
        print(f"Month {t}: Produce {x[t].x:.0f}, Setup={y[t].x:.0f}, Inventory={inventory[t].x:.0f}")
