"""
Extracted Code from inference.py - Instance 002
Transportation optimization problem
"""

from coptpy import *

env = Envr()
model = env.createModel("transportation")

# Supply nodes and demand nodes
supply = [50, 60, 40]  # Supply from factories
demand = [30, 40, 50, 30]  # Demand from warehouses
costs = [
    [8, 6, 10, 9],
    [9, 12, 13, 7],
    [14, 9, 16, 5]
]

# Decision variables: x[i][j] = units from supply i to demand j
x = []
for i in range(len(supply)):
    x.append([model.addVar(lb=0, name=f"x_{i}_{j}") for j in range(len(demand))])

# Objective: Minimize total cost
obj = sum(costs[i][j] * x[i][j] for i in range(len(supply)) for j in range(len(demand)))
model.setObjective(obj, sense=COPT.MINIMIZE)

# Supply constraints
for i in range(len(supply)):
    model.addConstr(sum(x[i][j] for j in range(len(demand))) <= supply[i], f"supply_{i}")

# Demand constraints
for j in range(len(demand)):
    model.addConstr(sum(x[i][j] for i in range(len(supply))) >= demand[j], f"demand_{j}")

# Solve
model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Minimum cost: {model.objval:.2f}")
    for i in range(len(supply)):
        for j in range(len(demand)):
            if x[i][j].x > 0:
                print(f"Ship from {i} to {j}: {x[i][j].x:.0f} units")
