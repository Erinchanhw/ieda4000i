"""
Extracted Code from inference.py - Instance 007
Resource allocation with multiple constraints
"""

from coptpy import *

env = Envr()
model = env.createModel("resource_allocation")

# Activities and resources
activities = range(4)
resources = range(3)

# Profit per activity
profit = [20, 15, 25, 18]

# Resource consumption per activity
consumption = [
    [2, 3, 1, 2],  # Resource 1
    [1, 2, 2, 1],  # Resource 2
    [3, 1, 2, 3]   # Resource 3
]

# Available resources
available = [100, 80, 120]

# Decision variables
x = [model.addVar(lb=0, name=f"activity_{a}") for a in activities]

# Objective: Maximize profit
obj = sum(profit[a] * x[a] for a in activities)
model.setObjective(obj, sense=COPT.MAXIMIZE)

# Resource constraints
for r in resources:
    model.addConstr(sum(consumption[r][a] * x[a] for a in activities) <= available[r], f"resource_{r}")

# Additional business constraints
model.addConstr(x[0] <= 2 * x[1], "ratio_constraint")
model.addConstr(x[2] >= x[3], "balance_constraint")
model.addConstr(x[0] + x[2] <= 50, "max_combined")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Maximum profit: ${model.objval:.2f}")
    for a in activities:
        print(f"Activity {a}: {x[a].x:.1f} units")
    
    # Check resource usage
    for r in resources:
        used = sum(consumption[r][a] * x[a].x for a in activities)
        print(f"Resource {r}: used {used:.1f} / {available[r]} available")
