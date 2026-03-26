"""
Extracted Code from inference.py - Instance 009
Cutting stock optimization
"""

from coptpy import *

env = Envr()
model = env.createModel("cutting_stock")

# Roll length: 100 units
roll_length = 100

# Orders: (length, demand)
orders = [(30, 15), (40, 10), (50, 8), (60, 12)]
patterns = []

# Generate simple patterns
for i in range(len(orders)):
    pattern = [0] * len(orders)
    pattern[i] = int(roll_length / orders[i][0])
    patterns.append(pattern)

# Variables: number of rolls cut using each pattern
y = [model.addVar(lb=0, vtype=COPT.INTEGER, name=f"pattern_{i}") for i in range(len(patterns))]

# Objective: Minimize total rolls used
obj = sum(y[i] for i in range(len(patterns)))
model.setObjective(obj, sense=COPT.MINIMIZE)

# Demand constraints
for j in range(len(orders)):
    demand = orders[j][1]
    model.addConstr(sum(patterns[i][j] * y[i] for i in range(len(patterns))) >= demand, 
                    f"demand_{j}")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Minimum rolls needed: {model.objval:.0f}")
    for i in range(len(patterns)):
        if y[i].x > 0:
            print(f"Pattern {i}: {y[i].x:.0f} rolls - cuts: {patterns[i]}")
