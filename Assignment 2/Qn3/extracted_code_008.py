"""
Extracted Code from inference.py - Instance 008
Blending optimization problem
"""

from coptpy import *

env = Envr()
model = env.createModel("blending")

# Ingredients
ingredients = range(4)
cost = [2.5, 1.8, 3.2, 2.0]

# Chemical composition (percentage)
composition = [
    [0.8, 0.1, 0.05, 0.05],  # Component A
    [0.2, 0.6, 0.1, 0.1],    # Component B
    [0.05, 0.1, 0.7, 0.15]    # Component C
]

# Required composition in final product
required = [0.4, 0.35, 0.25]  # For A, B, C

# Availability of each ingredient
availability = [100, 150, 80, 120]

# Variables
x = [model.addVar(lb=0, ub=availability[i], name=f"ingredient_{i}") for i in ingredients]

# Objective: Minimize cost
obj = sum(cost[i] * x[i] for i in ingredients)
model.setObjective(obj, sense=COPT.MINIMIZE)

# Total production
total = model.addVar(name="total")
model.addConstr(total == sum(x[i] for i in ingredients), "total_production")

# Composition constraints
for c in range(len(required)):
    model.addConstr(sum(composition[c][i] * x[i] for i in ingredients) >= required[c] * total, 
                    f"min_{c}")
    model.addConstr(sum(composition[c][i] * x[i] for i in ingredients) <= (required[c] + 0.05) * total,
                    f"max_{c}")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Minimum cost: ${model.objval:.2f}")
    print(f"Total production: {total.x:.1f} units")
    for i in ingredients:
        if x[i].x > 0:
            print(f"Ingredient {i}: {x[i].x:.1f} units (${cost[i]}/unit)")
