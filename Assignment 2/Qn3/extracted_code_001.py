"""
Extracted Code from inference.py - Instance 001
Optimization: Maximize profit from pill production
"""

from coptpy import *
import numpy as np

# Create COPT environment
env = Envr()
model = env.createModel("pill_production")

# Decision variables
large_pills = model.addVar(lb=0, name="large_pills")
small_pills = model.addVar(lb=0, name="small_pills")

# Objective: Minimize filler material
model.setObjective(2*large_pills + 1*small_pills, sense=COPT.MINIMIZE)

# Constraints
# Medicinal ingredients constraint
model.addConstr(3*large_pills + 2*small_pills <= 1000, "ingredients")
# Minimum large pills
model.addConstr(large_pills >= 100, "min_large")
# Small pills must be at least 60% of total
model.addConstr(small_pills >= 0.6 * (large_pills + small_pills), "min_small_percent")

# Solve
model.solve()

# Output results
print(f"Status: {model.status}")
if model.status == COPT.OPTIMAL:
    print(f"Optimal value: {model.objval:.2f}")
    print(f"Large pills: {large_pills.x:.0f}")
    print(f"Small pills: {small_pills.x:.0f}")
else:
    print("No optimal solution found")
