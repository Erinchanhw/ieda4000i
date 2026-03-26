"""
Extracted Code from inference.py - Instance 004
Investment portfolio optimization
"""

from coptpy import *
import math

env = Envr()
model = env.createModel("portfolio")

# Investment options
returns = [0.12, 0.08, 0.10, 0.06]  # Expected returns
risks = [0.15, 0.10, 0.12, 0.05]     # Risk levels
min_investment = [0, 5000, 0, 2000]   # Minimum investment per option
total_budget = 50000

# Variables
x = [model.addVar(lb=min_investment[i]/total_budget, ub=1, name=f"allocation_{i}") 
     for i in range(len(returns))]

# Objective: Maximize return
obj = sum(returns[i] * x[i] for i in range(len(returns)))
model.setObjective(obj, sense=COPT.MAXIMIZE)

# Budget constraint
model.addConstr(sum(x[i] for i in range(len(returns))) <= 1, "budget")

# Risk constraint
model.addConstr(sum(risks[i] * x[i] for i in range(len(returns))) <= 0.1, "risk")

# Diversification: at most 40% in any single investment
for i in range(len(returns)):
    model.addConstr(x[i] <= 0.4, f"max_allocation_{i}")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Maximum expected return: {model.objval:.2%}")
    print(f"Expected profit: ${model.objval * total_budget:.2f}")
    for i in range(len(returns)):
        if x[i].x > 0:
            print(f"Investment {i}: ${x[i].x * total_budget:.0f} ({x[i].x:.1%})")
