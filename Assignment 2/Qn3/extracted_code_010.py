"""
Extracted Code from inference.py - Instance 010
Project scheduling with resource constraints
"""

from coptpy import *

env = Envr()
model = env.createModel("project_scheduling")

# Activities and durations
activities = range(8)
duration = [3, 2, 4, 2, 3, 1, 2, 3]

# Precedence constraints: (before, after)
precedence = [(0, 2), (1, 2), (2, 4), (2, 5), (3, 6), (4, 7), (5, 7), (6, 7)]

# Resource requirements per activity
resource_req = [2, 1, 3, 2, 2, 1, 2, 1]
resource_limit = 4

# Variables
start = [model.addVar(lb=0, name=f"start_{a}") for a in activities]

# Objective: Minimize makespan
makespan = model.addVar(name="makespan")
model.setObjective(makespan, sense=COPT.MINIMIZE)

# Duration constraints
for a in activities:
    model.addConstr(start[a] + duration[a] <= makespan, f"completion_{a}")

# Precedence constraints
for before, after in precedence:
    model.addConstr(start[before] + duration[before] <= start[after], f"prec_{before}_{after}")

# Resource constraints (simplified: max at any time)
# This is a simplification - actual would need time discretization
for t in range(20):  # Check at discrete time points
    resource_used = 0
    for a in activities:
        # Check if activity is active at time t
        active = model.addVar(vtype=COPT.BINARY)
        model.addConstr(active >= (t - start[a]) / duration[a], f"active_{a}_{t}_lb")
        model.addConstr(active <= 1 + (t - start[a]) / duration[a], f"active_{a}_{t}_ub")
        resource_used += resource_req[a] * active
    model.addConstr(resource_used <= resource_limit, f"resource_t_{t}")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Minimum project duration: {makespan.x:.0f}")
    for a in activities:
        print(f"Activity {a}: Start={start[a].x:.0f}, End={start[a].x + duration[a]:.0f}")
