"""
Extracted Code from inference.py - Instance 005
Schedule optimization with time windows
"""

from coptpy import *

env = Envr()
model = env.createModel("scheduling")

# Jobs with processing times and deadlines
jobs = range(5)
processing_time = [3, 2, 4, 1, 3]
deadline = [5, 4, 8, 2, 6]
penalty = [10, 5, 8, 3, 7]

# Variables: start time for each job
start = [model.addVar(lb=0, name=f"start_{j}") for j in jobs]
completion = [model.addVar(name=f"completion_{j}") for j in jobs]
late = [model.addVar(lb=0, name=f"late_{j}") for j in jobs]
y = [[model.addVar(vtype=COPT.BINARY, name=f"order_{i}_{j}") for j in jobs] for i in jobs]

# Objective: Minimize lateness penalty
obj = sum(penalty[j] * late[j] for j in jobs)
model.setObjective(obj, sense=COPT.MINIMIZE)

# Completion time constraints
for j in jobs:
    model.addConstr(completion[j] == start[j] + processing_time[j], f"completion_{j}")
    model.addConstr(late[j] >= completion[j] - deadline[j], f"lateness_{j}")

# Sequencing constraints (ensure no overlap)
for i in jobs:
    for j in jobs:
        if i != j:
            M = 1000
            model.addConstr(start[i] + processing_time[i] <= start[j] + M * (1 - y[i][j]), f"seq_{i}_{j}")
            model.addConstr(start[j] + processing_time[j] <= start[i] + M * y[i][j], f"seq_{j}_{i}")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"Total lateness penalty: {model.objval:.2f}")
    for j in jobs:
        print(f"Job {j}: Start={start[j].x:.0f}, Completion={completion[j].x:.0f}, Lateness={max(0, completion[j].x - deadline[j]):.0f}")
