import numpy as np
from generator import generator
from engine import run_trial
import pickle

structures = ['control', 'alternative', 'hierarchy']

# --- set parameters ---
C = 6
# 20 sizes from 10 to 10010
N_range = np.arange(10, 10011, 500)
weight_dist = None

results_bag = {}
if weight_dist:
    print(f"Simulating for C = {C} and weight_dist = {weight_dist.capitalize()}")
else:
    print(f"Simulating for C = {C}")
for strc in structures:
    for N in N_range:
        if weight_dist:
            adj, w = generator(strc, C, N, weight_dist)  
        else:
            adj, w = generator(strc, C, N)
        results = run_trial(adj, w)
        indep = (strc, N)
        results_bag[indep] = results
        with open(f'filepath/C_{C}/{strc}', 'wb') as f:
            pickle.dump(results_bag, f)
        print(f'Saved results of simulation: {strc}; N={N}; C={C}')