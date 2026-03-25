import os
import numpy as np
from generator import generator
from engine import run_trial
import pickle

seed = np.random.default_rng(21) # seed for reproducibility in run_trial

structures = ['control', 'alternative', 'hierarchy']

# --- set parameters ---
C = 6
N_range = np.arange(10, 10011, 500) # 20 sizes from 10 to 10010
weight_dist = None

# --- ensure directory exists ---
save_dir = f'results/C_{C}'
os.makedirs(save_dir, exist_ok=True) 

# --- simulation loop ---

if weight_dist:
    print(f"Simulating for C = {C} and weight_dist = {weight_dist.capitalize()}")
else:
    print(f"Simulating for C = {C}")

for strc in structures:
    # 1. Reset the dictionary for each structure so data doesn't leak
    results_bag = {} 
    
    print(f"\nStarting structure: {strc}...")
    
    for N in N_range:
        # create graph
        adj, w = generator(strc, C, N, weight_dist=weight_dist)

        # run sim
        results = run_trial(adj, w, rng=seed)
        
        # 2. You don't need 'strc' in the key anymore since they are saved in separate files
        results_bag[N] = results 

        print(f"  Completed N={N}")

    # 3. Save ONCE per structure, outside the N loop, and add the .pkl extension
    file_path = os.path.join(save_dir, f'{strc}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(results_bag, f)
        
    print(f"Saved all results for {strc} to {file_path}")