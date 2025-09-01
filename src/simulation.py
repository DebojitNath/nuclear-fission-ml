import numpy as np


def simulate_fission(initial_neutrons=1, multiplication_constant=2.5, steps=10, run_id =1, seed=None):

    if seed is not None:
        np.random.seed(seed)

    results = []

    
    neutrons = initial_neutrons
    for step in range(steps):

        results.append({
            "run_id": run_id,
            "step": step,
            "neutrons": neutrons,
            "keff": -1 if step == 0 else (neutrons / prev_neutrons if prev_neutrons != 0 else 0)
        })

        prev_neutrons = neutrons


        base_survival = 0.8
        fluctuation = np.random.uniform(-0.1, 0.1)  
        crowding_factor = max(0, 1 - neutrons / 100)  
        survival_fraction = base_survival * crowding_factor + fluctuation
        survival_fraction = np.clip(survival_fraction, 0, 1)  

        neutrons = np.random.poisson(neutrons * multiplication_constant * survival_fraction)

    return results
