from simulation import simulate_fission
from save_results import save_to_csv, get_run_id
from visualization import plot_results, plot_from_csv, plot_aggregate_from_csv, plot_keff_from_results, plot_aggregate_keff_from_csv
import random

def multiple_run():
    n=int(input("Enter the number of times you want to generate the simulation: "))
    for i in range(n):
        initial_neutrons = random.randint(1,5)
        steps = 20
        multiplication_constant = random.uniform(0.85*2.5, 1.15*2.5)
        filename = "data/simulation_results.csv"
        run_id = get_run_id(filename)
        seed = random.randint(0, 10000)
        # run simulation
        results = simulate_fission(initial_neutrons, multiplication_constant, steps, run_id, seed)

        # save to csv
        save_to_csv(results, filename, run_id, mode='a')
    print("Simulation generated successfully...")
    print(f"{n} new runs added.")

def main():
    initial_neutrons = random.randint(1,5)
    steps = 20
    multiplication_constant = random.uniform(0.85*2.5, 1.15*2.5)
    filename = "data/simulation_results.csv"
    run_id = get_run_id(filename)
    seed = random.randint(0, 10000)
    # run simulation
    results = simulate_fission(initial_neutrons, multiplication_constant, steps, run_id, seed)

    # save to csv
    save_to_csv(results, filename, run_id, mode='a')
    
    # visualize
    print("Showing live simulation plot...")
    plot_results(results, title=f"Nuclear Fission Simulation - Run {run_id}")

    print("Showing saved results for this run...")
    plot_from_csv(filename, run=run_id)

    print("Showing aggregate trend across all runs...")
    plot_aggregate_from_csv(filename)

    print(f"Showing K_eff plot for Run {run_id}...")
    plot_keff_from_results(results, run_id)

    print("Showing aggregate K_eff plot across all runs...")
    plot_aggregate_keff_from_csv(filename)


if __name__ == "__main__":
    print("Enter 0 to generate single run and view visual data based on that \n\tOR \nEnter 1 to generate multiple run and save it to train the model...")
    choice = int(input("Enter your choice: "))
    if choice == 0:
        main()
    elif choice == 1:
        multiple_run()
    else:
        print("Please select a proper choice number...(0/1)")
