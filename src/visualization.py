import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results, title="Nuclear Fission Simulation"):
    df = pd.DataFrame(results)
    df.plot(x="step", y="neutrons", marker="o", title=title)
    plt.xlabel("Time Step")
    plt.ylabel("Number of Neutrons")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_from_csv(filename="data/simulation_results.csv", run=None):
    df = pd.read_csv(filename)

    if run is not None:
        df = df[df["run_id"] == run]
        df.plot(x="step", y="neutrons", marker="o", title=f"Run {run} Simulation")
    else:
        for r, group in df.groupby("run_id"):
            group.plot(x="step", y="neutrons", marker="o", label=f"Run {r}", legend=True)

    plt.xlabel("Time Step")
    plt.ylabel("Number of Neutrons")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_aggregate_from_csv(filename="data/simulation_results.csv"):
    df = pd.read_csv(filename)

    
    for run_id, group in df.groupby("run_id"):
        plt.plot(group["step"], group["neutrons"], color="gray", alpha=0.3)

    
    agg = df.groupby("step")["neutrons"].mean().reset_index()

    
    plt.plot(agg["step"], agg["neutrons"], marker="s", color="purple", linewidth=2,
             label="Average Neutrons")

    
    plt.title("Average Neutron Population Across Runs")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Neutrons")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_keff(results):
    neutrons = []
    for r in results:
        neutrons.append(r["neutrons"])

    keff = []
    for i in range(1, len(neutrons)):
        if neutrons[i-1] == 0:
            keff.append(0)
        else:
            keff.append(neutrons[i] / neutrons[i-1])
    return keff

def plot_keff_from_results(results, run_id):
    keff = compute_keff(results)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(keff)+1), keff, marker='s', linestyle='-', color='red')
    plt.title(f"K_eff per Step - Run {run_id}")
    plt.xlabel("Time Step")
    plt.ylabel("K_eff")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_aggregate_keff_from_csv(filename="data/simulation_results.csv"):
    df = pd.read_csv(filename)
    agg_keff = []


    for run_id, group in df.groupby("run_id"):
        neutrons = group["neutrons"].tolist()
        keff = []
        for i in range(1, len(neutrons)):
            if neutrons[i-1] == 0:
                keff.append(0)
            else:
                keff.append(neutrons[i] / neutrons[i-1])
        agg_keff.append(keff)


    max_len = 0
    for j in agg_keff:
        if len(j) > max_len:
            max_len = len(j)


    keff_matrix = []
    for k in agg_keff:
        x = k + [float('nan')] * (max_len - len(k))
        keff_matrix.append(x)

    keff_df = pd.DataFrame(keff_matrix)
    mean_keff = keff_df.mean(axis=0)

    plt.figure(figsize=(8,5))


    for k in agg_keff:
        plt.plot(range(1, len(k)+1), k, color="gray", alpha=0.3)

    plt.plot(range(1, len(mean_keff)+1), mean_keff, marker='o', color='purple', label="Mean K_eff")

    plt.legend()
    plt.title("Aggregate K_eff Across All Runs")
    plt.xlabel("Time Step")
    plt.ylabel("Average K_eff")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
