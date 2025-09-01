import pandas as pd
import os
def classify_run(keff):
    if keff == -1:
        return "initial"
    elif 0<= keff < 0.95:
        return "extinguished"     
    elif 0.95 <= keff <= 1.05 :
        return "equilibrium"      
    else:
        return "supercritical"


def gen_ml_dataset(raw_file = "data/simulation_results.csv", ml_file = "data/ml_dataset.csv"):
    if not os.path.exists(raw_file):
        print(f"[ERROR] Raw data file not found: {raw_file}")
        return
    
    df = pd.read_csv(raw_file)

    df = df.sort_values(by=["run_id", "step"]).reset_index(drop=True)

    if os.path.exists(ml_file):
        ml_df = pd.read_csv(ml_file)
        done_runs = set(ml_df["run_id"].unique())
    else:
        ml_df = pd.DataFrame()
        done_runs = set()

    new_runs = df[~df["run_id"].isin(done_runs)].copy()

    if not new_runs.empty:

        new_runs["prev_neutrons"] = new_runs.groupby("run_id")["neutrons"].shift(1)
        new_runs["prev_keff"] = new_runs.groupby("run_id")["keff"].shift(1)
        new_runs["next_neutrons"] = new_runs.groupby("run_id")["neutrons"].shift(-1)
        new_runs["next_keff"] = new_runs.groupby("run_id")["keff"].shift(-1)



        new_runs = new_runs.fillna(-1)
        new_runs["classification"] = new_runs["keff"].apply(classify_run)
    

        cols = [
            "run_id",
            "step",
            "prev_neutrons",
            "neutrons",
            "next_neutrons",
            "prev_keff",
            "keff",
            "next_keff",
            "classification"
        ]

        new_runs = new_runs[cols]
        

        if ml_df.empty:
            ml_df = new_runs
        else:
            ml_df = pd.concat([ml_df, new_runs], ignore_index=True)

        ml_df.to_csv(ml_file, index=False)
        
        print(f" Added {len(new_runs)} rows from {len(new_runs['run_id'].unique())} new runs...")
    else:
        print("No new runs to add...")




if __name__ == "__main__":
    gen_ml_dataset()