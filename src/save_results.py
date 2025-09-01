import csv
import os

def get_run_id(filename):
    if not os.path.exists(filename):
        return 1  
    
    with open(filename, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return 1
        else:
            return int(rows[-1]["run_id"])+1
        '''
        last_run = 0
        for i in rows:
            if i.get("run_id"):
                run_id = int(i["run_id"])
                if run_id > last_run:
                    last_run = run_id
        return last_run + 1
        '''
 

def save_to_csv(results, filename, run_id, mode="a"):
    file_exists = os.path.exists(filename)

    with open(filename, mode, newline="") as file:
        fieldnames = ["run_id", "step", "neutrons", "keff"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or mode == "w":
            writer.writeheader()

        for row in results:
            writer.writerow({
                "run_id": run_id,
                "step": row["step"],
                "neutrons": row["neutrons"],
                "keff": row["keff"]
            })